from astropy.time import Time
#use a snolab image and assume this time is valid for pixel times
from scipy.stats import binom
import numpy as np
from numba import njit
from scipy.stats import poisson
from numpy.lib.stride_tricks import as_strided

import os as _os

# Default detected-dipole count (legacy `dipole_coord_list.npz`). Used only as a
# fallback for callers that don't pass an explicit count or a coord-list file;
# the campaign and run_ccd_simulation always derive the true count from disk.
DEFAULT_N_DETECTED_TRAPS = 5171
TRAP_TRANSPORT_MODEL = 'phase_limited_v1v3'


def coord_list_for_tauhist(tauhistfile):
    """Map a tau-histogram filename to the matching dipole coordinate-list file.

    The detected-dipole count that seeds the baseline trap population lives in
    `dipole_coord_list{suffix}.npz` (written by run_charge_traps.py), where the
    only suffix that changes the raw dipole finding is `_minimal` (the minimal
    pipeline). The `_caldet` detection tag and the `_upper_limit` / threshold
    tags on the histogram do NOT change which dipoles were detected, so they are
    ignored here. Returns a path in the same directory as `tauhistfile`.
    """
    directory = _os.path.dirname(tauhistfile)
    base = _os.path.basename(tauhistfile)
    suffix = '_minimal' if 'minimal' in base else ''
    return _os.path.join(directory, f'dipole_coord_list{suffix}.npz')


def detected_trap_count(coord_list_path):
    """Number of detected dipoles in a `dipole_coord_list*.npz` file.

    Matches run_charge_traps.py:266 `total = sum(len(q_list) for q_list ...)` --
    the per-quadrant coordinate arrays summed across quadrants.
    """
    with np.load(coord_list_path, allow_pickle=True) as d:
        return int(sum(len(d[k]) for k in d.files))


def windowSum(data, windowsize, ndim):
    """Calculates the sliding-window sum of the N-dimensional data array."""
    window = (windowsize,) * ndim
    s = window + tuple(np.subtract(data.shape, window) + 1)
    return as_strided(data, shape=s, strides=data.strides * 2).sum(axis=tuple(np.arange(ndim)))

def expandWindow(badX, windowsize):
    """Expands coordinates of bad sliding windows into coordinates of all cells within."""
    ndim = len(badX)
    badX = np.transpose(badX)
    offsets = np.transpose(np.ones((windowsize,) * ndim).nonzero())
    expanded = np.concatenate([badX + offset for offset in offsets])
    if len(expanded) > 0:
        expanded = np.unique(expanded, axis=0)
    return tuple(expanded.T)

# def rateModel(shape, multiplier=1.0):
#     """Worst-case event rate distribution (adapted for 1D column arrays)."""
#     nX = shape[0]
#     uniform = np.full(shape, multiplier)
#     linX = np.linspace(0.0, 2.0, nX)
#     return np.maximum(uniform, linX)

# def rateModel(shape, multiplier=1):
#     """
#     return an array representing the worst-case event rate distribution across the cell array.
#     there are three cases: uniform, linear X-dependence, linear Y-dependence
#     linear X-dependence happens when spurious charge dominates
#     linear Y-dependence happens when dark current dominates
#     not clear this is the correct approach for 2+e events, but statistics are lower there so it doesn't really matter
#     multiplier: scale factor to apply to the uniform model, for a weaker cut
#     """
#     nX = shape[0]
#     uniform = np.full(shape, multiplier)
#     linX = np.linspace(0.0, 2.0, nX)
#     if len(shape)==1:
#         return np.maximum(uniform,linX)
#     else:
#         nY = shape[1]
#         linX = linX[:,np.newaxis]*np.ones((1,nY))
#         linY = np.linspace(0.0, 2.0, nY)
#         linY = linY[np.newaxis,:]*np.ones((nX,1))
#         return np.maximum.reduce([uniform, linX, linY])

def rateModel(shape, multiplier=1.0, mode="uniform"):
    """
    Event rate distribution model.
    mode="uniform": Flat background rate (removes bias for uniform simulations).
    mode="linear": Mimics SENSEI gradient where the right side of the CCD is noisier.
    """
    import numpy as np
    
    # Ensure shape is a tuple (findBadCells sometimes passes it differently)
    if isinstance(shape, int):
        shape = (shape,)
        
    # Create the unbiased flat base array
    uniform_array = np.full(shape, float(multiplier))
    
    if mode == "uniform":
        return uniform_array
        
    elif mode == "linear":
        # The X-axis (columns) is always the last dimension in both 1D and 2D shapes
        nX = shape[-1] 
        linX = np.linspace(0.0, 2.0, nX)
        
        # If the shape is 2D (rows, cols), broadcast the 1D gradient across all rows
        if len(shape) == 2:
            linX = np.broadcast_to(linX, shape)
            
        return np.maximum(uniform_array, linX * multiplier)
        
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'uniform' or 'linear'.")

def calculatePvals(goodCells, cellArrayList, windowsize=1):
    cellShape = cellArrayList[0].shape[:-1]
    nDim = len(cellShape)
    badCellsMask = np.ones(cellShape, dtype=bool)
    badCellsMask[goodCells] = 0
    cellShape = tuple(np.subtract(cellShape, windowsize) + 1)

    pvalList = np.zeros((len(cellArrayList),) + cellShape)
    rateList = np.zeros(len(cellArrayList))
    
    for i, cellArray in enumerate(cellArrayList):
        if windowsize == 1:
            denomArray = cellArray[..., 0]
            hitsArray = cellArray[..., 1]
            goodDenom = denomArray[goodCells].sum()
            goodHits = hitsArray[goodCells].sum()
        else:
            cellCopy = cellArray.copy()
            cellCopy[badCellsMask, :] = 0
            denomArray = cellCopy[..., 0]
            hitsArray = cellCopy[..., 1]
            goodDenom = denomArray.sum()
            goodHits = hitsArray.sum()
            if windowsize > 1:
                denomArray = windowSum(denomArray, windowsize, nDim)
                hitsArray = windowSum(hitsArray, windowsize, nDim)

        rateGood = goodHits / goodDenom if goodDenom > 0 else 0.0
        rateList[i] = rateGood
        nExpected = denomArray * rateModel(denomArray.shape) * rateGood
        pvalList[i] = poisson.sf(hitsArray - 0.5, nExpected)
        
    return rateList, pvalList

def addNeighbors(pvals, goodCells, badX, addCut):
    badX = np.reshape(badX, -1)
    for iseed in badX:
        for iadd in range(iseed - 1, -1, -1):
            if not goodCells[iadd] or pvals[iadd] > addCut: break
            goodCells[iadd] = False
        for iadd in range(iseed + 1, len(goodCells)):
            if not goodCells[iadd] or pvals[iadd] > addCut: break
            goodCells[iadd] = False

def find_very_hot_columns(unmasked_per_img, max_possible_unmasked, nHDUs=1):
    """Exact translation of the ROOT TH2 'very-hot col' threshold logic."""
    n_images, cols = unmasked_per_img.shape
    pcut = 0.5 / (nHDUs * cols)
    vhot_cols = []
    
    if n_images > 0:
        pv = np.exp(np.log(pcut) / n_images)
        # Recreate the 1D projection of the ROOT TH2 histogram (counts of unmasked pixels)
        coll, _ = np.histogram(unmasked_per_img.flatten(), bins=np.arange(max_possible_unmasked + 2))
        tot_coll = np.sum(coll)
        
        if tot_coll > 0:
            cumsum_prob = np.cumsum(coll / tot_coll)
            fcut_idx = np.nonzero(cumsum_prob > pv)[0]
            fcut = fcut_idx[0] if len(fcut_idx) > 0 else 0
            
            # Find columns where NO image has >= fcut unmasked pixels
            max_unmasked_per_col = np.max(unmasked_per_img, axis=0)
            vhot_cols = np.nonzero(max_unmasked_per_col < fcut)[0]
            

    return vhot_cols


def find_very_hot_pixels(pix_denom, n_images, nHDUs=1):
    """
    Finds pixels that are masked significantly more often than expected.
    Translates the very-hot pix threshold logic using binomial statistics.
    """
    if n_images == 0: 
        return ()
        
    rows, cols = pix_denom.shape
    nPix = rows * cols
    
    meanDenom = pix_denom.mean()
    pDenom = meanDenom / n_images
    pcut = 0.5 / (nHDUs * nPix)
    
    from scipy.stats import binom
    veryhotThresh = binom.ppf(q=pcut, n=n_images, p=pDenom)
    
    # Return as a tuple of arrays so it can be passed into goodCells index directly
    throw = np.nonzero(pix_denom < veryhotThresh)
    return throw 

def merge_hot_pixels_to_columns(bad_pix_list, n_rows):
    """
    Merges hot pixels into hot columns if they exceed density or spread thresholds.
    """
    maxHotPix = max(10, 0.05 * n_rows)
    maxHotPixRange = max(10, 0.1 * n_rows)
    
    pix2col = {}
    for r, c in bad_pix_list:
        if c not in pix2col:
            pix2col[c] = set()
        pix2col[c].add(r)
        
    new_bad_cols = []
    filtered_bad_pix = set(bad_pix_list)
    
    for c, rSet in pix2col.items():
        rList = sorted(list(rSet))
        # Condition: >3 pixels widely spread, OR > maxHotPix limit
        if (len(rList) > 2 and rList[-1] - rList[0] > maxHotPixRange) or len(rList) > maxHotPix:
            new_bad_cols.append(c)
            for r in rList:
                filtered_bad_pix.remove((r, c))
                
    return new_bad_cols, list(filtered_bad_pix)

def findBadCells(data, nCells, already_bad=None, nHDUs=1, doChunkCut=False):
    cellArrayList, nameList = zip(*data)
    
    # FIX: For 1e events, hotcol.py scales purely by nCells, not (nCells * nHDUs)
    pScales = [nCells] * len(data)  
    
    pCut = 0.5
    chunkCut = 0.5
    addCut = 5e-2
    nChunks = 16

    cellShape = cellArrayList[0].shape[:-1]
    goodCells = np.full(cellShape, True)
    if already_bad is not None and len(already_bad) > 0:
        goodCells[already_bad] = False

    nDim = len(cellShape)

    while True:
        while True:
            oldGood = goodCells.copy()
            for windowsize in range(1, 6):
                rateList, pvalList = calculatePvals(goodCells, cellArrayList, windowsize)
                pvals = np.min([pval * pscale for pval, pscale in zip(pvalList, pScales)], axis=0)
                badX = (pvals < pCut).nonzero()
                badX = expandWindow(badX, windowsize)
                goodCells[badX] = False

            rateList, pvalList = calculatePvals(goodCells, cellArrayList)
            pvals = np.min(pvalList, axis=0)
            if nDim == 1:
                badX = np.logical_not(goodCells).nonzero()
                addNeighbors(pvals, goodCells, badX, addCut)

            if np.count_nonzero(goodCells != oldGood) == 0: break

        if nDim == 1 and doChunkCut:
            # Chunk processing logic
            chunkPlist = np.zeros((len(cellArrayList), nChunks))
            for i, cellArray in enumerate(cellArrayList):
                rates = rateModel(cellArray[:, 0].shape) * rateList[i]
                chunks = zip(*[np.array_split(a, nChunks) for a in [cellArray, goodCells, rates]])
                for iChunk, (cData, cGood, cRates) in enumerate(chunks):
                    chunkPlist[i, iChunk] = poisson.sf(cData[cGood, 1].sum() - 0.5, (cData[cGood, 0] * cRates[cGood]).sum())
            
            chunkPvals = np.min([pval * pscale for pval, pscale in zip(chunkPlist, pScales)], axis=0)
            badChunks = (chunkPvals < chunkCut).nonzero()[0]
            if len(badChunks) == 0: break
            for iChunk in badChunks:
                np.copyto(np.array_split(goodCells, nChunks)[iChunk], False)
        else:
            break

    badX = np.transpose(np.logical_not(goodCells).nonzero())
    return badX.ravel() if badX.shape[1] == 1 else list(map(tuple, badX)), goodCells, pvalList


def hole_thermal_velocity(temperatures):
    """v_th = sqrt(3 k_B T / m_cond) for holes in cm/s.

    Constants match dipole.log_energy_cross_section (p-channel hole
    conductivity effective mass 0.41 m_e for 100-200 K)."""
    kb = 8.617333262e-5  # eV/K
    me = 0.510998950e6   # eV
    m_cond = 0.41 * me
    return 2.99792458e10 * np.sqrt(3 * kb * temperatures / m_cond)


def pixel_time(nsamp, delayH, delayIped, delayIsig, delaySW, delayRG, delayOG):		# Time to read 1 pixel
  return 5*int(delayH)+int(delayRG)+(int(delayIped)+int(delaySW)+int(delayIsig)+int(delayOG)+int(delayRG))*int(nsamp)

def pixel_time_vertical(nsamp,ncol, delayH, delayIped, delayIsig, delaySW, delayRG, delayOG,delayDG):		# Time to read 1 pixel
  pixel_time_hor=pixel_time(nsamp, delayH, delayIped, delayIsig, delaySW, delayRG, delayOG)
  return (pixel_time_hor+(int(delaySW)+int(delayDG)))*int(ncol)


def transplant_clusters(source_image, target_shape=(520, 3200), 
                                    count_threshold=20, max_aspect_ratio=3.0, 
                                    radius=20,exposure=72000,scale_factor = 1):
    from skimage.morphology import dilation, disk
    from scipy import ndimage
    from scipy.ndimage import distance_transform_edt
    import numpy as np
    
    # 1. Label and Measure
    labeled_array, num_features = ndimage.label(source_image > 0)
    cluster_slices = ndimage.find_objects(labeled_array)
    cluster_sums = ndimage.sum(source_image, labeled_array, index=np.arange(1, num_features + 1))
    
    new_image = np.zeros(target_shape, dtype=source_image.dtype)
    img_h, img_w = new_image.shape
    src_h, src_w = source_image.shape
    rng = np.random.default_rng()
    
    totClusters = num_features
    # Calculate how many we WANT to process
    target_num_clusters = int(np.round(scale_factor * (exposure / 72000) * num_features))
    
    # print(f"Exposure: {exposure}, Inserting approx {target_num_clusters} clusters out of {totClusters}")

    # --- FIX START: Randomize the selection ---
    # Create a list of all valid IDs (1 to num_features)
    all_label_ids = np.arange(1, num_features + 1)
    
    # Shuffle them randomly
    rng.shuffle(all_label_ids)
    
    # Select only the first N IDs from the shuffled list
    selected_labels = all_label_ids[:target_num_clusters]
    # --- FIX END ---

    # 2. Loop through the RANDOMLY SELECTED clusters
    for label_id in selected_labels:
        
        # --- FILTERS (Count & Shape) ---
        # Note: adjust index by -1 because cluster_sums is 0-indexed relative to labels
        if cluster_sums[label_id - 1] <= count_threshold:
            continue

        sl = cluster_slices[label_id - 1]
        y_slice, x_slice = sl
        height = y_slice.stop - y_slice.start
        width = x_slice.stop - x_slice.start
        
        if min(height, width) == 0: continue
        if max(height, width) / min(height, width) > max_aspect_ratio:
            continue

        cluster_mask = (labeled_array[sl] == label_id)
        
        # Find the maximum horizontal streak in any single row
        max_streak = np.max(np.sum(cluster_mask, axis=1))
        
        # Calculate how "solid" the cluster is compared to its bounding box
        fill_factor = np.sum(cluster_mask) / (height * width)
        
        # If it has a long horizontal line (> 30 pixels) AND is mostly empty space (< 50% solid),
        # it is a serial register hit attached to another event. Reject it entirely.
        if max_streak > 30 and fill_factor < 0.5:
            continue

        # ... (Rest of your code remains exactly the same) ...
        # ... EXPAND SLICE ...
        y_start_expanded = max(0, y_slice.start - radius)
        y_stop_expanded  = min(src_h, y_slice.stop + radius)
        x_start_expanded = max(0, x_slice.start - radius)
        x_stop_expanded  = min(src_w, x_slice.stop + radius)
        
        expanded_slice = (slice(y_start_expanded, y_stop_expanded), 
                          slice(x_start_expanded, x_stop_expanded))
        
        local_block = source_image[expanded_slice]
        if np.max(local_block) <= 100:
            continue

        local_labels = labeled_array[expanded_slice]
        
        # cluster_core_mask = (local_labels == label_id)
        cluster_core_mask = (local_labels == label_id) & (local_block > 100)
        # 1. Compute distance from the cluster (invert mask so cluster is 0)
        # 'distance_transform_edt' calculates distance to the nearest ZERO pixel.
        # So we invert: Cluster becomes False (0), Background becomes True (1).

        dist_map = distance_transform_edt(~cluster_core_mask)

        # 2. Threshold by radius to create the mask
        dilated_mask = dist_map <= radius

        foreign_clusters_mask = (local_labels > 0) & (local_labels != label_id)

        final_cluster_chunk = local_block * dilated_mask
        final_cluster_chunk[foreign_clusters_mask] = 0

        
     
        
        
        # --- PLACEMENT ---
        h, w = final_cluster_chunk.shape
        placed = False
        attempts = 0
        
        while not placed and attempts < 100:
            max_r = img_h - h
            max_c = img_w - w
            
            if max_r < 0 or max_c < 0: break

            r = rng.integers(0, max_r)
            c = rng.integers(0, max_c)
            
            target_area = new_image[r:r+h, c:c+w]
            
            if not np.any((target_area > 0) & (final_cluster_chunk > 0)):
                new_image[r:r+h, c:c+w] += final_cluster_chunk
                placed = True
            
            attempts += 1
            
    return new_image



def unbin_counts_conservative(binned_data):
    import numpy as np
    """
    Unbins a (rows, cols) array into (rows*32, cols) while:
    1. Conserving total counts (Sum unbinned == Original pixel).
    2. Keeping all values as integers.
    3. Randomly distributing remainders to avoid systematic bias.
    """
    rows, cols = binned_data.shape
    expansion = 32
    
    # 1. Calculate Base and Remainder
    # q: The base integer value for every sub-pixel
    # r: The number of extra '+1's we need to sprinkle in
    q = binned_data // expansion
    r = binned_data % expansion

    # 2. Create the Base Array
    # Repeat the quotient 32 times.
    # Shape becomes (20, 32, 3200) for easier manipulation
    unbinned_view = np.repeat(q[:, np.newaxis, :], expansion, axis=1)

    # 3. Generate Random Masks for Remainders
    # We need to add 1 to exactly 'r' pixels in each column-block.
    # We generate random noise, sort it, and select the top 'r' indices.
    
    # Create random noise (20, 32, 3200)
    rng = np.random.default_rng()
    noise = rng.random(unbinned_view.shape)
    
    # argsort along the expansion axis gives us random indices 0..31
    # We want to identify the slots that should receive the extra counts.
    # A simple way: Rank the noise. If the rank is < remainder, add 1.
    random_ranks = np.argsort(noise, axis=1)
    
    # Broadcast 'r' to match the shape (20, 1, 3200) for comparison
    r_broadcast = r[:, np.newaxis, :]
    
    # Create the boolean mask where we add the extra count
    # This selects exactly 'r' random locations per block
    add_mask = random_ranks < r_broadcast
    
    # 4. Apply and Reshape
    final_data = unbinned_view + add_mask.astype(int)
    
    # Flatten the middle dimension to get final (640, 3200)
    return final_data.reshape(rows * expansion, cols)


def inject_single_e(image, n_events=100, intensity=1, exclusion_mask=None):
    import numpy as np
    """
    Injects 'n' single-pixel events into the image.
    If 'exclusion_mask' is provided, events will NOT be placed where the mask is True.
    """
    rows, cols = image.shape
    rng = np.random.default_rng()
    
    if exclusion_mask is None:
        # Original behavior: Randomly select from the whole image
        random_rows = rng.integers(0, rows, size=n_events)
        random_cols = rng.integers(0, cols, size=n_events)
        np.add.at(image, (random_rows, random_cols), intensity)
    else:
        # New behavior: Select only from valid (unmasked) pixels
        
        # 1. Find coordinates of all safe pixels (where mask is False)
        # np.where returns a tuple of arrays (rows, cols)
        valid_rows, valid_cols = np.where(~exclusion_mask)
        
        n_valid = len(valid_rows)
        
        if n_valid == 0:
            print("Warning: No valid pixels available for injection! (Mask covers entire image)")
            return image
            
        # 2. Randomly select indices from the list of valid coordinates
        chosen_indices = rng.integers(0, n_valid, size=n_events)
        
        # 3. Retrieve the specific row/col coordinates for those indices
        r_coords = valid_rows[chosen_indices]
        c_coords = valid_cols[chosen_indices]
        
        # 4. Inject intensity
        np.add.at(image, (r_coords, c_coords), intensity)
    
    return image



def generate_halo_mask(image, threshold=100, radius=60):
    from scipy.ndimage import distance_transform_edt
    
    # 1. Find pixels that exceed the threshold
    hot_pixels = image > threshold
    
    # 2. Compute distance from every pixel to the nearest hot pixel
    # distance_transform_edt calculates the distance from non-zero pixels 
    # to the nearest zero pixel.
    # Therefore, we INVERT the mask: 
    #   - Hot pixels become 0 (Targets)
    #   - Background becomes 1 (Source)
    # The result is a map where every pixel contains its distance to the nearest hot pixel.
    dist_map = distance_transform_edt(~hot_pixels)
    
    # 3. Threshold the distance map
    # Pixels with distance <= radius are within the halo.
    mask = dist_map <= radius
    
    return mask







import numpy as np

def generate_column_bleed_mask(image, threshold=100, direction='down'):
    """
    Creates a mask flagging all pixels in the same column 'above' a high energy pixel.
    
    Parameters:
    -----------
    image : np.ndarray
        2D array of counts.
    threshold : float
        The count value above which a pixel is considered high energy.
    direction : str, 'up' or 'down'
        'up': Masks from the hot pixel towards row 0 (index 0).
        'down': Masks from the hot pixel towards the last row (index N).
        
    Returns:
    --------
    mask : np.ndarray (bool)
        Boolean mask where True indicates a masked pixel.
    """
    # 1. Identify the high energy pixels
    hot_pixels = image > threshold
    
    # 2. Vectorized directional masking using Cumulative Sums
    if direction == 'down':
        # Once a column hits a True, cumsum makes all subsequent rows > 0
        mask = np.cumsum(hot_pixels, axis=0) > 0
        
    elif direction == 'up':
        # Reverse the image vertically, apply the cumsum, and reverse it back.
        # This masks everything from the hot pixel up to row 0.
        mask = np.cumsum(hot_pixels[::-1, :], axis=0)[::-1, :] > 0
        
    else:
        raise ValueError("Direction must be 'up' or 'down'")
        
    return mask


def get_cluster(proc, min_pixs=2, max_pixs=None, min_total_value=3, max_total_value=None, return_count=False, return_median=False):
    import numpy as np
    from cv2 import connectedComponents
    
    m = proc > 0.7
    
    # connectedComponents returns the number of labels including the background (label 0)
    num_labels, clusters = connectedComponents(m.astype(np.uint8), connectivity=8)
    max_index = num_labels - 1
    
    if max_index == 0:
        if return_count: return 0
        if return_median: return np.nan
        return np.zeros_like(proc, dtype=bool)

    # --- OPTIMIZATION: Replace slow binned_statistic with ultra-fast np.bincount ---
    clusters_flat = clusters.ravel()
    values_flat = proc.ravel()
    
    # bincount calculates sums and counts instantly. 
    # Index 0 is background, so we slice [1:] to get labels 1 through max_index
    counts = np.bincount(clusters_flat)[1:]
    sums = np.bincount(clusters_flat, weights=values_flat)[1:]
    
    # The boolean conditions remain exactly the same
    conditions = (counts >= min_pixs) & (sums >= min_total_value)
    if max_pixs is not None:
        conditions &= (counts <= max_pixs)
    if max_total_value is not None:
        conditions &= (sums <= max_total_value)
        
    if return_count:
        return np.count_nonzero(conditions)
    if return_median:
        valid_sums = sums[conditions]
        if len(valid_sums) == 0:
            return np.nan
        return np.median(valid_sums)
        
    # Find valid label IDs (Add 1 because we sliced off the 0 background label earlier)
    valid_labels = np.nonzero(conditions)[0] + 1
    
    layer = np.isin(clusters, valid_labels)
    return layer

# def get_cluster(proc, min_pixs=2, max_pixs=None, min_total_value=3,max_total_value = None, return_count = False, return_median= False):#, min_max_value=None
#     from scipy.stats import binned_statistic
#     import numpy as np
#     from cv2 import connectedComponents
    
#     m = proc > 0.7
    
#     # connectedComponents returns the number of labels including the background (label 0)
#     num_labels, clusters = connectedComponents(m.astype(np.uint8))
    
#     # Subtract 1 to ignore the background
#     max_index = num_labels - 1
    
#     # --- FIX: Handle case with no clusters ---
#     if max_index == 0:
#         if return_count:
#             return 0
#         if return_median:
#             return np.nan # Or 0, depending on your preference
#         return np.zeros_like(proc, dtype=bool) # Return empty mask
#     # -----------------------------------------

#     all_clusters = clusters.flatten()
#     all_values = proc.flatten()
    
#     counts, bins, binned = binned_statistic(all_clusters, all_values, 'count', bins=max_index,
#                                             range=(1, max_index + 1))
#     sums, bins, binned = binned_statistic(all_clusters, all_values, 'sum', bins=max_index, range=(1, max_index + 1))
    
#     bins = bins[:-1]
    
#     if max_pixs is not None:
#         conditions = (counts >= min_pixs) & (sums >= min_total_value) & (counts <= max_pixs)
#     else: 
#         conditions = (counts >= min_pixs) & (sums >= min_total_value)
        
#     if max_total_value is not None:
#         conditions = conditions & (sums <= max_total_value)
        
#     if return_count:
#         return np.count_nonzero(conditions)
#     if return_median:
#         # Check if conditions is not empty to avoid warnings/errors
#         valid_sums = sums[conditions]
#         if len(valid_sums) == 0:
#             return np.nan
#         return np.median(valid_sums)
        
#     cond_bins = bins[conditions]

#     layer = np.isin(clusters, cond_bins)
#     return layer
        

def ratio_pixels(m, yx, action = None):
    import numpy as np
    arr = None
    if type(yx) == int and yx in [8, 9]:
        arr = [
            (0, 0),
            (1, 0),
            (-1, 0),
            (1, 1),
            (- 1, - 1),
            (0, - 1),
            (0, 1),
            (1, - 1),
            (- 1, + 1)
        ]
        if yx == 8:
            arr = arr[1:]  # drop [0,0]
    if type(yx) is dict:
        radii = yx['radius']
        from itertools import permutations
        arr = list(permutations(list(range(-radii, radii + 1)) * 2, 2))

    if type(yx) not in [int, dict]:
        if len(yx) != 2 or type(yx[0]) in [np.ndarray, list]:
            arr = yx

    if arr is not None:
        comb = [ratio_pixels(m, xy) for xy in arr]
        if action is None:
            return comb
        from functools import reduce
        if action in ['|', 'or', 'OR']:
            return reduce(lambda a, b: a | b, comb)
        if action in ['&', 'and', 'AND']:
            return reduce(lambda a, b: a & b, comb)
        if action in ['+']:
            return reduce(lambda a, b: a + b, comb)
        if action in ['-']:
            return reduce(lambda a, b: a - b, comb)
        if action in ['stack into 3d matrix', '3d']:
            return np.dstack(comb)

    m = np.array(m)
    ry, rx = yx
    from numpy import vstack, hstack
    def extra(line):
        line = line.copy()
        if line.dtype == bool:
            line[:] = False
        else:
            line[:] = -1
        return line
    rm2 = m
    if ry < 0:
        rm2 = vstack([extra(np.zeros((abs(ry), rm2.shape[1]), dtype=rm2.dtype)), rm2])
    elif ry > 0:
        rm2 = m[ry:]
    if rx < 0:
        rm2 = hstack([extra(np.zeros((rm2.shape[0], abs(rx)), dtype=rm2.dtype)), rm2])
    if rx > 0:
        rm2 = rm2[:, rx:]
    rm2 = rm2[:m.shape[0], :m.shape[1]]
    if rm2.shape[0] < m.shape[0]:
        rm2 = vstack([rm2, extra(np.zeros((m.shape[0] - rm2.shape[0], rm2.shape[1]), dtype=rm2.dtype))])
    if rm2.shape[1] < m.shape[1]:
        rm2 = hstack([rm2, extra(np.zeros((rm2.shape[0], m.shape[1]-rm2.shape[1]), dtype=rm2.dtype))])
    if rm2 is m:
        rm2 = m.copy()
    return rm2

@njit
def seed_numba(seed_value):
    """Seeds Numba's internal PRNG. Must be called from inside a JIT function."""
    np.random.seed(seed_value)

@njit
def fast_readout_numba(
    image,
    exp_acc,
    tpix_vertical,
    trap_rows,
    trap_cols,
    trap_emit_probs,
    trap_capture_alpha,
    trap_is_v3,
    trapped_charge,
):
    """
    JIT-compiled C-speed readout loop using a stationary padded buffer.

    Phase-limited V1/V3 trap transport: emission is allowed across the full row
    dwell, while capture and recapture are allowed only during one short phase
    overlap window. ``trap_emit_probs`` is 1-exp(-t_row/tau), and
    ``trap_capture_alpha`` is kc*t_phase, so P_capture(q) = 1-exp(-q*alpha).

    ``trap_is_v3`` splits the catalog by clock phase. A V3 trap is crossed by
    the packet on its way OUT of the row, after collecting the dwell's
    emission, so its own emitted carrier faces a same-step recapture roll. A
    V1 trap is crossed on the way IN: capture is checked on the arriving
    packet before the dwell's emission, and an emitted carrier exits over V3
    without recrossing the trap — it always escapes.
    """
    rows, cols = image.shape
    n_traps = len(trap_emit_probs)

    # 1. Pad image with empty rows at the top to simulate empty space shifting in
    padded_image = np.zeros((2 * rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            padded_image[r + rows, c] = image[r, c]

    output_stream = np.zeros(rows * cols, dtype=np.float64)
    out_idx = 0

    for t in range(rows):
        # 2. Readout the row that has reached the bottom
        read_row = 2 * rows - 1 - t
        for c in range(cols - 1, -1, -1):
            output_stream[out_idx] = padded_image[read_row, c]
            out_idx += 1

        # 3. Trap interactions on the newly shifted pixels
        for i in range(n_traps):
            tr = trap_rows[i]
            tc = trap_cols[i]

            # The pixel that just shifted into the trap's physical row 'tr'
            current_pixel_row = tr + rows - 1 - t

            if trap_is_v3[i] == 1:
                # V3: emit over the dwell, then one capture/recapture check as
                # the packet (incl. any just-emitted carrier) crosses on exit.
                if trapped_charge[i] > 0.0:
                    if np.random.random() < trap_emit_probs[i]:
                        padded_image[current_pixel_row, tc] += 1.0
                        trapped_charge[i] = 0.0

                q = padded_image[current_pixel_row, tc]
                if trapped_charge[i] <= 0.0 and q >= 1.0:
                    p_capture = 1.0 - np.exp(-q * trap_capture_alpha[i])
                    if np.random.random() < p_capture:
                        padded_image[current_pixel_row, tc] -= 1.0
                        trapped_charge[i] = 1.0
            else:
                # V1: capture is checked on the arriving packet first; an
                # emission during the dwell then escapes with no same-step
                # recapture (the packet exits over V3, never recrossing V1).
                q = padded_image[current_pixel_row, tc]
                if trapped_charge[i] <= 0.0 and q >= 1.0:
                    p_capture = 1.0 - np.exp(-q * trap_capture_alpha[i])
                    if np.random.random() < p_capture:
                        padded_image[current_pixel_row, tc] -= 1.0
                        trapped_charge[i] = 1.0

                if trapped_charge[i] > 0.0:
                    if np.random.random() < trap_emit_probs[i]:
                        padded_image[current_pixel_row, tc] += 1.0
                        trapped_charge[i] = 0.0

    # 4. Update Exposure Accumulator
    # (Vectorized the O(R^2 * C) addition loop into O(R * C))
    for sr in range(rows):
        add_val = (rows - 1 - sr) * tpix_vertical
        if add_val > 0:
            for sc in range(cols):
                exp_acc[sr, sc] += add_val

    # 5. Update the CCD state for the next exposure
    for r in range(rows):
        for c in range(cols):
            image[r, c] = padded_image[r, c]

    return output_stream


@njit
def fast_clear_numba(
    image,
    fast_shifts,
    fast_dwell,
    slow_shifts,
    slow_dwell,
    trap_rows,
    trap_cols,
    trap_emit_probs_fast,
    trap_emit_probs_slow,
    trap_capture_alpha,
    trap_is_v3,
    trapped_charge,
):
    """
    Clock the active area through the two-block clear recipe.

    Clear transport uses the same phase-limited V1/V3 model as readout. Capture
    is limited to the short phase-overlap dwell for every vertical shift; the
    emission probability is set by the full fast or slow clear dwell. The
    per-trap phase split matches ``fast_readout_numba``: V3 traps get a
    same-step recapture check on their own emission, V1 traps do not.
    """
    rows, cols = image.shape
    n_traps = len(trap_capture_alpha)
    total_shifts = fast_shifts + slow_shifts

    # Leading empty packets clock into the active area. Keeping the charge
    # stream stationary in this padded buffer avoids shifting a dense image
    # on every clear step.
    padded_image = np.zeros((total_shifts + rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            padded_image[r + total_shifts, c] = image[r, c]

    for t in range(total_shifts):
        emit_probs = trap_emit_probs_fast if t < fast_shifts else trap_emit_probs_slow

        for i in range(n_traps):
            tr = trap_rows[i]
            tc = trap_cols[i]

            # Packet shifted into physical trap row tr on this clear step.
            current_pixel_row = tr + total_shifts - 1 - t

            if trap_is_v3[i] == 1:
                if trapped_charge[i] > 0.0:
                    if np.random.random() < emit_probs[i]:
                        padded_image[current_pixel_row, tc] += 1.0
                        trapped_charge[i] = 0.0

                q = padded_image[current_pixel_row, tc]
                if trapped_charge[i] <= 0.0 and q >= 1.0:
                    p_capture = 1.0 - np.exp(-q * trap_capture_alpha[i])
                    if np.random.random() < p_capture:
                        padded_image[current_pixel_row, tc] -= 1.0
                        trapped_charge[i] = 1.0
            else:
                q = padded_image[current_pixel_row, tc]
                if trapped_charge[i] <= 0.0 and q >= 1.0:
                    p_capture = 1.0 - np.exp(-q * trap_capture_alpha[i])
                    if np.random.random() < p_capture:
                        padded_image[current_pixel_row, tc] -= 1.0
                        trapped_charge[i] = 1.0

                if trapped_charge[i] > 0.0:
                    if np.random.random() < emit_probs[i]:
                        padded_image[current_pixel_row, tc] += 1.0
                        trapped_charge[i] = 0.0

    # Charge still resident in the active area immediately after the final
    # shift. CCD.simulate_clear records it, then the clear boundary discards
    # all free surface charge while retaining trap occupancy.
    for r in range(rows):
        for c in range(cols):
            image[r, c] = padded_image[r, c]


@njit
def drain_traps_empty_numba(trap_taus, trap_capture_alpha, trap_is_v3,
                            trapped_charge, dwell, n_shifts):
    """
    Drain traps through the empty-packet tail of a continuous clear.

    A V1 trap's emission escapes freely, so it drains with the recapture-free
    closed form 1-exp(-T/tau) over T = dwell*n_shifts. A V3 trap's emitted
    carrier joins the (empty) passing packet and faces the same-gate recapture
    roll 1-exp(-alpha) on exit, independent of shift speed, so only a fraction
    exp(-alpha) of emissions escape: in the fast-shift limit (dwell << tau) the
    escape is a thinned Poisson process with rate exp(-alpha)/tau, giving
    P_drain = 1-exp(-T*exp(-alpha)/tau).
    """
    total_dwell = dwell * n_shifts
    n_traps = len(trap_taus)
    for i in range(n_traps):
        if trapped_charge[i] > 0.0:
            if trap_is_v3[i] == 1:
                escape_rate = np.exp(-trap_capture_alpha[i]) / trap_taus[i]
            else:
                escape_rate = 1.0 / trap_taus[i]
            p_emit = 1.0 - np.exp(-total_dwell * escape_rate)
            if np.random.random() < p_emit:
                trapped_charge[i] = 0.0



class CCD:
    def __init__(
        self,
        tpix_horizontal,
        tpix_vertical,
        tau_weights,
        tau_edges,
        pair_tau135,
        pair_sigma,
        runconditions='minos',
        trap_density_scale=1.0,
        packet_volume_um3=3.0,
        phase_capture_ticks=300.0,
        temperature_K=135.0,
        exp_indep_charge_mode='pre_readout',
        clear_mode='sequencer',
        binning_0h_factor=32.0,
        binning=1.0,
        n_detected_traps=DEFAULT_N_DETECTED_TRAPS,
        zero_exp_dep_rate=False,
        v3_phase_fraction=0.5,
    ):
        import numpy as np
        # self.original_image = np.copy(image_array)
        # self.exposure_images = np.zeros_like(sample_image,dtype=float)
        # self.exposure_images = []
        self.tpix_horizontal = tpix_horizontal
        self.tpix_vertical = tpix_vertical
        self.runconditions = runconditions
        valid_exp_indep_charge_modes = ('pre_readout', 'post_readout')
        if exp_indep_charge_mode not in valid_exp_indep_charge_modes:
            raise ValueError(
                f"Unknown exp_indep_charge_mode={exp_indep_charge_mode!r}; "
                f"expected one of {valid_exp_indep_charge_modes}"
            )
        self.exp_indep_charge_mode = exp_indep_charge_mode
        valid_clear_modes = (
            'instantaneous', 'sequencer', 'three_hour', 'binned_0h'
        )
        if clear_mode not in valid_clear_modes:
            raise ValueError(
                f"Unknown clear_mode={clear_mode!r}; "
                f"expected one of {valid_clear_modes}"
            )
        self.clear_mode = clear_mode
        # 'binned_0h' data-taking strategy: never run a hardware clear; instead a
        # binned (and therefore faster-readout) 0 h image is taken after every
        # real exposure, both resetting the array and serving as the 0 h
        # baseline. Binning shortens the readout, so the per-row trap dwell for
        # those 0 h images is tpix_vertical / binning_0h_factor.
        self.binning_0h_factor = float(binning_0h_factor)
        # Global readout binning factor. The caller has already divided tpix /
        # tpix_vertical by this; it is stored only for HDF5 provenance and the
        # per-file consistency guard (binning changes the readout dwell, so two
        # different binnings must not share an output directory).
        self.binning = float(binning)
        if runconditions == 'snolab':
            print("Using snolab run conditions, assuming 10x fewer high energy events")

        # Raw images are omitted by default to save RAM but can be explicitly requested
        self.reconstructed_images = [] 
        self.no_trap_images = []
        self.trap_bitmasks = []
        self.notrap_bitmasks = []
        self.exposures = []

        self.single_e_counts = []
        self.single_e_counts_no_traps = []
        self.single_e_counts_masked = []
        self.single_e_counts_masked_no_traps = []
        
        self.unmasked_pixels = []
        self.unmasked_pixels_no_traps = []

        # temp_scan_run1_clearseq.xml at the 15 MHz sequencer clock.
        self.clear_sequence = 'temp_scan_run1_clearseq.xml'
        self.clear_clock_hz = 15e6
        self.trap_transport_model = TRAP_TRANSPORT_MODEL
        self.phase_capture_ticks = float(phase_capture_ticks)
        if self.phase_capture_ticks < 0:
            raise ValueError('phase_capture_ticks must be non-negative')
        self.phase_capture_dwell_s = self.phase_capture_ticks / self.clear_clock_hz
        self.clear_vertical_phase_count = 6
        self.clear_horizontal_phase_count = 6
        self.clear_delay_vertical_ticks = 300
        self.clear_delay_horizontal_ticks = 150
        self.clear_delay_switch_ticks = 8
        self.clear_delay_reset_gate_ticks = 15
        self.clear_fast_shifts = 1500
        self.clear_fast_horizontal_steps = 10
        self.clear_slow_shifts = 10
        self.clear_slow_horizontal_steps = 3500

        vertical_ticks = (
            self.clear_vertical_phase_count * self.clear_delay_vertical_ticks
        )
        horizontal_step_ticks = (
            self.clear_delay_switch_ticks
            + self.clear_horizontal_phase_count * self.clear_delay_horizontal_ticks
            + self.clear_delay_reset_gate_ticks
        )
        self.clear_fast_dwell_s = (
            vertical_ticks
            + self.clear_fast_horizontal_steps * horizontal_step_ticks
        ) / self.clear_clock_hz
        self.clear_slow_dwell_s = (
            vertical_ticks
            + self.clear_slow_horizontal_steps * horizontal_step_ticks
        ) / self.clear_clock_hz
        self.clear_total_time_s = (
            self.clear_fast_shifts * self.clear_fast_dwell_s
            + self.clear_slow_shifts * self.clear_slow_dwell_s
        )

        # 'three_hour' clear: a data-taking strategy that runs the clear
        # continuously for 3 hours instead of the ~3.26 s standard recipe. After
        # the image flush (handled by fast_clear_numba) every packet is empty, so
        # the remaining time is modelled as continuous fast vertical shifts that
        # drain the traps (drain_traps_empty_numba). The shift count is derived
        # from the duration and the fast-shift dwell.
        self.clear_three_hour_seconds = 3.0 * 3600.0
        self.clear_three_hour_fast_shifts = int(
            round(self.clear_three_hour_seconds / self.clear_fast_dwell_s)
        )

        self.clear_occupied_traps_before = []
        self.clear_occupied_traps_after = []
        self.clear_surface_electrons_before = []
        self.clear_surface_electrons_after_transport = []


        self.UL_expdep = 8.19e-5 #e / pix / day

        self.UR_expdep = 4.36e-5 #e / pix / day

        self.LL_expdep = 6.88e-5 #e / pix / day

        self.LR_expdep = 8.23e-5 #e / pix / day

        self.UL_expindep = 12.23e-5 #e / pix / image

        self.UR_expindep = 9.94e-5 #e / pix / image

        self.LL_expindep = 7.53e-5 #e / pix / image

        self.LR_expindep = 6.52e-5 #e / pix / image
        self.exp_dep_rate = self.UR_expdep / (24 * 3600) #e / pix / s
        # Trap-only hypothesis test: zero the injected single-electron dark
        # current so trap emission is the sole exposure-dependent single-e
        # source. High-energy cosmic events and exposure-independent spurious
        # charge are untouched.
        self.zero_exp_dep_rate = bool(zero_exp_dep_rate)
        if self.zero_exp_dep_rate:
            self.exp_dep_rate = 0.0
        self.exp_indep_rate = self.UR_expindep #e / pix / image

        self.total_pix = (6144)* (1024)

        self.npix_per_quad = self.total_pix / 4


        self.nrow_quad = int(1024 /2)
        self.ncol_quad = int(6144 / 2)
        shape = (self.nrow_quad,self.ncol_quad)
        self.exposure_accumulator = np.zeros(shape)

        self.ccd_state = np.zeros(shape)
        



        self.exp_indep_events = self.exp_indep_rate * self.npix_per_quad


        # #generate trap locations
        # trap_density = (5171 / 4) / (self.nrow_quad * self.ncol_quad)
        # rng = np.random.default_rng()
        # self.trap_mask = rng.random(shape) < trap_density
        # num_traps = np.sum(self.trap_mask)


        # probs = np.array(tau_weights) / np.sum(tau_weights)
        # sampled_taus = rng.choice(tau_values, size=num_traps, p=probs)
        
        # # 3. Create Tau Map (Infinite tau for non-traps to prevent math errors)
        # self.tau_map = np.ones(shape, dtype=float) * np.inf
        # self.tau_map[self.trap_mask] = sampled_taus
        
        # # 4. Trapped Charge State (0.0 = Empty, 1.0 = Full)
        # self.trapped_charge = np.zeros(shape, dtype=float)


        # --- OPTIMIZATION 1: SPARSE TRAP STORAGE ---
        # Instead of a full boolean mask, store the coordinates of traps
        # trap_density calculation remains the same...
        # Baseline trap population = the detected dipole count (run_charge_traps
        # `dipole_coord_list*.npz`), spread over the four quadrants. Passed in so
        # it tracks the actual detection (e.g. 5171 legacy vs 9333 minimal)
        # instead of a stale literal; the upper-limit scale corrects this upward.
        self.n_detected_traps = int(n_detected_traps)
        baseline_trap_density = (self.n_detected_traps / 4) / (self.nrow_quad * self.ncol_quad)
        trap_density = baseline_trap_density * trap_density_scale
        if not 0 <= trap_density <= 1:
            raise ValueError(
                f"trap_density_scale={trap_density_scale} gives invalid "
                f"per-pixel trap probability {trap_density}"
            )
        self.trap_density = trap_density
        self.trap_density_scale = trap_density_scale
        rng = np.random.default_rng()
        self.trap_mask = rng.random(shape) < trap_density
        
        # Store indices (tuple of row_indices, col_indices) for fast access
        self.trap_indices = np.where(self.trap_mask)
        num_traps = len(self.trap_indices[0])
        
        # Store Tau values as a 1D array corresponding to the indices
        probs = np.array(tau_weights) / np.sum(tau_weights)
        
        # 1. Select a bin for each trap
        bin_indices = rng.choice(len(probs), size=num_traps, p=probs)
        # 2. Sample continuously (log-uniform) between the edges of the selected bin
        left_edges = tau_edges[bin_indices]
        right_edges = tau_edges[bin_indices + 1]
        self.trap_taus = np.exp(rng.uniform(np.log(left_edges), np.log(right_edges)))

        # --- SRH capture/recapture parameters ---
        # Each trap gets a capture cross-section resampled from the measured
        # (tau_e(135K), sigma) pairs nearest in log(tau), preserving the
        # empirical tau-sigma correlation. The per-carrier capture rate is
        # kc = sigma * v_th / V_packet, where V_packet is the effective
        # volume explored by a single carrier confined in a pixel well.
        pair_tau135 = np.asarray(pair_tau135, dtype=float)
        pair_sigma = np.asarray(pair_sigma, dtype=float)
        order = np.argsort(pair_tau135)
        sorted_logtau = np.log(pair_tau135[order])
        sorted_sigma = pair_sigma[order]
        K = min(20, len(sorted_sigma))
        ins = np.searchsorted(sorted_logtau, np.log(self.trap_taus))
        lo = np.clip(ins - K // 2, 0, len(sorted_sigma) - K)
        self.trap_sigmas = sorted_sigma[lo + rng.integers(0, K, size=num_traps)]

        self.packet_volume_um3 = packet_volume_um3
        self.temperature_K = temperature_K
        v_th = hole_thermal_velocity(temperature_K)  # cm/s
        packet_volume_cm3 = packet_volume_um3 * 1e-12
        self.trap_kc = self.trap_sigmas * v_th / packet_volume_cm3  # per-carrier capture rate [1/s]
        # The measured pocket-pumped catalog is treated as V1/V3 traps. V2
        # traps would be a separate, unmeasured population and are not included
        # in this baseline transport model.
        self.trap_capture_alpha = self.trap_kc * self.phase_capture_dwell_s
        # Per-trap clock-phase assignment. Pumping cannot distinguish V1 from
        # V3, but transport can: a V3 trap is crossed by the packet on row
        # EXIT (its own emission gets a same-step recapture roll), a V1 trap
        # on row ENTRY (its emission escapes over V3 with no recapture).
        # Bernoulli split with fraction `v3_phase_fraction` (1.0 reproduces
        # the pre-2026-07 all-V3 readout/clear kernel).
        self.v3_phase_fraction = float(v3_phase_fraction)
        if not 0.0 <= self.v3_phase_fraction <= 1.0:
            raise ValueError(
                f"v3_phase_fraction={v3_phase_fraction} must be in [0, 1]"
            )
        self.trap_is_v3 = (
            rng.random(num_traps) < self.v3_phase_fraction
        ).astype(np.uint8)
        self.readout_emit_probs = 1.0 - np.exp(-self.tpix_vertical / self.trap_taus)
        self.clear_fast_emit_probs = 1.0 - np.exp(-self.clear_fast_dwell_s / self.trap_taus)
        self.clear_slow_emit_probs = 1.0 - np.exp(-self.clear_slow_dwell_s / self.trap_taus)

        # Store trapped charge as a 1D array (much faster than 2D)
        self.trapped_charge_1d = np.zeros(num_traps, dtype=float)

        self.pix_denom = np.zeros(shape, dtype=int)      # number of times each pixel was unmasked (halo/bleed masks only)
        self.pix_hits = np.zeros(shape, dtype=int)       # number of 1‑e events in unmasked pixels
        self.hot_cols = set()                             # final hot column indices
        self.hot_pixels = set()                            # final hot pixel coordinates (after merging)





    # def charge_trap_interaction(self,current_image,dt):
    #     capture_efficiency = 1

    #     #capture process 
    #     # available_space = 1.0 - self.trapped_charge
    #     # potential_capture = current_image * capture_efficiency * self.trap_mask
    #     # actual_capture = np.minimum(potential_capture, available_space)
    #     # current_image -= actual_capture
    #     # self.trapped_charge += actual_capture
    #     candidates = (current_image >= 1) & (self.trap_mask) & (self.trapped_charge == 0)
    #     # We can treat this as a Binomial process: 
    #     # If capture_efficiency is 1.0, it simply takes the electron.
    #     # Note: This is simplified for 1e regime.
    #     captured_mask = candidates # In a more complex model, use np.random.binomial
    #     current_image[captured_mask] -= 1.0
    #     self.trapped_charge[captured_mask] += 1.0


    #     #now simulate decay
    #     # decay_factors = 1 - np.exp(-dt / self.tau_map)
    #     # released_charge = self.trapped_charge * decay_factors
    #     # self.trapped_charge -= released_charge
    #     # current_image += released_charge
    #     p_release = 1 - np.exp(-dt / self.tau_map)
    #     occupied_traps = self.trapped_charge > 0
    #     random_rolls = np.random.random(current_image.shape)
    #     release_mask = occupied_traps & (random_rolls < p_release)
    #     self.trapped_charge[release_mask] -= 1.0
    #     current_image[release_mask] += 1.0


    #     return current_image

    def charge_trap_interaction(self, current_image, dt):
        import numpy as np
        # The measured pocket-pumped traps are V1/V3 traps. During static
        # exposure the charge is parked under V2, so empty measured traps do not
        # capture from exposure charge. Occupied traps can still emit into their
        # pixel and then remain empty.
        if dt <= 0:
            return current_image

        occupied = self.trapped_charge_1d > 0
        if not np.any(occupied):
            return current_image

        p_release = 1.0 - np.exp(-dt / self.trap_taus)
        random_rolls = np.random.random(len(self.trap_taus))
        should_release = occupied & (random_rolls < p_release)

        releasing_rows = self.trap_indices[0][should_release]
        releasing_cols = self.trap_indices[1][should_release]
        current_image[releasing_rows, releasing_cols] += 1.0
        self.trapped_charge_1d[should_release] = 0.0

        return current_image



    def simulate_clear(self):
        import numpy as np

        if self.clear_mode == 'binned_0h':
            # No hardware clear in this mode: the binned 0 h readouts (issued by
            # take_fake_image after every real exposure) empty free charge and
            # reset the array. Trap occupancy carries over untouched.
            return

        self.clear_occupied_traps_before.append(
            int(np.count_nonzero(self.trapped_charge_1d))
        )
        self.clear_surface_electrons_before.append(float(np.sum(self.ccd_state)))

        if (
            self.clear_mode in ('sequencer', 'three_hour')
            and (
                np.any(self.ccd_state != 0)
                or np.any(self.trapped_charge_1d > 0)
            )
        ):
            # Transport resident image charge out with phase-limited V1/V3
            # capture and full-dwell emission. For 'three_hour' this is the
            # image-flush phase; the pure-emission empty tail follows below.
            fast_clear_numba(
                self.ccd_state,
                self.clear_fast_shifts,
                self.clear_fast_dwell_s,
                self.clear_slow_shifts,
                self.clear_slow_dwell_s,
                self.trap_indices[0],
                self.trap_indices[1],
                self.clear_fast_emit_probs,
                self.clear_slow_emit_probs,
                self.trap_capture_alpha,
                self.trap_is_v3,
                self.trapped_charge_1d,
            )

        if self.clear_mode == 'three_hour' and np.any(self.trapped_charge_1d > 0):
            # Remaining ~3 h of continuous fast shifts past empty packets,
            # drained analytically: recapture-free for V1 traps, recapture-
            # thinned for V3 traps (see drain_traps_empty_numba).
            drain_traps_empty_numba(
                self.trap_taus,
                self.trap_capture_alpha,
                self.trap_is_v3,
                self.trapped_charge_1d,
                self.clear_fast_dwell_s,
                self.clear_three_hour_fast_shifts,
            )

        self.clear_occupied_traps_after.append(
            int(np.count_nonzero(self.trapped_charge_1d))
        )
        self.clear_surface_electrons_after_transport.append(
            float(np.sum(self.ccd_state))
        )

        # The hardware clear removes free surface charge, not trapped charge.
        self.ccd_state.fill(0.0)




    def take_fake_image(self,exposure_time_hours,radius=60,store_image=False):
        import numpy as np
        from utils import approximate_electronize,get_qdata
        # from skimage.morphology import disk, binary_dilation


        exp = exposure_time_hours * 3600
        self.simulate_clear()
        self.exposures.append(exp)

        self.ccd_state = self.charge_trap_interaction(self.ccd_state,exp)

        
        exp_dep_events_expected = self.exp_dep_rate * exp * self.npix_per_quad
        n_exp_dep_events = np.random.poisson(exp_dep_events_expected)
        n_exp_indep_events = np.random.poisson(self.exp_indep_events)

        # print(self.npix_per_quad)
        # print(exp, exp_dep_events_expected, self.exp_indep_events)

        


        # print(f'Exposure-dependent events: {n_exp_dep_events}')
        # print(f'Exposure-independent events: {n_exp_indep_events}')
        #now generate fake image
        file = 'minos_image/proc_corr_proc_skp_72000secs_exp_run10_NSAMP_300_36.fits'

        q0 = get_qdata(file,0)

        q0 =approximate_electronize(q0,400)
        if self.runconditions == 'minos':
            q0_blank= transplant_clusters(q0.T, target_shape=(self.nrow_quad, self.ncol_quad),count_threshold=100, max_aspect_ratio=3.0,radius=radius,exposure=exp)
        elif self.runconditions == 'snolab':
            q0_blank= transplant_clusters(q0.T, target_shape=(self.nrow_quad, self.ncol_quad),count_threshold=100, max_aspect_ratio=3.0,radius=radius,exposure=exp/10)

        # footprint = disk(radius)

        # exclusion_mask = binary_dilation(q0_blank > 0, footprint)



        q0_fake = inject_single_e(
            q0_blank,
            n_events=n_exp_dep_events,
            intensity=1,
            exclusion_mask=None,
        )
        if self.exp_indep_charge_mode == 'pre_readout':
            q0_fake = inject_single_e(
                q0_fake,
                n_events=n_exp_indep_events,
                intensity=1,
                exclusion_mask=None,
            )
        # self.no_trap_images.append(q0_fake)

        self.ccd_state += q0_fake
        # if len(self.exposures) > 0:
        #     t1 = self.exposure_accumulator - self.exposures[-1]
        #     t2 = self.exposure_accumulator 
        #     # dt = t2 - t1
            
        # exp_image = np.zeros_like(q0_fake,dtype=float)

        # self.simulate_readout()
        # In 'binned_0h' mode the 0 h images are read out binned -> shorter
        # per-row dwell (faster readout), which is what resets the array in place
        # of a hardware clear.
        readout_tpix_vertical = None
        if self.clear_mode == 'binned_0h' and exposure_time_hours == 0:
            readout_tpix_vertical = self.tpix_vertical / self.binning_0h_factor
        img_trap = self.simulate_readout(tpix_vertical=readout_tpix_vertical)
        img_notrap = q0_fake.astype(np.float64, copy=True)

        if self.exp_indep_charge_mode == 'post_readout':
            # Spurious/readout-generated charge does not traverse active-area
            # traps. Add one shared realization after readout so the trap and
            # no-trap branches retain their common-random-number cancellation.
            post_readout_charge = np.zeros_like(img_trap)
            post_readout_charge = inject_single_e(
                post_readout_charge,
                n_events=n_exp_indep_events,
                intensity=1,
                exclusion_mask=None,
            )
            img_trap += post_readout_charge
            img_notrap += post_readout_charge
        
        if store_image:
            self.reconstructed_images.append(img_trap)
            self.no_trap_images.append(img_notrap)

        
        # --- MEMORY OPTIMIZATION: Compute local masks on the fly ---
        BIT_BASE1E, BIT_HALO, BIT_BLEED, BIT_BASE2E = 1, 2, 4, 32

        b_trap = np.zeros((self.nrow_quad, self.ncol_quad), dtype=np.uint8)
        b_trap[get_cluster(img_trap, min_pixs=1, max_pixs=1, min_total_value=1, max_total_value=1)] |= BIT_BASE1E
        b_trap[get_cluster(img_trap, min_pixs=1, max_pixs=2, min_total_value=2, max_total_value=2)] |= BIT_BASE2E
        b_trap[generate_halo_mask(img_trap, threshold=100, radius=60)] |= BIT_HALO
        b_trap[generate_column_bleed_mask(img_trap, threshold=100, direction='up')] |= BIT_BLEED
        self.trap_bitmasks.append(b_trap)

        b_notrap = np.zeros((self.nrow_quad, self.ncol_quad), dtype=np.uint8)
        b_notrap[get_cluster(img_notrap, min_pixs=1, max_pixs=1, min_total_value=1, max_total_value=1)] |= BIT_BASE1E
        b_notrap[get_cluster(img_notrap, min_pixs=1, max_pixs=2, min_total_value=2, max_total_value=2)] |= BIT_BASE2E
        b_notrap[generate_halo_mask(img_notrap, threshold=100, radius=60)] |= BIT_HALO
        b_notrap[generate_column_bleed_mask(img_notrap, threshold=100, direction='up')] |= BIT_BLEED
        self.notrap_bitmasks.append(b_notrap)

        # self.exposure_images.append(exp_image)

        


        self.exposure_accumulator += exp

    def process_run(self):
        import numpy as np
        from itertools import combinations
        
        print("\n--- Starting Masking ---")
        n_images = len(self.trap_bitmasks)
        shape = (self.nrow_quad, self.ncol_quad)
        


        BIT_BASE1E, BIT_HALO, BIT_BLEED, BIT_HOTCOL, BIT_HOTPIX, BIT_BASE2E = 1, 2, 4, 8, 16, 32
        
        print("1. Aggregating statistics using Split-Sample (A/B)...")

        unique_exposures = np.unique(self.exposures)
        
        # We will build independent data lists for Set A and Set B
        trap_col_data_A, notrap_col_data_A = [], []
        trap_col_data_B, notrap_col_data_B = [], []
        
        trap_pix_data_A, notrap_pix_data_A = [], []
        trap_pix_data_B, notrap_pix_data_B = [], []

        for exp in unique_exposures:
            # Find all image indices for this exposure
            all_idx = np.where(np.array(self.exposures) == exp)[0]
            
            # SPLIT THE DATA: Evens go to A, Odds go to B
            idx_A = all_idx[0::2] 
            idx_B = all_idx[1::2]
            
            def aggregate_subset(indices):
                t_denom_c, t_hits_c = np.zeros(self.ncol_quad), np.zeros(self.ncol_quad)
                nt_denom_c, nt_hits_c = np.zeros(self.ncol_quad), np.zeros(self.ncol_quad)
                t_denom_p, t_hits_p = np.zeros(shape, dtype=int), np.zeros(shape, dtype=int)
                nt_denom_p, nt_hits_p = np.zeros(shape, dtype=int), np.zeros(shape, dtype=int)
                
                for i in indices:
                    # Traps
                    b_t = self.trap_bitmasks[i]
                    u_t = (b_t & (BIT_HALO | BIT_BLEED)) == 0
                    b1e_t = (b_t & BIT_BASE1E) > 0
                    t_denom_c += np.sum(u_t, axis=0)
                    t_hits_c += np.sum(b1e_t & u_t, axis=0)
                    t_denom_p += u_t.astype(int)
                    t_hits_p += (b1e_t & u_t).astype(int)
                    
                    # No Traps
                    b_nt = self.notrap_bitmasks[i]
                    u_nt = (b_nt & (BIT_HALO | BIT_BLEED)) == 0
                    b1e_nt = (b_nt & BIT_BASE1E) > 0
                    nt_denom_c += np.sum(u_nt, axis=0)
                    nt_hits_c += np.sum(b1e_nt & u_nt, axis=0)
                    nt_denom_p += u_nt.astype(int)
                    nt_hits_p += (b1e_nt & u_nt).astype(int)
                    
                return (t_denom_c, t_hits_c, t_denom_p, t_hits_p), (nt_denom_c, nt_hits_c, nt_denom_p, nt_hits_p)

            # Aggregate Set A
            trap_A, notrap_A = aggregate_subset(idx_A)
            trap_col_data_A.append((np.column_stack((trap_A[0], trap_A[1])), f"Trap_{exp}s_A"))
            trap_pix_data_A.append((np.stack((trap_A[2], trap_A[3]), axis=-1), f"TrapPix_{exp}s_A"))
            notrap_col_data_A.append((np.column_stack((notrap_A[0], notrap_A[1])), f"NoTrap_{exp}s_A"))
            notrap_pix_data_A.append((np.stack((notrap_A[2], notrap_A[3]), axis=-1), f"NoTrapPix_{exp}s_A"))

            # Aggregate Set B
            trap_B, notrap_B = aggregate_subset(idx_B)
            trap_col_data_B.append((np.column_stack((trap_B[0], trap_B[1])), f"Trap_{exp}s_B"))
            trap_pix_data_B.append((np.stack((trap_B[2], trap_B[3]), axis=-1), f"TrapPix_{exp}s_B"))
            notrap_col_data_B.append((np.column_stack((notrap_B[0], notrap_B[1])), f"NoTrap_{exp}s_B"))
            notrap_pix_data_B.append((np.stack((notrap_B[2], notrap_B[3]), axis=-1), f"NoTrapPix_{exp}s_B"))


        print("2. Computing global hot pixel/column masks (Cross-Validation)...")
        
        def compute_masks(col_data, pix_data):
            bad_cols, _, _ = findBadCells(col_data, nCells=self.ncol_quad)
            if len(bad_cols) > 0:
                for item in pix_data:
                    item[0][:, bad_cols, :] = 0
            bad_pix_flat, _, _ = findBadCells(pix_data, nCells=self.nrow_quad*self.ncol_quad)
            new_cols, final_pix = merge_hot_pixels_to_columns(bad_pix_flat, self.nrow_quad)
            
            all_bad_cols = list(set(bad_cols) | set(new_cols))
            return all_bad_cols, final_pix
        
        bad_cols_tA, bad_pix_tA = compute_masks(trap_col_data_A, trap_pix_data_A)
        bad_cols_ntA, bad_pix_ntA = compute_masks(notrap_col_data_A, notrap_pix_data_A)

        bad_cols_tB, bad_pix_tB = compute_masks(trap_col_data_B, trap_pix_data_B)
        bad_cols_ntB, bad_pix_ntB = compute_masks(notrap_col_data_B, notrap_pix_data_B)

        
        # --- CROSS-APPLY MASKS ---
        for exp in unique_exposures:
            all_idx = np.where(np.array(self.exposures) == exp)[0]
            idx_A = all_idx[0::2]
            idx_B = all_idx[1::2]

            def apply_masks(indices, bitmasks, bad_cols, bad_pix):
                for i in indices:
                    if bad_cols:
                        bitmasks[i][:, bad_cols] |= BIT_HOTCOL
                    if bad_pix:
                        rs, cs = zip(*bad_pix)
                        bitmasks[i][rs, cs] |= BIT_HOTPIX
            
            # Apply Mask B to Images A
            apply_masks(idx_A, self.trap_bitmasks, bad_cols_tB, bad_pix_tB)
            apply_masks(idx_A, self.notrap_bitmasks, bad_cols_ntB, bad_pix_ntB)

            # Apply Mask A to Images B
            apply_masks(idx_B, self.trap_bitmasks, bad_cols_tA, bad_pix_tA)
            apply_masks(idx_B, self.notrap_bitmasks, bad_cols_ntA, bad_pix_ntA)

        print("3. Evaluating all mask permutations and extracting counts...")

        self.stats_trap = {}
        self.stats_notrap = {}
        
        mask_map = {'Halo': BIT_HALO, 'Bleed': BIT_BLEED, 'HotColumn': BIT_HOTCOL, 'HotPixel': BIT_HOTPIX}
        mask_keys = list(mask_map.keys())

        all_combos = []
        
        for L in range(0, len(mask_keys) + 1):
            for subset in combinations(mask_keys, L):
                combo_name = "+".join(subset) if subset else "None"
                all_combos.append(subset)
                self.stats_trap[combo_name] = {'counts': [], '2e_counts': [], 'unmasked_pix': []}
                self.stats_notrap[combo_name] = {'counts': [], '2e_counts': [], 'unmasked_pix': []}
                
        from cv2 import connectedComponents as _cc
        for i in range(n_images):
            b_trap = self.trap_bitmasks[i]
            b_notrap = self.notrap_bitmasks[i]

            # Pre-compute 2e cluster labels once per image; count via label-subtract in combo loop
            fp2e_t = (b_trap & BIT_BASE2E) > 0
            _, labels_2e_t = _cc(fp2e_t.astype(np.uint8), connectivity=8)
            n_total_2e_t = int(labels_2e_t.max())

            fp2e_nt = (b_notrap & BIT_BASE2E) > 0
            _, labels_2e_nt = _cc(fp2e_nt.astype(np.uint8), connectivity=8)
            n_total_2e_nt = int(labels_2e_nt.max())

            for subset in all_combos:
                combo_name = "+".join(subset) if subset else "None"
                mask_bits = sum(mask_map[name] for name in subset) if subset else 0

                # Trap Evaluation
                surviving_t = ((b_trap & BIT_BASE1E) > 0) & ((b_trap & mask_bits) == 0)
                unmasked_t = (b_trap & mask_bits) == 0

                bad_labels_t = np.unique(labels_2e_t[fp2e_t & ((b_trap & mask_bits) > 0)])
                n_bad_2e_t = int(np.count_nonzero(bad_labels_t > 0))

                self.stats_trap[combo_name]['counts'].append(np.count_nonzero(surviving_t))
                self.stats_trap[combo_name]['2e_counts'].append(n_total_2e_t - n_bad_2e_t)
                self.stats_trap[combo_name]['unmasked_pix'].append(np.count_nonzero(unmasked_t))

                # No-Trap Evaluation
                surviving_nt = ((b_notrap & BIT_BASE1E) > 0) & ((b_notrap & mask_bits) == 0)
                unmasked_nt = (b_notrap & mask_bits) == 0

                bad_labels_nt = np.unique(labels_2e_nt[fp2e_nt & ((b_notrap & mask_bits) > 0)])
                n_bad_2e_nt = int(np.count_nonzero(bad_labels_nt > 0))

                self.stats_notrap[combo_name]['counts'].append(np.count_nonzero(surviving_nt))
                self.stats_notrap[combo_name]['2e_counts'].append(n_total_2e_nt - n_bad_2e_nt)
                self.stats_notrap[combo_name]['unmasked_pix'].append(np.count_nonzero(unmasked_nt))

        print("--- Post-Run Analysis Complete ---")
    # def process_run(self):
    #     import numpy as np
    #     from itertools import combinations
        
    #     print("\n--- Starting Masking ---")
        
    #     n_images = len(self.reconstructed_images)
    #     shape = (self.nrow_quad, self.ncol_quad)
        
    #     clean_trap_stack = np.zeros(shape, dtype=float)
    #     clean_notrap_stack = np.zeros(shape, dtype=float)
        
    #     trap_local_masks = []
    #     notrap_local_masks = []

    #     # 1D Column tracking
    #     trap_denom = np.zeros(self.ncol_quad)
    #     trap_hits = np.zeros(self.ncol_quad)
    #     trap_unmasked_per_img = np.zeros((n_images, self.ncol_quad))

    #     notrap_denom = np.zeros(self.ncol_quad)
    #     notrap_hits = np.zeros(self.ncol_quad)
    #     notrap_unmasked_per_img = np.zeros((n_images, self.ncol_quad))

    #     # --- NEW: 2D Pixel tracking ---
    #     trap_pix_denom = np.zeros(shape, dtype=int)
    #     trap_pix_hits = np.zeros(shape, dtype=int)
        
    #     notrap_pix_denom = np.zeros(shape, dtype=int)
    #     notrap_pix_hits = np.zeros(shape, dtype=int)
        
    #     print("1. Computing local masks (Halo & Bleed) and building clean stacks...")
    #     for i in range(n_images):
    #         # --- Trap Images ---
    #         img_trap = self.reconstructed_images[i]
    #         halo_trap = generate_halo_mask(img_trap, threshold=100, radius=60)
    #         bleed_trap = generate_column_bleed_mask(img_trap, threshold=100, direction='up') 
    #         trap_local_masks.append({'Halo': halo_trap, 'Bleed': bleed_trap})
            
    #         clean_trap = img_trap.copy().astype(float)
    #         local_mask_trap = halo_trap | bleed_trap
    #         baseline_trap = np.median(img_trap[~local_mask_trap]) if np.any(~local_mask_trap) else 0
    #         clean_trap[local_mask_trap] = baseline_trap 
    #         clean_trap_stack += clean_trap

    #         unmasked_trap = ~local_mask_trap
    #         base_1e_trap = get_cluster(img_trap, min_pixs=1, max_pixs=1, min_total_value=1, max_total_value=1)
            
    #         # Column Updates
    #         trap_denom += np.sum(unmasked_trap, axis=0)
    #         trap_hits += np.sum(base_1e_trap & unmasked_trap, axis=0)
    #         trap_unmasked_per_img[i, :] = np.sum(unmasked_trap, axis=0)
            
    #         # --- NEW: Pixel Updates ---
    #         trap_pix_denom += unmasked_trap.astype(int)
    #         trap_pix_hits += (base_1e_trap & unmasked_trap).astype(int)
            
    #         # --- No Trap Images ---
    #         img_notrap = self.no_trap_images[i]
    #         halo_notrap = generate_halo_mask(img_notrap, threshold=100, radius=60)
    #         bleed_notrap = generate_column_bleed_mask(img_notrap, threshold=100, direction='up')
    #         notrap_local_masks.append({'Halo': halo_notrap, 'Bleed': bleed_notrap})
            
    #         clean_notrap = img_notrap.copy().astype(float)
    #         local_mask_notrap = halo_notrap | bleed_notrap
    #         baseline_notrap = np.median(img_notrap[~local_mask_notrap]) if np.any(~local_mask_notrap) else 0
    #         clean_notrap[local_mask_notrap] = baseline_notrap
    #         clean_notrap_stack += clean_notrap

    #         unmasked_notrap = ~local_mask_notrap
    #         base_1e_notrap = get_cluster(img_notrap, min_pixs=1, max_pixs=1, min_total_value=1, max_total_value=1)
            
    #         # Column Updates
    #         notrap_denom += np.sum(unmasked_notrap, axis=0)
    #         notrap_hits += np.sum(base_1e_notrap & unmasked_notrap, axis=0)
    #         notrap_unmasked_per_img[i, :] = np.sum(unmasked_notrap, axis=0)
            
    #         # --- NEW: Pixel Updates ---
    #         notrap_pix_denom += unmasked_notrap.astype(int)
    #         notrap_pix_hits += (base_1e_notrap & unmasked_notrap).astype(int)

    #     print("2. Computing global hot pixel/column masks...")
        
    #     # --- COLUMNS ---
    #     vhot_trap = find_very_hot_columns(trap_unmasked_per_img, max_possible_unmasked=self.nrow_quad)
    #     vhot_notrap = find_very_hot_columns(notrap_unmasked_per_img, max_possible_unmasked=self.nrow_quad)

    #     trap_data = [(np.column_stack((trap_denom, trap_hits)), "Trap")]
    #     bad_cols_trap, _, _ = findBadCells(trap_data, nCells=self.ncol_quad, already_bad=vhot_trap)
        
    #     notrap_data = [(np.column_stack((notrap_denom, notrap_hits)), "NoTrap")]
    #     bad_cols_notrap, _, _ = findBadCells(notrap_data, nCells=self.ncol_quad, already_bad=vhot_notrap)

    #     # --- NEW: PIXELS ---
    #     # 1. Very Hot Pixels Pre-Cut
    #     vhot_pix_trap = find_very_hot_pixels(trap_pix_denom, n_images)
    #     vhot_pix_notrap = find_very_hot_pixels(notrap_pix_denom, n_images)

    #     # 2. Zero out counts in already flagged hot columns (mimics hotcol.py behavior)
    #     if len(bad_cols_trap) > 0:
    #         trap_pix_denom[:, bad_cols_trap] = 0
    #         trap_pix_hits[:, bad_cols_trap] = 0
    #     if len(bad_cols_notrap) > 0:
    #         notrap_pix_denom[:, bad_cols_notrap] = 0
    #         notrap_pix_hits[:, bad_cols_notrap] = 0

    #     # 3. Find Bad Pixels (stack Denom and Hits on the last axis for 2D cell arrays)
    #     trap_pix_data = [(np.stack((trap_pix_denom, trap_pix_hits), axis=-1), "TrapPix")]
    #     bad_pix_trap, _, _ = findBadCells(trap_pix_data, nCells=self.nrow_quad*self.ncol_quad, already_bad=vhot_pix_trap)

    #     notrap_pix_data = [(np.stack((notrap_pix_denom, notrap_pix_hits), axis=-1), "NoTrapPix")]
    #     bad_pix_notrap, _, _ = findBadCells(notrap_pix_data, nCells=self.nrow_quad*self.ncol_quad, already_bad=vhot_pix_notrap)

    #     # 4. Merge dense hot pixels into columns
    #     new_cols_trap, final_pix_trap = merge_hot_pixels_to_columns(bad_pix_trap, self.nrow_quad)
    #     bad_cols_trap = list(set(bad_cols_trap) | set(new_cols_trap))

    #     new_cols_notrap, final_pix_notrap = merge_hot_pixels_to_columns(bad_pix_notrap, self.nrow_quad)
    #     bad_cols_notrap = list(set(bad_cols_notrap) | set(new_cols_notrap))



    #     # ==========================================
    #     print(f"Final Trap Masking: {len(bad_cols_trap)} Hot Columns, {len(final_pix_trap)} Hot Pixels")
    #     print(f"Final No-Trap Masking: {len(bad_cols_notrap)} Hot Columns, {len(final_pix_notrap)} Hot Pixels")
    #     # ==========================================


    #     # --- Create final boolean masks arrays ---
    #     global_hot_col_mask_trap = np.zeros(shape, dtype=bool)
    #     if len(bad_cols_trap) > 0: global_hot_col_mask_trap[:, bad_cols_trap] = True
            
    #     global_hot_col_mask_notrap = np.zeros(shape, dtype=bool)
    #     if len(bad_cols_notrap) > 0: global_hot_col_mask_notrap[:, bad_cols_notrap] = True
        
    #     global_hot_pix_mask_trap = np.zeros(shape, dtype=bool)
    #     for r, c in final_pix_trap: global_hot_pix_mask_trap[r, c] = True
            
    #     global_hot_pix_mask_notrap = np.zeros(shape, dtype=bool)
    #     for r, c in final_pix_notrap: global_hot_pix_mask_notrap[r, c] = True

    #     for i in range(n_images):
    #         trap_local_masks[i]['HotColumn'] = global_hot_col_mask_trap
    #         trap_local_masks[i]['HotPixel'] = global_hot_pix_mask_trap  # NEW
            
    #         notrap_local_masks[i]['HotColumn'] = global_hot_col_mask_notrap
    #         notrap_local_masks[i]['HotPixel'] = global_hot_pix_mask_notrap  # NEW
        
    #     # --- NEW: Add 'HotPixel' to combination keys ---
    #     self.stats_trap = {}
    #     self.stats_notrap = {}
        
    #     mask_keys = ['Halo', 'Bleed', 'HotColumn', 'HotPixel']
    #     all_combos = []
    #     for L in range(0, len(mask_keys) + 1):
    #         for subset in combinations(mask_keys, L):
    #             combo_name = "+".join(subset) if subset else "None"
    #             all_combos.append(subset)
    #             self.stats_trap[combo_name] = {'counts': [], 'unmasked_pix': []}
    #             self.stats_notrap[combo_name] = {'counts': [], 'unmasked_pix': []}
                
    #     print("3. Evaluating all mask permutations and extracting counts...")
    #     for i in range(n_images):
    #         # ... (The rest of your combination loop remains exactly the same) ...
    #         # ----------------------------------------------------
    #         # TRAP IMAGES
    #         # ----------------------------------------------------
    #         img_trap = self.reconstructed_images[i]
    #         masks_trap = trap_local_masks[i]
            
    #         base_1e_mask_trap = get_cluster(img_trap, min_pixs=1, max_pixs=1, min_total_value=1, max_total_value=1)
            
    #         for subset in all_combos:
    #             combo_name = "+".join(subset) if subset else "None"
                
    #             combined_mask = np.zeros(shape, dtype=bool)
    #             for name in subset:
    #                 combined_mask |= masks_trap[name]
                    
    #             surviving_counts = np.sum(base_1e_mask_trap & ~combined_mask)
    #             unmasked_pix = np.sum(~combined_mask)
                
    #             self.stats_trap[combo_name]['counts'].append(surviving_counts)
    #             self.stats_trap[combo_name]['unmasked_pix'].append(unmasked_pix)

    #         # ----------------------------------------------------
    #         # NO TRAP IMAGES
    #         # ----------------------------------------------------
    #         img_notrap = self.no_trap_images[i]
    #         masks_notrap = notrap_local_masks[i]
            
    #         base_1e_mask_notrap = get_cluster(img_notrap, min_pixs=1, max_pixs=1, min_total_value=1, max_total_value=1)
            
    #         for subset in all_combos:
    #             combo_name = "+".join(subset) if subset else "None"
                
    #             combined_mask = np.zeros(shape, dtype=bool)
    #             for name in subset:
    #                 combined_mask |= masks_notrap[name]
                    
    #             surviving_counts = np.sum(base_1e_mask_notrap & ~combined_mask)
    #             unmasked_pix = np.sum(~combined_mask)
                
    #             self.stats_notrap[combo_name]['counts'].append(surviving_counts)
    #             self.stats_notrap[combo_name]['unmasked_pix'].append(unmasked_pix)

    #     print("--- Post-Run Masking Complete ---")


    # def process_run(self):
    #     import numpy as np
    #     from itertools import combinations
        
    #     print("\n--- Starting Masking ---")
        
    #     n_images = len(self.reconstructed_images)
    #     shape = (self.nrow_quad, self.ncol_quad)
        
    #     # Stacks to hold the HEE-cleaned images
    #     clean_trap_stack = np.zeros(shape, dtype=float)
    #     clean_notrap_stack = np.zeros(shape, dtype=float)
        
    #     trap_local_masks = []
    #     notrap_local_masks = []

    #     trap_denom = np.zeros(self.ncol_quad)
    #     trap_hits = np.zeros(self.ncol_quad)
    #     trap_unmasked_per_img = np.zeros((n_images, self.ncol_quad))

    #     notrap_denom = np.zeros( self.ncol_quad)
    #     notrap_hits = np.zeros( self.ncol_quad)
    #     notrap_unmasked_per_img = np.zeros((n_images,  self.ncol_quad))


        
    #     print("1. Computing local masks (Halo & Bleed) and building clean stacks...")
    #     for i in range(n_images):
    #         # --- Trap Images ---
    #         img_trap = self.reconstructed_images[i]
    #         halo_trap = generate_halo_mask(img_trap, threshold=100, radius=60)
    #         # Make sure direction is 'up' based on your readout orientation!
    #         bleed_trap = generate_column_bleed_mask(img_trap, threshold=100, direction='up') 
    #         trap_local_masks.append({'Halo': halo_trap, 'Bleed': bleed_trap})
            
    #         # Clean image for stacking
    #         clean_trap = img_trap.copy().astype(float)
    #         local_mask_trap = halo_trap | bleed_trap
    #         baseline_trap = np.median(img_trap[~local_mask_trap]) if np.any(~local_mask_trap) else 0
    #         clean_trap[local_mask_trap] = baseline_trap 
    #         clean_trap_stack += clean_trap

    #         # Extract 1e hits and update column statistics for hotcol
    #         unmasked_trap = ~local_mask_trap
    #         base_1e_trap = get_cluster(img_trap, min_pixs=1, max_pixs=1, min_total_value=1, max_total_value=1)
    #         trap_denom += np.sum(unmasked_trap, axis=0)
    #         trap_hits += np.sum(base_1e_trap & unmasked_trap, axis=0)
    #         trap_unmasked_per_img[i, :] = np.sum(unmasked_trap, axis=0)


            
    #         # --- No Trap Images ---
    #         img_notrap = self.no_trap_images[i]
    #         halo_notrap = generate_halo_mask(img_notrap, threshold=100, radius=60)
    #         bleed_notrap = generate_column_bleed_mask(img_notrap, threshold=100, direction='up')
    #         notrap_local_masks.append({'Halo': halo_notrap, 'Bleed': bleed_notrap})
            
    #         # Clean image for stacking
    #         clean_notrap = img_notrap.copy().astype(float)
    #         local_mask_notrap = halo_notrap | bleed_notrap
    #         baseline_notrap = np.median(img_notrap[~local_mask_notrap]) if np.any(~local_mask_notrap) else 0
    #         clean_notrap[local_mask_notrap] = baseline_notrap
    #         clean_notrap_stack += clean_notrap

    #         # Extract 1e hits and update column statistics for hotcol
    #         unmasked_notrap = ~local_mask_notrap
    #         base_1e_notrap = get_cluster(img_notrap, min_pixs=1, max_pixs=1, min_total_value=1, max_total_value=1)
    #         notrap_denom += np.sum(unmasked_notrap, axis=0)
    #         notrap_hits += np.sum(base_1e_notrap & unmasked_notrap, axis=0)
    #         notrap_unmasked_per_img[i, :] = np.sum(unmasked_notrap, axis=0)

    #     print("2. Computing global hot pixel/column masks...")
    #     #Find "Very Hot" columns (dead/constant dark spikes)
    #     vhot_trap = find_very_hot_columns(trap_unmasked_per_img, max_possible_unmasked=self.nrow_quad)
    #     vhot_notrap = find_very_hot_columns(notrap_unmasked_per_img, max_possible_unmasked=self.nrow_quad)

    #     # Run the iterative statistical Poisson finder
    #     trap_data = [(np.column_stack((trap_denom, trap_hits)), "Trap")]
    #     bad_cols_trap, _, _ = findBadCells(trap_data, nCells=self.ncol_quad, already_bad=vhot_trap)
        
    #     notrap_data = [(np.column_stack((notrap_denom, notrap_hits)), "NoTrap")]
    #     bad_cols_notrap, _, _ = findBadCells(notrap_data, nCells=self.ncol_quad, already_bad=vhot_notrap)

    #     # Create the boolean mask arrays
    #     global_hot_mask_trap = np.zeros(shape, dtype=bool)
    #     if len(bad_cols_trap) > 0: global_hot_mask_trap[:, bad_cols_trap] = True
            
    #     global_hot_mask_notrap = np.zeros(shape, dtype=bool)
    #     if len(bad_cols_notrap) > 0: global_hot_mask_notrap[:, bad_cols_notrap] = True

    #     # Ensure the new mask key is appended to all local dictionaries
    #     for i in range(n_images):
    #         trap_local_masks[i]['HotColumn'] = global_hot_mask_trap
    #         notrap_local_masks[i]['HotColumn'] = global_hot_mask_notrap
        
    #     # --- Initialize storage dictionaries ---
    #     # Structure: self.stats_trap['Halo+Bleed']['counts'] -> List of counts per image
    #     self.stats_trap = {}
    #     self.stats_notrap = {}
        
    #     mask_keys = ['Halo', 'Bleed', 'HotColumn']
    #     all_combos = []
    #     for L in range(0, len(mask_keys) + 1):
    #         for subset in combinations(mask_keys, L):
    #             combo_name = "+".join(subset) if subset else "None"
    #             all_combos.append(subset)
    #             self.stats_trap[combo_name] = {'counts': [], 'unmasked_pix': []}
    #             self.stats_notrap[combo_name] = {'counts': [], 'unmasked_pix': []}
                
    #     print("3. Evaluating all mask permutations and extracting counts...")
    #     for i in range(n_images):
    #         # ----------------------------------------------------
    #         # TRAP IMAGES
    #         # ----------------------------------------------------
    #         img_trap = self.reconstructed_images[i]
    #         masks_trap = trap_local_masks[i]
    #         # masks_trap['HotColumn'] = global_hot_mask_trap
            
    #         # Find the true 1e- events ONCE on the unmasked image
    #         # get_cluster returns a boolean layer when return_count=False
    #         base_1e_mask_trap = get_cluster(img_trap, min_pixs=1, max_pixs=1, min_total_value=1, max_total_value=1)
            
    #         for subset in all_combos:
    #             combo_name = "+".join(subset) if subset else "None"
                
    #             # Combine the chosen masks using bitwise OR
    #             combined_mask = np.zeros(shape, dtype=bool)
    #             for name in subset:
    #                 combined_mask |= masks_trap[name]
                    
    #             # A single-e event survives if it was in the base mask AND is not covered by the combined mask
    #             surviving_counts = np.sum(base_1e_mask_trap & ~combined_mask)
    #             unmasked_pix = np.sum(~combined_mask)
                
    #             self.stats_trap[combo_name]['counts'].append(surviving_counts)
    #             self.stats_trap[combo_name]['unmasked_pix'].append(unmasked_pix)

    #         # ----------------------------------------------------
    #         # NO TRAP IMAGES
    #         # ----------------------------------------------------
    #         img_notrap = self.no_trap_images[i]
    #         masks_notrap = notrap_local_masks[i]
            
    #         base_1e_mask_notrap = get_cluster(img_notrap, min_pixs=1, max_pixs=1, min_total_value=1, max_total_value=1)
            
    #         for subset in all_combos:
    #             combo_name = "+".join(subset) if subset else "None"
                
    #             combined_mask = np.zeros(shape, dtype=bool)
    #             for name in subset:
    #                 combined_mask |= masks_notrap[name]
                    
    #             surviving_counts = np.sum(base_1e_mask_notrap & ~combined_mask)
    #             unmasked_pix = np.sum(~combined_mask)
                
    #             self.stats_notrap[combo_name]['counts'].append(surviving_counts)
    #             self.stats_notrap[combo_name]['unmasked_pix'].append(unmasked_pix)

    #     print("--- Post-Run Masking Complete ---")


        #counts
        
        # og_image_halo = generate_halo_mask(self.no_trap_images[-1],threshold=100,radius=60)
        # trap_image_halo = generate_halo_mask(self.reconstructed_images[-1],threshold=100,radius=60)


        # counts_1e_trap_nomask  = get_cluster(self.reconstructed_images[-1],min_pixs=1,max_pixs=1,min_total_value=1,max_total_value = 1,return_count=True)

        # counts_1e_og_nomask  = get_cluster(self.no_trap_images[-1],min_pixs=1,max_pixs=1,min_total_value=1,max_total_value = 1,return_count=True)


        # masked_trap_image = np.copy(self.reconstructed_images[-1]).astype(float)

        # masked_og_image = np.copy(self.no_trap_images[-1]).astype(float)


        # masked_og_image[og_image_halo] = np.nan
        # masked_trap_image[trap_image_halo] = np.nan

        # counts_1e_trap  = get_cluster(masked_trap_image,min_pixs=1,max_pixs=1,min_total_value=1,max_total_value = 1,return_count=True)
        # counts_1e_og  = get_cluster(masked_og_image,min_pixs=1,max_pixs=1,min_total_value=1,max_total_value = 1,return_count=True)

        # unmasked_pix_traps = np.sum(~trap_image_halo)

        # unmasked_pix_notraps = np.sum(~og_image_halo)


        # self.single_e_counts.append(counts_1e_trap_nomask)
        # self.single_e_counts_masked.append(counts_1e_trap)
        # self.single_e_counts_no_traps.append(counts_1e_og_nomask)


        # self.single_e_counts_masked_no_traps.append(counts_1e_og)

        # self.unmasked_pixels.append(unmasked_pix_traps)
        # self.unmasked_pixels_no_traps.append(unmasked_pix_notraps)


    def simulate_readout(self, tpix_vertical=None):
        import numpy as np

        # A binned readout clocks faster, so callers (e.g. the binned 0 h images
        # in 'binned_0h' mode) can pass a shorter per-row dwell; default is the
        # nominal full-frame readout time.
        if tpix_vertical is None:
            tpix_vertical = self.tpix_vertical

        # print(f"Starting Readout...")
        # image = self.ccd_state
        image = self.ccd_state.copy()
        rows, cols = image.shape

        if tpix_vertical == self.tpix_vertical:
            trap_emit_probs = self.readout_emit_probs
        else:
            trap_emit_probs = 1.0 - np.exp(-tpix_vertical / self.trap_taus)

        # Use the Numba JIT compiled C-loop to eliminate the python iteration bottleneck
        result_flat = fast_readout_numba(
            image, self.exposure_accumulator, tpix_vertical,
            self.trap_indices[0], self.trap_indices[1],
            trap_emit_probs, self.trap_capture_alpha, self.trap_is_v3,
            self.trapped_charge_1d
        )
        self.ccd_state[:] = image

        result_reconstructed= result_flat.reshape(rows, cols)
        result_reconstructed = np.flipud(np.fliplr(result_reconstructed))

        # self.reconstructed_images.append(result_reconstructed)

        return result_reconstructed


def run_single_trial(
    r,
    tpix,
    tpix_vertical,
    tau_weights,
    tau_edges,
    pair_tau135,
    pair_sigma,
    runconditions='snolab',
    outdir='./',
    trap_density_scale=1.0,
    packet_volume_um3=3.0,
    phase_capture_ticks=300.0,
    exp_indep_charge_mode='pre_readout',
    clear_mode='sequencer',
    binning_0h_factor=32.0,
    exposure_order='shuffled',
    n_detected_traps=DEFAULT_N_DETECTED_TRAPS,
    tauhistfile='',
    pairsfile='',
    binning=1.0,
    zero_exp_dep_rate=False,
    v3_phase_fraction=0.5,
):
    """This function contains everything needed for a single trial"""
    import os
    valid_exposure_orders = ('shuffled', 'ordered')
    if exposure_order not in valid_exposure_orders:
        raise ValueError(
            f"Unknown exposure_order={exposure_order!r}; "
            f"expected one of {valid_exposure_orders}"
        )
    # Guarantee a unique PRNG sequence for Numba across all forked child processes
    seed_numba(os.getpid() + (r * 10000))
    # Also seed the legacy global np.random stream (np.random.poisson in
    # take_fake_image, np.random.random in charge_trap_interaction): fork-started
    # workers otherwise inherit identical state, correlating trials within a
    # node (review F10). Trap/no-trap CRN pairing is unaffected.
    np.random.seed((os.getpid() + (r * 10000)) % (2 ** 32))
    
    import h5py
    import os

    os.makedirs(outdir, exist_ok=True)

    filename = outdir + f'ccd_traps_run{r}.h5'
    if os.path.exists(filename):
        with h5py.File(filename, 'r') as existing:
            existing_mode = existing.attrs.get(
                'exp_indep_charge_mode',
                'pre_readout',
            )
            existing_clear_mode = existing.attrs.get(
                'clear_mode',
                'instantaneous',
            )
            existing_exposure_order = existing.attrs.get(
                'exposure_order',
                'shuffled',
            )
            existing_trap_transport_model = existing.attrs.get(
                'trap_transport_model',
                '',
            )
            existing_phase_capture_ticks = existing.attrs.get(
                'phase_capture_ticks',
                np.nan,
            )
            # Files written before the V1/V3 phase split default to NaN so
            # they can never silently mix with post-split runs in one dir.
            existing_v3_phase_fraction = existing.attrs.get(
                'v3_phase_fraction',
                np.nan,
            )
            existing_binning = existing.attrs.get('binning', 1.0)
        if isinstance(existing_mode, bytes):
            existing_mode = existing_mode.decode()
        if isinstance(existing_clear_mode, bytes):
            existing_clear_mode = existing_clear_mode.decode()
        if isinstance(existing_exposure_order, bytes):
            existing_exposure_order = existing_exposure_order.decode()
        if isinstance(existing_trap_transport_model, bytes):
            existing_trap_transport_model = existing_trap_transport_model.decode()
        try:
            existing_phase_capture_ticks = float(existing_phase_capture_ticks)
        except (TypeError, ValueError):
            existing_phase_capture_ticks = np.nan
        try:
            existing_binning = float(existing_binning)
        except (TypeError, ValueError):
            existing_binning = np.nan
        if existing_mode != exp_indep_charge_mode:
            raise RuntimeError(
                f"{filename} was generated with exp_indep_charge_mode="
                f"{existing_mode!r}, not {exp_indep_charge_mode!r}. "
                "Use a separate output directory."
            )
        if existing_clear_mode != clear_mode:
            raise RuntimeError(
                f"{filename} was generated with clear_mode="
                f"{existing_clear_mode!r}, not {clear_mode!r}. "
                "Use a separate output directory."
            )
        if existing_exposure_order != exposure_order:
            raise RuntimeError(
                f"{filename} was generated with exposure_order="
                f"{existing_exposure_order!r}, not {exposure_order!r}. "
                "Use a separate output directory."
            )
        if existing_trap_transport_model != TRAP_TRANSPORT_MODEL:
            raise RuntimeError(
                f"{filename} was generated with trap_transport_model="
                f"{existing_trap_transport_model!r}, not {TRAP_TRANSPORT_MODEL!r}. "
                "Use a separate output directory or regenerate this file."
            )
        if not np.isclose(
            existing_phase_capture_ticks,
            float(phase_capture_ticks),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"{filename} was generated with phase_capture_ticks="
                f"{existing_phase_capture_ticks!r}, not {phase_capture_ticks!r}. "
                "Use a separate output directory or regenerate this file."
            )
        if not np.isclose(existing_binning, float(binning), rtol=0.0, atol=1.0e-12):
            raise RuntimeError(
                f"{filename} was generated with binning={existing_binning!r}, "
                f"not {binning!r}. Use a separate output directory."
            )
        try:
            existing_v3_phase_fraction = float(existing_v3_phase_fraction)
        except (TypeError, ValueError):
            existing_v3_phase_fraction = np.nan
        if not np.isclose(
            existing_v3_phase_fraction,
            float(v3_phase_fraction),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"{filename} was generated with v3_phase_fraction="
                f"{existing_v3_phase_fraction!r}, not {v3_phase_fraction!r} "
                "(NaN = pre-phase-split file). "
                "Use a separate output directory or regenerate this file."
            )
        return r

    CCDTest = CCD(
        tpix,
        tpix_vertical,
        tau_weights,
        tau_edges,
        pair_tau135,
        pair_sigma,
        runconditions=runconditions,
        trap_density_scale=trap_density_scale,
        packet_volume_um3=packet_volume_um3,
        phase_capture_ticks=phase_capture_ticks,
        exp_indep_charge_mode=exp_indep_charge_mode,
        clear_mode=clear_mode,
        binning_0h_factor=binning_0h_factor,
        binning=binning,
        n_detected_traps=n_detected_traps,
        zero_exp_dep_rate=zero_exp_dep_rate,
        v3_phase_fraction=v3_phase_fraction,
    )

    # Order of the full image sequence. With the fixed 0->4->6->10->20 ordering
    # ('ordered'), every 0 h image was always preceded by a 20 h image (maximal
    # trap fill), so the trap occupancy entering each exposure slot was
    # systematically biased -- inflating the 0 h trap excess and tilting the
    # fitted exposure-dependent rate. The default 'shuffled' order permutes the
    # whole 500-image sequence (100 of each exposure) together rather than
    # cycle-by-cycle: within a single 5-element permutation, sampling without
    # replacement leaves a residual anticorrelation between an image's exposure
    # and its predecessor's, whereas a full shuffle draws each predecessor from
    # the entire pool, decoupling the trap background (sourced by prior
    # exposures) from the current exposure to the ~1/N level. 'ordered'
    # reproduces the old fixed-cycle behaviour for comparison. A dedicated,
    # per-trial-seeded Generator keeps the shuffle reproducible without
    # perturbing the global np.random / Numba streams used by the physics.
    n_cycles = 100
    do_shuffle = (exposure_order == 'shuffled')
    shuffle_rng = np.random.default_rng(r)
    if clear_mode == 'binned_0h':
        # No hardware clear and no nominal 0 h slot. The real (long) exposures
        # are taken in 'shuffled' or fixed 'ordered' order, and a binned 0 h
        # image is taken after every one of them to reset the array (replacing
        # the clear) and provide the 0 h baseline.
        real_exposures = [4, 6, 10, 20]
        real_sequence = real_exposures * n_cycles
        if do_shuffle:
            shuffle_rng.shuffle(real_sequence)
        full_sequence = []
        for exposure in real_sequence:
            full_sequence.append(exposure)
            full_sequence.append(0)
    else:
        exposure_schedule = [0, 4, 6, 10, 20]
        full_sequence = exposure_schedule * n_cycles
        if do_shuffle:
            shuffle_rng.shuffle(full_sequence)
    for exposure in full_sequence:
        CCDTest.take_fake_image(exposure)

    CCDTest.process_run()

    # Memory Cleanup
    CCDTest.reconstructed_images = []
    CCDTest.no_trap_images = []
    CCDTest.trap_bitmasks = []
    CCDTest.notrap_bitmasks = []
    CCDTest.ccd_state = []

    with h5py.File(filename, 'w') as f:
        f.create_dataset('exposures',         data=np.array(CCDTest.exposures))
        f.create_dataset('trap_taus',         data=CCDTest.trap_taus)
        f.create_dataset('trap_sigmas',       data=CCDTest.trap_sigmas)
        f.create_dataset('trap_is_v3',        data=CCDTest.trap_is_v3)
        f.create_dataset('trap_indices_rows', data=CCDTest.trap_indices[0].astype(np.int32))
        f.create_dataset('trap_indices_cols', data=CCDTest.trap_indices[1].astype(np.int32))
        f.create_dataset('tau_weights',       data=np.array(tau_weights))
        f.create_dataset('tau_edges',         data=np.array(tau_edges))
        f.create_dataset(
            'clear_occupied_traps_before',
            data=np.array(CCDTest.clear_occupied_traps_before, dtype=np.int32),
        )
        f.create_dataset(
            'clear_occupied_traps_after',
            data=np.array(CCDTest.clear_occupied_traps_after, dtype=np.int32),
        )
        f.create_dataset(
            'clear_surface_electrons_before',
            data=np.array(CCDTest.clear_surface_electrons_before),
        )
        f.create_dataset(
            'clear_surface_electrons_after_transport',
            data=np.array(CCDTest.clear_surface_electrons_after_transport),
        )
        f.attrs['trap_density'] = CCDTest.trap_density
        f.attrs['trap_density_scale'] = CCDTest.trap_density_scale
        f.attrs['n_detected_traps'] = CCDTest.n_detected_traps
        f.attrs['exp_dep_rate'] = CCDTest.exp_dep_rate
        f.attrs['zero_exp_dep_rate'] = CCDTest.zero_exp_dep_rate
        f.attrs['packet_volume_um3'] = CCDTest.packet_volume_um3
        f.attrs['trap_transport_model'] = CCDTest.trap_transport_model
        f.attrs['v3_phase_fraction'] = CCDTest.v3_phase_fraction
        f.attrs['phase_capture_ticks'] = CCDTest.phase_capture_ticks
        f.attrs['phase_capture_dwell_s'] = CCDTest.phase_capture_dwell_s
        f.attrs['temperature_K'] = CCDTest.temperature_K
        f.attrs['exp_indep_charge_mode'] = CCDTest.exp_indep_charge_mode
        f.attrs['clear_mode'] = CCDTest.clear_mode
        f.attrs['clear_sequence'] = CCDTest.clear_sequence
        f.attrs['clear_clock_hz'] = CCDTest.clear_clock_hz
        f.attrs['clear_vertical_phase_count'] = CCDTest.clear_vertical_phase_count
        f.attrs['clear_horizontal_phase_count'] = CCDTest.clear_horizontal_phase_count
        f.attrs['clear_delay_vertical_ticks'] = CCDTest.clear_delay_vertical_ticks
        f.attrs['clear_delay_horizontal_ticks'] = CCDTest.clear_delay_horizontal_ticks
        f.attrs['clear_delay_switch_ticks'] = CCDTest.clear_delay_switch_ticks
        f.attrs['clear_delay_reset_gate_ticks'] = CCDTest.clear_delay_reset_gate_ticks
        f.attrs['clear_fast_shifts'] = CCDTest.clear_fast_shifts
        f.attrs['clear_fast_horizontal_steps'] = CCDTest.clear_fast_horizontal_steps
        f.attrs['clear_fast_dwell_s'] = CCDTest.clear_fast_dwell_s
        f.attrs['clear_slow_shifts'] = CCDTest.clear_slow_shifts
        f.attrs['clear_slow_horizontal_steps'] = CCDTest.clear_slow_horizontal_steps
        f.attrs['clear_slow_dwell_s'] = CCDTest.clear_slow_dwell_s
        f.attrs['clear_total_time_s'] = CCDTest.clear_total_time_s
        f.attrs['clear_three_hour_seconds'] = CCDTest.clear_three_hour_seconds
        f.attrs['clear_three_hour_fast_shifts'] = CCDTest.clear_three_hour_fast_shifts
        f.attrs['binning_0h_factor'] = CCDTest.binning_0h_factor
        f.attrs['binning'] = CCDTest.binning
        f.attrs['exposure_order'] = exposure_order
        # Seed-catalog provenance: which (tau, sigma) pairs / tau histogram
        # seeded this run, and the derived analysis flavor. Lets downstream
        # plotting self-route by the catalog that produced the data rather than
        # relying solely on the rundir path heuristic.
        f.attrs['tauhistfile'] = tauhistfile
        f.attrs['pairsfile'] = pairsfile
        f.attrs['flavor'] = (
            'minimal_caldet'
            if 'minimal' in f"{tauhistfile} {pairsfile}".lower()
            else 'legacy'
        )
        for group_name, stats in [('stats_trap', CCDTest.stats_trap), ('stats_notrap', CCDTest.stats_notrap)]:
            grp = f.create_group(group_name)
            for combo_name, d in stats.items():
                cg = grp.create_group(combo_name)
                cg.create_dataset('counts',      data=np.array(d['counts'],      dtype=np.int32))
                cg.create_dataset('2e_counts',   data=np.array(d['2e_counts'],   dtype=np.int32))
                cg.create_dataset('unmasked_pix',data=np.array(d['unmasked_pix'],dtype=np.int32))

    return r
