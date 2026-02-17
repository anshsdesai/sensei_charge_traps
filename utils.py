from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from scipy import ndimage

from matplotlib import colors
from matplotlib import cm, ticker

# plotting specifications
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                            AutoMinorLocator)
from matplotlib.offsetbox import AnchoredText
import numpy as np
#Options


def set_default_plotting_params(fontsize=12,goldenx=16):

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tck
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
    from matplotlib.offsetbox import AnchoredText
    #Options
    params = {'text.usetex' : False,
            'font.size' : fontsize,
            'font.family' : 'serif',
            'figure.autolayout': True
            }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.labelsize']=fontsize
    golden = (1 + 5 ** 0.5) / 2
    goldenx = goldenx
    goldeny = goldenx / golden
    plt.rcParams['figure.figsize']=(goldenx,goldeny)
    return

def zoomed_image(coord,image,desired_shape):
    xmax,ymax = image.shape
    dpx,dpy = coord
    xcenter = desired_shape[0] // 2
    ycenter = desired_shape[1] // 2
    upper_xbound =  desired_shape[0] - xcenter
    upper_ybound =  desired_shape[1] - ycenter


    lower_xbound = xcenter
    lower_ybound = ycenter
    # print('lower bounds')
    # print(lower_xbound,lower_ybound)
    # print('upper bounds')
    # print(upper_xbound,lower_ybound)
    # print('coordinate of event')
    # print(coord)
    # print('dpx, dpy')
    # print(dpx,dpy)
    

    if dpx > lower_xbound:
        xsub = lower_xbound
    else:
        xsub = dpx

    if dpy > lower_ybound:
        ysub = lower_ybound
    else:
        ysub = dpx


    if dpx < xmax - upper_xbound:
        xadd = upper_xbound
    else:
        xadd = xmax - dpx


    if dpy < ymax - upper_ybound:
        yadd = upper_ybound
    else:
        yadd = ymax - dpy

    # print('lower x index','upper x index')
    # print(dpx-xsub,dpx+xadd)
    # print('lower y index','upper y index')
    # print(dpy-ysub,dpy+yadd)
    z_image = np.copy(image)
    z_image = z_image[dpx-xsub:dpx+xadd,dpy-ysub:dpy+yadd]
    return z_image

def distance_bw_coords(coord1,coord2):
    dist = np.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)
    return dist



def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def get_qdata(filepath,q):
    with fits.open(filepath) as hdul:
        data = hdul[q].data
    return data

def approximate_electronize(data,zero_peak_val):
    if type(zero_peak_val) == str:
        zero_peak_val = float(zero_peak_val)
    data = data / zero_peak_val
    data = np.round(data)
    return data.astype(int)

def filter_qdata(data):
    hist,bins= np.histogram(data,np.arange(-2000,2000))
    from scipy.optimize import curve_fit
    bin_centers = (bins[:-1] + bins[1:]) / 2
    popt, pcov = curve_fit(gaussian, bin_centers, hist)
    sigma = popt[2]
    mu = popt[1]
    gaussianwidth = mu+3*sigma

    mask = (data > -1*gaussianwidth) & (data < gaussianwidth)
    image = np.copy(data)
    image[mask] = 0
    return image

def crop_qdata(data,xlower=2,xupper=512,ylower=8,yupper=3080):
    data = data[xlower:xupper,ylower:yupper]
    return data

    
import numpy as np

def crop_numpy_array(array: np.ndarray, center: tuple, size: tuple) -> np.ndarray:
    """
    Crops a 2D NumPy array around a given center coordinate to the desired size,
    handling edges appropriately.

    Parameters:
    - array: 2D numpy array
    - center: tuple (row, col) indicating the center of the crop
    - size: tuple (height, width) indicating the desired output size

    Returns:
    - Cropped 2D numpy array
    """
    n_rows, n_cols = array.shape
    center_row, center_col = center
    crop_height, crop_width = size

    # Calculate half sizes
    half_height = crop_height // 2
    half_width = crop_width // 2

    # Determine the crop window (clipping to array bounds)
    start_row = max(center_row - half_height, 0)
    end_row = min(start_row + crop_height, n_rows)
    start_row = max(end_row - crop_height, 0)  # adjust near bottom

    start_col = max(center_col - half_width, 0)
    end_col = min(start_col + crop_width, n_cols)
    start_col = max(end_col - crop_width, 0)  # adjust near right edge

    return array[start_row:end_row, start_col:end_col]





def transplant_clusters(source_image, target_shape=(520, 3200), 
                                    count_threshold=20, max_aspect_ratio=3.0, 
                                    radius=20,exposure=72000,scale_factor = 1):
    from skimage.morphology import dilation, disk
    
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
    
    print(f"Exposure: {exposure}, Inserting approx {target_num_clusters} clusters out of {totClusters}")

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

        # ... (Rest of your code remains exactly the same) ...
        # ... EXPAND SLICE ...
        y_start_expanded = max(0, y_slice.start - radius)
        y_stop_expanded  = min(src_h, y_slice.stop + radius)
        x_start_expanded = max(0, x_slice.start - radius)
        x_stop_expanded  = min(src_w, x_slice.stop + radius)
        
        expanded_slice = (slice(y_start_expanded, y_stop_expanded), 
                          slice(x_start_expanded, x_stop_expanded))
        
        local_block = source_image[expanded_slice]
        local_labels = labeled_array[expanded_slice]
        
        specific_cluster_mask = (local_labels == label_id)
        
        # REPLACEMENT HERE:
        # 1. Create the footprint (structuring element) for the radius
        # specific_cluster_mask is boolean, so dilation works as a binary dilation
        footprint = disk(radius) 
        dilated_mask = dilation(specific_cluster_mask, footprint)
        
        final_cluster_chunk = local_block * dilated_mask
        
        
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

    

class Event:
    def __init__(self,coord,image,eventtype,ccd,quad,dtph,sc_shifts,intensity,systemName,highcoord,lowcoord):
        self.coord = coord
        self.image = image
        self.event_type = eventtype
        self.ccd = ccd
        self.quad = quad
        self.dtph = dtph
        self.sc_shifts = sc_shifts
        self.intensity = intensity
        self.highcoord = highcoord
        self.lowcoord = lowcoord
        self.system=systemName

    def plotEvent(self,save=False,cmap=None,vmin=None,vmax=None):
        plt.imshow(self.image,cmap=cmap,vmin=vmin,vmax=vmax)

        ax = plt.gca()
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                text = ax.text(j, i, np.round(self.image[i, j],2), ha="center", va="center", color="red",fontsize=12)

        zi  = self.image
        coord = self.coord
        rowlength = zi.shape[0]
        collength = zi.shape[1]


        rowdiff = rowlength- rowlength//2
        rowticks = np.arange(rowlength)
        rowlabels = rowticks + coord[0] - rowdiff

        coldiff = zi.shape[1]- collength//2
        colticks = np.arange(0,collength)
        collabels = np.arange(len(colticks)) + coord[1] - coldiff

        if np.min(collabels) < 0:
            collabels += (0-np.min(collabels))

        if np.min(rowlabels) < 0:
            rowlabels += (0-np.min(rowlabels))

        if rowlength > 5:
            rowticks = rowticks[::2]
            rowlabels = rowlabels[::2]
        if collength > 5:
            colticks = colticks[::2]
            collabels = collabels[::2]



                

        plt.yticks(ticks = rowticks,labels=rowlabels)
        plt.ylabel('Row Coordinate')

        plt.xticks(ticks=colticks,labels=collabels)
        plt.xlabel('Column Coordinate')

        
        plt.title(f"{self.event_type} Event Candidate at {self.coord}, Mag = {self.intensity}")
        
        if not save:    
            plt.show()
        else:
            plt.savefig(f'plotting/eventType_{self.event_type}_coords{self.coord}_ccd{self.ccd}_quad{self.quad}.png')
        plt.close()
                
def get_fourier(image):
    import math
    import numpy as np

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    mag = 20*np.log(np.abs(fshift))

    # Fourier Transform along the first axis
    # Round up the size along this axis to an even number
    ny = int( math.ceil(image.shape[0] / 2.) * 2 )
    # We use rfft since we are processing real values
    ay = np.fft.rfft(image,ny, axis=0)
    #sum power along second axis
    ay = ay.real*ay.real + ay.imag*ay.imag
    ay = ay.sum(axis=1)/ay.shape[1]
    # Generate a list of frequencies
    fy = np.fft.rfftfreq(ny)

    nx = int( math.ceil(image.shape[1] / 2.) * 2 )
    ax = np.fft.rfft(image,nx,axis=1)
    ax = ax.real*ax.real + ax.imag*ax.imag
    ax = ax.sum(axis=0)/ax.shape[0]
    fx = np.fft.rfftfreq(nx)
    return [mag,fx,ax,fy,ay]

def image_properties(image,imagename=None,showimage=False,showHist=False,showcharge=False,showFourier=False,cmap='viridis',logimage=False):
    import matplotlib.pyplot as plt
    data = []
    avgs = []
    xlower = None
    xupper = None
    ylower = None
    yupper = None
    if imagename is not None:
        title = imagename
    else:
        title = ''

    hist_upper = int(np.nanmean(image) + 2000)
    hist_lower = int(np.nanmean(image) - 2000)
    if showimage:
        plt.imshow(image,cmap=cmap,vmin=0,vmax=hist_upper,origin='lower')
        plt.title(title)
        plt.show()
        
        plt.close()
    if logimage:
        log_friendly_image = np.copy(image)
        log_friendly_image[image <= 0] = 1
        plt.imshow(log_friendly_image,cmap=cmap,norm=colors.LogNorm(),origin='lower')
        plt.title(title)
        plt.show()
        plt.close()

    

    hist,bins= np.histogram(image,np.arange(hist_lower,hist_upper))
    mids = 0.5*(bins[1:] + bins[:-1])
    histmean = np.average(mids, weights=hist)
    var = np.average((mids - histmean)**2, weights=hist)

    histmean = np.round(histmean,3)
    var = np.round(var,3)


    if showHist:
        plt.stairs(hist,bins)
        plt.title(f'Mean = {histmean}, Variance = {var}')
        plt.show()
        plt.close()
    if showcharge:
        means = []
        for i in range(image.shape[0]):
            means.append(np.mean(image[i,:]))
        means = np.array(means)
        plt.plot(means)
        plt.xlabel('Row')
        plt.ylabel('Mean Column Charge')
        plt.show()
        plt.close()
        means = []
        for i in range(image.shape[1]):
            means.append(np.mean(image[:,i]))
        means = np.array(means)
        plt.plot(means)
        plt.xlabel('Column')
        plt.ylabel('Mean Row Charge')
        plt.show()
        plt.close()
    if showFourier:
        fourier_data = get_fourier(image)
        mag = fourier_data[0]
        fx = fourier_data[1]
        ax = fourier_data[2]
        fy = fourier_data[3]
        ay = fourier_data[4]

        # [mag,fx,ax,fy,ay]

        plt.imshow(mag,cmap=cmap)
        plt.show()
        plt.close()

        plt.plot(fx[1:],ax[1:],label = 'Horizontal')
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.yscale('log')
        plt.title('Horizontal Power Spectrum')
        plt.show()
        plt.close()
        plt.plot(fy[1:],ay[1:],label = 'Vertical')
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.yscale('log')

        plt.title('Vertical Power Spectrum')
        plt.show()
        plt.close()
    if showHist:
        return hist,bins
    return

def plot_histogram(image,bins):
    hist,bins = np.histogram(image,bins)
    plt.stairs(hist,bins)
    plt.show()
    plt.close()
    return hist,bins

def comparable(val1,val2,tolerance=1000):
    mag1 = np.abs(val1)
    mag2 = np.abs(val2)
    if mag1 < mag2 + tolerance and mag1 > mag2 - tolerance:
        return True
    else:
        return False


def generate_halo_mask(image, threshold=100, radius=60):
    import numpy as np
    from skimage.morphology import disk, dilation  # Updated import
    
    # 1. Find pixels that exceed the threshold
    hot_pixels = image > threshold
    
    # 2. Create a circular footprint
    footprint = disk(radius)
    
    # 3. Dilate the hot pixels
    # dilation() handles boolean inputs automatically
    mask = dilation(hot_pixels, footprint)
    
    return mask


def get_cluster(proc, min_pixs=2, max_pixs=None, min_total_value=3,max_total_value = None, return_count = False, return_median= False):#, min_max_value=None
    from scipy.stats import binned_statistic
    from cv2 import connectedComponents
    
    m = proc > 0.7
    
    # connectedComponents returns the number of labels including the background (label 0)
    num_labels, clusters = connectedComponents(m.astype(np.uint8))
    
    # Subtract 1 to ignore the background
    max_index = num_labels - 1
    
    # --- FIX: Handle case with no clusters ---
    if max_index == 0:
        if return_count:
            return 0
        if return_median:
            return np.nan # Or 0, depending on your preference
        return np.zeros_like(proc, dtype=bool) # Return empty mask
    # -----------------------------------------

    all_clusters = clusters.flatten()
    all_values = proc.flatten()
    
    counts, bins, binned = binned_statistic(all_clusters, all_values, 'count', bins=max_index,
                                            range=(1, max_index + 1))
    sums, bins, binned = binned_statistic(all_clusters, all_values, 'sum', bins=max_index, range=(1, max_index + 1))
    
    bins = bins[:-1]
    
    if max_pixs is not None:
        conditions = (counts >= min_pixs) & (sums >= min_total_value) & (counts <= max_pixs)
    else: 
        conditions = (counts >= min_pixs) & (sums >= min_total_value)
        
    if max_total_value is not None:
        conditions = conditions & (sums <= max_total_value)
        
    if return_count:
        return np.count_nonzero(conditions)
    if return_median:
        # Check if conditions is not empty to avoid warnings/errors
        valid_sums = sums[conditions]
        if len(valid_sums) == 0:
            return np.nan
        return np.median(valid_sums)
        
    cond_bins = bins[conditions]

    layer = np.isin(clusters, cond_bins)
    return layer
        

def ratio_pixels(m, yx, action = None):
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


# def getDipoleList(image_dir,q,plot=False):
#     from matplotlib.lines import Line2D
#     import glob
#     import re

#     colors = ['red','blue','green','orange']
#     dipole_list = []
#     for temp in ['135k','140k','150k','160k','170k','180k']:#['135k','140k','150k','160k','170k']:#,'180k']:

#         if 'minos' in image_dir:
#             eval = 400
#             minos=True
#             system = 'Minos'
#             ccd =2
#         else:
#             eval = 200
#             system = 'Cross1'
#             ccd = 1
#             minos = False

#         print(f'Using {system} settings' )
        
            
#         if plot:
#             fig = plt.figure(figsize=(12,8))  

#         for dtph in  np.array([500,1000,10000,100000,1000000,2500000]):
#             if minos:
#                 search_str = image_dir + f'proc*{temp}*_'  + '*_2_*' # +
#             else:
#                 search_str = image_dir + f'proc*{temp}*_'
#             try:
#                 imagefile = glob.glob(search_str)[0]
#                 print(imagefile)
#             except IndexError:
#                 #file is likely not there
#                 print('file not found')
#                 continue
#             # print(imagefile)
#             # try:
#             #     dtph = int(re.findall('dtph\d*_',imagefile)[0][4:-1])
#             # except IndexError:
#             #     dtph = 0

#             sc_shifts = int(re.findall('SC\d*_',imagefile)[0][2:-1])
#             image = get_qdata(imagefile,q)
#             image = crop_qdata(image)#,ylower=500,xlower=100)
#             meanimage = np.mean(image)
#             image = image - meanimage
#             image = approximate_electronize(image,eval)
#             meanimage = int(np.round(meanimage/ eval))
            
#             #for each row
            
        
#             for c,cutoff in enumerate([1500,2500,3500,4000]):
#                 coordlist = []
#                 # for q in [2]:
                

#                 for i in np.arange(1,image.shape[0]):

#                     test_row = i


#                     row1 = image[test_row,:]
#                     row0 = image[test_row-1,:]

#                     diff = row1 - row0


#                     #filter
#                     absdiff = np.abs(diff)
#                     potential_locations = np.where(absdiff > cutoff)[0]
#                     # potential_locations_check1 = len(np.where(absdiff > 1500)[0])
#                     # potential_locations_check2 = len(np.where(absdiff > 2500)[0])
#                     # if potential_locations_check2  > potential_locations_check1:
#                     #     print('what the fuck')
#                     #     print(imagefile)

                


#                     #for each location mark as a star or a dipole
#                     for col in potential_locations:
#                         coord = (test_row,col)
#                         coord_b = (test_row - 1,col)


                        
                        
                        

#                         if image[coord]*image[coord_b] < 0 and comparable(image[coord],image[coord_b],tolerance=1000):#possible candidate
#                                 zim =zoomed_image(coord,image,(4,5)) + meanimage
#                                 mag = np.abs(image[coord] - image[coord_b])
#                                 minc = coord_b if image[coord_b] < image[coord] else coord
#                                 maxc = coord_b if image[coord_b] > image[coord] else coord
#                                 dipole = Event(coord,zim,'Dipole',ccd,q,dtph,sc_shifts,mag,system,maxc,minc)
#                                 # if cutoff == 4000:
#                                 #     dipolelist.append(dipole)
            
#                         else: #star
#                             continue

#                         coordlist.append(coord)
#                 unique_items = list(set(coordlist)) 
#                 dipole_list += unique_items
#                 num_dipoles = len(unique_items)
#                 if plot:
#                     plt.scatter(dtph,num_dipoles,label=cutoff,color=colors[c])
            
            

#         if plot:
#             line1 = Line2D([0], [0], color=colors[0], lw=2, label='Threshold = 1500 ')
#             line2 = Line2D([0], [0], color=colors[1], lw=2, label='Threshold = 2500 ')
#             line3 = Line2D([0], [0], color=colors[2], lw=2, label='Threshold = 3500 ')
#             line4 = Line2D([0], [0], color=colors[3], lw=2, label='Threshold = 4000 ')

#             # Add the lines to the legend
#             plt.legend(handles=[line1,line2,line3,line4])
#             plt.xscale('log')
#             plt.ylabel('Number of Dipoles')
#             plt.xlabel("DTPH")
#             plt.title(f"{system} @ Temperature = {temp}")
#             plt.ylim(0,1200)
#             plt.show()
#             plt.close()
            
#     final_dipole_list = list(set(dipole_list))
#     return final_dipole_list


# def findDipoles(electronized_image,cutoffs):
#     image = electronized_image
#     meanimage = int(np.mean(image))
#     image = image - meanimage
#     dipole_list= {}
#     for c,cutoff in enumerate(cutoffs):
#         coordlist = []

#         for i in np.arange(1,image.shape[0]):
#             test_row = i


#             row1 = image[test_row,:]
#             row0 = image[test_row-1,:]

#             diff = row1 - row0


#             #filter
#             absdiff = np.abs(diff)
#             potential_locations = np.where(absdiff > cutoff)[0]
#             for col in potential_locations:
#                 coord = (test_row,col)
#                 coord_b = (test_row - 1,col)
#                 if image[coord]*image[coord_b] < 0 and comparable(image[coord],image[coord_b],tolerance=1000):#possible candidate
#                     zim =zoomed_image(coord,image,(4,5)) + meanimage
#                     mag = np.abs(image[coord] - image[coord_b])
#                     minc = coord_b if image[coord_b] < image[coord] else coord
#                     maxc = coord_b if image[coord_b] > image[coord] else coord
#                 else:
#                     #star or something
#                     continue
#                 coordlist.append(coord)

#         unique_items = list(set(coordlist)) 
#         dipole_list[cutoff] = unique_items
#     return dipole_list

#                     # dipole = Event(coord,zim,'Dipole',ccd,q,dtph,sc_shifts,mag,system,maxc,minc)



# def getDipoleList(image_dir,temperatures,goodquads,plot=False,cutoff=2500,cutoffs = [1500,2000,2500,3000,3500,4000,4500,5000]):
#     from matplotlib.lines import Line2D
#     import matplotlib.pyplot as plt
#     import matplotlib.colors as colors 
#     import re
#     import glob
#     if 'minos' in image_dir:
#         eval = 400
#         minos=True
#         system = 'Minos'
#         ccd =2
#         ccd_str = '*_2_*'
#     else:
#         eval = 200
#         system = 'Cross1'
#         ccd = 1
#         ccd_str = '*'
#         minos = False

#     norm = colors.Normalize(vmin=cutoffs[0],vmax=cutoffs[-1])
#     cmap= plt.cm.cividis
#     full_dipole_list = []
#     for q in goodquads:
#         dipole_list = []
#         for temperature in temperatures:
#             temp = f'{temperature}k'
#             search_str = image_dir + f'proc*{temp}*_' + '*dtph*' + ccd_str 
#             imagefiles = glob.glob(search_str)

#             if plot:
#                 fig = plt.figure(figsize=(12,8))
            
#             for imagefile in imagefiles:
#                 dtph = int(re.findall('dtph\d+_',imagefile)[0][4:-1])
#                 image = get_qdata(imagefile,q)
#                 image = crop_qdata(image)#,ylower=500,xlower=100)
#                 image = approximate_electronize(image,eval)
#                 dipoledict = findDipoles(image,cutoffs)
#                 if plot:
#                     for c,cf in enumerate(list(dipoledict.keys())):
#                         dp = dipoledict[cf]
#                         num_dipoles = len(dp)
#                         ts = dtph/(15*1e6)

#                         plt.scatter(ts,num_dipoles,label=cf,c=cf,cmap=cmap,norm=norm)



#             if plot:
#                 # line1 = Line2D([0], [0], color=colors[0], lw=2, label='Threshold = 1500 ')
#                 line_list = []
#                 for i,c in enumerate(cutoffs):

#                     line_list.append(Line2D([0], [0], color=cmap(norm(c)), lw=2, label=f'Threshold = {c}'))
        

#                 # Add the lines to the legend
#                 plt.legend(handles=line_list)
#                 plt.xscale('log')
#                 plt.ylabel('Number of Dipoles')
#                 plt.xlabel("$t_{ph}$")
#                 plt.title(f"{system} Quad {q+1} @ {temp}")
#                 plt.ylim(0,1500)
#                 plt.show()
#                 plt.close()

#             dipoles = dipoledict[cutoff]

            
#             dipole_list += dipoles


#         final_dipole_list = list(set(dipole_list))
#         full_dipole_list.append(final_dipole_list)

#     return full_dipole_list
    
    



        
# # def getdipoleSpectra(image_dir,dipole_coord_list,q,plotdipole=False,debug=False):
# #     from collections import defaultdict
# #     import glob
# #     import re
# #     dipole_dict = {}
# #     if 'minos' in image_dir:
# #         eval = 400
# #         minos=True
# #         system = 'Minos'
# #         ccd =2
# #         ccd_str = '*_2_*'
# #     else:
# #         eval = 200
# #         system = 'Cross1'
# #         ccd = 1
# #         ccd_str = '*'
# #         minos = False

# #     search_str = image_dir + f'proc*_' + '*dtph*' + ccd_str 

# #     imagefiles = glob.glob(search_str)
# #     temps = []
# #     dtphs = []

# #     for imagefile in imagefiles:
# #         dtph = int(re.findall('dtph\d+_',imagefile)[0][4:-1])
# #         temp = int(re.findall('_\d+k_',imagefile)[0][1:-2])
# #         temps.append(temp)
# #         dtphs.append(dtph)
# #     dtphs = np.array(dtphs)
# #     temps = np.array(temps)
# #     temps = np.unique(np.sort(temps))
# #     dtphs = np.unique(np.sort(dtphs))

# #     search_str = image_dir + f'proc*_{temp}k*' + f'*dtph{dtph}*' + ccd_str 



        
# #     for dpcoord in dipole_coord_list:
# #         dipole_dict[dpcoord] = {}
# #         for temp in temps:
# #             dipole_dict[dpcoord][temp] = {}
# #             intensities = []
# #             for dtph in dtphs:
            
# #                 search_str = image_dir + f'proc*_{temp}k*' + f'*dtph{dtph}*' + ccd_str 
# #                 if debug:
# #                     print(dtph,temp)
# #                     print(search_str)
# #                 if dtph == 7000 or dtph == 8000:
# #                     continue
# #                 imagefile = glob.glob(search_str)[0]
# #                 image = get_qdata(imagefile,q)
# #                 image = crop_qdata(image)
# #                 image = approximate_electronize(image,eval)
# #                 coord_b = (dpcoord[0]-1,dpcoord[1])
# #                 intensity = np.abs(image[dpcoord] -image[coord_b])
# #                 if plotdipole:
# #                     zim = zoomed_image(dpcoord,image,(5,6))
# #                     plt.figure()
# #                     ax = plt.gca()
# #                     plt.imshow(zim)
# #                     for i in range(zim.shape[0]):
# #                         for j in range(zim.shape[1]):
# #                             text = ax.text(j, i, np.round(zim[i, j],2), ha="center", va="center", color="red",fontsize=12)
# #                             # text = ax.text(6//2, 5//2-1, 'coord_b', ha="center", va="center", color="red")
# #                     # text = ax.text(6//2, 5//2, 'dpcoord', ha="center", va="center", color="red")
# #                     print("INTENSITY")
# #                     print(intensity)
# #                     plt.show()
# #                     plt.close()
                
                
# #                 intensities.append(intensity)
# #             intensities = np.array(intensities)
# #             dipole_dict[dpcoord][temp]['dtphs'] = dtphs
# #             dipole_dict[dpcoord][temp]['intensities'] = intensities


                


            
# #             # if temp not in dipole_dict.keys():
# #             #     dipole_dict[temp]= {}
# #             #     dipole_dict[temp]['dtphs']= []
# #             #     dipole_dict[temp]['intensities']= []


             
# #             # dipole_dict[temp]['dtphs'].append(dtph)
# #             # dipole_dict[temp]['intensities'].append(intensity)

# #     return dipole_dict
# def gauss(x,mean,sigma,constant):
#     return constant*np.exp(-0.5 * (((x-mean)**2) / (sigma**2)))


# def comparable_perc(a,b,perc=0.3):
#     if a == b:
#         return True
#     max_val = max(abs(a), abs(b))
#     if max_val == 0:
#         return False  # Avoid division by zero
#     percent_diff = abs(a - b) / max_val
#     return percent_diff < perc


# def findDipoles2(electronized_image,debug=False,useFit=False):
#     import ctypes
#     from ROOT import TH1D, TF1, TCanvas


#     dipole_list = []

#     hist_upper = int(np.nanmean(electronized_image) + 2000)
#     hist_lower = int(np.nanmean(electronized_image) - 2000)
#     nbins = 200
#     step_length = int((hist_upper - hist_lower) / nbins)
#     bins_ = np.arange(hist_lower,hist_upper,step_length)

#     hist,bins= np.histogram(electronized_image,bins_)
#     mids = 0.5*(bins[1:] + bins[:-1])
#     histmean = np.average(mids, weights=hist)

#     var = np.average((mids - histmean)**2, weights=hist)
#     histmean = np.round(histmean,2)
#     if useFit:
#         image_flat = electronized_image.flatten().astype(np.float64)

#         h = TH1D(f"temp",f"Charge", nbins, hist_lower, hist_upper)

#         weights = np.ones_like(image_flat)
#         # h.FillN(image_flat.size, image_flat, np.ones(image_flat.size))
#         # Get C-compatible pointers
#         x_ptr = image_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#         w_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

#         h.FillN(image_flat.size, x_ptr, w_ptr)

#         h.Fit("gaus","Q")
#         fit_function = h.GetFunction("gaus")
#         constant = fit_function.GetParameter(0)
#         mean = fit_function.GetParameter(1)
#         sigma = fit_function.GetParameter(2)
#     else:
#         sigma= np.sqrt(var)
        
#     sigma_cutoff =(3*sigma)**2
#     sigma_cutoff *= -1

#     if debug:
#         plt.figure()
#         xs = np.linspace(hist_lower,hist_upper,nbins)
#         plt.plot(xs,gauss(xs,mean,sigma,constant),lw=3)
#         plt.title(f"$\mu = {mean} \sigma={sigma}$")
#         plt.stairs(hist,bins)
#         plt.show()
#         plt.close()


#     median_charge_per_row = np.median(electronized_image,axis=1)

#     image = electronized_image.T -median_charge_per_row
#     image = image.T #image with median charge per row subtracted
#     for i in np.arange(1,image.shape[0]):



#         row1 = image[i,:]
#         row0 = image[i-1,:]

#         diff = row1 - row0
#         multipl =row1*row0
#         if debug:
#             print(multipl,sigma_cutoff)

#         potential_locations = np.where(multipl < sigma_cutoff)[0]

#     # potential_dipole_locations = np.where(image < sigma_cutoff)

#         if len(potential_locations) == 0:
#             continue
#         for col in potential_locations:

#             coord = (i,col)
#             coord_b = (i - 1,col)
#             charge1 = np.abs(image[coord])
#             charge2 = np.abs(image[coord_b])
#             if debug:
#                 print(charge1,charge2)

#             if comparable_perc(charge1,charge2):
#                 dipole_list.append(coord)
#             else:
#                 if debug:
#                     print('dipole did not match percentage threshold')
#                     print(coord)
#                 continue

#                 #star or bad dipole or something

#     dipole_list = list(set(dipole_list)) 

#     return dipole_list





#         #     if image[coord]*image[coord_b] < 0 and comparable(image[coord],image[coord_b],tolerance=1000):#possible candidate
#         #         zim =zoomed_image(coord,image,(4,5)) + meanimage
#         #         mag = np.abs(image[coord] - image[coord_b])
#         #         minc = coord_b if image[coord_b] < image[coord] else coord
#         #         maxc = coord_b if image[coord_b] > image[coord] else coord
#         #     else:
#         #         #star or something
#         #         continue
#         #     coordlist.append(coord)

#         # unique_items = list(set(coordlist)) 
#         # dipole_list[cutoff] = unique_items


# def getDipoleList2(image_dir,temperatures,goodquads,plot=False):

#     from collections import defaultdict
#     from matplotlib.lines import Line2D
#     import matplotlib.pyplot as plt
#     import matplotlib.colors as colors 
#     import re
#     import glob
#     # if 'minos' in image_dir:
#     eval = 400
#         # minos=True
#         # system = 'Minos'
#         # ccd =2
#     ccd_str = '*_2_*'
#     # else:
#     #     eval = 200
#     #     system = 'Cross1'
#     #     ccd = 1
#     #     ccd_str = '*'
#     #     minos = False

    

#     # norm = colors.Normalize(vmin=cutoffs[0],vmax=cutoffs[-1])
#     # cmap= plt.cm.cividis
#     full_dipole_list = []
#     total_dipoles = [0,0,0,0]
#     total_traps = [0,0,0,0]
#     for q in goodquads:
#         dipole_list = []
#         for temperature in temperatures:
#             print(f"temperature: {temperature}")
#             dipole_occurrences = defaultdict(set)
#             all_dipoles = []
#             temp = f'{temperature}k'
#             search_str = image_dir + f'proc*{temp}*_' + '*dtph*' + ccd_str 
#             imagefiles = glob.glob(search_str)

#             if plot:
#                 fig = plt.figure(figsize=(12,8))
            
#             for imagefile in imagefiles:
#                 dtph = int(re.findall('dtph\d+_',imagefile)[0][4:-1])
#                 image = get_qdata(imagefile,q)
#                 image = crop_qdata(image)#,ylower=500,xlower=100)
#                 image = approximate_electronize(image,eval)
#                 image_dipoles = findDipoles2(image)
#                 for dipole in image_dipoles:
#                     dipole_occurrences[tuple(dipole)].add(dtph)
#                 #need to filter dipoles further: dipoles that appear in more than two images at same temperature with different dtph 

#                 all_dipoles += image_dipoles
#             all_dipoles = list(set(all_dipoles))

#             good_dipoles = [coord for coord, dtphs in dipole_occurrences.items() if len(dtphs) > 1]
#             dipole_list += good_dipoles

#             print('# All Dipoles',len(all_dipoles))
#             print('# Good Dipoles',len(good_dipoles))
#             print("# Anomalous Dipoles",len(all_dipoles) - len(good_dipoles))
            

#         print(f'Number of total dipoles quadrant {q}: ',len(dipole_list))
#         total_dipoles[q] +=len(dipole_list)
#         final_dipole_list = list(set(dipole_list))
#         print(f'Number of traps quadrant {q}: ',len(final_dipole_list))
#         total_traps[q] +=len(final_dipole_list)

#         full_dipole_list.append(final_dipole_list)
#     print('Total Dipoles')
#     print(np.sum(total_dipoles))
#     print("Total Traps")
#     print(np.sum(total_traps))

#     return full_dipole_list
    



# def getDipoleSpectra(image_dir,goodquads,full_dipole_coord_list,absolute=False):
#     import glob
#     import re
    
#     full_dipole_dict = {}
#     for quad in goodquads:
#         dp_dict = {}
#         dipole_coord_list = full_dipole_coord_list[quad]
#         if 'minos' in image_dir:
#             eval = 400
#             minos=True
#             system = 'Minos'
#             ccd =2
#             ccd_str = '*_2_*'
#         else:
#             eval = 200
#             system = 'Cross1'
#             ccd = 1
#             ccd_str = '*'
#             minos = False
#         search_str = image_dir + f'proc*_' + '*dtph*' + ccd_str 
#         imagefiles = glob.glob(search_str)
#         temps = []
#         for dp in dipole_coord_list:
#             dp_dict[dp] = {}

#         for imagef in imagefiles:
#             dtph = int(re.findall('dtph\d+_',imagef)[0][4:-1])
#             temp = int(re.findall('_\d+k',imagef)[0][1:-1])
#             image = get_qdata(imagef,quad)
#             image = crop_qdata(image)
#             image = approximate_electronize(image,eval)

#             hist_upper = int(np.nanmean(image) + 2000)
#             hist_lower = int(np.nanmean(image) - 2000)
#             hist,bins= np.histogram(image,np.arange(hist_lower,hist_upper))
#             mids = 0.5*(bins[1:] + bins[:-1])
#             histmean = np.average(mids, weights=hist)
#             var = np.average((mids - histmean)**2, weights=hist)
#             sigma = np.sqrt(var)
            

#             for dp in dipole_coord_list:
#                 if temp not in dp_dict[dp].keys():
#                     dp_dict[dp][temp] = {}
#                     dp_dict[dp][temp]['intensities'] = []
#                     dp_dict[dp][temp]['dtphs'] = []
#                     dp_dict[dp][temp]['intensity_err'] = []


#                 coord_b = (dp[0]-1,dp[1])
#                 intensity = (image[dp] -image[coord_b]) / 2
#                 if absolute:
#                     intensity = np.abs(intensity)
                
#                 dp_dict[dp][temp]['intensities'].append(intensity)
#                 dp_dict[dp][temp]['dtphs'].append(dtph)
#                 dp_dict[dp][temp]['intensity_err'].append(sigma)


#         full_dipole_dict[quad] = dp_dict



#     return full_dipole_dict



# def getDipoleSpectra2(image_dir,goodquads,full_dipole_coord_list,absolute=True):
#     import glob
#     import re
    
#     full_dipole_dict = {}
#     for quad in goodquads:
#         dp_dict = {}
#         dipole_coord_list = full_dipole_coord_list[quad]
#         # if 'minos' in image_dir:
#         eval = 400
        
#         ccd_str = '*_2_*'
#         # else:
#         #     eval = 200
#         #     system = 'Cross1'
#         #     ccd = 1
#         #     ccd_str = '*'
#         #     minos = False
#         search_str = image_dir + f'proc*_' + '*dtph*' + ccd_str 
#         imagefiles = glob.glob(search_str)
#         temps = []
#         for dp in dipole_coord_list:
#             dp_dict[dp] = {}

#         for imagef in imagefiles:
#             dtph = int(re.findall('dtph\d+_',imagef)[0][4:-1])
#             temp = int(re.findall('_\d+k',imagef)[0][1:-1])
#             image = get_qdata(imagef,quad)
#             image = crop_qdata(image)
#             image = approximate_electronize(image,eval)

#             median_charge_per_row = np.median(image,axis=1)

#             image = image.T -median_charge_per_row
#             image = image.T #image with median charge per row subtracted



#             hist_upper = int(np.nanmean(image) + 2000)
#             hist_lower = int(np.nanmean(image) - 2000)
#             hist,bins= np.histogram(image,np.arange(hist_lower,hist_upper))
#             mids = 0.5*(bins[1:] + bins[:-1])
#             histmean = np.average(mids, weights=hist)
#             var = np.average((mids - histmean)**2, weights=hist)
#             sigma = np.sqrt(var) / 2
            

#             for dp in dipole_coord_list:
#                 if temp not in dp_dict[dp].keys():
#                     dp_dict[dp][temp] = {}
#                     dp_dict[dp][temp]['intensities'] = []
#                     dp_dict[dp][temp]['dtphs'] = []
#                     dp_dict[dp][temp]['intensity_err'] = []


#                 coord_b = (dp[0]-1,dp[1])
#                 intensity = (image[dp] -image[coord_b]) / 2
#                 if absolute:
#                     intensity = np.abs(intensity)
                
#                 dp_dict[dp][temp]['intensities'].append(intensity)
#                 dp_dict[dp][temp]['dtphs'].append(dtph)
#                 dp_dict[dp][temp]['intensity_err'].append(sigma)



#         #sort by dtph
#         for dp in dp_dict.keys():
#             for temp in dp_dict[dp].keys():
#                 intensities = np.array(dp_dict[dp][temp]['intensities'])
#                 intensity_err = np.array(dp_dict[dp][temp]['intensity_err'])

#                 dtphs = np.array(dp_dict[dp][temp]['dtphs'])
#                 seconds = dtphs / 15e6



#                 seconds = seconds[np.argsort(dtphs)]
#                 intensities = intensities[np.argsort(dtphs)]
#                 intensity_err = intensity_err[np.argsort(dtphs)]
#                 dtphs = dtphs[np.argsort(dtphs)]

#                 dp_dict[dp][temp]['intensities'] = intensities
#                 dp_dict[dp][temp]['intensity_err'] = intensity_err
#                 dp_dict[dp][temp]['seconds'] = seconds
#                 dp_dict[dp][temp]['dtphs'] = dtphs





#         full_dipole_dict[quad] = dp_dict



#     return full_dipole_dict

# def intensity_function(tph,coeff,tau):
#     npumps = 3000
#     d_t = 1
#     p_c = 1
#     return npumps*coeff*(np.exp(-tph / tau) - np.exp(-8 * (tph/tau)))


# def energy_cross_section(temps,E,logsigma):
#     kb = 8.6717333262e-5 #eV/K
#     h = 4.1135e-15 #eV/s
#     me = 0.511e6 #eV
#     ccms = 3e10
#     denom = 2*(me* np.sqrt(3) * (2 * np.pi)**(3/2))
#     kbT = kb * temps
#     scaling_factor =  (h**3) * (ccms**2) / denom
#     taus = (scaling_factor / (np.exp(logsigma)* (kbT)**2)) * np.exp(E / kbT)
#     return taus


# def log_energy_cross_section(temperatures,E,logsigma):
#     kb = 8.6717333262e-5 #eV/K
#     h = 4.1135e-15 #eV/s
#     me = 0.511e6 #eV
#     ccms = 3e10
#     denom = 2*(me* np.sqrt(3) * (2 * np.pi)**(3/2))
#     kbT = kb * temperatures
#     scaling_factor =  (h**3) * (ccms**2) / denom
#     logtaus = np.log(scaling_factor) - logsigma - 2 * np.log(kbT) + E  / kbT
#     return logtaus
    



# def fitDipoles(full_dipole_dict,r2_tol = 0.15,chi2_tol = 1000,usechi2=False,errobars=True):
#     from scipy.optimize import curve_fit
#     full_chi2_list = []
#     for q in list(full_dipole_dict.keys()):
#         chi_2_list = []
#         dipole_dict = full_dipole_dict[q]
#         for dp in list(dipole_dict.keys()):
#             dptest = dipole_dict[dp]
#             numgood = 0
#             temperatures = []
#             taus = []
#             tau_errs = []
#             for temp in list(dipole_dict[dp].keys()):
#                 if type(temp) != int:
#                     continue
#                 intensities = np.array(dipole_dict[dp][temp]['intensities'])
#                 dtphs = np.array(dipole_dict[dp][temp]['dtphs'])
#                 intensity_err = np.array(dipole_dict[dp][temp]['intensity_err'])
#                 seconds = dtphs / (15e6)
#                 seconds = seconds[np.argsort(dtphs)]
#                 intensities = intensities[np.argsort(dtphs)]
#                 intensity_err = intensity_err[np.argsort(dtphs)]

#                 dtphs = dtphs[np.argsort(dtphs)]
#                 #sort the ones in there already
#                 dipole_dict[dp][temp]['intensities']= intensities
#                 dipole_dict[dp][temp]['intensity_err']= intensity_err

#                 dipole_dict[dp][temp]['seconds']= seconds
#                 dipole_dict[dp][temp]['dtphs']= dtphs

#                 min_tph = np.min(seconds)
#                 max_tph = np.max(seconds)
#                 tau_estimate = max_tph
#                 dtpc_estimate = np.max(intensities) * 8 / 3_000 / 5.2
#                 # intensity_err = intensities - np.sqrt(intensities -0.25)


#                 if errobars:
#                     try:
#                         popt, pcov = curve_fit(intensity_function, seconds, intensities,sigma=intensity_err,p0=[dtpc_estimate,tau_estimate],bounds=([0, min_tph],[np.inf,max_tph]))
#                     except:
#                         # print('fit did not work')
#                         continue

#                 else:

#                     try:
#                         popt, pcov = curve_fit(intensity_function, seconds, intensities,p0=[dtpc_estimate,tau_estimate],bounds=([0, min_tph],[np.inf,max_tph]))
#                     except:
#                         # print('fit did not work')
#                         continue

#                 perr = np.sqrt(np.diag(pcov))

#                 second_grid =  np.geomspace(seconds[0],seconds[-1],100)
#                 fit = intensity_function(second_grid, *popt)
#                 pointwise_fit = intensity_function(seconds, *popt)

#                 ss_res = np.sum((intensities - pointwise_fit) ** 2)
#                 ss_tot = np.sum((intensities- np.mean(intensities)) ** 2)
#                 r2 = np.abs((1 - (ss_res / ss_tot)))
#                 chi2 = np.sum( (pointwise_fit - intensities)**2 / (intensities))
#                 chi_2_list.append(chi2)
#                 if usechi2:
#                     test = chi2 < 1000
#                 else:
#                     test = (r2 < 1+r2_tol and r2 > 1-r2_tol)
                
#                 if test:
#                     numgood+=1
#                     dptest[temp]['parameters'] = [popt,pcov,perr]
#                     dptest[temp]['r2'] = r2
#                     dptest[temp]['chi2'] = chi2
#                     tau = popt[1]
#                     tau_err = perr[1]
#                     temperatures.append(temp)
#                     taus.append(tau)
#                     tau_errs.append(tau_err)

                    
#             # if numgood >= 4:
#             #     dipole_dict[dp]['Good'] = True
#             # else:
#             dipole_dict[dp]['NumGood'] = numgood
            
#             temperatures = np.array(temperatures)
#             taus = np.array(taus)
#             tau_errs = np.array(tau_errs)

#             tau_errs = tau_errs[np.argsort(temperatures)]
#             taus = taus[np.argsort(temperatures)]
#             temperatures = temperatures[np.argsort(temperatures)]
#             goodResult =  all(earlier >= later for earlier, later in zip(taus, taus[1:]))
#             if len(taus) == 0:
#                 goodResult = False
#             if goodResult:
#                 # dipole_dict[dp]["Good"] = False #something went wrong with a fit?

#                 dipole_dict[dp]['temperatures'] = temperatures
#                 dipole_dict[dp]['taus'] = taus
#                 dipole_dict[dp]['tau_errs'] = tau_errs
#             # try:
#                 try:
#                     popt, pcov = curve_fit(energy_cross_section, temperatures, taus,sigma=tau_errs,bounds=([0,-100],[2,-1]))

#                     popt_log, pcov_log = curve_fit(log_energy_cross_section, temperatures, np.log(taus),sigma=tau_errs,bounds=([0,-100],[2,-1]))

#                     perr_log = np.sqrt(np.diag(pcov_log))

#                     perr = np.sqrt(np.diag(pcov))
#                     # energy = popt[0]
#                     logsigma = popt[1]
#                     if np.exp(logsigma) < 3e-1: #otherwise is at boundary and isn't a good fit. 

#                     # sigma = np.exp(sigma)

#                         dipole_dict[dp]['energy_fit_parameters'] = popt
#                         dipole_dict[dp]['energy_fit_error'] = perr
#                         dipole_dict[dp]['energy_fit_pcov'] = pcov

#                         dipole_dict[dp]['log_energy_fit_parameters'] = popt_log
#                         dipole_dict[dp]['log_energy_fit_error'] = perr_log
#                         dipole_dict[dp]['log_energy_fit_pcov'] = pcov_log
                        

#                         pointwise_taus = energy_cross_section(temperatures, *popt)
#                         # ss_res = np.sum((intensities - pointwise_fit) ** 2)
#                         # ss_tot = np.sum((intensities- np.mean(intensities)) ** 2)
#                         # r2 = np.abs((1 - (ss_res / ss_tot)))
#                         chi2 = np.sum( (pointwise_taus - taus)**2 / (taus))
#                         dipole_dict[dp]['energy_fit_chi2'] = chi2

#                         # chi_2_list.append(chi2)

                        


#                 except ValueError or RuntimeError:

#                     print('Value Error occured')
#                     print(temperatures,taus)

#                     # dipole_dict[dp]['Good'] = False
#         full_dipole_dict[q] = dipole_dict
#         full_chi2_list.append(chi_2_list)

#     return full_dipole_dict,full_chi2_list





# def plot_whole_images(imgdir,run,Log=False,showCharge=True,crop=True,plot=True,cmap='viridis'):
#     import numpy as np
#     import glob
#     import re

#     if 'minos' in imgdir:
#         all_files = glob.glob(imgdir + f'proc*{run}*_2_*.fits')
#         goodquads = [0,1,2,3]
#     else:   
#         all_files = glob.glob(imgdir + f'proc*{run}*.fits')
#         goodquads = [0,1,2,3]
#     file_nums = []
#     for f in all_files:
#         filenum = int(re.findall('_\d*.fits',f)[0][1:-5])
#         file_nums.append(filenum)
#     max_file_num = np.max(file_nums)+1
#     min_file_num = np.min(file_nums)
#         # fpath = f'test_images/2024-11-05_ansh_ppumping/proc_skp_temp_scan_{temp}_binned_NROW580_NBINROW1_NCOL3600_NBINCOL1_SC200000_vl-2.75_vh7.5_4.fits'

#     images = []
#     imagenames = []
#     whole_images = []
#     evals = []
#     calfile_not_found=False

#     for imn in np.arange(min_file_num,max_file_num):
#         if 'minos' in imgdir:
#             try:
#                 imagefile = glob.glob(imgdir + f'proc*{run}*_2_{imn}.fits')[0]
#                 try:
#                     calfile = glob.glob(imgdir + f'cal*{run}*_2_{imn}.xml')[0]
#                     import xml.etree.ElementTree as ET
#                     tree = ET.parse(calfile)
#                     root = tree.getroot()
#                     for child in root[0]:
#                         gain = (child.attrib.get('gain'))
#                         evals.append(gain)
#                 except:
#                     calfile_not_found= True
                        

#             except:
#                 continue
#         else:
#             try:
#                 imagefile = glob.glob(imgdir + f'proc*{run}*_{imn}.fits')[0]
#                 try:
#                     calfile = glob.glob(imgdir + f'cal*{run}*_{imn}.xml')[0]
#                     import xml.etree.ElementTree as ET
#                     tree = ET.parse(calfile)
#                     root = tree.getroot()
#                     for child in root[0]:
#                         gain = (child.attrib.get('gain'))
#                         evals.append(gain)
#                 except:
#                     calfile_not_found= True
#             except:
#                 continue
#         fpath = imagefile
#         if calfile_not_found:
#             if 'minos' in imgdir:
#                 evals.append(400)
#             else:
#                 evals.append(200)


#         rowbin = int(re.findall('NBINROW\d+_',imagefile)[0][7:-1])
#         if rowbin > 1:
#             minrow = 1
#             extra = 0
#         else:
#             minrow  = 2
#             extra = 0
#         colbin = int(re.findall('NBINCOL\d+_',imagefile)[0][7:-1])
#         totcol = int(re.findall('NCOL\d+_',imagefile)[0][4:-1])
#         totrow = int(re.findall('NROW\d+_',imagefile)[0][4:-1])

#         max_row = (totrow // rowbin) - ((totrow - 513) // rowbin) + extra
#         max_col = (totcol // colbin) - ((totcol - 3080) // colbin)

#         if colbin > 8:
#             min_col = 1
#         else:
#             min_col = 8 // colbin
        

        
#         # if 'NBINROW1' in imagefile:
#         #     rowfac = 1
        
#         # elif 'NBINROW2' in imagefile:
#         #     rowfac = 2
#         # elif 'NBINCOL1' in imagefile:
#         #     colfac = 1
#         # elif 'NBINCOL2' in imagefile:
#         #     colfac = 2

#         # elif 'NBINROW4' in imagefile:
#         #     rowfac = 4
#         # elif 'NBINCOL4' in imagefile:
#         #     colfac = 4
#         image_quads = []
#         for q in goodquads:
#             image = get_qdata(fpath,q)
#             if len(evals)  == (len(goodquads) + 1):
#                 gain = evals[q+1]
#             else:
#                 gain = evals[0]
#             image = approximate_electronize(image,gain)

#             # image = crop_qdata(image,xlower=2,xupper=512,ylower=8,yupper=3080)
#             if crop:
#                 image = crop_qdata(image,xlower=minrow,xupper=max_row,ylower=min_col,yupper=max_col)
#             if Log:
#                 image[image <= 0] = 1

#             # image_properties(image,imagename=fpath,showimage=True,showHist=False,showcharge=False,showFourier=False,cmap='viridis',logimage=True)
#             images.append(image)
#             image_quads.append(image)
#         imagenames.append(fpath)
#         q0 = np.flip(image_quads[0],axis=1) #quad 0
#         q0 = np.flip(q0,axis=0)
#         q1 = image_quads[1] # quad 1
#         q1 = np.flip(q1,axis=0)
#         q2 = np.flip(image_quads[2],axis=1) # quad 2
#         q2 = np.flip(q2,axis=0)

#         q3 = image_quads[3] # quad 3
#         q3 = np.flip(q3,axis=0)
        

#         # swap
#         # temp = q2
#         # q2 = q3
#         # q3 = temp
#         # q2 = np.flip(q2,axis=1)
#         # q3 = np.flip(q3,axis=1)
#         # q0 = np.flip(q0,axis=1)
#         # q1 = np.flip(q1,axis=1)
#         q0 = np.flip(q0,axis=0)
#         q1 = np.flip(q1,axis=0)

#         top = np.concatenate((q0,q2),axis=0)
#         bottom = np.concatenate((q1,q3),axis=0)

#         whole = np.concatenate((top,bottom),axis=1)
#         if plot:
#             if Log:
#                 plt.imshow(whole,norm=colors.LogNorm(),cmap=cmap)
#                 plt.title("log " + fpath)
#             else:
#                 plt.imshow(whole)
#                 plt.title(fpath)
#             ax = plt.gca() 
#             imlocs = [(0.02,0.6),(0.52,0.6),(0.02,0.1),(0.52,0.1)]
#             if showCharge:
#                 for i,image in enumerate(image_quads):
#                     imloc = imlocs[i]
#                     hist_upper = int(np.nanmean(image) + 2000)
#                     hist_lower = int(np.nanmean(image) - 2000)
#                     hist,bins= np.histogram(image,np.arange(hist_lower,hist_upper))
#                     mids = 0.5*(bins[1:] + bins[:-1])
#                     histmean = np.average(mids, weights=hist)
#                     var = np.average((mids - histmean)**2, weights=hist)
#                     histmean = np.round(histmean,2)

#                     var = np.round(var,2)

#                     plt.text(imloc[0],imloc[1],f"avg charge = {histmean}, var = {var}",transform=ax.transAxes,color='black',fontsize=12,bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))
#             plt.show()
#             plt.close()
#         whole_images.append(whole)

        
#         # break
#     return imagenames,whole_images



            






            

        



        
# def plot_individual_quads(imgdir,run,goodquads=[0,1,2,3],hist=False,logimage=False):
#     import numpy as np
#     import re
#     import glob
#     # run = 'star_half_measurement_run2'
#     if 'minos' in imgdir:
#         all_files = glob.glob(imgdir + f'proc*{run}*_2_*.fits')
       
#     else:   
#         all_files = glob.glob(imgdir + f'proc*{run}*.fits')
       
#     file_nums = []
#     for f in all_files:
#         filenum = int(re.findall('_\d*.fits',f)[0][1:-5])
#         file_nums.append(filenum)
#     max_file_num = np.max(file_nums)+1
#     min_file_num = np.min(file_nums)
#         # fpath = f'test_images/2024-11-05_ansh_ppumping/proc_skp_temp_scan_{temp}_binned_NROW580_NBINROW1_NCOL3600_NBINCOL1_SC200000_vl-2.75_vh7.5_4.fits'

#     print(min_file_num,max_file_num)
#     images = []
#     imagenames = []
#     for imn in np.arange(min_file_num,max_file_num):
#         if 'minos' in imgdir:
#             try:
#                 imagefile = glob.glob(imgdir + f'proc*{run}*_2_{imn}.fits')[0]
#             except:
#                 continue
#             eval = 400
#         else:
#             try:
#                 imagefile = glob.glob(imgdir + f'proc*{run}*_{imn}.fits')[0]
#             except:
#                 continue
#             eval = 200
#         fpath = imagefile

#         rowbin = int(re.findall('NBINROW\d+_',imagefile)[0][7:-1])
#         if rowbin > 1:
#             minrow = 1
#             extra = 0
#         else:
#             minrow  = 2
#             extra = 0
#         colbin = int(re.findall('NBINCOL\d+_',imagefile)[0][7:-1])
#         totcol = int(re.findall('NCOL\d+_',imagefile)[0][4:-1])
#         totrow = int(re.findall('NROW\d+_',imagefile)[0][4:-1])

#         max_row = (totrow // rowbin) - ((totrow - 513) // rowbin) + extra
#         max_col = (totcol // colbin) - ((totcol - 3080) // colbin)

#         if colbin > 8:
#             min_col = 1
#         else:
#             min_col = 8 // colbin
        

        
#         # if 'NBINROW1' in imagefile:
#         #     rowfac = 1
        
#         # elif 'NBINROW2' in imagefile:
#         #     rowfac = 2
#         # elif 'NBINCOL1' in imagefile:
#         #     colfac = 1
#         # elif 'NBINCOL2' in imagefile:
#         #     colfac = 2

#         # elif 'NBINROW4' in imagefile:
#         #     rowfac = 4
#         # elif 'NBINCOL4' in imagefile:
#         #     colfac = 4

#         for q in goodquads:
#             image = get_qdata(fpath,q)

#             image = approximate_electronize(image,eval)

#             # image = crop_qdata(image,xlower=2,xupper=512,ylower=8,yupper=3080)
            
#             image = crop_qdata(image,xlower=minrow,xupper=max_row,ylower=min_col,yupper=max_col)
#             print('****************************************************')
#             print(f'******************QUADRANT {q+1}********************')
#             print('****************************************************')
#             image_properties(image,imagename=fpath,showimage=True,showHist=hist,showcharge=False,showFourier=False,cmap='viridis',logimage=logimage)
#             images.append(image)
#             imagenames.append(fpath)
#         # break
#     return imagenames,images
