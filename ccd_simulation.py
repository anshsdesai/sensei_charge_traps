from astropy.time import Time
#use a snolab image and assume this time is valid for pixel times

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
        # 1. Compute distance from the cluster (invert mask so cluster is 0)
        # 'distance_transform_edt' calculates distance to the nearest ZERO pixel.
        # So we invert: Cluster becomes False (0), Background becomes True (1).

        dist_map = distance_transform_edt(~specific_cluster_mask)

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


def get_cluster(proc, min_pixs=2, max_pixs=None, min_total_value=3,max_total_value = None, return_count = False, return_median= False):#, min_max_value=None
    from scipy.stats import binned_statistic
    import numpy as np
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


class CCD:
    def __init__(self,tpix_horizontal,tpix_vertical,tau_weights, tau_values):
        import numpy as np
        # self.original_image = np.copy(image_array)
        # self.exposure_images = np.zeros_like(sample_image,dtype=float)
        # self.exposure_images = []
        self.tpix_horizontal = tpix_horizontal
        self.tpix_vertical = tpix_vertical

        self.reconstructed_images = []
        self.no_trap_images = []
        self.exposures = []

        self.single_e_counts = []
        self.single_e_counts_no_traps = []
        self.single_e_counts_masked = []
        self.single_e_counts_masked_no_traps = []
        
        self.unmasked_pixels = []
        self.unmasked_pixels_no_traps = []


        self.UL_expdep = 8.19e-5 #e / pix / day

        self.UR_expdep = 4.36e-5 #e / pix / day

        self.LL_expdep = 6.88e-5 #e / pix / day

        self.LR_expdep = 8.23e-5 #e / pix / day

        self.UL_expindep = 12.23e-5 #e / superpix / image

        self.UR_expindep = 9.94e-5 #e / superpix / image

        self.LL_expindep = 7.53e-5 #e / superpix / image

        self.LR_expindep = 6.52e-5 #e / superpix / day
        self.exp_dep_rate = self.UR_expdep / (24 * 3600) #e / pix / s
        self.exp_indep_rate = self.UR_expindep/ 32 #e / pix / image

        self.total_pix = (6144)* (1024)

        self.npix_per_quad = self.total_pix / 4


        self.nrow_quad = int(1024 /2)
        self.ncol_quad = int(6144 / 2)
        shape = (self.nrow_quad,self.ncol_quad)
        self.exposure_accumulator = np.zeros(shape)

        self.ccd_state = np.zeros(shape)
        



        self.exp_indep_events = np.round(self.exp_indep_rate * self.npix_per_quad)


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
        trap_density = (5171 / 4) / (self.nrow_quad * self.ncol_quad)
        rng = np.random.default_rng()
        self.trap_mask = rng.random(shape) < trap_density
        
        # Store indices (tuple of row_indices, col_indices) for fast access
        self.trap_indices = np.where(self.trap_mask)
        num_traps = len(self.trap_indices[0])
        
        # Store Tau values as a 1D array corresponding to the indices
        probs = np.array(tau_weights) / np.sum(tau_weights)
        self.trap_taus = rng.choice(tau_values, size=num_traps, p=probs)
        
        # Store trapped charge as a 1D array (much faster than 2D)
        self.trapped_charge_1d = np.zeros(num_traps, dtype=float)





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
        # --- OPTIMIZATION 2: SPARSE INTERACTION ---
        
        # 1. Extract charge ONLY at trap locations
        # We use the pre-calculated indices to grab values directly
        charge_at_traps = current_image[self.trap_indices]
        
        # 2. Capture Logic (Vectorized on 1D arrays)
        # Check: Pixel has charge (>=1) AND Trap is empty (==0)
        can_capture = (charge_at_traps >= 1.0) & (self.trapped_charge_1d == 0.0)
        
        # Get indices of traps that are actually capturing right now
        # We filter our master index list by the boolean 'can_capture'
        capturing_rows = self.trap_indices[0][can_capture]
        capturing_cols = self.trap_indices[1][can_capture]
        
        # Update the image and the trap state
        current_image[capturing_rows, capturing_cols] -= 1.0
        self.trapped_charge_1d[can_capture] += 1.0

        # 3. Release Logic
        # Calculate probability P = 1 - exp(-dt/tau)
        # We use the scalar dt directly (no need for an array of dt)
        p_release = 1.0 - np.exp(-dt / self.trap_taus)
        
        # Generate random rolls only for the traps (N=1300), not the image (N=1.5M)
        n_traps = len(self.trap_taus)
        random_rolls = np.random.random(n_traps)
        
        # Check: Trap has charge (>0) AND Roll is successful
        should_release = (self.trapped_charge_1d > 0) & (random_rolls < p_release)
        
        # Get indices of traps that are releasing
        releasing_rows = self.trap_indices[0][should_release]
        releasing_cols = self.trap_indices[1][should_release]
        
        # Update image and trap state
        current_image[releasing_rows, releasing_cols] += 1.0
        self.trapped_charge_1d[should_release] -= 1.0

        return current_image




    def take_fake_image(self,exposure_time_hours,radius=60):
        import numpy as np
        from utils import approximate_electronize,get_qdata
        # from skimage.morphology import disk, binary_dilation


        exp = exposure_time_hours * 3600
        self.exposures.append(exp)

        
        exp_dep_events = self.exp_dep_rate * exp * self.npix_per_quad #assume not much during readout -- maybe not a great assumption but whatever

        # print(self.npix_per_quad)
        exp_dep_events = np.round(exp_dep_events)
        # print(exp,exp_dep_events,self.exp_indep_events)




        n_singlee_events = int(exp_dep_events + self.exp_indep_events)

        n_singlee_events = np.random.poisson(n_singlee_events)

        


        print(f'Number of single electron events from rate to inject: {n_singlee_events}')
        #now generate fake image
        file = 'minos_image/proc_corr_proc_skp_72000secs_exp_run10_NSAMP_300_36.fits'

        q0 = get_qdata(file,0)

        q0 =approximate_electronize(q0,400)
        q0_blank= transplant_clusters(q0.T, target_shape=(self.nrow_quad, self.ncol_quad),count_threshold=100, max_aspect_ratio=3.0,radius=radius,exposure=exp)

        # footprint = disk(radius)

        # exclusion_mask = binary_dilation(q0_blank > 0, footprint)




        q0_fake = inject_single_e(q0_blank, n_events=n_singlee_events, intensity=1,exclusion_mask=None)
        self.no_trap_images.append(q0_fake)

        self.ccd_state += q0_fake
        if len(self.exposures) > 0:
            t1 = self.exposure_accumulator - self.exposures[-1]
            t2 = self.exposure_accumulator 
            # dt = t2 - t1
            self.ccd_state = self.charge_trap_interaction(self.ccd_state,self.tpix_vertical)
        # exp_image = np.zeros_like(q0_fake,dtype=float)

        self.simulate_readout()

        #counts
        
        og_image_halo = generate_halo_mask(self.no_trap_images[-1],threshold=100,radius=60)
        trap_image_halo = generate_halo_mask(self.reconstructed_images[-1],threshold=100,radius=60)


        counts_1e_trap_nomask  = get_cluster(self.reconstructed_images[-1],min_pixs=1,max_pixs=1,min_total_value=1,max_total_value = 1,return_count=True)

        counts_1e_og_nomask  = get_cluster(self.no_trap_images[-1],min_pixs=1,max_pixs=1,min_total_value=1,max_total_value = 1,return_count=True)


        masked_trap_image = np.copy(self.reconstructed_images[-1]).astype(float)

        masked_og_image = np.copy(self.no_trap_images[-1]).astype(float)


        masked_og_image[og_image_halo] = np.nan
        masked_trap_image[trap_image_halo] = np.nan

        counts_1e_trap  = get_cluster(masked_trap_image,min_pixs=1,max_pixs=1,min_total_value=1,max_total_value = 1,return_count=True)
        counts_1e_og  = get_cluster(masked_og_image,min_pixs=1,max_pixs=1,min_total_value=1,max_total_value = 1,return_count=True)

        unmasked_pix_traps = np.sum(~trap_image_halo)

        unmasked_pix_notraps = np.sum(~og_image_halo)


        self.single_e_counts.append(counts_1e_trap_nomask)
        self.single_e_counts_masked.append(counts_1e_trap)
        self.single_e_counts_no_traps.append(counts_1e_og_nomask)


        self.single_e_counts_masked_no_traps.append(counts_1e_og)

        self.unmasked_pixels.append(unmasked_pix_traps)
        self.unmasked_pixels_no_traps.append(unmasked_pix_notraps)

        #simulate a clear
        self.ccd_state *= 0

        # self.exposure_images.append(exp_image)

        


        self.exposure_accumulator += exp





    def simulate_readout(self):
        import numpy as np

        
        print(f"Starting Readout...")
        # image = self.ccd_state
        image = self.ccd_state.copy()
        serial_register = np.zeros(image.shape[1])


        rows, cols = image.shape
        
        output_stream = []

        for r in range(rows):
            serial_register = image[-1, :].copy() #this represents a vertical shift of charge to the serial register

            image[1:, :] = image[:-1, :]
            image[0, :] = 0.0
            # line_buffer = []
            index = int(-1 -r )
            t2 = self.exposure_accumulator+ self.tpix_vertical 
            t1 = self.exposure_accumulator
            dt = t2 - t1

            # image = self.charge_trap_interaction(image,dt)
            image = self.charge_trap_interaction(image,self.tpix_vertical)
            self.exposure_accumulator[:index,:] += self.tpix_vertical


            #would need this for serial register traps
            # for c in range(cols):
            #     # index2 = int(-1 -c)
            #     val = serial_register[-1]
            #     serial_register[1:] = serial_register[:-1]
            #     serial_register[0] = 0.0
            #     # self.exposure_image[:index,:index2] += self.tpix_horizontal #don't necessarily need to keep track of this, assuming no traps in serial register
            #     line_buffer.append(val)

            # output_stream.extend(line_buffer)
            output_stream.extend(serial_register[::-1])

        result_flat = np.array(output_stream)
        result_reconstructed= result_flat.reshape(rows, cols)
        result_reconstructed = np.flipud(np.fliplr(result_reconstructed))

        self.reconstructed_images.append(result_reconstructed)

        return 







