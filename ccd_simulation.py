import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from astropy.io import fits
from utils import *
from tqdm.autonotebook import tqdm




from collections import Counter
image_dir_search = 'proc/*.fits'
temperatures = []


for image in glob.glob(image_dir_search):
    temp = re.findall('_\d+k',image)[0][1:-1]
    if 'dtph' not in image:
        continue
    # if temp not in temperatures:
   
    temperatures.append(temp)
temperatures_strs = np.array(temperatures)

print(Counter(temperatures_strs))
# temperatures = np.unique(np.sort(temperatures))
temps = np.sort(np.array([int(t) for t in temperatures_strs]))



from dipole import *
import pickle
goodQuads = [0,1,2,3]
image_dir = 'proc/'
test_temps = temperatures


try:
    with open('dipole_coord_list.pkl','rb') as infile:
        full_dipole_coord_list = pickle.load(infile)
except FileNotFoundError:
    full_dipole_coord_list = getDipoleList2(image_dir,test_temps,goodQuads)
    with open('dipole_coord_list.pkl','wb') as outfile:
        pickle.dump(full_dipole_coord_list,outfile)

        
try:
    with open('dipole_spectra.pkl','rb') as infile:
        dipole_spectra = pickle.load(infile)
except FileNotFoundError:
    dipole_spectra = getDipoleSpectra2(image_dir,goodQuads,full_dipole_coord_list)
    with open('dipole_spectra.pkl','wb') as outfile:
        pickle.dump(dipole_spectra,outfile)

useIntensityErr = True
wellBehavedThreshold = 4
threshold_str = f'_{wellBehavedThreshold}'
intensity_str = '_err' if useIntensityErr else ''
try:
    with open(f'fit_dipole_spectra{intensity_str}{threshold_str}.pkl','rb') as infile:
        fit_dipole_spectra = pickle.load(infile)

except FileNotFoundError:
    fit_dipole_spectra = fitTrapIntensity(dipole_spectra,useIntensityErr=useIntensityErr,wellBehavedThreshold=wellBehavedThreshold)
    with open(f'fit_dipole_spectra{intensity_str}{threshold_str}.pkl','wb') as outfile:
        pickle.dump(fit_dipole_spectra,outfile)


tau_at_135s = []
for q in [0,1,2,3]:
    
    dpkeys = list(fit_dipole_spectra[q])
    for i in range(len(dpkeys)):
        
        # dpkeys = list(fit1_dipole_spectra[q])
        # dp = random.choice(dpkeys)
        dp = dpkeys[i]
        if type(dp) != tuple:
            continue
        testdp = fit_dipole_spectra[q][dp]
        if testdp['WellBehavedTrap'] and not testdp['EnergyFitFailed']:
            testdpfit = testdp['EnergyFitInfo']

            # p_value = testdpfit['p_value'] 
            # chi_squared = testdpfit['chi2']
            # r2 = testdpfit['r_squared']

            # pvals.append(p_value)
            # chi2s.append(chi_squared)
            # r2s.append(r2)
            # red_chi2s.append( testdpfit['reduced_chi2'])


            # fit_is_good = chi_squared < 100

            # fit_is_good = p_value < 0.05
            fit_is_good = testdp["GoodEnergyFit"]
            # fit_is_good = fit_is_good and 
            # maxtau = np.max(testdpfit['taus'])
            # maxtaus.append(maxtau)
            # fit_is_good = fit_is_good
            # fit_is_good = True


            if fit_is_good:
                cs = testdpfit['BestFitCrossSection']
                cserr = testdpfit['BestFitCrossSectionErr']
                e = testdpfit['BestFitEnergy']
                e_err = testdpfit['BestFitEnergyErr']

                 # temperatures = np.linspace(120,180,100)
                logtau_at_135 = log_energy_cross_section(135,e,np.log(cs))

                tau_at_135 = np.exp(logtau_at_135)
                tau_at_135s.append(tau_at_135)
              

                
                  
bins = np.geomspace(1e-5,1e9,100)
tau_at_135s = np.array(tau_at_135s)  
hist,bin_edges = np.histogram(tau_at_135s,bins)          
bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
# stuff =  plt.hist(tau_at_135s,bins=bins)
# plt.stairs(hist,bins)
# plt.xlabel("$\\tau_e$")
# plt.xscale('log')
# plt.ylabel("Counts")
# plt.show()
# plt.close()


tau_weights = hist
tau_values = bin_centers


from astropy.time import Time
#use a snolab image and assume this time is valid for pixel times

def pixel_time(nsamp, delayH, delayIped, delayIsig, delaySW, delayRG, delayOG):		# Time to read 1 pixel
  return 5*int(delayH)+int(delayRG)+(int(delayIped)+int(delaySW)+int(delayIsig)+int(delayOG)+int(delayRG))*int(nsamp)

def pixel_time_vertical(nsamp,ncol, delayH, delayIped, delayIsig, delaySW, delayRG, delayOG,delayDG):		# Time to read 1 pixel
  pixel_time_hor=pixel_time(nsamp, delayH, delayIped, delayIsig, delaySW, delayRG, delayOG)
  return (pixel_time_hor+(int(delaySW)+int(delayDG)))*int(ncol)


snolab_dir = '/data/analyses/snolab_run1/'
file = 'proc_corr_proc_skp_sensei_2023-02-14_135K_run7_commissioning_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR10800_5_83.fits'
with fits.open(snolab_dir +file) as hdul:
    q = hdul[0]
    header = q.header
    print(list(header.keys()))
    nrow=header['NROW']
    ncol=header['NCOL']
    exposure=header['EXPOSURE']
    nsamp=header['NSAMP']
    delayH=header['HIERARCH DELAY_H_OVERLAP']
    delayRG=header['HIERARCH DELAY_RG_WIDTH']
    delayIped=header['HIERARCH DELAY_INTEG_PED']
    delaySW=header['HIERARCH DELAY_SWHIGH']
    delayIsig=header['HIERARCH DELAY_INTEG_SIG']
    delayOG=header['HIERARCH DELAY_OG_LOW']
    delayDG=header['HIERARCH DELAY_DG_LOW']


    
#     print(list(q.header.keys()))
#     print(q.header['DATESTART'])
#     print(q.header['EXPOSURE'])

#     dte_start = q.header['DATESTART']
#     dte_end = q.header['DATEEND']
#     t = Time([dte_start,dte_end], format='isot')
#     readout_time = t.jd[1] - t.jd[0]
#     print(readout_time * 24 * 3600)
#     readout_time =readout_time * 24 * 3600
# # singleReadTime = smask.readout_time - exp
# # seconds_per_row = (singleReadTime)/nRow
# # seconds_per_pixel = seconds_per_row/nCol

# singleReadTime = readout_time - 72000

# seconds_per_row = (singleReadTime)/nRow
# seconds_per_pixel = seconds_per_row/nCol
tpix=pixel_time(nsamp, delayH, delayIped, delayIsig, delaySW, delayRG, delayOG) / 15e6
tpix_vertical=pixel_time_vertical(nsamp,ncol, delayH, delayIped, delayIsig, delaySW, delayRG, delayOG,delayDG)/15e6





class CCD:
    def __init__(self,tpix_horizontal,tpix_vertical,tau_weights, tau_values):
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
        from skimage.morphology import disk, binary_dilation


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

        footprint = disk(radius)

        exclusion_mask = binary_dilation(q0_blank > 0, footprint)




        q0_fake = inject_single_e(q0_blank, n_events=n_singlee_events, intensity=1,exclusion_mask=exclusion_mask)
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





# def simulate_readout(image,readout_time_hours = 7,tpix_horizontal,tpix_vertical):
#     total_seconds = readout_time_hours *3600
#     npix = image.shape[0] * image.shape[1]
#     rows,cols = image.shape
#     for r in range(rows):
#         serial_register = image[-1, :].copy()
#         image_area[0, :] = 0.0

# CCDTest.simulate_readout()


import pickle
for r in range(1,10):
    filename = f'ccd_traps_run{r}.pkl'

    CCDTest = CCD(tpix,tpix_vertical,tau_weights,tau_values)

    for i in tqdm(range(100)):
        CCDTest.take_fake_image(0) #0h exposure
        CCDTest.take_fake_image(4) #4h exposure
        CCDTest.take_fake_image(6) #6h exposure
        CCDTest.take_fake_image(10) #10h exposure

        CCDTest.take_fake_image(20) #20h exposure
    with open(filename,'wb') as f:
        pickle.dump(CCDTest,f)
        
    del CCDTest

        # counts1e_notraps.append(counts_1e_og)
        # counts1e_traps.append(counts_1e_trap)
    

