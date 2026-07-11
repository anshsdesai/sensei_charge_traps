# dipole_new.py -- MINIMAL signed-refit synthesis (derived from dipole.py).
# Keeps: signed intensities, constant pedestal (fit_offset default True),
# temporal pair-noise errors (getDipoleSpectra2 error_model='physical'),
# absolute_sigma default True, robust/relaxed finder, simple SRH law.
# Adds: amplitude sign-consistency classification before the SRH fit.
# Drops: regional covariance / pumping overdispersion (live in signed_refit_*),
# intrinsic-scatter budget and outlier rejection.
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

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


def gauss(x,mean,sigma,constant):
    return constant*np.exp(-0.5 * (((x-mean)**2) / (sigma**2)))


def comparable_perc(a,b,perc=0.3):
    if a == b:
        return True
    max_val = max(abs(a), abs(b))
    if max_val == 0:
        return False  # Avoid division by zero
    percent_diff = abs(a - b) / max_val
    return percent_diff < perc

def findDipoles2(electronized_image,debug=False,useFit=False,robust_sigma=False,symmetry_perc=0.3):
    # robust_sigma: estimate the detection threshold from a MAD on the
    #   row-median-subtracted image, so dark-current nonuniformity at high T
    #   does not silently desensitize the finder.
    # symmetry_perc: lobe-symmetry requirement; None disables it (the readout
    #   pedestal makes real dipoles asymmetric at high T).

    dipole_list = []

    hist_upper = int(np.nanmean(electronized_image) + 2000)
    hist_lower = int(np.nanmean(electronized_image) - 2000)
    nbins = 200
    step_length = int((hist_upper - hist_lower) / nbins)
    bins_ = np.arange(hist_lower,hist_upper,step_length)

    hist,bins= np.histogram(electronized_image,bins_)
    mids = 0.5*(bins[1:] + bins[:-1])
    histmean = np.average(mids, weights=hist)

    var = np.average((mids - histmean)**2, weights=hist)
    histmean = np.round(histmean,2)
    if useFit:
        from scipy.optimize import curve_fit
        try:
            # Provide initial guesses: [mean, sigma, constant]
            p0 = [histmean, np.sqrt(var), np.max(hist)]
            popt, _ = curve_fit(gauss, mids, hist, p0=p0)
            mean, sigma, constant = popt
        except RuntimeError:
            # Fallback to standard statistics if the fit fails to converge
            mean = histmean
            sigma = np.sqrt(var)
            constant = np.max(hist)
    else:
        sigma= np.sqrt(var)
        mean = histmean
        constant = np.max(hist)
        
    sigma_cutoff =(3*sigma)**2
    sigma_cutoff *= -1

    if debug:
        plt.figure()
        xs = np.linspace(hist_lower,hist_upper,nbins)
        plt.plot(xs,gauss(xs,mean,sigma,constant),lw=3)
        plt.title(f"$\mu = {mean} \sigma={sigma}$")
        plt.stairs(hist,bins)
        plt.show()
        plt.close()


    median_charge_per_row = np.median(electronized_image,axis=1)

    image = electronized_image.T -median_charge_per_row
    image = image.T #image with median charge per row subtracted

    if robust_sigma:
        # MAD on the subtracted image: insensitive to defects and to the
        # broad dark-current tails that inflate the histogram width at high T.
        sigma = 1.4826 * np.nanmedian(np.abs(image - np.nanmedian(image)))
        sigma_cutoff = -1 * (3 * sigma) ** 2

    # Fully vectorize the row-by-row product search
    multipl = image[1:, :] * image[:-1, :]
    potential_rows, potential_cols = np.where(multipl < sigma_cutoff)
    actual_rows = potential_rows + 1
    
    for r, c in zip(actual_rows, potential_cols):
        potential_locations = [c] # Preserve loop logic format

        if len(potential_locations) == 0:
            continue
        for col in potential_locations:

            coord = (r, col)
            coord_b = (r - 1, col)
            charge1 = np.abs(image[coord])
            charge2 = np.abs(image[coord_b])
            if debug:
                print(charge1,charge2)

            if symmetry_perc is None or comparable_perc(charge1,charge2,perc=symmetry_perc):
                dipole_list.append(coord)
            else:
                if debug:
                    print('dipole did not match percentage threshold')
                    print(coord)
                continue

                #star or bad dipole or something

    dipole_list = list(set(dipole_list)) 

    return dipole_list

import numpy as np
import matplotlib.pyplot as plt

def histogram_around_point(array, center, size, bins, plot = False):
    """
    Plots or returns a histogram of a square region around a point in a 2D NumPy array.

    Parameters:
    - array: 2D numpy array
    - center: (row, col) tuple
    - size: size of the square region (default 50x50)
    - bins: number of histogram bins
    - plot: whether to plot the histogram (default: True)

    Returns:
    - hist: histogram counts
    - bin_edges: edges of the histogram bins
    """
    n_rows, n_cols = array.shape
    half = size // 2
    row, col = center

    # Handle edges by clipping
    start_row = max(row - half, 0)
    end_row = min(row + half, n_rows)
    start_col = max(col - half, 0)
    end_col = min(col + half, n_cols)

    region = array[start_row:end_row, start_col:end_col]
    region_flat = region.flatten()

    # Flatten the region and create histogram
    hist, bin_edges = np.histogram(region_flat, bins=bins)
    region_std = np.std(region_flat)

    if plot:
        plt.hist(region.flatten(), bins=bins, edgecolor='black')
        plt.title(f"Histogram around point ({row}, {col})")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()

    return hist, bin_edges, region_std

def getDipoleList2(image_dir,temperatures,goodquads,plot=False,robust_sigma=False,
                   symmetry_perc=0.3,image_files=None):
    from utils import get_qdata,crop_qdata,approximate_electronize

    from collections import defaultdict
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors 
    import re
    import glob
    # if 'minos' in image_dir:
    eval = 400
        # minos=True
        # system = 'Minos'
        # ccd =2
    ccd_str = '*_2_*'
    # else:
    #     eval = 200
    #     system = 'Cross1'
    #     ccd = 1
    #     ccd_str = '*'
    #     minos = False

    

    # norm = colors.Normalize(vmin=cutoffs[0],vmax=cutoffs[-1])
    # cmap= plt.cm.cividis
    full_dipole_list = []
    total_dipoles = [0,0,0,0]
    total_traps = [0,0,0,0]
    frozen_imagefiles = None if image_files is None else sorted(str(path) for path in image_files)
    for q in goodquads:
        dipole_list = []
        for temperature in temperatures:
            # print(f"temperature: {temperature}")
            dipole_occurrences = defaultdict(set)
            all_dipoles = []
            temp = f'{temperature}k'
            if frozen_imagefiles is None:
                search_str = image_dir + f'proc*{temp}*_' + '*dtph*' + ccd_str
                selected_imagefiles = glob.glob(search_str)
            else:
                selected_imagefiles = [
                    path for path in frozen_imagefiles
                    if re.search(fr'_{temperature}k_', path)
                ]

            if plot:
                fig = plt.figure(figsize=(12,8))
            
            for imagefile in selected_imagefiles:
                dtph = int(re.findall(r'dtph\d+_',imagefile)[0][4:-1])
                image = get_qdata(imagefile,q)
                image = crop_qdata(image)#,ylower=500,xlower=100)
                # print(imagefile)
                image = approximate_electronize(image,eval)
                image_dipoles = findDipoles2(image,robust_sigma=robust_sigma,symmetry_perc=symmetry_perc)
                for dipole in image_dipoles:
                    dipole_occurrences[tuple(dipole)].add(dtph)
                #need to filter dipoles further: dipoles that appear in more than two images at same temperature with different dtph 

                all_dipoles += image_dipoles
            all_dipoles = list(set(all_dipoles))

            good_dipoles = [coord for coord, dtphs in dipole_occurrences.items() if len(dtphs) > 1]
            dipole_list += good_dipoles

            print('# All Dipoles',len(all_dipoles))
            print('# Good Dipoles',len(good_dipoles))
            print("# Anomalous Dipoles",len(all_dipoles) - len(good_dipoles))
            

        print(f'Number of total dipoles quadrant {q}: ',len(dipole_list))
        total_dipoles[q] +=len(dipole_list)
        final_dipole_list = list(set(dipole_list))
        print(f'Number of traps quadrant {q}: ',len(final_dipole_list))
        total_traps[q] +=len(final_dipole_list)

        full_dipole_list.append(final_dipole_list)
    print('Total Dipoles')
    print(np.sum(total_dipoles))
    print("Total Traps")
    print(np.sum(total_traps))

    return full_dipole_list

def getDipoleSpectra2(image_dir,goodquads,full_dipole_coord_list,absolute=False,
                      error_model='physical',noise_table=None,image_files=None):
    # error_model='patch'   : legacy — intensity_err is the spatial sigma of the
    #                         local 34x34 patch. That is pixel-to-pixel
    #                         nonuniformity, not the temporal fluctuation of a
    #                         fixed pair, and overestimates it ~2.5x; kept for
    #                         reproducibility.
    # error_model='physical': intensity_err is the per-point temporal noise of
    #                         I=(a-b)/2:
    #                           sigma_I^2 = sigma_base^2(T,quad) + (S_a + S_b)/4
    #                         where sigma_base is the measured trap-free temporal
    #                         pair noise from noise_table[(temp, quad)] (electrons;
    #                         includes baseline shot + read noise) and S is each
    #                         pixel's signal charge above the row median (its own
    #                         Poisson shot, relevant for bright points). The patch
    #                         sigma is still stored separately as 'patch_sigma'.
    if error_model == 'physical' and noise_table is None:
        raise ValueError("error_model='physical' requires noise_table {(temp, quad): sigma_base_e}")
    import glob
    from tqdm.autonotebook import tqdm
    import ctypes
    # from ROOT import TH1D, TF1, TCanvas
    from scipy.optimize import curve_fit
    import re
    from utils import get_qdata,crop_qdata,approximate_electronize,crop_numpy_array
    
    full_dipole_dict = {}
    for q in tqdm(range(len(goodquads))):
        quad = goodquads[q]
        dp_dict = {}
        dipole_coord_list = full_dipole_coord_list[quad]
        # if 'minos' in image_dir:
        eval = 400
        
        ccd_str = '*_2_*'
        # else:
        #     eval = 200
        #     system = 'Cross1'
        #     ccd = 1
        #     ccd_str = '*'
        #     minos = False
        if image_files is None:
            search_str = image_dir + f'proc*_' + '*dtph*' + ccd_str
            selected_imagefiles = glob.glob(search_str)
        else:
            selected_imagefiles = sorted(str(path) for path in image_files)
        temps = []
        for dp in dipole_coord_list:
            dp_dict[dp] = {}

        for f in tqdm(range(len(selected_imagefiles))):
            imagef = selected_imagefiles[f]
            dtph = int(re.findall(r'dtph\d+_',imagef)[0][4:-1])
            temp = int(re.findall(r'_\d+k',imagef)[0][1:-1])
            image = get_qdata(imagef,quad)
            image = crop_qdata(image)
            image = approximate_electronize(image,eval)

            # Pre-subtraction charge: needed for Poisson shot noise of the
            # dipole pixels themselves (dominant at high temperature).
            raw_charge = image.copy()

            median_charge_per_row = np.median(image,axis=1)

            image = image.T -median_charge_per_row
            image = image.T #image with median charge per row subtracted



            hist_upper = int(np.nanmean(image) + 2000)
            hist_lower = int(np.nanmean(image) - 2000)
            hist,bins= np.histogram(image,np.arange(hist_lower,hist_upper))
            mids = 0.5*(bins[1:] + bins[:-1])
            histmean = np.average(mids, weights=hist)
            var = np.average((mids - histmean)**2, weights=hist)
            sigma_image = np.sqrt(var)
            

            for dp in dipole_coord_list:
           

                if temp not in dp_dict[dp].keys():
                    dp_dict[dp][temp] = {}
                    dp_dict[dp][temp]['intensities'] = []
                    dp_dict[dp][temp]['dtphs'] = []
                    dp_dict[dp][temp]['intensity_err'] = []
                    dp_dict[dp][temp]['poisson_err'] = []
                    dp_dict[dp][temp]['patch_sigma'] = []
                    dp_dict[dp][temp]['image_sigma'] = sigma_image


                coord_b = (dp[0]-1,dp[1])
                intensity = (image[dp] -image[coord_b]) / 2
                if absolute:
                    intensity = np.abs(intensity)

                bins = np.arange(-1000,1000,2000/100)
                hist,bins,sigma_hist = histogram_around_point(image, dp, size=35, bins=bins, plot=False)
                mids = 0.5*(bins[1:] + bins[:-1])
                histmean = np.average(mids, weights=hist)
                histmean = np.abs(histmean)
                sigma_poisson = np.sqrt(histmean)

                # mids = 0.5*(bins[1:] + bins[:-1])
                # popt, pcov = curve_fit(gauss, mids, hist, p0=[np.mean(hist), sigma_hist,np.max(hist)])
                # sigma = popt[1]


           
                # image_flat = cropped_image.flatten().astype(np.float64)

                # h = TH1D(f"temp",f"Charge", nbins, hist_lower, hist_upper)

                # weights = np.ones_like(image_flat)
                # # h.FillN(image_flat.size, image_flat, np.ones(image_flat.size))
                # # Get C-compatible pointers
                # x_ptr = image_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                # w_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

                # h.FillN(image_flat.size, x_ptr, w_ptr)

                # h.Fit("gaus","Q")
                # fit_function = h.GetFunction("gaus")
                # constant = fit_function.GetParameter(0)
                # mean = fit_function.GetParameter(1)
                # sigma = fit_function.GetParameter(2)


                
                if error_model == 'physical':
                    sigma_base = noise_table[(temp, quad)]
                    signal_a = max(float(raw_charge[dp]) - float(median_charge_per_row[dp[0]]), 0.0)
                    signal_b = max(float(raw_charge[coord_b]) - float(median_charge_per_row[coord_b[0]]), 0.0)
                    sigma_point = np.sqrt(sigma_base**2 + (signal_a + signal_b) / 4.0)
                else:
                    sigma_point = sigma_hist

                dp_dict[dp][temp]['intensities'].append(intensity)
                dp_dict[dp][temp]['dtphs'].append(dtph)
                dp_dict[dp][temp]['intensity_err'].append(sigma_point)
                dp_dict[dp][temp]['poisson_err'].append(sigma_poisson)
                dp_dict[dp][temp]['patch_sigma'].append(sigma_hist)


                # dp_dict[dp][temp]['intensity_err_fit'].append(sigma)



        #sort by dtph
        for dp in dp_dict.keys():
            for temp in dp_dict[dp].keys():
                intensities = np.array(dp_dict[dp][temp]['intensities'])
                intensity_err = np.array(dp_dict[dp][temp]['intensity_err'])
                poisson_err =  np.array(dp_dict[dp][temp]['poisson_err'])
                patch_sigma = np.array(dp_dict[dp][temp]['patch_sigma'])

                dtphs = np.array(dp_dict[dp][temp]['dtphs'])
                seconds = dtphs / 15e6



                order = np.argsort(dtphs)
                seconds = seconds[order]
                intensities = intensities[order]
                intensity_err = intensity_err[order]
                poisson_err = poisson_err[order]
                patch_sigma = patch_sigma[order]

                dtphs = dtphs[order]

                dp_dict[dp][temp]['intensities'] = intensities
                dp_dict[dp][temp]['intensity_err'] = intensity_err
                dp_dict[dp][temp]['poisson_err'] = poisson_err
                dp_dict[dp][temp]['patch_sigma'] = patch_sigma

                dp_dict[dp][temp]['seconds'] = seconds
                dp_dict[dp][temp]['dtphs'] = dtphs



        full_dipole_dict[quad] = dp_dict



    return full_dipole_dict

def intensity_function(tph,coeff,tau):
    npumps = 3000
    d_t = 1
    p_c = 1
    return npumps*coeff*(np.exp(-tph / tau) - np.exp(-8 * (tph/tau)))


# Peak value of exp(-x) - exp(-8x), reached at x = ln(8)/7: used for initial guesses.
INTENSITY_SHAPE_PEAK = 0.650
INTENSITY_SHAPE_PEAK_X = np.log(8.0) / 7.0


def intensity_function_offset(tph, coeff, tau, offset):
    """Pumped-dipole model plus a t_ph-independent pedestal.

    The pedestal is generated during readout (the trap defers charge from the
    dark-current background packets clocked through its physical pixel), so it
    is identical in every image of the t_ph scan. Both coeff and offset are
    signed: the pumped dipole orientation is set by the sub-pixel trap position
    while the pedestal orientation is set by the readout direction, and the two
    need not agree.
    """
    return intensity_function(tph, coeff, tau) + offset


#hole effective masses (in units of the electron rest mass) for p-channel
#silicon between 100 and 200 K (Green 1990), as in arXiv:2406.18502
M_COND_HOLE = 0.41
M_DENS_HOLE = 0.94

def hole_thermal_velocity(temperatures):
    """v_th = sqrt(3 k_B T / m_cond) for holes, in cm/s."""
    kb = 8.617333262e-5 #eV/K
    me = 0.510998950e6 #eV
    ccms = 2.99792458e10 #cm/s
    return ccms * np.sqrt(3 * kb * temperatures / (M_COND_HOLE * me))

def log_energy_cross_section(temperatures,E,logsigma):
    #SRH emission time tau_e = exp(E/kT) / (sigma v_th N_v) with
    #v_th = sqrt(3 kT / m_cond) and N_v = 2 (2 pi m_dens kT / h^2)^(3/2)
    kb = 8.617333262e-5 #eV/K
    h = 4.135667696e-15 #eV s
    me = 0.510998950e6 #eV
    ccms = 2.99792458e10 #cm/s
    denom = 2 * np.sqrt(3) * (2 * np.pi)**(3/2) * (M_DENS_HOLE * me)**(3/2) / np.sqrt(M_COND_HOLE * me)
    kbT = kb * temperatures
    scaling_factor =  (h**3) * (ccms**2) / denom
    logtaus = np.log(scaling_factor) - logsigma - 2 * np.log(kbT) + (E  / kbT)
    return logtaus


def fit_energy_cross_section(good_temperatures, good_taus, good_tau_errs,
                             good_signs, wellBehavedThreshold,
                             errors_are_absolute):
    from scipy.optimize import curve_fit
    from scipy.stats import chi2
    logtaus = np.log(good_taus)
    logtauerr = good_tau_errs / good_taus
    # Orientation consistency: a single physical trap pumps in one
    # direction, so all significant amplitudes should share a sign.
    n_pos = int(np.sum(good_signs > 0))
    n_neg = int(np.sum(good_signs < 0))
    single_orientation = (n_pos == 0) or (n_neg == 0)
    result = {
        'n_positive_temps': n_pos,
        'n_negative_temps': n_neg,
        'OrientationConsistent': bool(single_orientation),
        'OrientationClass': None,
        'WellBehavedTrap': None,
        'EnergyFitFailed': None,
        'GoodEnergyFit': None,
        'popt': None,
        'perr': None,
        'pcov': None,
        'chi_squared': None,
        'reduced_chi_squared': None,
        'p_value': None,
        'r2': None,
    }
    if n_pos >= 2 and n_neg >= 2:
        result['OrientationClass'] = 'dual_response'
    elif not single_orientation:
        result['OrientationClass'] = 'ambiguous_sign_conflict'
    elif n_pos > 0:
        result['OrientationClass'] = 'single_positive'
    elif n_neg > 0:
        result['OrientationClass'] = 'single_negative'
    else:
        result['OrientationClass'] = 'no_good_fit'
    if not single_orientation:
        result['GoodEnergyFit'] = False

    
    result['WellBehavedTrap'] = True if len(good_temperatures) >= wellBehavedThreshold else False

    if result['WellBehavedTrap'] and single_orientation:
        try:
            popt, pcov = curve_fit(
                log_energy_cross_section,
                good_temperatures,
                logtaus,
                sigma=logtauerr,
                bounds=([0,-100],[2,-1]),
                absolute_sigma=errors_are_absolute,
            )
            perr = np.sqrt(np.diag(pcov))
            result['EnergyFitFailed'] = False
        except:
            result['EnergyFitFailed'] = True
            return result
        log_tau_fit = log_energy_cross_section(good_temperatures, *popt)
        residuals = np.log(good_taus) - log_tau_fit

        chi_squared = np.sum((residuals / logtauerr)**2)
        dof = len(good_taus) - len(popt)
        reduced_chi_squared = chi_squared/dof
        # p_value = 1 - chi2.cdf(chi_squared, dof)
        p_value = chi2.sf(chi_squared, dof)

        # p_value = 1 - chi2.cdf(chi_squared, dof)
        ss_res = np.sum((residuals) ** 2)

        ss_tot = np.sum((np.log(good_taus)- np.mean(np.log(good_taus))) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        
        
        goodness_of_fit =  p_value > 0.05
        

        rtol = 0.25

        goodness_of_fit = (r2 < 1 + rtol) & (r2 > 1 - rtol)

        # Dof-insensitive reduced-chi2 threshold instead of a p>0.05 GOF.
        # The p-value GOF is biased against well-sampled traps: with ~15
        # precise tau points it has the power to reject on the ~1% high-T
        # Arrhenius lean, so pass-rate falls 65%->21% as temperatures grow
        # and the BEST-measured traps are preferentially cut. A reduced-chi2
        # cut does not get stricter with more points; it still removes the
        # genuinely non-Arrhenius blends (reduced chi2 >> threshold).
        srh_reduced_chi2_max = 10
        goodness_of_fit = reduced_chi_squared < srh_reduced_chi2_max
        #energy boundary
        if popt[0] <= 1e-5 or popt[0] > 10:
            goodness_of_fit = False

        if popt[1] == -100 or popt[1] == -1:
            goodness_of_fit = False

        result['GoodEnergyFit'] = True if goodness_of_fit else False
        result['popt'] = popt
        result['perr'] = perr
        result['pcov'] = pcov
        result['chi_squared'] = chi_squared
        result['reduced_chi_squared'] = reduced_chi_squared
        result['p_value'] = p_value
        result['r2'] = r2

    return result


# (minimal pipeline) robust_energy_fit and estimate_intrinsic_dispersion
# removed: no intrinsic-scatter budget and no automatic outlier rejection.
def constant_fit_r2(y, y_err=None):
    # Step 1: Fit the best constant (weighted or unweighted)
    if y_err is None:
        c = np.mean(y)
    else:
        weights = 1 / y_err**2
        c = np.sum(y * weights) / np.sum(weights)

    y_pred = np.full_like(y, c)
    
    # Step 2: Calculate R²
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return c, r2


def fitTrapIntensity(full_dipole_dict,useIntensityErr=True,wellBehavedThreshold=4,
                     fit_offset=True,errors_are_absolute=True,
                     delta_chi2_threshold=11.83,delta_chi2_table=None):
    # fit_offset=True: fit the signed 3-parameter model
    #   I(t_ph) = 3000*coeff*(exp(-t_ph/tau) - exp(-8 t_ph/tau)) + offset
    # and replace the two max-intensity threshold cuts with the statistically
    # meaningful pumped-amplitude significance |coeff|/sigma_coeff >= 3 (with a
    # pedestal in the data, "max intensity" no longer measures the pumped signal).
    # Requires spectra extracted with absolute=False (signed intensities).
    # errors_are_absolute=True tells scipy that intensity_err and tau_err are
    # measured standard deviations. Otherwise curve_fit rescales parameter
    # covariance by reduced chi-square, which changes the significance and
    # relative-error cuts even though the supplied errors are already absolute.
    print(f"Requiring at least {wellBehavedThreshold} Good Temperature Fits")
    from scipy.optimize import curve_fit
    from scipy.stats import chi2,linregress
    from tqdm.autonotebook import tqdm
    goodquads = list(full_dipole_dict.keys())
    for quad in range(len(goodquads)):
        q = goodquads[quad]
        dipole_dict = full_dipole_dict[q]
        dplist = list(dipole_dict.keys())
        for d in tqdm(range(len(dplist))):
            dp = dplist[d]
            dptest = dipole_dict[dp]
            good_temperatures = []
            good_taus = []
            good_tau_errs = []
            good_signs = []
            for temp in list(dptest.keys()):
                if type(temp) != int:
                    continue
                seconds = dipole_dict[dp][temp]['seconds']
                intensities = dipole_dict[dp][temp]['intensities']
                if useIntensityErr:
                    intensity_err = dipole_dict[dp][temp]['intensity_err']
                else:
                    intensity_err = dipole_dict[dp][temp]['poisson_err']


                min_tph = np.min(seconds)
                max_tph = np.max(seconds)

                if fit_offset:
                    model_fn = intensity_function_offset
                    offset_estimate = float(np.median(intensities))
                    deviations = intensities - offset_estimate
                    k_peak = int(np.argmax(np.abs(deviations)))
                    coeff_estimate = deviations[k_peak] / (3_000 * INTENSITY_SHAPE_PEAK)
                    tau_estimate = float(np.clip(seconds[k_peak] / INTENSITY_SHAPE_PEAK_X, 1e-8, 1000))
                    p0 = [coeff_estimate, tau_estimate, offset_estimate]
                    fit_bounds = ([-np.inf, 1e-8, -np.inf], [np.inf, 1000, np.inf])
                else:
                    model_fn = intensity_function
                    tau_estimate = seconds[np.argmax(intensities)]
                    dtpc_estimate = np.max(intensities) * 8 / 3_000 / 5.2
                    p0 = [dtpc_estimate, tau_estimate]
                    fit_bounds = ([0, 1e-8], [np.inf, 1000])

                try:

                    popt, pcov = curve_fit(
                        model_fn,
                        seconds,
                        intensities,
                        sigma=intensity_err,
                        p0=p0,
                        bounds=fit_bounds,
                        absolute_sigma=errors_are_absolute,
                        maxfev=20000,
                    )
                    dipole_dict[dp][temp]['IntensityFitFailed'] = False
                    # else:
                    #     popt, pcov = curve_fit(intensity_function, seconds, intensities,sigma=poisson_err,p0=[dtpc_estimate,tau_estimate],bounds=([0, min_tph],[np.inf,max_tph]))
                    #     dipole_dict[dp][temp]['IntensityFitFailed'] = False


                except:
                    # #try fit without errorbars
                    # dipole_dict[dp][temp]['FitFailed'] = 0
                    # try:
                    #     popt, pcov = curve_fit(intensity_function, seconds, intensities,p0=[dtpc_estimate,tau_estimate],bounds=([0, min_tph],[np.inf,max_tph]))
                    # except:
                    # print('fit did not work')
                    dipole_dict[dp][temp]['IntensityFitFailed'] = True
                    dipole_dict[dp][temp]['GoodIntensityFit'] = False
                    continue

                
                const, const_lin_r2 = constant_fit_r2(intensities, y_err=intensity_err)
                slope, intercept, r, p, std_err = linregress(seconds, intensities)
                lin_r2 = r**2



                residuals = intensities - model_fn(seconds, *popt)
                chi_squared = np.sum((residuals / intensity_err)**2)
                dof = len(intensities) - len(popt)
                reduced_chi_squared = chi_squared/dof
                #
                p_value = 1 - chi2.cdf(chi_squared, dof)
                #chi2.sf(chi_squared, dof)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((intensities- np.mean(intensities)) ** 2)
                r2 = 1 - (ss_res / ss_tot)



                rtol = 0.25

                goodness_of_fit_test = (r2 < 1 + rtol) & (r2 > 1 - rtol)

                goodness_of_fit_test = p_value > 0.05 if useIntensityErr else reduced_chi_squared < 500
                    
                # goodness_of_fit_test = reduced_chi_squared < 500

                # print(f"Chi-squared: {chi_squared:.2f}, DoF: {dof}, p-value: {p_value:.4f}")
                dipole_dict[dp][temp]['GoodIntensityFit'] = True if goodness_of_fit_test else False

                perr = np.sqrt(np.diag(pcov))

                if fit_offset:
                    amplitude_significance = np.abs(popt[0]) / perr[0] if perr[0] > 0 else 0.0
                    dipole_dict[dp][temp]['amplitude_significance'] = amplitude_significance
                    if amplitude_significance < 3:
                        dipole_dict[dp][temp]['GoodIntensityFit'] = False
                    # Delta-chi2 of the pumped+offset model vs the best constant,
                    # used as a secondary guard against spike fits where the
                    # covariance underestimates sigma_A (behind the
                    # amplitude-significance and >=4-temperature cuts).
                    #
                    # IMPORTANT: this is an UNCALIBRATED guard, NOT a calibrated
                    # significance. Under the no-pumping null (coeff=0) tau is an
                    # unidentified nuisance that is optimized over, so Wilks'
                    # theorem does not apply and delta_chi2 is NOT chi2_2
                    # distributed. The default 11.83 is therefore NOT a true
                    # "3-sigma for 2 parameters" cut: the real null tail is
                    # heavier (empirically ~1.9% of control nulls exceed it,
                    # i.e. ~2.3 sigma, not 0.27%). For a calibrated, per-temperature
                    # false-positive rate, pass delta_chi2_table={temperature: thr}
                    # built from trap-free control pairs (see
                    # signed_refit_detection_calibration.py).
                    weights = 1.0 / intensity_err**2
                    const_best = np.sum(intensities * weights) / np.sum(weights)
                    chi2_const = np.sum(((intensities - const_best) / intensity_err) ** 2)
                    delta_chi2 = chi2_const - chi_squared
                    dipole_dict[dp][temp]['delta_chi2_vs_constant'] = delta_chi2
                    if delta_chi2_table is not None:
                        threshold = float(delta_chi2_table.get(int(temp), delta_chi2_threshold))
                    else:
                        threshold = float(delta_chi2_threshold)
                    dipole_dict[dp][temp]['delta_chi2_threshold'] = threshold
                    if delta_chi2 < threshold:
                        dipole_dict[dp][temp]['GoodIntensityFit'] = False
                else:
                    if np.max(intensities) < 3 * np.mean(intensity_err):
                        dipole_dict[dp][temp]['GoodIntensityFit'] = False
                    sigma_image = dipole_dict[dp][temp]['image_sigma']
                    if np.max(intensities) < 3 * sigma_image:
                        dipole_dict[dp][temp]['GoodIntensityFit'] = False

                rel_error = perr[1]/popt[1]
                if rel_error > 0.5:
                    dipole_dict[dp][temp]['GoodIntensityFit'] = False

                # rel_erros = np.abs(residuals) / intensities
                # if np.nanmean(rel_erros) > 0.5:
                #     dipole_dict[dp][temp]['GoodIntensityFit'] = False

                dipole_dict[dp][temp]['fit_p_value'] = p_value
                dipole_dict[dp][temp]['fit_chi_squared'] = chi_squared
                dipole_dict[dp][temp]['fit_reduced_chi_squared'] = reduced_chi_squared

                dipole_dict[dp][temp]['fit_r_squared'] = r2
                dipole_dict[dp][temp]['fit_lin_r_squared'] = lin_r2
                dipole_dict[dp][temp]['fit_const_lin_r_squared'] = const_lin_r2

                dipole_dict[dp][temp]['fit_coeff'] = popt[0]
                dipole_dict[dp][temp]['fit_tau'] = popt[1]

                dipole_dict[dp][temp]['fit_coeff_err'] = perr[0]
                dipole_dict[dp][temp]['fit_tau_err'] = perr[1]
                dipole_dict[dp][temp]['fit_covariance_matrix'] = pcov
                dipole_dict[dp][temp]['fit_errors_are_absolute'] = errors_are_absolute
                if fit_offset:
                    dipole_dict[dp][temp]['fit_offset'] = popt[2]
                    dipole_dict[dp][temp]['fit_offset_err'] = perr[2]

                if dipole_dict[dp][temp]['GoodIntensityFit']:
                    good_temperatures.append(temp)
                    good_taus.append(popt[1])
                    good_tau_errs.append(perr[1])
                    good_signs.append(np.sign(popt[0]))


            good_temperatures=np.array(good_temperatures)
            good_taus=np.array(good_taus)
            good_tau_errs=np.array(good_tau_errs)
            good_signs=np.array(good_signs)
            energy_fit = fit_energy_cross_section(
                good_temperatures,
                good_taus,
                good_tau_errs,
                good_signs,
                wellBehavedThreshold,
                errors_are_absolute,
            )
            dipole_dict[dp]['n_positive_temps'] = energy_fit['n_positive_temps']
            dipole_dict[dp]['n_negative_temps'] = energy_fit['n_negative_temps']
            dipole_dict[dp]['OrientationConsistent'] = energy_fit['OrientationConsistent']
            dipole_dict[dp]['OrientationClass'] = energy_fit['OrientationClass']
            if not energy_fit['OrientationConsistent']:
                dipole_dict[dp]['GoodEnergyFit'] = False

            
            dipole_dict[dp]['WellBehavedTrap'] = energy_fit['WellBehavedTrap']

            if dipole_dict[dp]['WellBehavedTrap'] and energy_fit['OrientationConsistent']:
                if energy_fit['EnergyFitFailed']:
                    dipole_dict[dp]['EnergyFitFailed'] = True
                    
                    continue
                popt = energy_fit['popt']
                perr = energy_fit['perr']
                pcov = energy_fit['pcov']
                chi_squared = energy_fit['chi_squared']
                reduced_chi_squared = energy_fit['reduced_chi_squared']
                p_value = energy_fit['p_value']
                r2 = energy_fit['r2']
                dipole_dict[dp]['EnergyFitFailed'] = False
                


                dipole_dict[dp]['GoodEnergyFit'] = energy_fit['GoodEnergyFit']
                # if dipole_dict[dp]['GoodEnergyFit']:
                dipole_dict[dp]['energy_BestFitEnergy'] = popt[0]
                dipole_dict[dp]['energy_BestFitEnergyErr'] = perr[0]
                dipole_dict[dp]['energy_r_squared'] = r2

                dipole_dict[dp]['energy_chi2'] = chi_squared
                dipole_dict[dp]['energy_reduced_chi2'] = reduced_chi_squared
                dipole_dict[dp]['energy_p_value'] = p_value

                dipole_dict[dp]['energy_BestFitCrossSection'] = np.exp(popt[1])
                dipole_dict[dp]['energy_BestFitCrossSectionErr'] = perr[1] * np.exp(popt[1])
                dipole_dict[dp]['energy_CovarianceMatrix'] = pcov
                dipole_dict[dp]['energy_temperatures'] = good_temperatures
                dipole_dict[dp]['energy_taus'] = good_taus
                dipole_dict[dp]['energy_tau_errs'] = good_tau_errs
                dipole_dict[dp]['energy_fit_errors_are_absolute'] = errors_are_absolute


    return full_dipole_dict

def fitTrapIntensity_cutflow(full_dipole_dict, useIntensityErr=True, wellBehavedThreshold=4):
    print(f"Requiring at least {wellBehavedThreshold} Good Temperature Fits")
    from scipy.optimize import curve_fit
    from scipy.stats import chi2, linregress
    from tqdm.autonotebook import tqdm
    
    # Dictionary to accumulate rejection reasons
    rejection_summary = {}

    goodquads = list(full_dipole_dict.keys())
    
    for quad in range(len(goodquads)):
        q = goodquads[quad]
        dipole_dict = full_dipole_dict[q]
        dplist = list(dipole_dict.keys())
        
        for d in tqdm(range(len(dplist))):
            dp = dplist[d]
            dptest = dipole_dict[dp]
            good_temperatures = []
            good_taus = []
            good_tau_errs = []
            
            for temp in list(dptest.keys()):
                if type(temp) != int:
                    continue
                seconds = dipole_dict[dp][temp]['seconds']
                intensities = dipole_dict[dp][temp]['intensities']
                intensity_err = dipole_dict[dp][temp]['intensity_err'] if useIntensityErr else dipole_dict[dp][temp]['poisson_err']
                
                min_tph = np.min(seconds)
                max_tph = np.max(seconds)
                tau_estimate = seconds[np.argmax(intensities)]
                dtpc_estimate = np.max(intensities) * 8 / 3_000 / 5.2
                
                rejection_reasons = []  # To store the rejection reasons
                
                try:
                    popt, pcov = curve_fit(intensity_function, seconds, intensities, sigma=intensity_err, 
                                           p0=[dtpc_estimate, tau_estimate], bounds=([0, 1e-8], [np.inf, 1000]))
                    dipole_dict[dp][temp]['IntensityFitFailed'] = False

                except Exception as e:
                    dipole_dict[dp][temp]['IntensityFitFailed'] = True
                    dipole_dict[dp][temp]['GoodIntensityFit'] = False
                    rejection_reasons.append(f"Fit failed with error: {str(e)}")
                    continue

                const, const_lin_r2 = constant_fit_r2(intensities, y_err=intensity_err)
                slope, intercept, r, p, std_err = linregress(seconds, intensities)
                lin_r2 = r ** 2

                residuals = intensities - intensity_function(seconds, *popt)
                chi_squared = np.sum((residuals / intensity_err) ** 2)
                dof = len(intensities) - len(popt)
                reduced_chi_squared = chi_squared / dof
                p_value = 1 - chi2.cdf(chi_squared, dof)
                ss_res = np.sum((intensities - intensity_function(seconds, *popt)) ** 2)
                ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
                r2 = 1 - (ss_res / ss_tot)

                rtol = 0.25
                goodness_of_fit_test = (r2 < 1 + rtol) & (r2 > 1 - rtol)
                goodness_of_fit_test = p_value > 0.05 if useIntensityErr else reduced_chi_squared < 500
                dipole_dict[dp][temp]['GoodIntensityFit'] = True if goodness_of_fit_test else False

                # Additional Rejection Criteria
                if np.max(intensities) < 3 * np.mean(intensity_err):
                    rejection_reasons.append("Max intensity less than 3 times the mean intensity error")
                    dipole_dict[dp][temp]['GoodIntensityFit'] = False
                sigma_image = dipole_dict[dp][temp]['image_sigma']
                if np.max(intensities) < 3 * sigma_image:
                    rejection_reasons.append("Max intensity less than 3 times image sigma")
                    dipole_dict[dp][temp]['GoodIntensityFit'] = False
                
                perr = np.sqrt(np.diag(pcov))
                rel_error = perr[1] / popt[1]
                if rel_error > 0.5:
                    rejection_reasons.append(f"Relative error for tau coefficient > 0.5")
                    dipole_dict[dp][temp]['GoodIntensityFit'] = False
                
                # Accumulate rejection reasons in the summary
                if not dipole_dict[dp][temp]['GoodIntensityFit']:
                    for reason in rejection_reasons:
                        if reason not in rejection_summary:
                            rejection_summary[reason] = 0
                        rejection_summary[reason] += 1
                    continue

                dipole_dict[dp][temp]['fit_p_value'] = p_value
                dipole_dict[dp][temp]['fit_chi_squared'] = chi_squared
                dipole_dict[dp][temp]['fit_reduced_chi_squared'] = reduced_chi_squared
                dipole_dict[dp][temp]['fit_r_squared'] = r2
                dipole_dict[dp][temp]['fit_lin_r_squared'] = lin_r2
                dipole_dict[dp][temp]['fit_const_lin_r_squared'] = const_lin_r2
                dipole_dict[dp][temp]['fit_coeff'] = popt[0]
                dipole_dict[dp][temp]['fit_tau'] = popt[1]
                dipole_dict[dp][temp]['fit_coeff_err'] = perr[0]
                dipole_dict[dp][temp]['fit_tau_err'] = perr[1]
                dipole_dict[dp][temp]['fit_covariance_matrix'] = pcov

                if dipole_dict[dp][temp]['GoodIntensityFit']:
                    good_temperatures.append(temp)
                    good_taus.append(popt[1])
                    good_tau_errs.append(perr[1])

            # After processing all temperatures for the current dipole, check if enough good fits
            good_temperatures = np.array(good_temperatures)
            good_taus = np.array(good_taus)
            good_tau_errs = np.array(good_tau_errs)
            logtaus = np.log(good_taus)
            logtauerr = good_tau_errs / good_taus

            dipole_dict[dp]['WellBehavedTrap'] = True if len(good_temperatures) >= wellBehavedThreshold else False

            if dipole_dict[dp]['WellBehavedTrap']:
                try:
                    popt, pcov = curve_fit(log_energy_cross_section, good_temperatures, logtaus, 
                                           sigma=logtauerr, bounds=([0, -100], [2, -1]))
                    perr = np.sqrt(np.diag(pcov))
                    dipole_dict[dp]['EnergyFitFailed'] = False
                except Exception as e:
                    dipole_dict[dp]['EnergyFitFailed'] = True
                    print(f"Energy fit failed for {dp} due to: {e}")
                    continue

                log_tau_fit = log_energy_cross_section(good_temperatures, *popt)
                residuals = np.log(good_taus) - log_tau_fit
                chi_squared = np.sum((residuals / logtauerr) ** 2)
                dof = len(good_taus) - len(popt)
                reduced_chi_squared = chi_squared / dof
                p_value = chi2.sf(chi_squared, dof)
                ss_res = np.sum((residuals) ** 2)
                ss_tot = np.sum((np.log(good_taus) - np.mean(np.log(good_taus))) ** 2)
                r2 = 1 - (ss_res / ss_tot)

                # Energy fit rejection criteria
                goodness_of_fit = p_value > 0.05
                rtol = 0.25
                goodness_of_fit = (r2 < 1 + rtol) & (r2 > 1 - rtol)
                goodness_of_fit = reduced_chi_squared < 5

                if popt[0] <= 1e-5 or popt[0] > 10:
                    goodness_of_fit = False
                if popt[1] == -100 or popt[1] == -1:
                    goodness_of_fit = False

                dipole_dict[dp]['GoodEnergyFit'] = True if goodness_of_fit else False

                dipole_dict[dp]['energy_BestFitEnergy'] = popt[0]
                dipole_dict[dp]['energy_BestFitEnergyErr'] = perr[0]
                dipole_dict[dp]['energy_r_squared'] = r2
                dipole_dict[dp]['energy_chi2'] = chi_squared
                dipole_dict[dp]['energy_reduced_chi2'] = reduced_chi_squared
                dipole_dict[dp]['energy_p_value'] = p_value
                dipole_dict[dp]['energy_BestFitCrossSection'] = np.exp(popt[1])
                dipole_dict[dp]['energy_BestFitCrossSectionErr'] = perr[1] * np.exp(popt[1])
                dipole_dict[dp]['energy_CovarianceMatrix'] = pcov
                dipole_dict[dp]['energy_temperatures'] = good_temperatures
                dipole_dict[dp]['energy_taus'] = good_taus
                dipole_dict[dp]['energy_tau_errs'] = good_tau_errs

    # Print the rejection summary at the end
    print("\nRejection Summary:")
    for reason, count in rejection_summary.items():
        print(f"  {reason}: {count} times")

    return full_dipole_dict



def plotRandomDipoleSpectra(fit_dipole_spectra,quads,n=10):
    import random
    cmap = plt.cm.RdBu
    import matplotlib.colors as colors
    norm = colors.Normalize(vmin=125,vmax=210)
    for q in quads:
        for i in range(n):
            print()
            dpkeys = list(fit_dipole_spectra[q])
            dp = random.choice(dpkeys)
            plt.figure()
            plt.xlabel('Seconds')
            plt.ylabel("Intensity ")
            plt.xscale('log')
            title = f'Trap @ Quad {q+1}: {int(dp[0]),int(dp[1])}'
            plt.title(title)
            testdp = fit_dipole_spectra[q][dp]
            if testdp['WellBehavedTrap']:
                for temp in testdp.keys():
                    if type(temp) != int:
                        continue
                    dipole = testdp[temp]
                    if dipole['GoodIntensityFit']:
                
                        plt.scatter(dipole['seconds'],dipole['intensities'],color=cmap(norm(temp)))
                        plt.errorbar(dipole['seconds'],dipole['intensities'],yerr=dipole['intensity_err'],color=cmap(norm(temp)),ls='None')

                        seconds = np.geomspace(np.min(dipole['seconds']),np.max(dipole['seconds']),100)
                        fit_ints = intensity_function(seconds,dipole['fit_coeff'],dipole['fit_tau'])
                        plt.plot(seconds,fit_ints,ls='-',color=cmap(norm(temp)))
                sm = plt.cm.ScalarMappable(cmap=cmap)
                sm.set_clim(vmin=125, vmax=210)
                ax = plt.gca()
                colorbar = plt.colorbar(sm,ax=ax)
            plt.show()
            plt.close()
            


def monte_carlo_distance_histograms(
    n_points,
    grid_width,
    grid_height,
    num_samples=1000,
    num_bins=100,
    confidence_level=0.90,
    spline_smoothing=0.5,  # Lower = smoother
    return_spline=True
):
    
    """
    Monte Carlo simulation to estimate the distribution of pairwise distances
    for randomly distributed coordinates in a bounded grid.
    
    Returns:
        bin_centers: centers of distance bins
        mean_hist: mean histogram over samples
        ci_lower: lower bound of confidence interval
        ci_upper: upper bound of confidence interval
        spline (optional): spline fit of mean histogram
    """
    from scipy.spatial.distance import pdist
    from scipy.interpolate import UnivariateSpline
    from tqdm.autonotebook import tqdm

    montecarlo_hists = []
    
    max_distance = np.hypot(grid_width, grid_height)
    bin_edges = np.linspace(0, max_distance, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    for _ in tqdm(range(num_samples), desc="Monte Carlo simulation"):
        random_points = np.column_stack((
            np.random.uniform(0, grid_width, n_points),
            np.random.uniform(0, grid_height, n_points)
        ))
        distances = pdist(random_points)
        hist, _ = np.histogram(distances, bins=bin_edges, density=True)
        montecarlo_hists.append(hist)

    montecarlo_hists = np.array(montecarlo_hists)
    mean_hist = montecarlo_hists.mean(axis=0)

    # Compute confidence intervals
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    ci_lower = np.percentile(montecarlo_hists, lower_percentile, axis=0)
    ci_upper = np.percentile(montecarlo_hists, upper_percentile, axis=0)

    if return_spline:
        spline = UnivariateSpline(bin_centers, mean_hist, s=spline_smoothing)
        return bin_centers, mean_hist, ci_lower, ci_upper,montecarlo_hists, spline
    else:
        return bin_centers, mean_hist, ci_lower, ci_upper,montecarlo_hists
