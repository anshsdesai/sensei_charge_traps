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

def findDipoles2(electronized_image,debug=False,useFit=False):

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

            if comparable_perc(charge1,charge2):
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

def getDipoleList2(image_dir,temperatures,goodquads,plot=False):
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
    for q in goodquads:
        dipole_list = []
        for temperature in temperatures:
            # print(f"temperature: {temperature}")
            dipole_occurrences = defaultdict(set)
            all_dipoles = []
            temp = f'{temperature}k'
            search_str = image_dir + f'proc*{temp}*_' + '*dtph*' + ccd_str 
            imagefiles = glob.glob(search_str)

            if plot:
                fig = plt.figure(figsize=(12,8))
            
            for imagefile in imagefiles:
                dtph = int(re.findall(r'dtph\d+_',imagefile)[0][4:-1])
                image = get_qdata(imagefile,q)
                image = crop_qdata(image)#,ylower=500,xlower=100)
                # print(imagefile)
                image = approximate_electronize(image,eval)
                image_dipoles = findDipoles2(image)
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

def getDipoleSpectra2(image_dir,goodquads,full_dipole_coord_list,absolute=True):
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
        search_str = image_dir + f'proc*_' + '*dtph*' + ccd_str 
        imagefiles = glob.glob(search_str)
        temps = []
        for dp in dipole_coord_list:
            dp_dict[dp] = {}

        for f in tqdm(range(len((imagefiles)))):
            imagef = imagefiles[f]
            dtph = int(re.findall(r'dtph\d+_',imagef)[0][4:-1])
            temp = int(re.findall(r'_\d+k',imagef)[0][1:-1])
            image = get_qdata(imagef,quad)
            image = crop_qdata(image)
            image = approximate_electronize(image,eval)

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


                
                dp_dict[dp][temp]['intensities'].append(intensity)
                dp_dict[dp][temp]['dtphs'].append(dtph)
                dp_dict[dp][temp]['intensity_err'].append(sigma_hist)
                dp_dict[dp][temp]['poisson_err'].append(sigma_poisson)


                # dp_dict[dp][temp]['intensity_err_fit'].append(sigma)



        #sort by dtph
        for dp in dp_dict.keys():
            for temp in dp_dict[dp].keys():
                intensities = np.array(dp_dict[dp][temp]['intensities'])
                intensity_err = np.array(dp_dict[dp][temp]['intensity_err'])
                poisson_err =  np.array(dp_dict[dp][temp]['poisson_err'])

                dtphs = np.array(dp_dict[dp][temp]['dtphs'])
                seconds = dtphs / 15e6



                seconds = seconds[np.argsort(dtphs)]
                intensities = intensities[np.argsort(dtphs)]
                intensity_err = intensity_err[np.argsort(dtphs)]
                poisson_err = poisson_err[np.argsort(dtphs)]

                dtphs = dtphs[np.argsort(dtphs)]

                dp_dict[dp][temp]['intensities'] = intensities
                dp_dict[dp][temp]['intensity_err'] = intensity_err
                dp_dict[dp][temp]['poisson_err'] = poisson_err

                dp_dict[dp][temp]['seconds'] = seconds
                dp_dict[dp][temp]['dtphs'] = dtphs



        full_dipole_dict[quad] = dp_dict



    return full_dipole_dict

def intensity_function(tph,coeff,tau):
    npumps = 3000
    d_t = 1
    p_c = 1
    return npumps*coeff*(np.exp(-tph / tau) - np.exp(-8 * (tph/tau)))


def log_energy_cross_section(temperatures,E,logsigma):
    kb = 8.6717333262e-5 #eV/K
    h = 4.1135e-15 #eV/s
    me = 0.511e6 #eV
    ccms = 3e10
    denom = 2*(me* np.sqrt(3) * (2 * np.pi)**(3/2))
    kbT = kb * temperatures
    scaling_factor =  (h**3) * (ccms**2) / denom
    logtaus = np.log(scaling_factor) - logsigma - 2 * np.log(kbT) + (E  / kbT)
    return logtaus




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


def fitTrapIntensity(full_dipole_dict,useIntensityErr=True,wellBehavedThreshold=4):
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
                tau_estimate = seconds[np.argmax(intensities)]
                dtpc_estimate = np.max(intensities) * 8 / 3_000 / 5.2
               
                try:
                    
                    popt, pcov = curve_fit(intensity_function, seconds, intensities,sigma=intensity_err,p0=[dtpc_estimate,tau_estimate],bounds=([0, 1e-8],[np.inf,1000]))
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



                residuals = intensities - intensity_function(seconds, *popt)
                chi_squared = np.sum((residuals / intensity_err)**2)
                dof = len(intensities) - len(popt)
                reduced_chi_squared = chi_squared/dof
                # 
                p_value = 1 - chi2.cdf(chi_squared, dof)
                #chi2.sf(chi_squared, dof)
                ss_res = np.sum((intensities - intensity_function(seconds, *popt)) ** 2)
                ss_tot = np.sum((intensities- np.mean(intensities)) ** 2)
                r2 = 1 - (ss_res / ss_tot)



                rtol = 0.25

                goodness_of_fit_test = (r2 < 1 + rtol) & (r2 > 1 - rtol)

                goodness_of_fit_test = p_value > 0.05 if useIntensityErr else reduced_chi_squared < 500
                    
                # goodness_of_fit_test = reduced_chi_squared < 500

                # print(f"Chi-squared: {chi_squared:.2f}, DoF: {dof}, p-value: {p_value:.4f}")
                dipole_dict[dp][temp]['GoodIntensityFit'] = True if goodness_of_fit_test else False
                if np.max(intensities) < 3 * np.mean(intensity_err):
                    dipole_dict[dp][temp]['GoodIntensityFit'] = False
                sigma_image = dipole_dict[dp][temp]['image_sigma']
                if np.max(intensities) < 3 * sigma_image:
                    dipole_dict[dp][temp]['GoodIntensityFit'] = False

                perr = np.sqrt(np.diag(pcov))
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

                if dipole_dict[dp][temp]['GoodIntensityFit']:
                    good_temperatures.append(temp)
                    good_taus.append(popt[1])
                    good_tau_errs.append(perr[1])


            good_temperatures=np.array(good_temperatures)
            good_taus=np.array(good_taus)
            good_tau_errs=np.array(good_tau_errs)
            logtaus = np.log(good_taus)
            logtauerr = good_tau_errs / good_taus

            
            dipole_dict[dp]['WellBehavedTrap'] = True if len(good_temperatures) >= wellBehavedThreshold else False

            if dipole_dict[dp]['WellBehavedTrap']:
                try:
                    popt, pcov = curve_fit(log_energy_cross_section, good_temperatures, logtaus,sigma=logtauerr,bounds=([0,-100],[2,-1]))
                    perr = np.sqrt(np.diag(pcov))
                    dipole_dict[dp]['EnergyFitFailed'] = False
                except:
                    dipole_dict[dp]['EnergyFitFailed'] = True
                    
                    continue
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

                goodness_of_fit = reduced_chi_squared < 5
                #energy boundary
                if popt[0] <= 1e-5 or popt[0] > 10:
                    goodness_of_fit = False

                if popt[1] == -100 or popt[1] == -1:
                    goodness_of_fit = False
                


                dipole_dict[dp]['GoodEnergyFit'] = True if goodness_of_fit else False
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