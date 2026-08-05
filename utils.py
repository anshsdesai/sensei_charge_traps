from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
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

import h5py
import numpy as np

def _save_dict_to_hdf5(group, d):
    """Recursively saves a dictionary to an HDF5 group."""
    for key, val in d.items():
        # Convert key to string, prepending 'temp_' if it's an integer
        key_str = f"temp_{key}" if isinstance(key, int) else str(key)
        
        if isinstance(val, dict):
            sub_grp = group.create_group(key_str)
            _save_dict_to_hdf5(sub_grp, val)
        elif isinstance(val, (list, np.ndarray)):
            arr = np.array(val)
            if arr.dtype == object:
                continue  # Skip arrays of objects to avoid HDF5 TypeError
            group.create_dataset(key_str, data=arr)
        elif val is None:
            group.attrs[key_str] = "NONE_VALUE"
        else:
            group.attrs[key_str] = val

def _load_dict_from_hdf5(group):
    """Recursively loads a dictionary from an HDF5 group."""
    d = {}
    for key, item in group.items():
        orig_key = int(key.split('_')[1]) if key.startswith('temp_') else key
        if isinstance(item, h5py.Group):
            d[orig_key] = _load_dict_from_hdf5(item)
        else:
            d[orig_key] = item[()] # Extracts datasets to arrays/scalars
            
    for key, val in group.attrs.items():
        orig_key = int(key.split('_')[1]) if key.startswith('temp_') else key
        d[orig_key] = None if val == "NONE_VALUE" else val
        
    return d

def save_spectra_hdf5(spectra_dict, filename='dipole_spectra.h5'):
    """Saves the nested dipole spectra dictionary to an HDF5 file."""
    with h5py.File(filename, 'w') as f:
        for quad, dp_dict in spectra_dict.items():
            quad_grp = f.create_group(f"quad_{quad}")
            for dp, temp_dict in dp_dict.items():
                dp_grp = quad_grp.create_group(f"dp_{dp[0]}_{dp[1]}")
                _save_dict_to_hdf5(dp_grp, temp_dict)

def load_spectra_hdf5(filename='dipole_spectra.h5'):
    """Loads the nested dipole spectra dictionary from an HDF5 file."""
    spectra_dict = {}
    with h5py.File(filename, 'r') as f:
        for quad_name, quad_grp in f.items():
            quad = int(quad_name.split('_')[1])
            spectra_dict[quad] = {}
            for dp_name, dp_grp in quad_grp.items():
                _, x, y = dp_name.split('_')
                dp = (int(x), int(y))
                spectra_dict[quad][dp] = _load_dict_from_hdf5(dp_grp)
    return spectra_dict


# ---------------------------------------------------------------------------
# Fit-catalog aggregation + fast caches
#
# load_spectra_hdf5 (above) reads every per-temperature 'seconds'/'intensities'
# array for every dipole via _load_dict_from_hdf5 -- across ~9000+ dipoles x 23
# temperatures this is what makes loading a fit_dipole_spectra*.h5 catalog take
# ~2.5 minutes, even though every downstream consumer immediately discards most
# of it via its own WellBehavedTrap/GoodEnergyFit filtering. run_charge_traps.py
# already pays that cost once to have fit_dipole_spectra in hand; the functions
# below reduce it there, in one pass, to three small artifacts that a notebook
# can load directly without ever calling load_spectra_hdf5 itself.
# ---------------------------------------------------------------------------

MEASUREMENT_TEMPERATURES = [
    125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180,
    183, 185, 187, 190, 193, 195, 197, 200, 203, 207, 210,
]


def aggregate_trap_fits(fit_dipole_spectra, log_energy_cross_section,
                        measurement_temperatures=None, quads=(0, 1, 2, 3)):
    """Reduce the fit catalog to the arrays the figures consume.

    ``log_energy_cross_section`` is passed explicitly (rather than imported at
    module level) because this module doesn't own a flavor choice between
    dipole.py and dipole_new.py -- callers resolve that themselves.

    Returns a dict with:
      - ``energy_crossSections``: list of (sigma, E, sigma_err, E_err, avg_good_temp)
      - ``energy_covariances``: parallel list of the 2x2 [E, logsigma] fit
        covariance per good trap (or None if unavailable); same order/length as
        ``energy_crossSections``. Used to draw the correlated (E, sigma)
        uncertainty and the covariance-propagated tau(T) band.
      - ``tau_temp_fits``: list of [temps, taus, tau_errs] per good trap
      - ``tau_temperatures``: {T: {'measured': [...], 'extrapolated': [...]}}
      - ``cross_sections`` / ``energies``: arrays from energy_crossSections
      - ``maxtaus``: per-trap max measured tau
    """
    if measurement_temperatures is None:
        measurement_temperatures = MEASUREMENT_TEMPERATURES

    energy_crossSections = []
    energy_covariances = []
    tau_temp_fits = []
    maxtaus = []
    tau_temperatures = {t: {'measured': [], 'extrapolated': []}
                        for t in measurement_temperatures}

    for q in quads:
        if q not in fit_dipole_spectra:
            continue
        for dp in list(fit_dipole_spectra[q]):
            if not isinstance(dp, tuple):
                continue
            testdp = fit_dipole_spectra[q][dp]
            # Minimal pipeline (dipole_new.py) only writes 'EnergyFitFailed' when
            # WellBehavedTrap AND single_orientation; a missing key means no energy
            # fit was attempted, which is equivalent to a failed fit.
            if not (testdp['WellBehavedTrap'] and not testdp.get('EnergyFitFailed', True)):
                continue

            maxtaus.append(np.max(testdp['energy_taus']))

            if not testdp["GoodEnergyFit"]:
                continue

            cs = testdp['energy_BestFitCrossSection']
            cserr = testdp['energy_BestFitCrossSectionErr']
            e = testdp['energy_BestFitEnergy']
            e_err = testdp['energy_BestFitEnergyErr']
            avg_good_temp = np.mean(testdp['energy_temperatures'])

            for t in measurement_temperatures:
                tau = np.exp(log_energy_cross_section(t, e, np.log(cs)))
                if t in testdp['energy_temperatures']:
                    tau_temperatures[t]['measured'].append(tau)
                else:
                    tau_temperatures[t]['extrapolated'].append(tau)

            energy_crossSections.append((cs, e, cserr, e_err, avg_good_temp))
            # 2x2 [E, logsigma] covariance from the Arrhenius fit (popt order),
            # as stored by dipole(_new).py. None if this catalog predates it.
            cov = testdp.get('energy_CovarianceMatrix')
            energy_covariances.append(np.array(cov, dtype=float)
                                      if cov is not None else None)
            tau_temp_fits.append([testdp['energy_temperatures'],
                                  testdp['energy_taus'],
                                  testdp['energy_tau_errs']])

    energy_crossSections_arr = np.array(energy_crossSections)
    cross_sections = energy_crossSections_arr[:, 0] if len(energy_crossSections) else np.array([])
    energies = energy_crossSections_arr[:, 1] if len(energy_crossSections) else np.array([])

    print(f"Number of good energy fits: {len(energy_crossSections)}")
    return {
        'energy_crossSections': energy_crossSections,
        'energy_covariances': energy_covariances,
        'tau_temp_fits': tau_temp_fits,
        'tau_temperatures': tau_temperatures,
        'cross_sections': cross_sections,
        'energies': energies,
        'maxtaus': maxtaus,
    }


def agg_cache_path(fit_source):
    """Derive the aggregate-cache filename from a fit_dipole_spectra*.h5 path."""
    return os.path.splitext(fit_source)[0] + '_agg.npz'


def amp_cache_path(fit_source):
    """Derive the amplitude-vs-temperature cache filename from a fit_dipole_spectra*.h5 path."""
    return os.path.splitext(fit_source)[0] + '_amp_by_temp.h5'


def example_cache_path(fit_source):
    """Derive the example-traps cache filename from a fit_dipole_spectra*.h5 path."""
    return os.path.splitext(fit_source)[0] + '_example_traps.h5'


def save_trap_fit_aggregate(agg, path):
    """Cache aggregate_trap_fits' output. agg's fields are ragged (variable-length
    per-trap arrays) and contain None entries (missing covariances), so it's
    pickled inside the npz rather than stored as plain arrays -- matching this
    repo's existing allow_pickle=True npz convention (e.g. mc_dist*.npz)."""
    np.savez(path, agg=np.array(agg, dtype=object))


def load_trap_fit_aggregate(path):
    """Load a cache written by save_trap_fit_aggregate."""
    return np.load(path, allow_pickle=True)['agg'].item()


def build_fit_catalog_caches(fit_dipole_spectra, log_energy_cross_section,
                             measurement_temperatures=None, quads=(0, 1, 2, 3),
                             num_example_traps=75):
    """Reduce an in-memory fit_dipole_spectra dict (as returned by
    load_spectra_hdf5 / dp.fitTrapIntensity) to three artifacts in a single pass
    -- the same walk aggregate_trap_fits already does, plus two more schema-
    compatible reductions collected along the way:

      - ``agg``: identical to aggregate_trap_fits' return value.
      - ``amp_by_temp``: {quad: {(row, col): {'WellBehavedTrap': True,
        <temp>: {'fit_coeff', 'fit_coeff_err', 'GoodIntensityFit'}, ...}}} for
        every WellBehavedTrap dipole (regardless of energy-fit outcome) --
        schema-compatible with fit_dipole_spectra, so
        plot_amplitude_vs_temperature can consume it unmodified.
      - ``example_traps``: same schema, keeping the full per-temperature curve
        data ('seconds', 'intensities', 'fit_coeff', 'fit_tau', 'fit_offset',
        error fields) for only the first ``num_example_traps`` traps meeting
        plot_example_traps' own gate (WellBehavedTrap, not EnergyFitFailed,
        GoodEnergyFit) -- schema-compatible, so plot_example_traps can consume
        it unmodified.

    Caching these three (rather than reloading the full catalog each time) is
    the point: load_spectra_hdf5 materializes every per-temperature 'seconds'/
    'intensities' array for every dipole -- ~9000+ dipoles x 23 temperatures --
    which is what makes loading a fit_dipole_spectra*.h5 catalog take ~2.5
    minutes; a downstream notebook that only needs these three small
    reductions never has to pay that cost again once run_charge_traps.py has
    written them (it already pays it once, to have fit_dipole_spectra in hand
    at all).
    """
    if measurement_temperatures is None:
        measurement_temperatures = MEASUREMENT_TEMPERATURES

    energy_crossSections = []
    energy_covariances = []
    tau_temp_fits = []
    maxtaus = []
    tau_temperatures = {t: {'measured': [], 'extrapolated': []}
                        for t in measurement_temperatures}

    amp_by_temp = {q: {} for q in quads}
    example_traps = {q: {} for q in quads}
    num_examples_kept = 0

    for q in quads:
        if q not in fit_dipole_spectra:
            continue
        for dp in list(fit_dipole_spectra[q]):
            if not isinstance(dp, tuple):
                continue
            testdp = fit_dipole_spectra[q][dp]

            well_behaved = bool(testdp.get('WellBehavedTrap', False))
            if well_behaved:
                temp_entry = {'WellBehavedTrap': True}
                for key, data in testdp.items():
                    if isinstance(key, int) and data.get('GoodIntensityFit', False):
                        temp_entry[key] = {
                            'fit_coeff': data['fit_coeff'],
                            'fit_coeff_err': data['fit_coeff_err'],
                            'GoodIntensityFit': True,
                        }
                amp_by_temp[q][dp] = temp_entry

            # Minimal pipeline (dipole_new.py) only writes 'EnergyFitFailed' when
            # WellBehavedTrap AND single_orientation; a missing key means no energy
            # fit was attempted, which is equivalent to a failed fit.
            if not (well_behaved and not testdp.get('EnergyFitFailed', True)):
                continue

            maxtaus.append(np.max(testdp['energy_taus']))

            if not testdp["GoodEnergyFit"]:
                continue

            if num_examples_kept < num_example_traps:
                example_traps[q][dp] = testdp
                num_examples_kept += 1

            cs = testdp['energy_BestFitCrossSection']
            cserr = testdp['energy_BestFitCrossSectionErr']
            e = testdp['energy_BestFitEnergy']
            e_err = testdp['energy_BestFitEnergyErr']
            avg_good_temp = np.mean(testdp['energy_temperatures'])

            for t in measurement_temperatures:
                tau = np.exp(log_energy_cross_section(t, e, np.log(cs)))
                if t in testdp['energy_temperatures']:
                    tau_temperatures[t]['measured'].append(tau)
                else:
                    tau_temperatures[t]['extrapolated'].append(tau)

            energy_crossSections.append((cs, e, cserr, e_err, avg_good_temp))
            cov = testdp.get('energy_CovarianceMatrix')
            energy_covariances.append(np.array(cov, dtype=float)
                                      if cov is not None else None)
            tau_temp_fits.append([testdp['energy_temperatures'],
                                  testdp['energy_taus'],
                                  testdp['energy_tau_errs']])

    energy_crossSections_arr = np.array(energy_crossSections)
    cross_sections = energy_crossSections_arr[:, 0] if len(energy_crossSections) else np.array([])
    energies = energy_crossSections_arr[:, 1] if len(energy_crossSections) else np.array([])

    print(f"Number of good energy fits: {len(energy_crossSections)}")
    agg = {
        'energy_crossSections': energy_crossSections,
        'energy_covariances': energy_covariances,
        'tau_temp_fits': tau_temp_fits,
        'tau_temperatures': tau_temperatures,
        'cross_sections': cross_sections,
        'energies': energies,
        'maxtaus': maxtaus,
    }
    return agg, amp_by_temp, example_traps


def build_and_save_fit_catalog_caches(fit_dipole_spectra, fit_source, log_energy_cross_section,
                                      **kwargs):
    """Build the three fit-catalog caches (see build_fit_catalog_caches) from an
    already-loaded ``fit_dipole_spectra`` dict and write them alongside
    ``fit_source`` (used only to derive the cache filenames)."""
    agg, amp_by_temp, example_traps = build_fit_catalog_caches(
        fit_dipole_spectra, log_energy_cross_section, **kwargs)
    save_trap_fit_aggregate(agg, agg_cache_path(fit_source))
    save_spectra_hdf5(amp_by_temp, amp_cache_path(fit_source))
    save_spectra_hdf5(example_traps, example_cache_path(fit_source))
    return agg, amp_by_temp, example_traps
