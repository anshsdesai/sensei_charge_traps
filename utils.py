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

