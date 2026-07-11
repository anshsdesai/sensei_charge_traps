from astropy.io import fits
import sys
sys.path.insert(0, '.')
from ccd_simulation import pixel_time, pixel_time_vertical

f = './snolab_image/proc_corr_proc_skp_sensei_2023-02-14_135K_run7_commissioning_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR10800_5_83.fits'
with fits.open(f) as hdul:
    h = hdul[0].header
    ncol = h['NCOL']; nsamp = h['NSAMP']
    delayH=h['HIERARCH DELAY_H_OVERLAP']; delayRG=h['HIERARCH DELAY_RG_WIDTH']
    delayIped=h['HIERARCH DELAY_INTEG_PED']; delaySW=h['HIERARCH DELAY_SWHIGH']
    delayIsig=h['HIERARCH DELAY_INTEG_SIG']; delayOG=h['HIERARCH DELAY_OG_LOW']
    delayDG=h['HIERARCH DELAY_DG_LOW']

tpix = (pixel_time(nsamp, delayH, delayIped, delayIsig, delaySW, delayRG, delayOG)/15e6)
tpix_v = (pixel_time_vertical(nsamp, ncol, delayH, delayIped, delayIsig, delaySW, delayRG, delayOG, delayDG)/15e6)
print(f"ncol={ncol} nsamp={nsamp}")
print(f"tpix (horizontal, 1 pixel) = {tpix:.6e} s")
print(f"tpix_vertical (1 row transfer) = {tpix_v:.6e} s")
print(f"total readout time 512 rows = {512*tpix_v:.4f} s")
