"""Decompose the 789-vs-1147 attrition and test fixes, on the existing fit h5.

For every well-behaved trap, recompute the Arrhenius fit under scenarios:
  A as-is (reproduce GoodEnergyFit, reduced_chi2 < 5)
  B + peak-resolved cut: drop good points whose dipole peak t*=0.297*tau lies
    beyond the sampled window at that temperature (unconstrained ramps)
  C B + iterative outlier rejection (drop |resid| > 0.3 dex, refit, need >=4 left)
  D B + intrinsic log-tau dispersion floor of 0.15 dex added in quadrature
Report characterized counts, legacy recovery, and E/sigma medians.
"""
import numpy as np
import h5py
from scipy.optimize import curve_fit
from scipy.stats import chi2
from dipole import log_energy_cross_section, INTENSITY_SHAPE_PEAK_X

PEAK_FACTOR = INTENSITY_SHAPE_PEAK_X  # t_peak = tau * ln(8)/7

def load():
    traps = {}
    with h5py.File('fit_dipole_spectra_signed_err_4.h5','r') as h5:
        for qn, qg in h5.items():
            for dn, dg in qg.items():
                pts = []
                for tn in dg.keys():
                    if not tn.startswith('temp_'): continue
                    t = int(tn.split('_')[1]); tg = dg[tn]
                    if not bool(tg.attrs.get('GoodIntensityFit', False)): continue
                    tau = float(tg.attrs['fit_tau']); err = float(tg.attrs['fit_tau_err'])
                    smax = float(np.max(tg['seconds'][()]))
                    pts.append((t, tau, err, smax))
                if len(pts) >= 4:
                    traps[(qn,dn)] = pts
    return traps

def arrhenius(temps, taus, errs):
    logtau = np.log(taus); logerr = errs/taus
    try:
        popt,_ = curve_fit(log_energy_cross_section, temps.astype(float), logtau,
                           sigma=logerr, bounds=([0,-100],[2,-1]))
    except Exception:
        return None
    resid = logtau - log_energy_cross_section(temps.astype(float), *popt)
    rchi2 = np.sum((resid/logerr)**2)/max(len(taus)-2,1)
    return popt, rchi2, resid

def good_energy(popt, rchi2, thresh=5.0):
    if popt is None: return False
    if rchi2 >= thresh: return False
    if popt[0] <= 1e-5 or popt[0] > 10: return False
    return True

def scenario(traps, mode):
    ch许=0; res={'char':0,'E':[],'logsig':[]}
    char_keys=set()
    for key, pts in traps.items():
        temps=np.array([p[0] for p in pts]); taus=np.array([p[1] for p in pts])
        errs=np.array([p[2] for p in pts]); smax=np.array([p[3] for p in pts])
        if mode in ('B','C','D'):
            keep = taus*PEAK_FACTOR <= smax
            if keep.sum() < 4: continue
            temps,taus,errs,smax = temps[keep],taus[keep],errs[keep],smax[keep]
        fit = arrhenius(temps,taus,errs)
        if fit is None: continue
        popt,rchi2,resid = fit
        if mode=='C':
            for _ in range(3):
                keep = np.abs(resid)/np.log(10) <= 0.3
                if keep.sum()<4 or keep.all(): break
                temps,taus,errs = temps[keep],taus[keep],errs[keep]
                fit=arrhenius(temps,taus,errs)
                if fit is None: break
                popt,rchi2,resid=fit
            if fit is None: continue
        if mode=='D':
            logtau=np.log(taus); logerr=np.sqrt((errs/taus)**2 + (0.15*np.log(10))**2)
            try:
                popt,_=curve_fit(log_energy_cross_section,temps.astype(float),logtau,
                                 sigma=logerr,bounds=([0,-100],[2,-1]))
            except Exception: continue
            resid=logtau-log_energy_cross_section(temps.astype(float),*popt)
            rchi2=np.sum((resid/logerr)**2)/max(len(taus)-2,1)
        if good_energy(popt,rchi2):
            res['char']+=1; res['E'].append(popt[0]); res['logsig'].append(popt[1])
            char_keys.add(key)
    res['keys']=char_keys
    return res

if __name__=='__main__':
    traps=load()
    print(f'well-behaved traps (>=4 good intensity fits): {len(traps)}')
    legacy=set()
    with h5py.File('fit_dipole_spectra_err_4.h5','r') as h5:
        for qn,qg in h5.items():
            for dn,dg in qg.items():
                a=dg.attrs
                if bool(a.get('WellBehavedTrap',False)) and not bool(a.get('EnergyFitFailed',True)) and bool(a.get('GoodEnergyFit',False)):
                    legacy.add((qn,dn))
    for mode in ['A','B','C','D']:
        r=scenario(traps,mode)
        rec=len(r['keys']&legacy)
        E=np.array(r['E']); ls=np.array(r['logsig'])
        print(f"[{mode}] characterized={r['char']:5d}  legacy-recovered={rec:5d}/{len(legacy)}  "
              f"medE={np.median(E):.3f}eV med_lnsig={np.median(ls):+.2f}")
