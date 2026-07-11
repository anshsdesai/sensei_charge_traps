"""Compare legacy vs new (scenario D) characterized populations at the level that
feeds the science: tau_135 distribution and E distribution."""
import numpy as np
import h5py
from scipy.optimize import curve_fit
from dipole import log_energy_cross_section, INTENSITY_SHAPE_PEAK_X

def tau135(E, logsig):
    return float(np.exp(log_energy_cross_section(np.array([135.0]), E, logsig)[0]))

# Legacy characterized E/sigma straight from stored attrs
legacy_E=[]; legacy_t=[]
with h5py.File('fit_dipole_spectra_err_4.h5','r') as h5:
    for qn,qg in h5.items():
        for dn,dg in qg.items():
            a=dg.attrs
            if bool(a.get('WellBehavedTrap',False)) and not bool(a.get('EnergyFitFailed',True)) and bool(a.get('GoodEnergyFit',False)):
                E=float(a['energy_BestFitEnergy']); ls=float(np.log(float(a['energy_BestFitCrossSection'])))
                legacy_E.append(E); legacy_t.append(tau135(E,ls))

# New scenario D: peak-cut + 0.15dex dispersion floor energy fit
new_E=[]; new_t=[]
with h5py.File('fit_dipole_spectra_signed_err_4.h5','r') as h5:
    for qn,qg in h5.items():
        for dn,dg in qg.items():
            pts=[]
            for tn in dg.keys():
                if not tn.startswith('temp_'): continue
                t=int(tn.split('_')[1]); tg=dg[tn]
                if not bool(tg.attrs.get('GoodIntensityFit',False)): continue
                tau=float(tg.attrs['fit_tau']); err=float(tg.attrs['fit_tau_err'])
                smax=float(np.max(tg['seconds'][()]))
                if tau*INTENSITY_SHAPE_PEAK_X <= smax:  # peak-resolved
                    pts.append((t,tau,err))
            if len(pts)<4: continue
            temps=np.array([p[0] for p in pts],float); taus=np.array([p[1] for p in pts]); errs=np.array([p[2] for p in pts])
            logtau=np.log(taus); logerr=np.sqrt((errs/taus)**2+(0.15*np.log(10))**2)
            try:
                popt,_=curve_fit(log_energy_cross_section,temps,logtau,sigma=logerr,bounds=([0,-100],[2,-1]))
            except Exception: continue
            resid=logtau-log_energy_cross_section(temps,*popt)
            rchi2=np.sum((resid/logerr)**2)/max(len(taus)-2,1)
            if rchi2<5 and 1e-5<popt[0]<=10:
                new_E.append(popt[0]); new_t.append(tau135(popt[0],popt[1]))

for name,E,t in [('LEGACY',legacy_E,legacy_t),('NEW(D)',new_E,new_t)]:
    E=np.array(E); t=np.array(t); lt=np.log10(t)
    print(f"\n{name}: N={len(E)}")
    print(f"  E [eV]   median {np.median(E):.3f}  16/84% [{np.percentile(E,16):.3f},{np.percentile(E,84):.3f}]  frac>0.4eV {np.mean(E>0.4):.3f}")
    print(f"  log10 tau135  median {np.median(lt):+.2f}  16/84% [{np.percentile(lt,16):+.2f},{np.percentile(lt,84):+.2f}]")
    print(f"  tau135 frac >1s {np.mean(t>1):.3f}  >1hr {np.mean(t>3600):.3f}  >1day {np.mean(t>86400):.3f}  >1yr {np.mean(t>3.15e7):.3f}")
