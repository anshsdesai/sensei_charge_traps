"""Apply the most-correct energy-fit criterion to the signed catalog and report before/after.

Stage-2 (energy) re-selection only; the per-temperature intensity fits are read
as-is from the signed h5. Estimates the intrinsic dispersion from the data, then
fits each well-behaved trap with robust_energy_fit and evaluates GoodEnergyFit at
several thresholds.
"""
import numpy as np
import h5py
from scipy.stats import chi2 as chi2dist
from dipole import robust_energy_fit, estimate_intrinsic_dispersion, log_energy_cross_section

def tau135(E, logsig):
    return float(np.exp(log_energy_cross_section(np.array([135.0]), E, logsig)[0]))

def load_signed():
    traps = {}
    with h5py.File('fit_dipole_spectra_signed_err_4.h5','r') as h5:
        for qn,qg in h5.items():
            for dn,dg in qg.items():
                T=[]; tau=[]; err=[]; smax=[]
                for tn in dg.keys():
                    if not tn.startswith('temp_'): continue
                    t=int(tn.split('_')[1]); tg=dg[tn]
                    if not bool(tg.attrs.get('GoodIntensityFit',False)): continue
                    T.append(float(t)); tau.append(float(tg.attrs['fit_tau']))
                    err.append(float(tg.attrs['fit_tau_err'])); smax.append(float(np.max(tg['seconds'][()])))
                if len(T)>=4:
                    traps[(qn,dn)] = (np.array(T),np.array(tau),np.array(err),np.array(smax))
    return traps

def load_legacy_characterized():
    keys=set()
    with h5py.File('fit_dipole_spectra_err_4.h5','r') as h5:
        for qn,qg in h5.items():
            for dn,dg in qg.items():
                a=dg.attrs
                if bool(a.get('WellBehavedTrap',False)) and not bool(a.get('EnergyFitFailed',True)) and bool(a.get('GoodEnergyFit',False)):
                    keys.add((qn,dn))
    return keys

if __name__=='__main__':
    traps = load_signed()
    legacy = load_legacy_characterized()
    print(f'well-behaved (>=4 good intensity fits): {len(traps)}')
    print(f'legacy characterized: {len(legacy)}')

    # 1) Measure intrinsic dispersion from low-T points (model-clean regime).
    sigma_int, n_lowT = estimate_intrinsic_dispersion(
        [(T,tau,err) for (T,tau,err,smax) in traps.values()])
    print(f'\nintrinsic dispersion measured from {n_lowT} traps low-T (<=165K): '
          f'sigma_int = {sigma_int:.3f} dex')

    # 2) Robust energy fit per trap with peak-in-window + outlier rejection.
    fits={}
    for key,(T,tau,err,smax) in traps.items():
        r = robust_energy_fit(T,tau,err,seconds_max=smax,sigma_int_dex=sigma_int)
        if r is not None:
            fits[key]=r

    # 3) Evaluate GoodEnergyFit at several thresholds.
    print('\n criterion                         | characterized | legacy-recovered | medE  | med_log10tau135 | frac>1hr')
    def report(name, good_fn):
        E=[]; lt=[]; keys=set()
        for key,r in fits.items():
            if not (1e-5 < r['energy'] <= 10): continue
            if good_fn(r):
                keys.add(key); E.append(r['energy']); lt.append(np.log10(tau135(r['energy'],r['log_sigma'])))
        E=np.array(E); lt=np.array(lt)
        t=10**lt
        print(f' {name:32s} | {len(keys):13d} | {len(keys&legacy):5d}/{len(legacy):<10d} | '
              f'{np.median(E):.3f} | {np.median(lt):+.2f}            | {np.mean(t>3600):.3f}')
        return keys
    k_p05  = report('p-value > 0.05',           lambda r: chi2dist.sf(r['chi2'],r['dof'])>0.05)
    k_p01  = report('p-value > 0.01',           lambda r: chi2dist.sf(r['chi2'],r['dof'])>0.01)
    k_rc2  = report('reduced chi2 < 2',          lambda r: r['reduced_chi2']<2)
    k_rc3  = report('reduced chi2 < 3',          lambda r: r['reduced_chi2']<3)

    print(f'\nlegacy comparison: legacy had 2135 characterized, medE 0.281 eV, med log10 tau135 +0.64, frac>1hr 0.024')
    # how many legacy traps are recovered vs genuinely fail even the loosest
    loose = k_p01 | k_rc3
    print(f'legacy traps NOT recovered by any criterion: {len(legacy - loose)}')
