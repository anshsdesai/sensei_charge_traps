"""Most-correct energy criterion v2: no blanket peak cut; robust outlier rejection
(relative to each trap's own SRH line) handles spurious ramps; intrinsic dispersion
estimated post-rejection over the full temperature range."""
import numpy as np
import h5py
from scipy.stats import chi2 as chi2dist
from dipole import robust_energy_fit, log_energy_cross_section

def tau135(E,ls): return float(np.exp(log_energy_cross_section(np.array([135.0]),E,ls)[0]))

traps={}
with h5py.File('fit_dipole_spectra_signed_err_4.h5','r') as h5:
    for qn,qg in h5.items():
        for dn,dg in qg.items():
            T=[];tau=[];err=[]
            for tn in dg.keys():
                if not tn.startswith('temp_'): continue
                t=int(tn.split('_')[1]);tg=dg[tn]
                if not bool(tg.attrs.get('GoodIntensityFit',False)): continue
                T.append(float(t));tau.append(float(tg.attrs['fit_tau']));err.append(float(tg.attrs['fit_tau_err']))
            if len(T)>=4: traps[(qn,dn)]=(np.array(T),np.array(tau),np.array(err))
legacy=set()
with h5py.File('fit_dipole_spectra_err_4.h5','r') as h5:
    for qn,qg in h5.items():
        for dn,dg in qg.items():
            a=dg.attrs
            if bool(a.get('WellBehavedTrap',False)) and not bool(a.get('EnergyFitFailed',True)) and bool(a.get('GoodEnergyFit',False)):
                legacy.add((qn,dn))

# estimate sigma_int post-rejection (no peak cut), median reduced chi2 = 1
def median_rchi2(sig):
    rs=[]
    for T,ta,er in traps.values():
        r=robust_energy_fit(T,ta,er,sigma_int_dex=sig,max_resid_dex=0.4,n_sigma=4.0)
        if r is not None: rs.append(r['reduced_chi2'])
    return np.median(rs)
lo,hi=0.0,0.5
for _ in range(36):
    mid=(lo+hi)/2
    if median_rchi2(mid)>1.0: lo=mid
    else: hi=mid
sigma_int=(lo+hi)/2
print(f'sigma_int (post-rejection, full range) = {sigma_int:.3f} dex')

fits={}
for k,(T,ta,er) in traps.items():
    r=robust_energy_fit(T,ta,er,sigma_int_dex=sigma_int,max_resid_dex=0.4,n_sigma=4.0)
    if r is not None and 1e-5<r['energy']<=10: fits[k]=r

keys=set([k for k,r in fits.items() if chi2dist.sf(r['chi2'],r['dof'])>0.05])
E=np.array([fits[k]['energy'] for k in keys]); ls=np.array([fits[k]['log_sigma'] for k in keys])
t=np.array([tau135(fits[k]['energy'],fits[k]['log_sigma']) for k in keys]); lt=np.log10(t)
print(f'\ncharacterized = {len(keys)}   legacy-recovered = {len(keys&legacy)}/{len(legacy)}')
print(f'  medE={np.median(E):.3f}eV  med_log10tau135={np.median(lt):+.2f}  frac>1s={np.mean(t>1):.3f}  frac>1hr={np.mean(t>3600):.4f}  frac>1day={np.mean(t>86400):.4f}')
print(f'LEGACY: 2135, medE 0.281, med_log10tau135 +0.64, frac>1s 0.954, frac>1hr 0.024, frac>1day 0.015')

unrec=[k for k in legacy if k not in keys]
have=[k for k in unrec if k in fits]
print(f'\nunrecovered legacy: {len(unrec)} ; of these have-a-fit-but-fail-pvalue: {len(have)}; no-valid-fit: {len(unrec)-len(have)}')
if have:
    rchi=np.array([fits[k]['reduced_chi2'] for k in have])
    nused=np.array([fits[k]['n_used'] for k in have])
    print(f'  failing ones: median reduced chi2 {np.median(rchi):.1f}, median n_used {np.median(nused):.0f}')
# legacy quality of unrecovered
leg_rchi=[]
with h5py.File('fit_dipole_spectra_err_4.h5','r') as h5:
    for k in unrec:
        a=h5[k[0]][k[1]].attrs
        if 'energy_reduced_chi2' in a: leg_rchi.append(float(a['energy_reduced_chi2']))
print(f'  legacy reduced chi2 of unrecovered: median {np.median(leg_rchi):.2f}')
