"""Why do ~885 legacy traps fail, and is a full-range dispersion the right choice?"""
import numpy as np
import h5py
from scipy.stats import chi2 as chi2dist
from dipole import robust_energy_fit, estimate_intrinsic_dispersion, log_energy_cross_section

def tau135(E,ls): return float(np.exp(log_energy_cross_section(np.array([135.0]),E,ls)[0]))

traps={}
with h5py.File('fit_dipole_spectra_signed_err_4.h5','r') as h5:
    for qn,qg in h5.items():
        for dn,dg in qg.items():
            T=[];tau=[];err=[];smax=[]
            for tn in dg.keys():
                if not tn.startswith('temp_'): continue
                t=int(tn.split('_')[1]);tg=dg[tn]
                if not bool(tg.attrs.get('GoodIntensityFit',False)): continue
                T.append(float(t));tau.append(float(tg.attrs['fit_tau']))
                err.append(float(tg.attrs['fit_tau_err']));smax.append(float(np.max(tg['seconds'][()])))
            if len(T)>=4: traps[(qn,dn)]=(np.array(T),np.array(tau),np.array(err),np.array(smax))
legacy=set()
with h5py.File('fit_dipole_spectra_err_4.h5','r') as h5:
    for qn,qg in h5.items():
        for dn,dg in qg.items():
            a=dg.attrs
            if bool(a.get('WellBehavedTrap',False)) and not bool(a.get('EnergyFitFailed',True)) and bool(a.get('GoodEnergyFit',False)):
                legacy.add((qn,dn))

# dispersion two ways
s_low,_=estimate_intrinsic_dispersion([(T,ta,er) for (T,ta,er,sm) in traps.values()], t_max=165)
s_all,_=estimate_intrinsic_dispersion([(T,ta,er) for (T,ta,er,sm) in traps.values()], t_max=999)
print(f'sigma_int low-T(<=165) = {s_low:.3f} dex ; full-range = {s_all:.3f} dex')

def run(sig):
    keys=set(); E=[]; lt=[]
    fits={}
    for k,(T,tau,err,smax) in traps.items():
        r=robust_energy_fit(T,tau,err,seconds_max=smax,sigma_int_dex=sig)
        if r is None or not (1e-5<r['energy']<=10): continue
        fits[k]=r
        if chi2dist.sf(r['chi2'],r['dof'])>0.05:
            keys.add(k); E.append(r['energy']); lt.append(np.log10(tau135(r['energy'],r['log_sigma'])))
    return keys,np.array(E),np.array(lt),fits

for label,sig in [('low-T',s_low),('full-range',s_all)]:
    keys,E,lt,fits=run(sig)
    t=10**lt
    print(f'\n[{label} sigma={sig:.3f}] characterized={len(keys)} legacy-rec={len(keys&legacy)}/{len(legacy)} '
          f'medE={np.median(E):.3f} med_log10tau135={np.median(lt):+.2f} frac>1hr={np.mean(t>3600):.3f} frac>1day={np.mean(t>86400):.4f}')

# diagnose still-unrecovered legacy under full-range
keys,E,lt,fits=run(s_all)
unrec=[k for k in legacy if k not in keys]
print(f'\nunrecovered legacy under full-range: {len(unrec)}')
# why: among those that HAVE a fit, look at n_used, reduced chi2, and whether high-T points dominate
nfit=0; rchi=[]; nhigh_frac=[]; tau135_unrec=[]
for k in unrec:
    if k in fits:
        nfit+=1; r=fits[k]; rchi.append(r['reduced_chi2'])
        T=traps[k][0]; nhigh_frac.append(np.mean(T>165))
        tau135_unrec.append(np.log10(tau135(r['energy'],r['log_sigma'])))
rchi=np.array(rchi)
print(f'  have a fit but fail p>0.05: {nfit}; median reduced chi2 {np.median(rchi):.1f} (so genuinely poor SRH fit)')
print(f'  their median high-T point fraction: {np.median(nhigh_frac):.2f}')
print(f'  their tau135 (log10) median {np.median(tau135_unrec):+.2f}  frac>1hr {np.mean(np.array(tau135_unrec)>np.log10(3600)):.3f}')
# Were these marginal in legacy? check legacy energy reduced chi2
leg_rchi=[]
with h5py.File('fit_dipole_spectra_err_4.h5','r') as h5:
    for k in unrec:
        qn,dn=k; a=h5[qn][dn].attrs
        if 'energy_reduced_chi2' in a: leg_rchi.append(float(a['energy_reduced_chi2']))
leg_rchi=np.array(leg_rchi)
print(f'  legacy energy reduced chi2 of these: median {np.median(leg_rchi):.2f}, frac>2 {np.mean(leg_rchi>2):.2f} (legacy passed them at <5)')
