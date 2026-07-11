"""Are the unrecovered legacy traps a real correction (pedestal-biased long tail)
or wrongly rejected? Compare legacy vs new tau(T) and tau135."""
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dipole import robust_energy_fit, log_energy_cross_section

def tau135(E,ls): return float(np.exp(log_energy_cross_section(np.array([135.0]),E,ls)[0]))

# new good intensity taus
new={}
with h5py.File('fit_dipole_spectra_signed_err_4.h5','r') as h5:
    for qn,qg in h5.items():
        for dn,dg in qg.items():
            pts={}
            for tn in dg.keys():
                if not tn.startswith('temp_'): continue
                t=int(tn.split('_')[1]);tg=dg[tn]
                if bool(tg.attrs.get('GoodIntensityFit',False)):
                    pts[t]=(float(tg.attrs['fit_tau']),float(tg.attrs['fit_tau_err']))
            if len(pts)>=4: new[(qn,dn)]=pts

# legacy characterized: their stored E, tau135, and per-T good taus
legacy={}
with h5py.File('fit_dipole_spectra_err_4.h5','r') as h5:
    for qn,qg in h5.items():
        for dn,dg in qg.items():
            a=dg.attrs
            if not(bool(a.get('WellBehavedTrap',False)) and not bool(a.get('EnergyFitFailed',True)) and bool(a.get('GoodEnergyFit',False))): continue
            E=float(a['energy_BestFitEnergy']); ls=float(np.log(float(a['energy_BestFitCrossSection'])))
            pts={}
            for tn in dg.keys():
                if not tn.startswith('temp_'): continue
                t=int(tn.split('_')[1]);tg=dg[tn]
                if bool(tg.attrs.get('GoodIntensityFit',False)):
                    pts[t]=float(tg.attrs['fit_tau'])
            legacy[(qn,dn)]={'E':E,'ls':ls,'tau135':tau135(E,ls),'pts':pts}

# recompute new fit per trap
def newfit(k):
    p=new.get(k)
    if not p: return None
    T=np.array(sorted(p)); ta=np.array([p[t][0] for t in T]); er=np.array([p[t][1] for t in T])
    return robust_energy_fit(T,ta,er,sigma_int_dex=0.061,max_resid_dex=0.4,n_sigma=4.0)

# classify recovered vs unrecovered (p>0.05)
from scipy.stats import chi2 as chi2dist
recovered=set(); unrec=set()
for k in legacy:
    r=newfit(k)
    if r and 1e-5<r['energy']<=10 and chi2dist.sf(r['chi2'],r['dof'])>0.05: recovered.add(k)
    else: unrec.add(k)

# aggregate: tau135 legacy vs new for unrecovered (where new fit exists)
ll=[]; ln=[]
for k in unrec:
    r=newfit(k)
    if r and 1e-5<r['energy']<=10:
        ll.append(np.log10(legacy[k]['tau135'])); ln.append(np.log10(tau135(r['energy'],r['log_sigma'])))
ll=np.array(ll); ln=np.array(ln)
print(f'unrecovered with new fit: {len(ll)}')
print(f'  legacy log10 tau135 median {np.median(ll):+.2f} ; new median {np.median(ln):+.2f} ; median shift {np.median(ln-ll):+.2f} dex')
print(f'  legacy frac>1hr {np.mean(ll>np.log10(3600)):.3f} ; new frac>1hr {np.mean(ln>np.log10(3600)):.3f}')

# pick 6 unrecovered examples that legacy called long-lived
cands=sorted([k for k in unrec if k in new and legacy[k]['tau135']>1], key=lambda k:-legacy[k]['tau135'])[:6]
fig,axes=plt.subplots(2,3,figsize=(16,9))
kb=8.617333262e-5
for ax,k in zip(axes.ravel(),cands):
    Lp=legacy[k]['pts']; Np=new[k]
    Tl=np.array(sorted(Lp)); tl=np.array([Lp[t] for t in Tl])
    Tn=np.array(sorted(Np)); tn=np.array([Np[t][0] for t in Tn]); te=np.array([Np[t][1] for t in Tn])
    ax.plot(1/(kb*Tl),np.log10(tl),'s',color='gray',label='legacy tau(T)')
    ax.errorbar(1/(kb*Tn),np.log10(tn),yerr=te/tn/np.log(10),fmt='o',color='C0',label='new tau(T)')
    # legacy line
    Tg=np.linspace(min(Tn.min(),Tl.min()),max(Tn.max(),Tl.max()),50)
    ax.plot(1/(kb*Tg),log_energy_cross_section(Tg,legacy[k]['E'],legacy[k]['ls'])/np.log(10),'-',color='gray',alpha=0.7,label=f"legacy E={legacy[k]['E']:.2f}")
    r=newfit(k)
    if r: ax.plot(1/(kb*Tg),log_energy_cross_section(Tg,r['energy'],r['log_sigma'])/np.log(10),'-',color='C0',alpha=0.7,label=f"new E={r['energy']:.2f}")
    ax.set_xlabel('1/kT'); ax.set_ylabel('log10 tau'); ax.legend(fontsize=7); ax.set_title(str(k),fontsize=8)
plt.tight_layout(); plt.savefig('figures/unrecovered_examples.png',dpi=140); print('saved figures/unrecovered_examples.png')
