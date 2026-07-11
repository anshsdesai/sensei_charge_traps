import numpy as np
import sys
sys.path.insert(0, '.')
from dipole import log_energy_cross_section  # logtau(T, E, logsigma)

kb = 8.617333262e-5
T0 = 135.0

def tau_at(T, E, sigma):
    return np.exp(log_energy_cross_section(np.array([T]), E, np.log(sigma))[0])

def sigma_for(E, tau135, T=135.0):
    # invert: logtau = C - logsigma - 2log(kT) + E/kT  ->  logsigma = ... - logtau
    # use the function at logsigma=0 to get the constant part
    logtau_ref = log_energy_cross_section(np.array([T]), E, 0.0)[0]  # logsigma=0
    logsigma = logtau_ref - np.log(tau135)
    return np.exp(logsigma)

print("=== Literature anchor: defect with E and sigma~1e-15, predicted tau(135K) ===")
for E in [0.25, 0.29, 0.30, 0.32, 0.44]:
    for sig in [1e-15, 1e-14]:
        print(f"  E={E:.2f} eV, sigma={sig:.0e} -> tau(135K) = {tau_at(135.0,E,sig):.3e} s")

print("\n=== Degeneracy slope at fixed tau(135K): d log10(sigma)/dE ===")
slope = 1.0/(kb*T0)/np.log(10)
print(f"  {slope:.1f} decades per eV  (so DeltaE=0.1 eV -> {slope*0.1:.1f} decades in sigma)")

print("\n=== Our catalog: sigma at the literature energy E in [0.28,0.31] ===")
for nm, f in [('legacy','trap_tau135_sigma_pairs.npz'),
              ('minimal','trap_tau135_sigma_pairs_minimal_caldet.npz')]:
    d = np.load(f); E=d['energy']; s=d['sigma']; t=d['tau135']
    m = (E>=0.28)&(E<=0.31)
    print(f"  {nm}: n(E in 0.28-0.31)={m.sum()}  median sigma={np.median(s[m]):.2e}  "
          f"median tau135={np.median(t[m]):.2f}s")
    # what sigma would E=0.295, tau135=median predict?
    print(f"         model sigma for (E=0.295, tau135={np.median(t[m]):.2f}) = "
          f"{sigma_for(0.295, np.median(t[m])):.2e}")

print("\n=== full catalog E and sigma summary ===")
for nm, f in [('legacy','trap_tau135_sigma_pairs.npz'),
              ('minimal','trap_tau135_sigma_pairs_minimal_caldet.npz')]:
    d = np.load(f); E=d['energy']; s=d['sigma']
    print(f"  {nm}: E p10/50/90 = {np.percentile(E,10):.3f}/{np.median(E):.3f}/{np.percentile(E,90):.3f}  "
          f"log10sig p10/50/90 = {np.percentile(np.log10(s),10):.1f}/{np.median(np.log10(s)):.1f}/{np.percentile(np.log10(s),90):.1f}")
    frac_big = np.mean(s>3e-16)
    print(f"         frac sigma>3e-16 (near/above lit) = {frac_big:.2f}")
