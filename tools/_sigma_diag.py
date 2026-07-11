import numpy as np

leg = np.load('trap_tau135_sigma_pairs.npz')
mn = np.load('trap_tau135_sigma_pairs_minimal_caldet.npz')

for nm, d in [('legacy', leg), ('minimal', mn)]:
    s = d['sigma']; e = d['energy']; t = d['tau135']
    print(f"=== {nm}: n={len(s)}")
    print(f"  log10(sigma): mean {np.mean(np.log10(s)):.2f} median {np.median(np.log10(s)):.2f}")
    # correlation of log sigma with energy
    m = np.isfinite(s) & np.isfinite(e) & (s > 0)
    r = np.corrcoef(np.log10(s[m]), e[m])[0, 1]
    print(f"  corr(log10 sigma, energy) = {r:.3f}")
    # split by energy
    hi = e > np.median(e)
    print(f"  median sigma  low-E {np.median(s[~hi]):.2e}  high-E {np.median(s[hi]):.2e}")

# capture probability proxy: k_c proportional to sigma; show ratio of mean sigma
print("\nmean sigma ratio minimal/legacy:", np.mean(mn['sigma'])/np.mean(leg['sigma']))
print("median sigma ratio minimal/legacy:", np.median(mn['sigma'])/np.median(leg['sigma']))

# fraction of traps with very large sigma
for nm, d in [('legacy', leg), ('minimal', mn)]:
    s = d['sigma']
    print(f"{nm}: frac sigma>1e-15 = {np.mean(s>1e-15):.3f}, frac>1e-14 = {np.mean(s>1e-14):.3f}")
