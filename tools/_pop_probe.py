import numpy as np

def summ(name, f):
    d = np.load(f)
    print("===", name, "keys:", list(d.keys()))
    for k in d.keys():
        a = d[k]
        if a.ndim == 0:
            print(f"  {k}: scalar {a}")
            continue
        print(f"  {k}: shape {a.shape} min {np.nanmin(a):.3e} max {np.nanmax(a):.3e} sum {np.nansum(a):.3e}")

# tau histograms
for nm, f in [('legacy_hist', 'tau_at_135k_hist.npz'),
              ('minimal_hist', 'tau_at_135k_hist_minimal_caldet.npz')]:
    summ(nm, f)

print("\n--- tau histogram detail (counts per bin, edges) ---")
for nm, f in [('legacy', 'tau_at_135k_hist.npz'),
              ('minimal', 'tau_at_135k_hist_minimal_caldet.npz')]:
    d = np.load(f)
    # try common key names
    keys = list(d.keys())
    print(nm, keys)

print("\n--- sigma pairs ---")
for nm, f in [('legacy', 'trap_tau135_sigma_pairs.npz'),
              ('minimal', 'trap_tau135_sigma_pairs_minimal_caldet.npz')]:
    d = np.load(f)
    print("===", nm, "keys:", list(d.keys()))
    for k in d.keys():
        a = d[k]
        try:
            print(f"  {k}: n={a.shape} median={np.nanmedian(a):.3e} "
                  f"p10={np.nanpercentile(a,10):.3e} p90={np.nanpercentile(a,90):.3e}")
        except Exception as e:
            print(f"  {k}: {a}")
