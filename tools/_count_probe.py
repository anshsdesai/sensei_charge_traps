import numpy as np, glob, os

print("=== detected dipoles (coord lists) ===")
for f in ['dipole_coord_list.npz', 'dipole_coord_list_minimal.npz']:
    d = np.load(f, allow_pickle=True)
    tot = 0
    detail = []
    for k in d.keys():
        a = d[k]
        try:
            n = len(a)
        except TypeError:
            n = a.shape[0] if hasattr(a, 'shape') else 0
        detail.append((k, n)); tot += n
    print(f"{f}: keys={detail} total={tot}")

print("\n=== characterized (tau135 pairs) ===")
for f in ['trap_tau135_sigma_pairs.npz', 'trap_tau135_sigma_pairs_minimal_caldet.npz']:
    d = np.load(f)
    print(f"{f}: n={len(d['tau135'])}")

print("\n=== ratio characterized/detected ===")
det_leg = sum(len(np.load('dipole_coord_list.npz', allow_pickle=True)[k]) for k in np.load('dipole_coord_list.npz', allow_pickle=True).keys())
det_min = sum(len(np.load('dipole_coord_list_minimal.npz', allow_pickle=True)[k]) for k in np.load('dipole_coord_list_minimal.npz', allow_pickle=True).keys())
char_leg = len(np.load('trap_tau135_sigma_pairs.npz')['tau135'])
char_min = len(np.load('trap_tau135_sigma_pairs_minimal_caldet.npz')['tau135'])
print(f"legacy : detected {det_leg}, characterized {char_leg}, ratio {char_leg/det_leg:.3f}")
print(f"minimal: detected {det_min}, characterized {char_min}, ratio {char_min/det_min:.3f}")
