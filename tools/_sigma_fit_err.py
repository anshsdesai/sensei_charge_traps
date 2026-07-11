import h5py, numpy as np, itertools

f = 'fit_dipole_spectra_minimal_caldet_err_4.h5'
with h5py.File(f, 'r') as h:
    keys = list(h.keys())
    print('top keys (first 10):', keys[:10], 'n=', len(keys))
    # find a group and print its contents
    grp = None
    for k in keys:
        if isinstance(h[k], h5py.Group):
            grp = k; break
    if grp is not None:
        print('example group', grp, '->', list(h[grp].keys()))
        for kk in h[grp]:
            it = h[grp][kk]
            if isinstance(it, h5py.Dataset):
                print('   ', kk, it.shape, it.dtype, '=', np.array(it).ravel()[:6])
            else:
                print('   [grp]', kk, list(it.keys()))
