import h5py, numpy as np

f = 'fit_dipole_spectra_minimal_caldet_err_4.h5'
# discover the field names in a dp group
with h5py.File(f, 'r') as h:
    g0 = h['quad_0']
    first = list(g0.keys())[0]
    print('dp fields:', list(g0[first].keys()))
    d = g0[first]
    for k in d.keys():
        v = d[k]
        if isinstance(v, h5py.Dataset):
            print('  ', k, np.array(v).shape, '=', np.array(v).ravel()[:6])

    logsig_err_dec, rho, ntemps = [], [], []
    for q in h.keys():
        for name in h[q].keys():
            dp = h[q][name]
            if 'energy_CovarianceMatrix' not in dp.keys():
                continue
            pc = np.array(dp['energy_CovarianceMatrix'])
            if pc.shape != (2, 2) or not np.all(np.isfinite(pc)) or pc[0,0] <= 0 or pc[1,1] <= 0:
                continue
            logsig_err_dec.append(np.sqrt(pc[1,1])/np.log(10))   # marginal logsigma sd, decades
            rho.append(pc[0,1]/np.sqrt(pc[0,0]*pc[1,1]))
            if 'energy_temperatures' in dp.keys():
                ntemps.append(len(np.array(dp['energy_temperatures'])))

dec=np.array(logsig_err_dec); rho=np.array(rho); ntemps=np.array(ntemps)
print(f"\nn traps with energy covariance = {len(dec)}")
print(f"marginal logsigma uncertainty (decades): median={np.median(dec):.2f}  "
      f"p10={np.percentile(dec,10):.2f}  p90={np.percentile(dec,90):.2f}")
print(f"  frac sigma uncertainty >1 decade: {np.mean(dec>1):.2f}   >2 decades: {np.mean(dec>2):.2f}")
print(f"E-logsigma fit correlation |rho|: median={np.median(np.abs(rho)):.4f}  "
      f"frac>0.95: {np.mean(np.abs(rho)>0.95):.2f}  frac>0.99: {np.mean(np.abs(rho)>0.99):.2f}")
if len(ntemps):
    print(f"n temperatures per fit: median={int(np.median(ntemps))}  min={ntemps.min()}  max={ntemps.max()}")
    # sigma uncertainty vs number of temps
    for lo,hi in [(4,5),(6,8),(9,30)]:
        m=(ntemps>=lo)&(ntemps<=hi)
        if m.sum():
            print(f"   ntemps {lo}-{hi}: n={m.sum()} median sigma-uncert {np.median(dec[m]):.2f} dec, median |rho| {np.median(np.abs(rho[m])):.4f}")
