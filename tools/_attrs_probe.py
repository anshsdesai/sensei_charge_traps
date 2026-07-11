import h5py, glob, os, numpy as np

files = {
 'minimal_clear3h_pre': 'campaign/minos_baseline_vp3_expind_pre_clear3h_shuf_minimal_caldet/ccd_traps_run0.h5',
 'legacy_clear3h_pre':  'campaign/minos_baseline_vp3_expind_pre_clear3h_shuf/ccd_traps_run0.h5',
}
keys = ['tpix_vertical','tpix_horizontal','packet_volume_um3','temperature_K',
        'clear_mode','exp_indep_charge_mode','runconditions','n_detected_traps',
        'trap_density','clear_fast_dwell_s','clear_three_hour_fast_shifts',
        'clear_total_time_s','binning']
for name, f in files.items():
    if not os.path.exists(f):
        print(name, "MISSING", f); continue
    with h5py.File(f,'r') as h:
        print("===", name)
        for k in keys:
            v = h.attrs.get(k, '<absent>')
            print(f"  {k}: {v}")
        # trap arrays present?
        ds = [d for d in h.keys()]
        if 'trap_sigmas' in h:
            s = h['trap_sigmas'][:]
            print(f"  n_traps_placed={len(s)} median_sigma={np.median(s):.3e} mean_sigma={np.mean(s):.3e}")
        if 'trap_taus' in h:
            t = h['trap_taus'][:]
            print(f"  median_tau={np.median(t):.3e}")
        if 'trap_kc' in h:
            kc = h['trap_kc'][:]
            print(f"  median_kc={np.median(kc):.3e} 1/s   mean_kc={np.mean(kc):.3e}")
