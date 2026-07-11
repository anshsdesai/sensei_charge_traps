import h5py, json, numpy as np

V_TH_135 = 2.99792458e10 * np.sqrt(3*8.617333262e-5*135.0/(0.41*0.510998950e6))  # cm/s
V_PACKET = 3.0e-12  # cm^3
NROW, NCOL = 512, 3072
NPIX = NROW * NCOL
TPIX_V = 49.09696  # s, from _timing_probe

def predict(run_h5, label):
    with h5py.File(run_h5, 'r') as h:
        sig = h['trap_sigmas'][:]
        tau = h['trap_taus'][:]
    kc = sig * V_TH_135 / V_PACKET            # 1/s per carrier
    lam_e = 1.0 / tau
    # saturation capture prob for q=1 packet over one 49 s dwell
    tot = kc + lam_e
    p_sat = (kc / tot) * (1.0 - np.exp(-TPIX_V * tot))
    n = len(sig)
    print(f"=== {label}: n_traps/quadrant={n}")
    print(f"  median kc={np.median(kc):.1f}/s  frac saturated(p>0.9)={np.mean(p_sat>0.9):.3f}"
          f"  frac sub-sat(p<0.5)={np.mean(p_sat<0.5):.3f}")
    print(f"  sum p_sat = {p_sat.sum():.1f}  (vs n={n}: efficiency {p_sat.sum()/n:.3f})")
    return p_sat, n

# charged-pixel density a trap encounters at 0h (pre-mode spurious ~ no-trap 0h SER)
RATE0 = 9.93e-5
MEAN_TR = NROW / 2.0  # avg real packets a trap sees = its row index, uniform in [0,512)

for f, lab in [('campaign/minos_baseline_vp3_expind_pre_clear3h_shuf_minimal_caldet/ccd_traps_run0.h5','minimal'),
               ('campaign/minos_baseline_vp3_expind_pre_clear3h_shuf/ccd_traps_run0.h5','legacy')]:
    p_sat, n = predict(f, lab)
    mu = MEAN_TR * RATE0                      # expected charged packets seen per trap
    # each trap removes ~1 single-e if it captures its first charged packet (then holds it)
    D = (p_sat * (1.0 - np.exp(-mu))).sum()   # captures (= single-e removed) per image
    deficit_rate = D / NPIX
    print(f"  PREDICT deficit: {D:.1f} captures/image  ->  rate {deficit_rate:.3e}\n")

print("OBSERVED clear3h pre 0h deficit:  minimal 2.60e-5,  legacy 1.17e-5")
print(f"v_th(135)={V_TH_135:.3e} cm/s")
