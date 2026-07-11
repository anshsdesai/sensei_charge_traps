"""Instrumented trial: tally the fate of every reservoir emission, tau-resolved.

Replaces the closed-form (1-e^{-(t+T_read)/tau}) with the measured kernel response.
Config = minos/upper/clearseq/pre (vp3), read from an existing campaign HDF5 so the
trap population/parameters match exactly. Counters per tau-bin per exposure:
  E_int          integration-phase emissions (charge_trap_interaction)
  E_read_counted readout emissions at shift t<tr  (land in a still-to-be-read packet)
  E_read_lost    readout emissions at shift t>=tr (copied back, lost to next clear)
  selfrecap      emit then immediately recapture same step (net zero)
  recap_other    capture of trap-origin charge emitted earlier (cancels a prior emission)
  sink           capture of injected (non-trap-origin) charge (real single-e sink)
"""
import os, glob
os.environ["MPLBACKEND"] = "Agg"
import numpy as np, h5py
from numba import njit
from astropy.io import fits
import ccd_simulation as cs
from ccd_simulation import CCD, pixel_time, pixel_time_vertical

# ---- timing (bin1) from the reference FITS ----
fn = sorted(glob.glob("snolab_image/*.fits"))[0]
_h1 = fits.getheader(fn, 1)
h = _h1 if 'HIERARCH DELAY_H_OVERLAP' in _h1 else fits.getheader(fn, 0)
g = lambda k: h['HIERARCH ' + k]
nsamp = int(h.get('NSAMP', 300)); ncol = 3072
dH, dRG, dIp, dSW, dIs, dOG, dDG = (g('DELAY_H_OVERLAP'), g('DELAY_RG_WIDTH'),
    g('DELAY_INTEG_PED'), g('DELAY_SWHIGH'), g('DELAY_INTEG_SIG'), g('DELAY_OG_LOW'), g('DELAY_DG_LOW'))
tpix = pixel_time(nsamp, dH, dIp, dIs, dSW, dRG, dOG) / 15e6
tpv = pixel_time_vertical(nsamp, ncol, dH, dIp, dIs, dSW, dRG, dOG, dDG) / 15e6

# ---- config from an existing upper HDF5 ----
ref = "campaign/minos_upper_vp3_expind_pre_clearseq_shuf_minimal_caldet/ccd_traps_run0.h5"
with h5py.File(ref) as f:
    n_det = int(f.attrs['n_detected_traps']); tds = float(f.attrs['trap_density_scale'])
    pv = float(f.attrs['packet_volume_um3']); pct = float(f.attrs['phase_capture_ticks'])
    tauhist = f.attrs['tauhistfile']; pairsf = f.attrs['pairsfile']
    tauhist = tauhist.decode() if isinstance(tauhist, bytes) else tauhist
    pairsf = pairsf.decode() if isinstance(pairsf, bytes) else pairsf
td = np.load(tauhist); tau_weights, tau_edges = td['hist'], td['bin_edges']
pd = np.load(pairsf); pair_tau135, pair_sigma = pd['tau135'], pd['sigma']

# ---- tau bins (decadal) ----
TAU_EDGES = np.logspace(-3, 12, 16)   # 15 bins, 1/decade
NBIN = len(TAU_EDGES) - 1


@njit(cache=True)
def readout_instr(image, image_org, exp_acc, tpix_vertical, trap_rows, trap_cols,
                  emit_probs, cap_alpha, trapped, tau_bin,
                  c_rc, c_rl, c_src, c_srl, c_ro, c_sk):
    rows, cols = image.shape
    n = len(emit_probs)
    pim = np.zeros((2 * rows, cols)); por = np.zeros((2 * rows, cols))
    for r in range(rows):
        for c in range(cols):
            pim[r + rows, c] = image[r, c]; por[r + rows, c] = image_org[r, c]
    out = np.zeros(rows * cols); k = 0
    for t in range(rows):
        rr = 2 * rows - 1 - t
        for c in range(cols - 1, -1, -1):
            out[k] = pim[rr, c]; k += 1
        for i in range(n):
            tr = trap_rows[i]; tc = trap_cols[i]; b = tau_bin[i]
            cpr = tr + rows - 1 - t
            just_c = False; just_l = False
            if trapped[i] > 0.0:
                if np.random.random() < emit_probs[i]:
                    pim[cpr, tc] += 1.0; por[cpr, tc] += 1.0; trapped[i] = 0.0
                    if t < tr:
                        c_rc[b] += 1; just_c = True
                    else:
                        c_rl[b] += 1; just_l = True
            q = pim[cpr, tc]
            if trapped[i] <= 0.0 and q >= 1.0:
                if np.random.random() < 1.0 - np.exp(-q * cap_alpha[i]):
                    if por[cpr, tc] > 0.0:
                        por[cpr, tc] -= 1.0
                        if just_c:
                            c_src[b] += 1
                        elif just_l:
                            c_srl[b] += 1
                        else:
                            c_ro[b] += 1
                    else:
                        c_sk[b] += 1
                    pim[cpr, tc] -= 1.0; trapped[i] = 1.0
    for sr in range(rows):
        av = (rows - 1 - sr) * tpix_vertical
        if av > 0:
            for sc in range(cols):
                exp_acc[sr, sc] += av
    for r in range(rows):
        for c in range(cols):
            image[r, c] = pim[r, c]; image_org[r, c] = por[r, c]
    return out


def attach(ccd):
    ccd._tau_bin = np.clip(np.digitize(ccd.trap_taus, TAU_EDGES) - 1, 0, NBIN - 1).astype(np.int64)
    ccd._int_org = np.zeros_like(ccd.ccd_state)
    ccd._log = []   # per-image dict

    def cti(current_image, dt):
        ccd._int_org[:] = 0.0
        if dt <= 0:
            ccd._cur = dict(E_int=np.zeros(NBIN))
            return current_image
        occ = ccd.trapped_charge_1d > 0
        p = 1.0 - np.exp(-dt / ccd.trap_taus)
        rel = occ & (np.random.random(len(ccd.trap_taus)) < p)
        rr = ccd.trap_indices[0][rel]; cc = ccd.trap_indices[1][rel]
        current_image[rr, cc] += 1.0
        ccd._int_org[rr, cc] += 1.0
        ccd.trapped_charge_1d[rel] = 0.0
        e_int = np.bincount(ccd._tau_bin[rel], minlength=NBIN).astype(float)
        ccd._cur = dict(E_int=e_int)
        return current_image

    def readout(tpix_vertical=None):
        if tpix_vertical is None:
            tpix_vertical = ccd.tpix_vertical
        image = ccd.ccd_state.copy()
        org = ccd._int_org.copy()
        rows, cols = image.shape
        ep = ccd.readout_emit_probs if tpix_vertical == ccd.tpix_vertical \
            else 1.0 - np.exp(-tpix_vertical / ccd.trap_taus)
        rc = np.zeros(NBIN, np.int64); rl = np.zeros(NBIN, np.int64)
        src = np.zeros(NBIN, np.int64); srl = np.zeros(NBIN, np.int64)
        ro = np.zeros(NBIN, np.int64); sk = np.zeros(NBIN, np.int64)
        flat = readout_instr(image, org, ccd.exposure_accumulator, tpix_vertical,
                             ccd.trap_indices[0], ccd.trap_indices[1], ep, ccd.trap_capture_alpha,
                             ccd.trapped_charge_1d, ccd._tau_bin, rc, rl, src, srl, ro, sk)
        ccd.ccd_state[:] = image; ccd._int_org[:] = org
        d = ccd._cur
        d.update(E_read_counted=rc.astype(float), E_read_lost=rl.astype(float),
                 sr_counted=src.astype(float), sr_lost=srl.astype(float),
                 recap_other=ro.astype(float), sink=sk.astype(float),
                 occ_before=int((ccd.trapped_charge_1d > 0).sum()))
        ccd._log.append(d)
        return np.flipud(np.fliplr(flat.reshape(rows, cols)))

    ccd.charge_trap_interaction = cti
    ccd.simulate_readout = readout


# ---- build CCD (minos, sequencer, pre, vp3) ----
BIN = float(os.environ.get("BIN", "1"))
tpv_used = tpv / BIN
print(f"tpix_vertical={tpv_used:.3f}s (BIN={BIN:g})  config: n_det={n_det} tds={tds:.3f} pv={pv} pct={pct}")
ccd = CCD(tpix, tpv_used, tau_weights, tau_edges, pair_tau135, pair_sigma,
          runconditions='minos', trap_density_scale=tds, packet_volume_um3=pv,
          phase_capture_ticks=pct, exp_indep_charge_mode='pre_readout',
          clear_mode='sequencer', binning=1.0, n_detected_traps=n_det)
print(f"actual num_traps = {len(ccd.trap_indices[0])}")
attach(ccd)

# ---- run schedule ----
N_CYCLES = int(os.environ.get("NCYC", "60"))
rng = np.random.default_rng(0)
seq = [0, 4, 6, 10, 20] * N_CYCLES
rng.shuffle(seq)


def net_1e(bt, bn):
    """unmasked Δ1e and halo+bleed-masked Δ1e from the per-image bitmasks."""
    t1 = (bt & 1) != 0; n1 = (bn & 1) != 0
    keep_t = t1 & ((bt & 2) == 0) & ((bt & 4) == 0)
    keep_n = n1 & ((bn & 2) == 0) & ((bn & 4) == 0)
    return int(t1.sum()) - int(n1.sum()), int(keep_t.sum()) - int(keep_n.sum())


for i, e in enumerate(seq):
    ccd.take_fake_image(e)
    d = ccd._log[-1]; d['exp'] = e
    du, dm = net_1e(ccd.trap_bitmasks[-1], ccd.notrap_bitmasks[-1])
    d['d1e_unmasked'] = du; d['d1e_hbmasked'] = dm
    ccd.trap_bitmasks[-1] = None; ccd.notrap_bitmasks[-1] = None  # free memory
    if (i + 1) % 50 == 0:
        print(f"  image {i+1}/{len(seq)}  occ={d['occ_before']}  d1e_unmasked={du}  d1e_hb={dm}")

# ---- aggregate over steady-state images (skip first 40%) ----
warm = int(0.4 * len(ccd._log))
keys = ['E_int', 'E_read_counted', 'E_read_lost', 'sr_counted', 'sr_lost', 'recap_other', 'sink']
ss = ccd._log[warm:]
print(f"\nSteady-state: images {warm}..{len(ccd._log)}  mean occ={np.mean([d['occ_before'] for d in ss]):.0f} "
      f"of {len(ccd.trap_indices[0])} traps ({100*np.mean([d['occ_before'] for d in ss])/len(ccd.trap_indices[0]):.0f}%)")
np.savez("_instr_counters.npz",
         tau_resolved={k: np.array([d[k] for d in ss]).sum(0) for k in keys},
         net_escaped=np.array([(d['E_int'] + d['E_read_counted'] - d['sr_counted'] - d['recap_other']) for d in ss]).sum(0),
         ntrap=len(ccd.trap_indices[0]), tau_edges=TAU_EDGES, keys=keys, allow_pickle=True)

print(f"\nGROSS FLOWS per image (mean), and the fate FUNNEL:")
print(f"{'exp':>4}{'nimg':>5}{'E_int':>8}{'Eread':>8}{'selfrec':>8}{'escape':>8}{'recapO':>8}"
      f"{'NETcnt':>8}{'sink':>8}{'d1e_un':>8}{'d1e_hb':>8}")
for e in [0, 4, 6, 10, 20]:
    sub = [d for d in ss if d['exp'] == e]
    if not sub:
        continue
    m = lambda k: np.mean([d[k].sum() if hasattr(d[k], 'sum') else d[k] for d in sub])
    Eread = m('E_read_counted') + m('E_read_lost')
    selfrec = m('sr_counted') + m('sr_lost')
    escape = Eread - selfrec
    netcnt = m('E_int') + m('E_read_counted') - m('sr_counted') - m('recap_other')
    print(f"{e:>4}{len(sub):>5}{m('E_int'):>8.0f}{Eread:>8.0f}{selfrec:>8.0f}{escape:>8.0f}"
          f"{m('recap_other'):>8.0f}{netcnt:>8.0f}{m('sink'):>8.0f}"
          f"{np.mean([d['d1e_unmasked'] for d in sub]):>8.0f}{np.mean([d['d1e_hbmasked'] for d in sub]):>8.0f}")
print("\nLEGEND: Eread=readout emissions; selfrec=immediately self-recaptured (same step);")
print("escape=Eread-selfrec; NETcnt=E_int+Eread_counted-sr_counted-recapO (trap e- reaching readout,")
print("pre merge/mask); sink=captures of INJECTED 1e (real sink); d1e_un/d1e_hb=ACTUAL net Δ single-e")
print("count (unmasked / halo+bleed-masked). Survival = d1e / gross emission.")
