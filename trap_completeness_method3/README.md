# Trap Completeness Method 3 Workspace

This directory is the execution workspace for the Method 3 completeness study described in
[`../trap_completeness_method.md`](../trap_completeness_method.md). The goal is to build a
transparent, inspectable detection model for pocket-pumped traps:

1. model the dipole intensity curve from the paper's analytic equation,
2. calibrate measurement noise from the raw `proc/` FITS files,
3. calibrate the amplitude prior from characterized traps,
4. compute per-temperature detection probability by injection-recovery,
5. combine temperatures into a characterization probability for `tau_135` and `E`.

The directory is designed for token-efficient work with Codex or other agents. Each stage has a
scoped packet in `agents/`; the packet tells the agent what to read, what to run, what to produce,
and where to write a concise result summary.

## Directory Layout

```text
trap_completeness_method3/
  README.md                 # global roadmap and status
  AGENT_PROTOCOL.md         # rules for token-efficient, non-black-box work
  agents/                   # one scoped work packet per stage
  notebooks/                # exploratory notebooks; notebooks call src/ once logic stabilizes
  src/                      # small reusable functions extracted from notebooks
  cache/                    # generated artifacts; contents are ignored by git
```

## Canonical Inputs

- Raw pocket-pumping images: `../proc/*.fits`
- Primary characterized-trap file: `../fit_dipole_spectra_err_4.h5`
- Sensitivity comparison file: `../fit_dipole_spectra_err_3.h5`
- Dipole coordinates: `../dipole_coord_list.npz`
- Core analysis code: `../dipole.py`, `../utils.py`
- Conceptual parent document: `../trap_completeness_method.md`

## Canonical Runtime

Use the WSL conda environment `sensei_charge_traps`, not the Windows-side
`sensei_charge_traps_new` environment. Preferred command:

```bash
/home/ansh/miniforge3/bin/conda run -n sensei_charge_traps python <script>.py
```

The environment's Python executable is:

```bash
/home/ansh/miniforge3/envs/sensei_charge_traps/bin/python
```

## Canonical Assumptions

- Paper-consistent characterization threshold: `n_good = 4`
- Sensitivity comparison threshold: `n_good = 3`
- Target operating temperature: `T0 = 135.0 K`
- Pump count: `N_pumps = 3000`
- Local-noise patch convention: call `histogram_around_point(size=35)`, whose source slice is
  effectively `34 x 34` for interior points (`row-half:row+half` with `half=17`)
- Heavy outputs live under `cache/` or existing ignored result directories and are referenced from
  stage packets rather than pasted into chat.

## Dependency Chain

| Stage | Packet | Depends on | Gate |
|---|---|---|---|
| 00 | `agents/00_data_inventory.md` | none | FITS/HDF5/interpreter inventory complete |
| 01 | `agents/01_hdf5_records_audit.md` | 00 | characterized trap records summarized |
| 02 | `agents/02_fits_noise_parity.md` | 00, 01 | FITS-derived local noise matches stored `intensity_err`; `image_sigma` field semantics are identified |
| 03 | `agents/03_trap_free_noise_map.md` | 02 | unbiased `p_sigma(sigma | T)` artifact created |
| 04 | `agents/04_intensity_error_scaling.md` | 01, 02, 03 | injection noise scaling chosen and justified |
| 05 | `agents/05_amplitude_prior.md` | 01 | amplitude prior and correlations summarized |
| 06 | `agents/06_single_curve_recovery.md` | 04, 05 | one fake-trap walkthrough is fully inspectable |
| 07 | `agents/07_single_temperature_pdet.md` | 06 | one-temperature `p_det` grid plus cutflow looks sane |
| 08 | `agents/08_full_pdet_grid.md` | 03, 04, 05, 07 | full `p_det(tau, A, T)` grid has metadata and cutflows |
| 09 | `agents/09_characterization_probability.md` | 08 | `P(characterized | tau_135, E)` map created |
| 10 | `agents/10_validation_sensitivity.md` | 09 | validation and sensitivity summaries complete |

Do not proceed past Stage 02 until the noise-parity gate is satisfied. The gate is that
FITS-derived local patch noise must reproduce HDF5 `intensity_err`; HDF5 `image_sigma` must be
understood separately as the whole-image threshold-like field. This protects the entire Method 3
selection function from a silent mismatch between FITS processing and stored HDF5 quantities.

## Current Status

| Stage | Status | Result Summary |
|---|---|---|
| 00 Data inventory | Completed | 1004 FITS scanned; 23/23 expected temperatures present; extra 230 K scan files noted; inventory cached |
| 01 HDF5 records audit | Completed | 2135 characterized traps for `n_good = 4` and 2517 for `n_good = 3`; all 23 temperature `seconds` grids are internally consistent; `fit_*` attrs are missing only for the 896 `IntensityFitFailed` temperature records; Stage 02 confirmed sampled `200 K` records have a 29-point grid |
| 02 FITS noise parity | Completed | Local patch sigma reproduces HDF5 `intensity_err` exactly over 1428 sampled rows; HDF5 `image_sigma` is a whole-image per-temperature/quadrant field, not local noise; Stage 03 may proceed using the local patch definition validated against `intensity_err` |
| 03 Trap-free noise map | Completed | Built `cache/03_noise_map_v1.h5` from 577200 trap-free local patches over 481 CCD2 dwell FITS; 92 temperature/quadrant groups; min group count 5400; known-trap exclusion and tail-count checks passed |
| 04 Intensity-error scaling | Completed | Scanned 2,487,251 HDF5 intensity points; `intensity_err` matches the local-noise role while `image_sigma` is a separate global threshold field; chose Stage 06+ noise model `sigma ~ p_sigma(sigma | T, quadrant, dtph)` from `cache/03_noise_map_v1.h5` |
| 05 Amplitude prior | Completed | Built `cache/05_amplitude_prior_v1.npz` from 13,963 high-confidence amplitude records covering all 2,135 `n_good = 4` traps; `fit_coeff` central width is a factor 2.89, trap-level depth correlations are weak (`max |rho| = 0.150`), `P_c(T)` falls to 0.681 at 210 K, and fainter-by-2/fainter-by-4 sensitivity variants are cached |
| 06 Single-curve recovery | Completed | Ran deterministic 160 K, quadrant 0 fake-curve walkthrough with 5 tau cases and 128 realizations each; near-band cases passed at 0.969 and 0.953, outside-band cases failed with explicit controlling cuts; artifacts cached |
| 07 Single-temperature `p_det` | Completed | Built `cache/07_single_temperature_pdet_160K_v1.h5` on a `10 tau x 7 A x 3 sigma` grid with 80 realizations per point; bright peak-reachable traps reach median `p_det = 0.95`, low-amplitude unreachable long-tau cases stay at max `0.0125`, cutflow sums pass, and alternate-seed max delta is `0.025` |
| 08 Full `p_det` grid | Completed | Built April-only `200 K` production `cache/08_pdet_grid_v1.h5` with marginalized `p_det(T, tau, A)` on a `23 x 55 x 35` grid and 100 realizations per point; all required checks passed; pilot artifacts retained |
| 09 Characterization probability | Completed | Built `cache/09_characterization_probability_v1.h5` from production Stage 08 grid and default Stage 05 amplitude prior; `161 x 121` `P(characterized | tau_135, E)` map plus `n_good = 3` sensitivity cached; Poisson-binomial and known-trap validation passed |
| 10 Validation and sensitivity | Completed | Stage 09 map validated against `2135` known `n_good = 4` traps (`99.86%` at `P_4 >= 0.8`); observed-`E` conditional completeness is `>=95%` for `tau_135 = 5.62e-4` to `668 s` under the default high-confidence amplitude prior; faint-by-2/faint-by-4 and excluding-160/170 K sensitivities cached; final statement written |
| 11 Naive efficiency closure | Completed | Naive measured/extrapolated curve explained: baseline Method 3 over-predicts the `1e-4..1e-2 s` plateau (0.93 vs 0.155) because real bright curves at `T >= 165 K` fail the chi-square GOF cut at 44-93% (median p-value ~0 at `T >= 175 K`) while the idealized injection fails at ~5%; hybrid with the empirical per-temperature GOF survival closes the curve (mean abs eff diff 0.053); flags that the long-`tau_135` completeness reach relies on high-T detections the Stage 08 model treats too optimistically |
| 12 High-T misfit mechanism | Completed | High-T GOF failures explained: real `t_ph`-independent dipole floor (64 e- at 150 K -> ~700 e- at >=175 K) from the trap acting as a CTI/deferred-charge defect on the high-T dark-current background during readout; `model + offset` restores chi2red ~0.15 at all temperatures; floor/sigma crosses 1 at 160-165 K matching the pass-rate collapse; patch `intensity_err` confirmed ~2.5x above true temporal noise at all T |

## How To Work With Codex

Start a task by naming the packet and limiting context:

> Work on `trap_completeness_method3/agents/02_fits_noise_parity.md`. Use results from stages 00
> and 01 if present. Fill the packet's `Results` section and update the status row in this README.

For large computations, ask Codex to first run a small diagnostic subset and write the result. Only
scale up after the packet's required checks pass.
