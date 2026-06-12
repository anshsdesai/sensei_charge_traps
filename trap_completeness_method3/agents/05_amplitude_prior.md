# 05 Amplitude Prior

## Objective

Calibrate the trap-amplitude model using fitted `fit_coeff` values and test whether amplitude is
independent of `E`, `tau`, and `tau_135`.

## Why This Matters

Method 3's main residual assumption is the amplitude prior. A long-lived trap is hard to detect
because its peak may be unreachable, not because it is intrinsically faint, but the analysis must
test whether observed amplitudes support that assumption.

## Inputs

- `agents/01_hdf5_records_audit.md`
- `../fit_dipole_spectra_err_4.h5`
- `../fit_dipole_spectra_err_3.h5`
- `../dipole.py`
- `../utils.py`

## Procedure

1. Extract `fit_coeff`, `fit_coeff_err`, `fit_tau`, `fit_tau_err`, `GoodIntensityFit`, temperature,
   `E`, `log_sigma`, and `tau_135` for characterized traps.
2. Convert fitted coefficient to amplitude using `A = 3000 * fit_coeff`.
3. Summarize amplitude breadth by temperature and for bright, high-confidence traps.
4. Estimate common temperature scaling `P_c(T)` from bright traps.
5. Estimate trap-level depth spread after removing the common temperature scaling.
6. Test correlations of amplitude/depth with `E`, `fit_tau`, and `tau_135`.
7. Define the default amplitude prior and at least two sensitivity variants.

## Required Checks

- Report whether `fit_coeff` spans a factor of a few or orders of magnitude.
- Report rank correlations with `E`, `fit_tau`, and `tau_135`.
- Explicitly state whether the `D_t` independent of `E/tau` assumption looks acceptable.
- Define fainter-amplitude sensitivity variants if the observed prior is truncated at threshold.

## Outputs

- `cache/05_amplitude_prior_v1.npz`
- `cache/05_amplitude_prior_summary.json`
- Optional plots under `cache/figures/`

## Stop Conditions

- Stop if `fit_coeff` is absent or unreliable.
- Stop if strong amplitude correlations invalidate the default independence assumption without a
  replacement model.

## Results

Completed 2026-05-20T13:35:53-07:00.

Command used:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/ansh/miniforge3/bin/conda run -n sensei_charge_traps python trap_completeness_method3/src/amplitude_prior.py
```

Input artifacts:

- `agents/01_hdf5_records_audit.md`
- `../fit_dipole_spectra_err_4.h5`
- `../fit_dipole_spectra_err_3.h5`
- `../dipole.py`
- `../utils.py`

Output artifacts:

- `cache/05_amplitude_prior_v1.npz`
- `cache/05_amplitude_prior_summary.json`
- `cache/figures/05_depth_prior_hist.png`
- `cache/figures/05_pc_temperature_scaling.png`
- `src/amplitude_prior.py`

Short numerical summary:

- Primary `n_good = 4` HDF5 provided `15233` usable `GoodIntensityFit` amplitude records from
  `2135` characterized traps.
- The high-confidence prior subset required `fit_coeff / fit_coeff_err >= 5`, `fit_tau_err /
  fit_tau <= 0.25`, and sampled-peak fraction `>= 0.5`. It retained `13963` records and still
  covered all `2135` characterized traps.
- `A = 3000 * fit_coeff` has median `2587.64 e-`, p05/p95 `1261.33`/`3643.41 e-`, and
  p95/p05 factor `2.89` over all primary good-fit records.
- After removing the common temperature scaling, the default trap-depth prior has median
  `2511.61 e-`, p05/p95 `1367.87`/`3206.87 e-`, and p95/p05 factor `2.34`.
- The robust common temperature factor `P_c(T)`, normalized to `P_c(135 K)=1`, has p95/p05
  factor `1.74`; it rises to `1.26` at `160 K` and falls to `0.681` at `210 K`.
- Sensitivity `n_good = 3` file was also scanned and contains `16379` usable good-fit amplitude
  records from `2517` characterized traps.

Rank correlations:

- Observation-level `log(A)` over all primary good-fit records: `rho(E)=0.0681`,
  `rho(log fit_tau)=0.1515`, `rho(log tau_135)=0.0828`.
- Observation-level temperature-corrected `log(D_t)` over the high-confidence subset:
  `rho(E)=0.1490`, `rho(log fit_tau)=0.0264`, `rho(log tau_135)=0.1581`.
- Trap-level median temperature-corrected `log(D_t)`: `rho(E)=0.1386`,
  `rho(log median fit_tau)=0.0822`, `rho(log tau_135)=0.1501`.

Default prior and sensitivity variants:

- Default: empirical trap-depth samples from `default_depth_electrons_at_pc135` in
  `cache/05_amplitude_prior_v1.npz`, with amplitudes at temperature `T` reconstructed as
  `A(T) = D_t * pc_temperature_factor(T)`.
- Faint sensitivity variants: `faint_0p5_depth_electrons_at_pc135` and
  `faint_0p25_depth_electrons_at_pc135`, shifting the same empirical depth samples fainter by
  factors of `2` and `4`.

Required checks:

- Report whether `fit_coeff` spans a factor of a few or orders of magnitude: PASS. p95/p05 is
  `2.89`, so the central observed width is a factor of a few, not orders of magnitude. The full
  min/max span is `18.05`, driven by tails.
- Report rank correlations with `E`, `fit_tau`, and `tau_135`: PASS, listed above and stored in
  `cache/05_amplitude_prior_summary.json`.
- Explicitly state whether the `D_t` independent of `E/tau` assumption looks acceptable: PASS.
  The largest trap-level absolute Spearman correlation is `0.1501`, so no strong dependence is
  seen in the characterized sample.
- Define fainter-amplitude sensitivity variants if the observed prior is truncated at threshold:
  PASS. The default prior is explicitly conditional on detected/high-confidence records, so the
  fainter-by-2 and fainter-by-4 variants should be propagated through Stage 08/10.

Stop conditions encountered:

- None. `fit_coeff` is present for usable `GoodIntensityFit` records, and no strong amplitude
  correlation forces a replacement model.

## Open Questions

- `P_c(T)` is not perfectly flat; the inferred factor drops to `0.681` by `210 K`, so later stages
  should use the stored temperature factor rather than a scalar normalization.
- Because the empirical amplitude prior is selected on detected, high-confidence records, the
  final completeness statement should remain conditional on the amplitude-prior assumption and
  include the `faint_0p5` and `faint_0p25` sensitivity variants.
