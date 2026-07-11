# Signed-Pipeline Refit — Change Audit

> **Physics-audit status (2026-06-13):** This document is a change summary, not
> an approval of the new selections. See `SIGNED_REFIT_PHYSICS_AUDIT.md`. In
> particular, the supplied-error covariance needed `absolute_sigma=True`, the
> current scalar pair-noise formula is not yet validated as a per-point error
> model, and the proposed robust energy criterion increases decoy acceptance.

Audit of all changes made while rebuilding the dipole intensity analysis with
signed intensities, a readout pedestal term, physically correct error bars, and a
re-examined goodness-of-fit criterion. Written 2026-06-12 for review **before**
anything propagates to the published catalog or figures.

## How to read this

- **Section 1** is the one-paragraph summary.
- **Section 2** lists changes to *production* code that the pipeline runs. These
  are the ones to audit carefully. **Every new behavior is behind a non-default
  flag**, so the legacy `run_charge_traps.py` path produces byte-identical results
  unless explicitly switched on — except for one deliberate default change in
  `run_ccd_simulation.py` (Section 2.2), which is unrelated to the trap fitting.
- **Sections 3–6** are additive: new functions, a new driver, new artifacts, paper
  text, and the Method-3 study. Nothing here overwrites legacy outputs.
- **Section 7** lists throwaway diagnostic scripts (safe to delete).
- **Section 8** is the scientific findings; **Section 9** is the open decision and
  what is *not* yet done.

---

## 1. Summary

The per-temperature dipole fits failed above ~165 K because a real,
`t_ph`-independent **readout pedestal** (deferred dark-current charge at the trap
pixel, growing with temperature) is not in the 2-parameter model, so the χ² test
rejected bright hot curves. I added the pedestal as a fitted offset, switched the
intensity definition to **signed**, and replaced the spatial-patch error bar with
the **physically correct temporal pair noise** (~2.5× smaller). With these, the
fits are good at all temperatures (reduced χ² ≈ 1) and injection shows the fit is
unbiased. The smaller, correct error bars then exposed that the old
`reduced χ² < 5` energy-fit cut was calibrated to inflated errors; I built a
more-correct energy criterion (exact linear SRH fit + data-measured intrinsic
dispersion + robust outlier rejection). One open issue remains (Section 9): the
hot-temperature τ values lean ~0.1–0.2 dex below the straight SRH line, which
churns ~650 traps from the catalog. **No legacy artifact has been overwritten;
the new criterion is not yet wired into the production fit function.**

---

## 2. Production code changes (audit these)

### 2.1 `dipole.py` — all new behavior gated behind non-default flags

| Function | Change | Default (legacy) behavior preserved? |
|---|---|---|
| `findDipoles2` | New args `robust_sigma=False`, `symmetry_perc=0.3`. `robust_sigma=True` sets the detection threshold from a MAD on the row-median-subtracted image instead of the histogram width. `symmetry_perc=None` disables the lobe-amplitude-match cut. | Yes — defaults reproduce the old finder. |
| `getDipoleList2` | New args `robust_sigma=False`, `symmetry_perc=0.3` passed through to `findDipoles2`. | Yes. |
| `getDipoleSpectra2` | New args `error_model='patch'`, `noise_table=None`. With `error_model='physical'`, `intensity_err` becomes `sqrt(sigma_base(T,quad)^2 + (S_a+S_b)/4)` where `sigma_base` comes from `noise_table` and `S_{a,b}` are the two pixels' charge above the row median. The old spatial-patch σ is still computed and stored under the new key `patch_sigma`. Captures `raw_charge` before row-median subtraction for the shot-noise term. | Yes — `error_model='patch'` reproduces the old `intensity_err`. The `absolute=True/False` switch is unchanged. **Note:** a new `patch_sigma` dataset is now written into every spectra record. |
| `intensity_function_offset` (NEW) | `N_pumps*coeff*(exp(-t/τ) − exp(-8t/τ)) + offset`; both `coeff` and `offset` signed. Plus module constants `INTENSITY_SHAPE_PEAK`, `INTENSITY_SHAPE_PEAK_X`. | Additive; `intensity_function` unchanged. |
| `fitTrapIntensity` | New arg `fit_offset=False`. When `True`: fits the 3-parameter offset model; replaces the two max-intensity threshold cuts with an **amplitude-significance** cut `|coeff|/σ_coeff ≥ 3`; adds a **Δχ² ≥ 11.83** guard versus the constant-only model (catches spike fits); stores `fit_offset`, `fit_offset_err`, `amplitude_significance`, `delta_chi2_vs_constant`. The `p_value > 0.05` and `σ_τ/τ ≤ 0.5` cuts are unchanged. | Yes — `fit_offset=False` reproduces the legacy 2-parameter fit and cut set exactly. |

The energy-fit stage of `fitTrapIntensity` (the per-trap Arrhenius fit with
`reduced χ² < 5`) is **unchanged**. The new energy criterion lives in standalone
functions (Section 3) and is **not yet called by `fitTrapIntensity`** — see
Section 9.

The SRH physics function `log_energy_cross_section` (the T⁻² prefactor) is
**unchanged**. An earlier idea to fit an empirical prefactor power was discarded as
unphysical (it is degenerate with E over our temperature range).

### 2.2 `run_ccd_simulation.py` — one deliberate default change

- `--num_workers` default changed from `cpu_count() - 1` to `cpu_count() // 2`,
  with a comment. **Reason:** each worker holds ~2.3 GB; the old default exhausted
  32 GB RAM during a campaign and blocked WSL. This is the only change here that
  alters default behavior, and it is unrelated to the trap fitting. Explicit
  `--num_workers` still overrides.
- (The `--exp-indep-charge-mode` argument visible in this file was added by you,
  not by this work.)

---

## 3. New reusable functions in `dipole.py` (not yet wired into the pipeline)

- `robust_energy_fit(temperatures, taus, tau_errs, ...)` — the SRH law is linear in
  `(log σ, E)`, so this is an exact weighted linear least squares (no `curve_fit`).
  Adds (a) an intrinsic log-τ dispersion `sigma_int_dex` in quadrature with the
  measurement error, and (b) robust one-at-a-time outlier rejection of points more
  than `max(n_sigma·σ, max_resid_dex)` off the trap's own line, never below
  `min_points`. Returns E, log σ, errors, reduced χ², n_used.
- `estimate_intrinsic_dispersion(trap_points, ...)` — bisects `sigma_int` so the
  *median* reduced χ² of the population equals 1, measuring the genuine scatter of
  traps about the SRH law from the data itself.

These are exercised only by the analysis scripts in Section 7. They are the
proposed replacement for the `reduced χ² < 5` energy cut, pending the Section 9
decision.

---

## 4. New pipeline driver and regenerated artifacts (versioned; legacy untouched)

`run_signed_pipeline.py` (NEW) regenerates, skipping any stage whose output exists:

| Output (NEW filename) | Size | Produced by |
|---|---|---|
| `pair_noise_table.npz` | 3 KB | temporal pair-noise σ_base per (T, quadrant) |
| `dipole_coord_list_signed.npz` | 150 KB | `getDipoleList2(robust_sigma=True, symmetry_perc=None)` → 9333 candidates (legacy: 5171) |
| `dipole_spectra_signed.h5` | 831 MB | `getDipoleSpectra2(absolute=False, error_model='physical')` |
| `fit_dipole_spectra_signed_err_4.h5` | 1.2 GB | `fitTrapIntensity(fit_offset=True)` |

The legacy `dipole_coord_list.npz`, `dipole_spectra.h5`, and
`fit_dipole_spectra_err_4.h5` are **not modified**. This transitional driver is
intended to be folded into `run_charge_traps.py` as defaults once the approach is
approved, after which it would be deleted.

Decoy control (false-characterization test): `run_decoy_control.py` →
`decoy_spectra_signed.h5`, `decoy_fit_signed.h5`.

---

## 5. Paper changes — `paper/paper.tex`

All edits are in the measurement/identification/analysis sections. Numbers that
will change after the refit are flagged inline with `% PENDING` comments and have
**not** been updated yet.

- Pocket-pumping section: note that the dipole intensity now **retains its sign**.
- Eq. (`I_fit`): added the pedestal term `+ I_0` and a new paragraph explaining its
  readout / deferred-dark-current origin and why omitting it rejects hot curves.
- Trap identification: new paragraph on the robust-σ threshold and dropping the
  lobe-symmetry requirement.
- New paragraph defining the temporal-pair-noise error model and noting the spatial
  patch σ overestimates it ~2.5×.
- "Good trap" criteria: amplitude-significance cut replaces the two max-intensity
  cuts.
- Method-3 paragraph: injection now includes the pedestal and the temporal noise.
- `% PENDING` flags on: 5171 / 60,029 dipoles, 3379 good, 2121 well-behaved, and the
  `10⁻³–10⁴ s` completeness bounds.

`paper/paper.pdf` shows as modified in git but has **not** been rebuilt by this
work; treat the PDF as stale.

---

## 6. Method-3 completeness study additions (`trap_completeness_method3/`)

New, self-contained; explains the original odd efficiency curve.

- `agents/11_naive_efficiency_closure.md`, `src/naive_efficiency_closure.py`,
  `src/observed_cutflow.py` — closure test: the naive measured/extrapolated
  efficiency curve is reproduced once the empirical per-temperature good-fit rate
  is folded in (the curve is a selection artifact, not trap physics).
- `agents/12_high_temp_misfit.md`, `src/high_temp_misfit_diagnostics.py` —
  established the pedestal mechanism (constant-offset fixes χ² at all T).
- `README.md` updated with stage 11 and 12 rows; cache artifacts under `cache/`.

---

## 7. Diagnostic / investigation scripts (throwaway — safe to delete)

Root-level, written to investigate and validate; not part of the pipeline:
`validate_signed_refit.py`, `diagnose_validation.py`, `tau_bias_injection_test.py`,
`pedestal_vs_tph_test.py`, `deviation_stratify.py`, `energy_fit_scenarios.py`,
`population_stability.py`, `apply_energy_criterion.py`, `diagnose_unrecovered.py`,
`final_criterion.py`, `unrecovered_examples.py`, `finder_null_test.py`,
`smoke_test_offset_fit.py`, plus logs `signed_pipeline_log.txt`,
`signed_fit_log.txt`, `decoy_control_log.txt`, and `figures/unrecovered_examples.png`.

(The discarded empirical-prefactor-power idea was never written to disk — that
command was rejected before it ran, so no such script exists.)

---

## 8. What the investigation established

1. The hot-temperature failures are a real readout pedestal, not noise: a charge
   deficit at the trap pixel with charge deferred along the readout direction,
   growing with temperature; constant in `t_ph`.
2. The offset model restores reduced χ² ≈ 1 at all temperatures; injection shows
   the recovered τ is unbiased given a constant pedestal.
3. Correct temporal pair noise is ~32–39 e⁻, vs the ~190 e⁻ patch σ (~2.5× smaller).
4. Finder false-characterization rate (decoy control): 0.8% (random), 2.9%
   (structured non-pumping sites).
5. Aggregate population is stable and better-sampled at high energy
   (median E 0.28→0.30 eV; fraction E>0.4 eV 1.6%→4.2%).

---

## 9. Open decision and what is NOT done

**Open (blocks everything downstream):** hot-temperature τ leans ~0.1–0.2 dex below
the straight SRH line (curved Arrhenius), failing ~650 of 2135 legacy traps under
the correct error bars and halving the long-lived tail (τ₁₃₅ > 1 hr: 0.024→0.011).
Cause is either a `t_ph`-dependent pedestal (dark current accumulated during the
pump train — instrumental, fixable) or genuine σ(T) (physics, quote as systematic).
This is the question posed to you.

**Not done yet:**
- The new energy criterion (`robust_energy_fit`) is **not** wired into
  `fitTrapIntensity`; the production energy stage still uses `reduced χ² < 5`.
- `run_charge_traps.py` is **not** modified; the signed path is a separate driver.
- Downstream products are **not** regenerated: `tau_at_<T>k_hist.npz`,
  `trap_tau135_sigma_pairs.npz`, the Method-3 grids, `prob_of_measuring.pdf`,
  `efficiency_completeness.pdf`, and the `% PENDING` paper numbers.
- Paper PDF not rebuilt.

**Reversibility:** every legacy artifact is intact; reverting means discarding the
`*_signed*` files and the new scripts, and reverting the gated edits in `dipole.py`,
the one default in `run_ccd_simulation.py`, and the `paper.tex` text.
