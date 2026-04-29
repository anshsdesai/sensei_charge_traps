# Agent 02: Intensity Fit Audit

## Goal
Audit the per-temperature intensity fitting stage in `getDipoleSpectra2` and `fitTrapIntensity`, quantify which cuts dominate selection, and determine whether the current goodness-of-fit criteria are rejecting traps because of genuine model mismatch or because the uncertainty model is being used too formally.

## Primary Questions
- Is the local `35x35` spread being used appropriately as a fit uncertainty?
- Is the `p_value > 0.05` cut too aggressive relative to the roughness of the uncertainty estimate?
- Are the brightness and `tau`-precision cuts removing genuinely bad fits, or are they redundant with the p-value gate?
- What should be recommended as the default fit-quality policy?

## Files To Read
- `dipole.py`
- `fit_dipole_spectra_err_4.h5`
- any user-provided rerun summaries or updated result files

## Current Repo Facts
- Current cached fit-stage counts:

| Stage | Count |
| --- | ---: |
| Traps with any good temperature fit | 3365 |
| Traps with `>= 4` good temperature fits | 2514 |
| Traps with `GoodEnergyFit` | 2135 |

- Current cached temperature-fit statistics:

| Metric | Value |
| --- | ---: |
| Total temperature groups in cache | 118933 |
| Temperature groups with `GoodIntensityFit=True` | 20121 |
| Mean good temperatures per trap | 3.89 |
| Median good temperatures per trap | 3 |
| Mean good temperatures per well-behaved trap | 7.26 |
| Median good temperatures per well-behaved trap | 6 |

- Current cut pressure indicators from the cached output:

| Condition proxy | Count |
| --- | ---: |
| `p_value <= 0.05` | 62350 |
| `max(intensity) < 3 * mean(intensity_err)` | 35160 |
| `max(intensity) < 3 * image_sigma` | 36905 |
| `tau_err / tau > 0.5` | 10881 |

- A key observed pattern:
  - If the p-value requirement is conceptually removed while preserving the SNR and `tau`-precision cuts, the count of traps with `>= 4` good temperature fits rises from `2514` to `4702`.

## Intensity-Fit Logic To Audit
1. Intensity definition in `getDipoleSpectra2`.
   - Verify `intensity = (image[dp] - image[coord_b]) / 2`.
   - Confirm whether absolute value handling is intended for the downstream fit.
2. Uncertainty definition.
   - `intensity_err` is populated from the standard deviation of the surrounding `35x35` region.
   - Treat this as a rough local spread estimate unless strong evidence shows it behaves like a formal Gaussian sigma.
3. Goodness-of-fit implementation in `fitTrapIntensity`.
   - Review the temperature-level gates:
     - `p_value > 0.05`
     - `max(intensity) > 3 * mean(intensity_err)`
     - `max(intensity) > 3 * image_sigma`
     - `tau_err / tau < 0.5`
4. Coupling between fit errors and fit quality.
   - `curve_fit(..., sigma=intensity_err)` is used without `absolute_sigma=True`.
   - The stored parameter errors are therefore scaled by reduced chi-square.
   - The `tau_err / tau` cut is not independent of the residual-based fit quality.

## Required Output 1: Temperature-Fit Cutflow Table
Produce a table with one row per cut or cut combination, using the current cache first and rerun outputs later.

Minimum rows:

| Selection variant | Traps with `>= 4` good temperatures |
| --- | ---: |
| Baseline current cache | 2514 |
| p-value threshold `0.001` | 2701 |
| p-value threshold `0.01` | 2596 |
| p-value threshold `0.05` | 2514 |
| p-value threshold `0.1` | 2436 |
| No p-value requirement, keep other cuts | 4702 |

Clarify that these alternate counts are derived from the current cached fit statistics and must be validated on rerun if code changes materially.

## Required Output 2: Example Accepted/Rejected Spectra
Prepare examples in three categories:
1. Clear good fit
2. Clear bad fit
3. Borderline p-value-only rejection

For each example, record:
- quadrant
- coordinate
- temperature
- number of `dtph` points
- `fit_tau`
- `fit_tau_err`
- `fit_p_value`
- `fit_reduced_chi_squared`
- peak intensity
- local noise scale

At least one example should show a visually plausible fit that fails only on p-value.

## Required Output 3: Recommendation On P-Value Usage
End with one of these recommendations:
- Keep p-value as a hard cut
- Keep p-value, but downgrade it to a review flag when other quality indicators are good
- Replace strict p-value gating with a different fit-quality policy because the uncertainty model is too rough for formal chi-square selection

The recommendation must explicitly reference:
- whether the uncertainty model is formal or approximate
- whether the current p-value gate is the dominant recall limiter

## Exact User Rerun Request
Ask the user to run at least:
1. Baseline current pipeline
2. One fit-quality variant:
   - either a p-value threshold grid
   - or a mode where p-value is reported but not enforced

Ask the user to paste back:
- total traps with any good temperature fit
- total traps with `>= 4` good temperature fits
- total traps with `GoodEnergyFit`
- temperature-fit cutflow counts
- 10 to 20 example trap summaries for borderline cases
- if possible, updated `.h5` outputs or compact CSV summaries

## Handoff Notes For Later Agents
- Agent 03 needs to know whether changing temperature-fit policy alters the final energy-fit population materially.
- Agent 05 should treat p-value policy as the central fit-quality decision unless reruns show the detection stage is the larger recall bottleneck.
