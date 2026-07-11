# Signed Refit Step 10 SRH Validation

- Pipeline version: `signed-refit-srh-v1`
- Artifact: `signed_refit_srh_fits_v1.h5`
- Artifact SHA-256: `4330a5ada27cd5da528426aa2f42a8d297de0ce0b61377ab03fdd11419de6e1d`
- Acceptance status: **PASS**
- Primary variant: `no_160_170`.
- Model: simple p-channel SRH emission from a level above the valence edge.
- Intrinsic scatter: fixed to zero; no outlier rejection.
- SRH-consistent classification: profile deviance `p>=0.05`.

## Population cutflow

- Step 9 single-trap inputs: 2,703.
- Primary fit successes: 2,407.
- Primary SRH-consistent sites: 1,287.

| Status | Full | No 160/170 K | Primary |
|---|---:|---:|---:|
| `not_step9_single` | 5,538 | 5,538 | 5,538 |
| `insufficient_points` | 0 | 264 | 264 |
| `optimizer_failed` | 0 | 0 | 0 |
| `prediction_outside_profile` | 2 | 4 | 4 |
| `parameter_boundary` | 17 | 28 | 28 |
| `fit_success_non_srh` | 1,489 | 1,120 | 1,120 |
| `srh_consistent` | 1,195 | 1,287 | 1,287 |

## Acquisition-family systematic

- 160/170 K pooled residual sites: 1,881.
- Pooled residual median: +0.0360 dex.
- Site-bootstrap 99% interval: [+0.0298, +0.0477] dex.
- Residual trigger: False.
- Paired fit population: 2,398.
- Full/no-family median energy: 0.2857/0.2886 eV.
- Median energy shift: +0.0029 eV; combined one-sigma uncertainty 0.0010 eV.
- Paired median no-family-minus-full ln(sigma): +0.0000.
- Population trigger: True.
- Primary decision: `no_160_170`.

## Residual diagnostics

- Primary fitted points outside the measured pump-peak window: 2,127/25,703 (8.28%).
- Primary median/maximum per-site leverage: 0.439/0.998.
- Residual versus |amplitude| Spearman rho: -0.171 (p=1.83e-165).
- Residual versus |pedestal| Spearman rho: -0.083 (p=5.69e-40).
- Residual versus intensity-fit p-value Spearman rho: +0.070 (p=2.82e-29).
- High-temperature median residuals exceeding 0.10 dex: 0.

No high-temperature point is removed or assigned intrinsic scatter. Any listed deviation remains a documented simple-SRH failure mode.

## Legacy comparison

- Legacy well-behaved sites: 2,514.
- Legacy well-behaved entering current Step 10: 1,563.
- Legacy well-behaved excluded before Step 10: 951.
- Legacy good-energy sites: 2,135.
- Legacy good-energy and current primary SRH-consistent: 711.
- Common-population median current-minus-legacy energy: +0.0117 eV.
- Common-population median current-minus-legacy ln(sigma): +0.8275.

Legacy well-behaved Step 9 exclusions:

| Current classification | Sites |
|---|---:|
| `ambiguous_sign_conflict` | 212 |
| `dual_response` | 208 |
| `insufficient_significant_temperatures` | 373 |
| `no_significant_temperature` | 158 |

## Figures

- `figures/signed_refit_srh/srh_residuals_by_temperature.png`
- `figures/signed_refit_srh/family_energy_comparison.png`
- `figures/signed_refit_srh/legacy_current_energy_comparison.png`

## Acceptance gate

- PASS: energy and cross-section intervals profile the calibrated tau likelihoods.
- PASS: one fixed simple-SRH consistency criterion is used without intrinsic scatter or outlier removal.
- PASS: acquisition-family promotion was evaluated before selecting the primary variant.
- PASS: high-temperature deviations are retained and reported rather than tuned away.
- PASS: non-SRH and failed sites remain explicit classifications.
