# Signed Refit Finder Calibration

- Calibration version: `signed-refit-finder-calibration-v2`
- Finder version: `signed-refit-finder-v1`
- Gain used: `400` ADU/e- (provisionally accepted by the analysis owner on 2026-06-13)
- Acceptance status: **PASS**

## Predeclared scan

| Configuration | Noise | Lobe rule | Threshold | Balance | Persistence | Trail isolation |
|---|---|---|---:|---:|---:|---|
| `legacy_reference` | legacy | product | 3.0 sigma | 0.30 | 2 | no |
| `robust_product_no_balance_p2` | robust | product | 3.0 sigma | none | 2 | no |
| `robust_separate_3sigma_p2` | robust | separate | 3.0 sigma | none | 2 | no |
| `robust_separate_2p5_balance_p2` | robust | separate | 2.5 sigma | 0.50 | 2 | no |
| `robust_separate_2p5_balance_p3` | robust | separate | 2.5 sigma | 0.50 | 3 | no |
| `robust_separate_2p5_balance_p3_isolated` | robust | separate | 2.5 sigma | 0.50 | 3 | yes |

The scan was fixed before looking at the final candidate count. The operating point maximizes real-residual injection completeness where the sampled expected peak is at least 4.0 robust image sigma, subject to ordinary and structured-null gates. Product-only rules are retained as historical comparisons but are not production-admissible because they can accept one sub-threshold lobe. Candidate count is reported only as a consequence, not as a selection input.

A v1 pilot incorrectly multiplied the 3000-cycle pump shape by `N_PUMPS` a second time and therefore clipped injection probabilities to one. After that unit bug was fixed, its A>=0.10 completeness gate was also found to be physically impossible: A=0.10 peaks near 195 e- while the measured per-image thresholds are typically 420-1050 e-. Version v2 uses the dimensionless sampled peak-SNR rule above and adds A=0.80 so every representative temperature has strong injections.

## Noise-estimator comparison

| T (K) | Robust median sigma (e-) | Legacy median sigma (e-) | Legacy/robust |
|---:|---:|---:|---:|
| 125 | 210.16 | 221.28 | 1.053 |
| 130 | 215.35 | 227.01 | 1.054 |
| 135 | 219.42 | 231.58 | 1.055 |
| 140 | 223.13 | 235.49 | 1.055 |
| 145 | 224.61 | 237.19 | 1.056 |
| 150 | 224.61 | 238.49 | 1.062 |
| 155 | 222.76 | 236.33 | 1.061 |
| 160 | 348.78 | 370.56 | 1.062 |
| 165 | 215.72 | 229.93 | 1.066 |
| 170 | 325.80 | 345.83 | 1.061 |
| 175 | 202.75 | 217.74 | 1.074 |
| 180 | 196.44 | 213.23 | 1.085 |
| 183 | 189.03 | 205.65 | 1.088 |
| 185 | 186.81 | 201.95 | 1.081 |
| 187 | 183.84 | 199.77 | 1.087 |
| 190 | 179.77 | 195.62 | 1.088 |
| 193 | 178.65 | 192.03 | 1.075 |
| 195 | 169.02 | 183.29 | 1.084 |
| 197 | 178.65 | 190.82 | 1.068 |
| 200 | 177.17 | 190.04 | 1.073 |
| 203 | 174.95 | 187.68 | 1.073 |
| 207 | 169.76 | 180.70 | 1.064 |
| 210 | 166.05 | 178.39 | 1.074 |

## Completeness-purity tradeoff

| Configuration | Strong-signal completeness | Ordinary E2E FPR | Horizontal-trigger vertical FPR | Near-defect FPR | Horizontal-axis FPR | Union candidates | Production gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `legacy_reference` | 79.9888% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 5,171 | FAIL |
| `robust_product_no_balance_p2` | 97.2196% | 0.0000% | 0.0000% | 0.0000% | 4.3594% | 9,329 | FAIL |
| `robust_separate_3sigma_p2` | 91.1635% | 0.0000% | 0.0000% | 0.0000% | 0.0116% | 5,341 | PASS |
| `robust_separate_2p5_balance_p2` | 94.3150% | 0.0000% | 0.0000% | 0.0000% | 0.1159% | 8,241 | PASS |
| `robust_separate_2p5_balance_p3` | 91.7759% | 0.0000% | 0.0000% | 0.0000% | 0.0580% | 7,080 | PASS |
| `robust_separate_2p5_balance_p3_isolated` | 66.7564% | 0.0000% | 0.0000% | 0.0000% | 0.0580% | 5,957 | PASS |

The ordinary controls were deliberately masked away from the preliminary candidate union in Step 2, so their zero rate is conditional and is not, by itself, a full-field purity estimate. The near-defect and horizontal controls provide the adversarial checks, and the quoted binomial upper bounds retain finite-sample uncertainty.

The horizontal-axis control applies the vertical-pair covariance and Step 6 threshold to horizontal-pair curves. It is intentionally a stress test for non-pumping structure, not a claim that horizontal curves have independently calibrated p-values.

## Selected operating point

Selected: `robust_separate_2p5_balance_p2`.

It recovers 94.3150% of injections whose sampled expected peak is >=4.0 robust image sigma. Its complete finder-plus-profile false-positive rates are 0.0000% on ordinary controls, 0.0000% at horizontal-trigger coordinates using the vertical pair, 0.0000% near defects, and 0.1159% for the true horizontal-axis negative control.

For these aggregate rates, the corresponding one-sided 95% binomial upper bounds are 0.0008%, 0.0347%, 0.0008%, and 0.1966%.

### Completeness by temperature and amplitude

| T (K) | A=0.03 | A=0.06 | A=0.10 | A=0.20 | A=0.40 | A=0.80 |
|---:|---:|---:|---:|---:|---:|---:|
| 125 | 0.0000% | 0.0000% | 0.0366% | 3.5278% | 64.5264% | 98.5352% |
| 145 | 0.0000% | 0.0122% | 0.0610% | 2.2705% | 54.0649% | 98.1934% |
| 170 | 0.0000% | 0.0000% | 0.0488% | 0.4028% | 14.2456% | 88.7085% |
| 183 | 0.0000% | 0.0366% | 0.2563% | 7.7393% | 77.5269% | 99.4873% |
| 200 | 0.0122% | 0.0732% | 0.5859% | 16.5283% | 90.3442% | 99.7681% |
| 210 | 0.0000% | 0.0610% | 0.4150% | 17.9443% | 91.3452% | 99.7681% |

### End-to-end null rate by temperature

| T (K) | Ordinary | Horizontal-trigger vertical | Near defect | Horizontal axis |
|---:|---:|---:|---:|---:|
| 125 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 130 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 135 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 140 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 145 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 150 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 155 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 160 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 165 | 0.0000% | 0.0000% | 0.0000% | 0.2667% |
| 170 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 175 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 180 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 183 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 185 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 187 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 190 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 193 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 195 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 197 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| 200 | 0.0000% | 0.0000% | 0.0000% | 0.2667% |
| 203 | 0.0000% | 0.0000% | 0.0000% | 1.0667% |
| 207 | 0.0000% | 0.0000% | 0.0000% | 0.8000% |
| 210 | 0.0000% | 0.0000% | 0.0000% | 0.2667% |

## Charge-balance and trail diagnostics

The relaxed balance requirement limits the magnitude mismatch between opposite lobes to 50%. Requiring three dwell detections changes the all-temperature union from 8,241 to 7,080 sites. Adding the 20-row trail-isolation diagnostic changes it further to 5,957.

The isolation rule counts additional >=2.5-sigma pixels in the same column within 20 rows of the pair; it is a diagnostic for deferred trails and crowded defects, not a claim that every non-isolated site is nonphysical.

## Acceptance gate

- PASS: finder completeness was measured with binomial transfer injections placed on real, held-out residual image pairs.
- PASS: the full finder decision was intersected with the stored Step 6 profile-tau empirical p-value for ordinary, horizontal-trigger, and near-defect null controls; a horizontal-axis finder-plus-profile negative control was added independently.
- PASS: both lobes must be separately significant in the frozen production configuration.
- PASS: the operating point obeys the predeclared completeness and false-positive gates.
- PASS: final trap count was not an optimization target.

Frozen configuration: `signed_refit_finder_config.json`.

## Figures

- `figures/signed_refit_finder/completeness_purity_tradeoff.png`
- `figures/signed_refit_finder/selected_completeness_by_temperature.png`
