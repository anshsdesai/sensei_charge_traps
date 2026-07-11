# Signed Refit Detection-Significance Calibration

- Calibration version: `signed-refit-detection-calibration-v1`
- Profile fitter: `signed-refit-profile-tau-v1`
- Noise-model SHA-256: `d07dfec56bc8b5cad98282fe7a1c3c2fd3e5c157af660325338b7cd87535f39a`
- Controls SHA-256: `a8de148506e2d8f863844d903904c0b1c93f2e8785f14be92ee8a9a015b7fea6`
- Acceptance status: **PASS**

## Statistic and decision rule

The ranking statistic is the largest generalized-least-squares `delta chi-square` improvement over a constant curve while scanning the complete frozen 801-point log-`tau` grid. Because `tau` is undefined when the amplitude is zero, no Wilks-theorem chi-square interpretation is used.

A candidate-temperature fit passes when its finite-sample empirical `p <= 0.001` against the calibration controls for that temperature. Equivalent temperature-specific statistic thresholds are:

| T (K) | Delta chi-square threshold | Independent evaluation FPR |
|---:|---:|---:|
| 125 | 25.218 | 0.049% |
| 130 | 29.906 | 0.037% |
| 135 | 27.121 | 0.012% |
| 140 | 24.133 | 0.110% |
| 145 | 22.155 | 0.085% |
| 150 | 23.158 | 0.085% |
| 155 | 21.162 | 0.110% |
| 160 | 25.599 | 0.061% |
| 165 | 19.684 | 0.159% |
| 170 | 18.336 | 0.122% |
| 175 | 21.562 | 0.110% |
| 180 | 38.447 | 0.085% |
| 183 | 49.758 | 0.159% |
| 185 | 21.249 | 0.098% |
| 187 | 32.276 | 0.073% |
| 190 | 37.033 | 0.073% |
| 193 | 49.786 | 0.049% |
| 195 | 24.514 | 0.037% |
| 197 | 24.930 | 0.037% |
| 200 | 29.869 | 0.073% |
| 203 | 25.944 | 0.085% |
| 207 | 30.098 | 0.085% |
| 210 | 23.552 | 0.073% |

Each threshold is the lowest observed calibration statistic whose add-one finite-sample tail probability is at most 0.001. With 8,192 references this is normally the seventh-largest value. A disjoint 8,192 curves at each temperature are used only for evaluation.

## False-positive budget

- Preliminary candidate-temperature tests: 214,659 (9,333 sites x 23 temperatures).
- Target ordinary-null budget: at most 214.7 false temperature fits in that complete preliminary set.
- This is an intensity-fit budget, not a final false-trap claim. Step 7 must include finder selection, and later steps require multiple temperature fits and sign/SRH consistency.

## Independent ordinary-null evaluation

- Aggregate FPR: 0.0812%.
- Maximum temperature FPR: 0.1587%.
- Quadrant FPRs: Q0=0.0722%, Q1=0.0701%, Q2=0.1189%, Q3=0.0637%.
- Region FPR range: 0.0170%-0.1698%.
- Sites passing at least four temperatures: 2 of 8,192.
- 95% upper bound projected to 9,333 sites for at least four temperature passes: 7.17 sites.

## Look-elsewhere effect

- The old `delta_chi2 >= 11.83` rule accepts 1.972% of independent ordinary nulls.
- The calibrated target is 0.100%.
- The empirical thresholds are therefore substantially higher and vary with the actual dwell grid and acquisition condition.

## Structured controls

- Persistent horizontal-trigger sites: 375 sites / 8,625 curves; FPR 0.846%; sites with >=4 passes 5.
- Near-defect vertical sites: 15,420 sites / 354,660 curves; FPR 0.075%; sites with >=4 passes 1.

Horizontal-trigger coordinates were selected by persistence in at least two dwell images but evaluated using the normal vertical-pair profile fit. Near-defect coordinates lie outside all v2 masks but within 5 pixels of a persistent-defect mask.

## Acceptance gate

- PASS: the independent ordinary-null FPR meets the stated budget.
- PASS: rates remain within predefined temperature, quadrant, and region stability limits.
- PASS: horizontal-trigger and near-defect stress controls remain below the predefined 1% ceiling.
- PASS: thresholds and empirical p-values include the full tau look-elsewhere search without a Wilks-theorem claim.
