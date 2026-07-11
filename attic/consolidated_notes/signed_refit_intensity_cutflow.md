# Signed Refit Step 9 Intensity Cutflow

- Pipeline version: `signed-refit-intensity-v1`
- Fit artifact: `signed_refit_intensity_fits_v1.h5`
- Fit artifact SHA-256: `c7b3c1bb6e757e17de94238f0f29b29275fc2bfb24704b29c5e1bd1b165e5b92`
- Acceptance status: **PASS**
- Candidate sites: 8,241.
- Accepted candidate-temperature fits: 38,000.
- Final single-trap sites: 2,703.
- Prevented restorations of prior conflict/structured sites: 254.

A fit is attempted only after the exact finite-sample empirical `p <= 0.001` test and `|D_t P_c| <= 1` on the null-covariance fit. Characterization then requires converged signal-dependent covariance, a physical final amplitude, a two-sided profile interval, and no boundary or multimodal flag.

## Temperature cutflow

| T (K) | Candidates | Detected physical | Accepted | Quality fraction |
|---:|---:|---:|---:|---:|
| 125 | 8,241 | 272 | 109 | 40.07% |
| 130 | 8,241 | 2,029 | 151 | 7.44% |
| 135 | 8,241 | 3,379 | 186 | 5.50% |
| 140 | 8,241 | 3,345 | 676 | 20.21% |
| 145 | 8,241 | 2,806 | 2,196 | 78.26% |
| 150 | 8,241 | 2,735 | 2,577 | 94.22% |
| 155 | 8,241 | 2,874 | 2,608 | 90.74% |
| 160 | 8,241 | 2,437 | 2,251 | 92.37% |
| 165 | 8,241 | 3,092 | 2,629 | 85.03% |
| 170 | 8,241 | 2,602 | 2,234 | 85.86% |
| 175 | 8,241 | 2,338 | 2,009 | 85.93% |
| 180 | 8,241 | 2,097 | 1,836 | 87.55% |
| 183 | 8,241 | 1,921 | 1,721 | 89.59% |
| 185 | 8,241 | 2,001 | 1,764 | 88.16% |
| 187 | 8,241 | 1,956 | 1,721 | 87.99% |
| 190 | 8,241 | 1,894 | 1,639 | 86.54% |
| 193 | 8,241 | 1,768 | 1,482 | 83.82% |
| 195 | 8,241 | 1,839 | 1,654 | 89.94% |
| 197 | 8,241 | 1,991 | 1,645 | 82.62% |
| 200 | 8,241 | 1,923 | 1,672 | 86.95% |
| 203 | 8,241 | 1,950 | 1,790 | 91.79% |
| 207 | 8,241 | 1,909 | 1,810 | 94.81% |
| 210 | 8,241 | 1,912 | 1,640 | 85.77% |

## Temperature diagnostics

- 125 K: 109/272 physical detections are characterizable; the dominant fit rejection is `interval_not_two_sided` (137).
- 130 K: 151/2,029 physical detections are characterizable; the dominant fit rejection is `interval_not_two_sided` (1,814).
- 135 K: 186/3,379 physical detections are characterizable; the dominant fit rejection is `interval_not_two_sided` (3,145).
- 140 K: 676/3,345 physical detections are characterizable; the dominant fit rejection is `interval_not_two_sided` (2,627).

The 130-140 K loss is therefore a scan-window effect: most detected profiles do not bracket both sides of the Delta-chi-square interval. They remain detected in the artifact but are not assigned a lifetime for Step 10.

## Rejection reasons by temperature

| T (K) | accepted | empirical_detection_fail | null_amplitude_nonphysical | variance_not_converged | final_amplitude_nonphysical | interval_not_two_sided | boundary_limited | multimodal_profile | profile_fit_exception |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 125 | 109 | 7,892 | 77 | 17 | 0 | 137 | 0 | 9 | 0 |
| 130 | 151 | 6,140 | 72 | 37 | 1 | 1,814 | 0 | 26 | 0 |
| 135 | 186 | 4,788 | 74 | 21 | 7 | 3,145 | 0 | 20 | 0 |
| 140 | 676 | 4,483 | 413 | 14 | 19 | 2,627 | 0 | 9 | 0 |
| 145 | 2,196 | 4,325 | 1,110 | 12 | 53 | 530 | 0 | 15 | 0 |
| 150 | 2,577 | 4,315 | 1,191 | 12 | 62 | 65 | 0 | 19 | 0 |
| 155 | 2,608 | 4,256 | 1,111 | 45 | 153 | 34 | 0 | 34 | 0 |
| 160 | 2,251 | 4,075 | 1,729 | 80 | 40 | 35 | 0 | 31 | 0 |
| 165 | 2,629 | 4,340 | 809 | 295 | 94 | 27 | 0 | 47 | 0 |
| 170 | 2,234 | 4,314 | 1,325 | 267 | 25 | 32 | 0 | 44 | 0 |
| 175 | 2,009 | 5,235 | 668 | 165 | 40 | 48 | 0 | 76 | 0 |
| 180 | 1,836 | 5,533 | 611 | 160 | 25 | 56 | 0 | 20 | 0 |
| 183 | 1,721 | 5,727 | 593 | 94 | 47 | 51 | 0 | 8 | 0 |
| 185 | 1,764 | 5,673 | 567 | 97 | 42 | 53 | 0 | 45 | 0 |
| 187 | 1,721 | 5,757 | 528 | 115 | 45 | 62 | 0 | 13 | 0 |
| 190 | 1,639 | 5,863 | 484 | 119 | 46 | 80 | 0 | 10 | 0 |
| 193 | 1,482 | 5,986 | 487 | 154 | 41 | 82 | 0 | 9 | 0 |
| 195 | 1,654 | 5,947 | 455 | 74 | 23 | 66 | 0 | 22 | 0 |
| 197 | 1,645 | 5,822 | 428 | 111 | 34 | 187 | 0 | 14 | 0 |
| 200 | 1,672 | 5,895 | 423 | 79 | 40 | 117 | 0 | 15 | 0 |
| 203 | 1,790 | 5,853 | 438 | 56 | 31 | 52 | 0 | 21 | 0 |
| 207 | 1,810 | 5,908 | 424 | 59 | 18 | 3 | 0 | 19 | 0 |
| 210 | 1,640 | 6,057 | 272 | 56 | 21 | 151 | 0 | 44 | 0 |

## Quadrant cutflow

| Quadrant | Candidates | Accepted fits | Final single-trap sites |
|---:|---:|---:|---:|
| 0 | 2,390 | 9,734 | 641 |
| 1 | 2,495 | 11,677 | 859 |
| 2 | 1,611 | 7,797 | 589 |
| 3 | 1,745 | 8,792 | 614 |

## Final orientation classes

| Class | Sites |
|---|---:|
| `ambiguous_sign_conflict` | 411 |
| `dual_response` | 341 |
| `insufficient_significant_temperatures` | 898 |
| `no_significant_temperature` | 3,886 |
| `single_orientation_negative` | 960 |
| `single_orientation_positive` | 1,743 |
| `structured_background_overlap` | 2 |

## Null-control check

- Independent ordinary-null aggregate empirical FPR: 0.093%.
- Maximum per-temperature ordinary-null FPR: 0.195%.
- Frozen calibration status: PASS.

## Visual checks

- `figures/signed_refit_intensity/accepted_intensity_fits.png`
- `figures/signed_refit_intensity/rejected_intensity_fits.png`

## Acceptance gate

- PASS: all artifacts use new versioned names and pinned hashes.
- PASS: every accepted fit maps to stored manifest source files, covariance, variance calibration, and profile.
- PASS: no temperature has an unexplained fit-processing collapse.
- PASS: ordinary-null false-positive rates remain within the frozen calibration gate.
