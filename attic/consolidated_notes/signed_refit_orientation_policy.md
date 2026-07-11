# Signed Refit Orientation Policy

- Validation version: `signed-refit-orientation-validation-v2`
- Policy version: `signed-refit-orientation-v2`
- Acceptance status: **PASS**

## Frozen policy

A temperature contributes an orientation only when its complete profile-tau search has empirical `p <= 0.001` under the Step 6 calibration. Insignificant temperatures are retained in the artifact but ignored for orientation consistency.

- Fewer than four significant temperatures: `insufficient_significant_temperatures`.
- At least four significant temperatures, all positive or all negative: `single_orientation_positive` or `single_orientation_negative`.
- At least four significant temperatures with exactly one accepted minority-sign fit: `ambiguous_sign_conflict`.
- At least two accepted positive and two accepted negative fits: `dual_response`.
- A vertical pair sharing either lobe pixel with the frozen persistent-horizontal morphology list: `structured_background_overlap`, regardless of sign consistency.
- Both conflict classes are excluded from a single-trap SRH fit and remain published as auditable classifications.

Step 9 must recompute these labels using its definitive accepted-temperature mask. It may remove an unaccepted temperature from the sign test, but it may not combine accepted opposite signs or restore a persistent-horizontal overlap to the single-trap class.

## Candidate results

- Step 7 candidate sites: 8,241.
- Single-orientation eligible: 3,313.
- Ambiguous one-sign-conflict sites: 419.
- Dual-response sites: 467.
- Persistent-horizontal pixel overlaps: 2.

| Classification | Sites |
|---|---:|
| `no_significant_temperature` | 3,552 |
| `insufficient_significant_temperatures` | 488 |
| `single_orientation_positive` | 2,145 |
| `single_orientation_negative` | 1,168 |
| `ambiguous_sign_conflict` | 419 |
| `dual_response` | 467 |
| `structured_background_overlap` | 2 |

## Injection efficiency

- Injection sites: 512.
- Sites with at least four detected true-signal temperatures: 512.
- Correct single-orientation efficiency conditional on that eligibility: 99.609%.
- Sign accuracy among accepted active-temperature fits: 100.000%.
- Injection sites overlapping persistent-horizontal morphology: 0.

| Scenario | Injections | Orientation eligible | Correct single rate |
|---|---:|---:|---:|
| `A=0.40,width=4` | 64 | 64 | 100.000% |
| `A=0.40,width=8` | 64 | 64 | 100.000% |
| `A=0.40,width=12` | 64 | 64 | 100.000% |
| `A=0.40,width=23` | 64 | 64 | 100.000% |
| `A=0.80,width=4` | 64 | 64 | 100.000% |
| `A=0.80,width=8` | 64 | 64 | 98.438% |
| `A=0.80,width=12` | 64 | 64 | 98.438% |
| `A=0.80,width=23` | 64 | 64 | 100.000% |

## Null and structured controls

| Class | Sites | Raw >=4 significant | Raw single orientation | Finder-selected union | Structured overlap | Final single orientation | Final rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ordinary` | 16,384 | 4 | 3 | 5 | 17 | 0 | 0.0000% |
| `horizontal_trigger_vertical` | 375 | 5 | 5 | 2 | 375 | 0 | 0.0000% |
| `near_defect` | 15,420 | 1 | 0 | 3 | 12 | 0 | 0.0000% |
| `horizontal_axis` | 375 | 95 | 38 | 375 | 375 | 0 | 0.0000% |

The candidate amplitudes in this report use the frozen null-covariance detection fit solely to calibrate the sign policy. They are not the Step 9 definitive amplitudes or uncertainties. The final artifact must use the signal-dependent covariance and rerun this classifier.

The horizontal-axis class is intentionally selected by a horizontal finder and therefore measures whether sign consistency alone rejects coherent non-pumping structure. It is not included in the vertical-catalog false-positive gate. Production null rates require entry through the frozen vertical finder union. Version v1 failed because it omitted that conditioning and treated all horizontal-axis stress sites as vertical candidates.

## Visual inspection

The sign-changing examples plot every fitted amplitude in gray and mark empirically significant temperatures in red. The inspected examples show coherent positive and negative temperature bands or a single isolated conflicting fit; no sign is silently converted.

- `figures/signed_refit_orientation/sign_changing_candidate_examples.png`
- `figures/signed_refit_orientation/orientation_signal_null_rates.png`

## Acceptance gate

- PASS: conditional injection efficiency is 99.609% (required >= 95.0%).
- PASS: accepted active-fit sign accuracy is 100.000% (required >= 99.0%).
- PASS: maximum end-to-end vertical-null single-orientation rate is 0.0000% (required <= 0.1%).
- PASS: classifier logic makes a single-trap label impossible when any accepted opposite-sign temperature is present.
- PASS: persistent response sharing a pixel in the non-pumping horizontal direction is retained as structured background, not a single vertical trap.
- PASS: ambiguous and dual-response labels and per-temperature signs remain stored for auditing.
