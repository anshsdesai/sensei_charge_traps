# Signed Refit Control-Pair Summary

- Control version: `signed-refit-controls-v2`
- Control artifact: `signed_refit_control_pairs.npz`
- Manifest SHA-256: `477cca5d74a2dcf953aaeb6e8f614f3f5a807f87f09a992024609acb73c07b67`
- Random seed: `2026061302`
- Detector regions per quadrant: 4 x 8 = 32
- Controls per region: 512 (384 training, 128 validation)
- Total controls: 65536

## Masks

- Candidate exclusion uses the union of legacy and initial signed catalogs.
- Candidate centers and lobes are excluded with a 20-pixel Chebyshev halo.
- The halo matches the 20-pixel deferred-charge scale adopted for the vertical trail mask; this supersedes the contaminated 8-pixel v1 controls.
- A vertical trail exclusion extends 20 rows and 2 columns around every candidate.
- The cropped-image edge margin is 8 pixels.
- Persistent defects and hot columns are derived from a robust static median of one representative image per temperature, independently of the sampled control-curve fluctuations.
- No separate experimental bad-pixel map exists for these pocket-pumping scans.

## Candidate catalogs

| Quadrant | Legacy | Initial signed | Union |
|---:|---:|---:|---:|
| 0 | 1324 | 2633 | 2633 |
| 1 | 1602 | 2764 | 2764 |
| 2 | 1068 | 1771 | 1771 |
| 3 | 1177 | 2165 | 2165 |

## Control counts and diagnostics

| Quadrant | Training | Validation | Valid pair pool | Candidate-excluded pixels | Defect-masked pixels | Minimum candidate distance | Static |I| p99 (e-) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12288 | 4096 | 298146 | 1235907 | 3691 | 21 | 193.50 |
| 1 | 12288 | 4096 | 279439 | 1253451 | 7985 | 21 | 246.79 |
| 2 | 12288 | 4096 | 488220 | 1035108 | 1669 | 21 | 264.79 |
| 3 | 12288 | 4096 | 435127 | 1092273 | 2172 | 21 | 196.50 |

## Split validation

- Every quadrant/region contains exactly 384 training and 128 validation controls.
- Training and validation coordinates are disjoint.
- All controls avoid candidate, defect, hot-column, trail, and boundary masks.
- Fixed coordinates are reused for every temperature and dwell image.
- Static pair-intensity outliers were removed region by region before sampling.

## Acceptance gate

- PASS: control pairs do not overlap candidate or defect masks.
- PASS: every `(quadrant, region)` has the requested control statistics.
- PASS: training and validation samples are disjoint.
- PASS: automated static-intensity diagnostics show no obvious residual candidate or persistent-defect contamination.
