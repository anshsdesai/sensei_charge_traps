# Signed Refit Real-Candidate Variance Closure

- Closure version: `signed-refit-real-candidate-variance-closure-v2`
- Noise-model SHA-256: `d07dfec56bc8b5cad98282fe7a1c3c2fd3e5c157af660325338b7cd87535f39a`
- Candidate-coordinate SHA-256: `a6381cd46454b79ab0bfe1888e3472fd7b6bafe97733b5527e506eeda9dbd6c7`
- Orientation-validation SHA-256: `ca98e27569103373c2d8ba3fe15dba1931788d4777b87f19b93daebf513fcf6e`
- Lobe-order contract: `I=(image[row,col]-image[row-1,col])/2`
- Electronization scale: 400.0 ADU/e- (global).
- Acceptance status: **PASS**
- Step 8 orientation-only single-trap sites: 3,313.
- Physical-amplitude single-trap sites used here: 3,109.

## Split and selection

Candidate coordinates are split by a fixed coordinate hash before their curves are examined. The sample is restricted to the frozen Step 8 single-trap policy after requiring the paper-model coefficient `|D_t P_c| <= 1`: at least four empirically significant physical temperatures, no accepted opposite-sign temperature, and no persistent-horizontal pixel overlap. No residual-goodness criterion enters this selection. Each split is sampled uniformly across fitted-amplitude quartiles at every temperature.

- Calibration fits used for overdispersion: 1999.
- Evaluation input/eligible fits: 11403/9526 (83.54%).

## Pumping overdispersion

- Null-fit amplitude edges: 0.2497, 0.4822, 0.7387.
- Measured multipliers `phi`: 8.3117, 16.0634, 30.0039, 23.0403.
- Any upper-bound hit: False.

The frozen amplitude-bin multiplier applies only to `3000 q(1-q)`. The bin is chosen from the null-covariance detection amplitude before the signal-dependent refit. Null covariance scale and excess pair-shot variance remain separate terms.

- Method: two-fold coordinate cross-fit; reported final edges are arithmetic fold means and final factors are geometric fold means.

| Held-out split | Null-fit amplitude bin | Median |A| | Fits | phi | Refit width |
|---:|---:|---:|---:|---:|---:|

| 0 | 1 | 0.1638 | 223 | 9.1211 | 0.999 |
| 0 | 2 | 0.3763 | 243 | 15.3086 | 0.996 |
| 0 | 3 | 0.5724 | 277 | 30.0039 | 1.000 |
| 0 | 4 | 0.8904 | 273 | 19.1758 | 1.000 |
| 1 | 1 | 0.1609 | 209 | 7.5742 | 1.002 |
| 1 | 2 | 0.3745 | 245 | 16.8555 | 1.000 |
| 1 | 3 | 0.5659 | 267 | 30.0039 | 1.000 |
| 1 | 4 | 0.8891 | 262 | 27.6836 | 1.001 |

## Evaluation closure

- Aggregate residual width: 0.993.
- Aggregate nominal `p<0.05` rate: 9.54%.
- Fitted-amplitude width spread: 0.080.

| Fitted-amplitude quartile | Median |A| | Fits | Closure width | p<0.05 |
|---:|---:|---:|---:|---:|
| 1 | 0.1745 | 2382 | 1.004 | 9.99% |
| 2 | 0.4371 | 2381 | 1.004 | 10.25% |
| 3 | 0.6368 | 2381 | 1.022 | 11.68% |
| 4 | 0.9190 | 2382 | 0.942 | 6.26% |

## Acceptance gate

- PASS: real-candidate residual width closes in every fitted-amplitude quartile.
- PASS: no material amplitude trend remains after the measured pumping term.
- PASS: amplitude-stratified overdispersion was fixed on disjoint coordinates.

Nominal chi-square tails remain a goodness-of-fit diagnostic; Step 6 empirical calibration continues to define detection significance.
