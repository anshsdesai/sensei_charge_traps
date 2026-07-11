# Signed Refit Real-Candidate Variance Closure

- Closure version: `signed-refit-real-candidate-variance-closure-v1`
- Noise-model SHA-256: `d07dfec56bc8b5cad98282fe7a1c3c2fd3e5c157af660325338b7cd87535f39a`
- Candidate-coordinate SHA-256: `6c885bb20686854272727e635e6317a5de012acc1b07bf0732295488ed77e71b`
- Lobe-order contract: `I=(image[row,col]-image[row-1,col])/2`
- Electronization scale: 400.0 ADU/e- (global).
- Acceptance status: **FAIL**

## Split and selection

Candidate coordinates are split by a fixed coordinate hash before their curves
are examined. The original Step 6 empirical threshold selects pump-shaped
curves. A candidate must be significant at four or more temperatures with at
least 80% orientation consistency, and its current sign must match that dominant
orientation. Each split is sampled uniformly across fitted-amplitude quartiles
at every temperature.

- Calibration fits used for overdispersion: 1200.
- Evaluation input/eligible fits: 5727/4647 (81.14%).

## Pumping overdispersion

- Calibration closure width with independent cycles (`phi=1`): 1.812.
- Measured multiplier `phi`: 20.0000.
- Upper-bound hit: True.

The multiplier applies only to `3000 q(1-q)`. Null covariance scale and excess pair-shot variance remain separate terms.

## Evaluation closure

- Aggregate residual width: 1.056.
- Aggregate nominal `p<0.05` rate: 7.94%.
- Fitted-amplitude width spread: 0.302.

| Fitted-amplitude quartile | Median |A| | Fits | Closure width | p<0.05 |
|---:|---:|---:|---:|---:|
| 1 | 0.1955 | 1162 | 1.038 | 8.86% |
| 2 | 0.4385 | 1161 | 1.130 | 7.41% |
| 3 | 0.5338 | 1162 | 0.862 | 3.27% |
| 4 | 0.8339 | 1162 | 1.164 | 12.22% |

## Acceptance gate

- **FAIL:** overdispersion hit the calibration upper bound; amplitude quartile 2 width 1.130; amplitude quartile 3 width 0.862; amplitude quartile 4 width 1.164; amplitude quartile 4 p05 0.122; amplitude width spread 0.302

Nominal chi-square tails remain a goodness-of-fit diagnostic; Step 6 empirical calibration continues to define detection significance.
