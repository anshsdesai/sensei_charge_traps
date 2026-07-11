# Signed Refit Empirical Noise Model

- Noise-model version: `signed-refit-noise-v2`
- Artifact: `signed_refit_noise_model.h5`
- Artifact SHA-256: `d07dfec56bc8b5cad98282fe7a1c3c2fd3e5c157af660325338b7cd87535f39a`
- Covariance matrices: 2944
- Mapping: exact temperature and quadrant, then the candidate's 4 x 8 cropped-detector region.
- Training controls per covariance: 384.
- Held-out validation controls per covariance: 128; not used here.

## Estimator

1. Remove each control pair's median across the dwell scan.
2. Remove the training ensemble's median null template at each dwell point.
3. Winsorize each dwell coordinate at 5.0 robust sigma.
4. Estimate Oracle Approximating Shrinkage covariance toward a scaled identity.
5. Apply a relative eigenvalue floor of 1e-08.

Classical and robust sample covariance matrices are also stored for audit.

## Numerical validation

- Condition number: min 1.09, median 10.7, max 613.
- OAS shrinkage: min 0.00593, median 0.0599, max 0.911.
- Minimum regularized eigenvalue: 372.786 (electrons)^2.
- Null-template pump projection delta-chi-square max/p95: 0.121/0.016.
- Null-template conditional |z| max/p95: 0.347/0.126.
- Every covariance is finite, positive definite, invertible, and matched to its scan's unique sorted dtph grid.

## Observed dependence

- Per-dwell sigma (e-): min 27.1, median 35.1, p95 41.9, max 47.1.
- Maximum absolute off-diagonal correlation: min 0.0119, median 0.516, p95 0.919, max 0.969.
- Robust/classical covariance fractional Frobenius difference: min 8.36e-10, median 7.96e-09, p95 0.203, max 0.978.
- Spearman residual-RMS versus static pair brightness: min -0.146, median 0.0172, p95 0.105, max 0.208.
- Spearman residual-RMS versus generated-charge background: min -0.164, median 0.0134, p95 0.114, max 0.221.

The HDF5 file stores per-temperature, quadrant, region, and dwell sigma/correlation values so temperature, quadrant, region, and dtph dependence remain explicit.

## Acceptance gate

- PASS: every production scan has an invertible regional covariance model.
- PASS: condition numbers and shrinkage strengths are recorded.
- PASS: detector-region and acquisition dependence are retained rather than collapsed into one scalar.
- PASS: the empirical null template has negligible projection onto every scan's pump-shape family.
- PASS: the model was frozen without examining candidate acceptance changes.

Held-out residual closure is intentionally deferred to Step 4.
