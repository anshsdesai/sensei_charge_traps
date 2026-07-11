# Signed Refit Profile-Tau Fitter Validation

- Validation version: `signed-refit-profile-validation-v1`
- Profile fitter version: `signed-refit-profile-tau-v1`
- Noise-model SHA-256: `d07dfec56bc8b5cad98282fe7a1c3c2fd3e5c157af660325338b7cd87535f39a`
- Random seed: `2026061305`
- Realizations per scenario: 300
- Acceptance status: **PASS**

## Model-conditional synthetic bias and coverage

| Scenario | T (K) | Q/R | tau true (s) | A true | Median bias (dex) | 68% coverage | Two-sided | Sign | Boundary | Multimodal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short-positive | 125 | 0/0 | 0.0003 | +0.180 | +0.0039 | 70.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| short-negative | 145 | 1/10 | 0.003 | -0.150 | -0.0052 | 68.3% | 100.0% | 100.0% | 0.0% | 0.0% |
| mid-positive | 183 | 2/20 | 0.03 | +0.140 | -0.0038 | 66.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| long-negative | 190 | 3/7 | 0.3 | -0.160 | +0.0028 | 69.3% | 100.0% | 100.0% | 0.0% | 0.0% |
| warm-positive | 203 | 0/24 | 0.001 | +0.120 | -0.0068 | 66.7% | 100.0% | 100.0% | 0.0% | 0.0% |
| warm-negative | 210 | 3/15 | 0.1 | -0.140 | +0.0065 | 64.0% | 100.0% | 100.0% | 0.0% | 0.0% |

- Aggregate 68% profile coverage under the assumed Gaussian covariance: 67.39%.
- Maximum absolute scenario median bias: 0.0068 dex.
- Minimum sign recovery: 100.00%.
- Minimum two-sided interval rate: 100.00%.
- Maximum 401-versus-1201-grid fitted-tau difference after continuous refinement: 5.74e-07 dex.

This is an algebra/model-conditional test: draws come from the same Gaussian
covariance used by the fitter. Empirical coverage against the observed
heavy-tailed PDF is tested separately in
`signed_refit_variance_validation.md` using real held-out residuals.

The tested tau values span 0.0003-0.3 s, both amplitude signs, warm and cold scans, 18- and 25-point dwell grids, nonzero pedestals, and exact regional covariance/null templates from the frozen v2 noise model.

## Boundary and multimodal behavior

- Out-of-window long-tau injection: fit tau=10.3 s; boundary_limited=True; upper_interval_limited=True.
- Low-signal noise realization: multimodal=True; competitive modes=4; delta chi-square=1.656.

These cases retain profile limits and flags instead of receiving a symmetric Gaussian tau error.

## Nonlinear curve-fit comparison

- Nonlinear attempts: 1080; failure rate 0.09%.
- Best-start full-covariance nonlinear versus profile median absolute difference: 4.19e-07 dex.
- Full-covariance nonlinear initial-start spread p95: 5.475 dex.
- Current diagonal-error nonlinear minus covariance-profile median: +0.0011 dex; p95 absolute difference 0.081 dex.

The profile fitter has no initial tau guess: it evaluates the complete log-tau grid and refines the global minimum. The full-covariance nonlinear comparison checks numerical agreement when its local optimizer reaches the same minimum; the diagonal comparison illustrates the effect of the legacy error treatment.

## Acceptance gate

- PASS: synthetic tau bias is below the predefined 0.03 dex limit.
- PASS: profile intervals have nominal coverage when the assumed Gaussian
  covariance is the data-generating model.
- PASS: the solution is independent of nonlinear initial guesses and stable under profile-grid refinement.
- PASS: boundary-limited and competing-minimum cases are explicitly flagged.

Step 5 does not assign a dipole-detection threshold. The empirical significance calibration remains Step 6.
