# Signed Refit Candidate-Variance Validation

- Validation version: `signed-refit-candidate-variance-validation-v1`
- Variance model: `signed-refit-candidate-variance-v1`
- Signal-dependent fitter: `signed-refit-profile-tau-signal-variance-v1`
- Noise-model SHA-256: `d07dfec56bc8b5cad98282fe7a1c3c2fd3e5c157af660325338b7cd87535f39a`
- Acceptance status: **PASS**

## Physical covariance

For the paper model, the per-cycle transfer probability is `q=|A|[exp(-t/tau)-exp(-8t/tau)]`. The added pumping variance is `3000 q(1-q)`, because a transferred charge contributes directly to `I=(a-b)/2`. The null covariance is retained for read/background noise. An optional measured excess pair-charge term adds `max((a+b)_candidate-(a+b)_control,0)/4` to the diagonal.

## Null covariance scale

The first 64 held-out controls in every region calibrate one conservative temperature scale; the other 64 evaluate it. Scale factors are never estimated from candidates.

| T (K) | Calibration width | Covariance scale | Evaluation width | p<0.05 |
|---:|---:|---:|---:|---:|
| 125 | 1.017 | 1.0344 | 1.005 | 6.628% |
| 130 | 1.026 | 1.0528 | 0.991 | 5.676% |
| 135 | 1.022 | 1.0453 | 0.997 | 6.055% |
| 140 | 1.021 | 1.0418 | 1.000 | 5.933% |
| 145 | 1.016 | 1.0331 | 1.004 | 6.140% |
| 150 | 1.020 | 1.0401 | 1.001 | 6.091% |
| 155 | 1.018 | 1.0367 | 1.000 | 6.421% |
| 160 | 1.021 | 1.0431 | 0.996 | 5.469% |
| 165 | 1.021 | 1.0420 | 1.000 | 6.177% |
| 170 | 1.000 | 1.0000 | 1.005 | 6.934% |
| 175 | 1.017 | 1.0341 | 1.001 | 6.055% |
| 180 | 1.028 | 1.0558 | 1.009 | 6.128% |
| 183 | 1.034 | 1.0698 | 1.050 | 6.042% |
| 185 | 1.013 | 1.0258 | 1.001 | 5.994% |
| 187 | 1.034 | 1.0688 | 1.010 | 4.919% |
| 190 | 1.031 | 1.0634 | 1.000 | 5.750% |
| 193 | 1.036 | 1.0732 | 0.998 | 5.444% |
| 195 | 1.013 | 1.0258 | 1.037 | 6.091% |
| 197 | 1.028 | 1.0577 | 1.001 | 5.615% |
| 200 | 1.037 | 1.0760 | 1.001 | 5.591% |
| 203 | 1.031 | 1.0636 | 1.008 | 5.554% |
| 207 | 1.007 | 1.0136 | 1.004 | 5.811% |
| 210 | 1.034 | 1.0697 | 0.984 | 4.407% |

The scale corrects the mild variance deficit from the robust finite-sample covariance estimate. Heavy analytical tails remain visible and still require the empirical Step 6 detection calibration.

## Real-residual injection

- Evaluation fits: 1152.
- Aggregate 68% interval coverage: 59.72%.
- Characterization-eligible fits: 707 (61.37%); eligible coverage 68.60%.
- Aggregate median tau bias: +0.0039 dex.
- Variance-iteration convergence: 99.39%.

| |A| true | Fits | Detection | Identifiable | All coverage | Eligible coverage | Eligible bias (dex) | Closure width |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 384 | 0.78% | 0.52% | 42.19% | 50.00% | -0.1475 | 0.976 |
| 0.10 | 384 | 84.38% | 83.59% | 67.45% | 67.60% | +0.0039 | 0.989 |
| 0.30 | 384 | 100.00% | 100.00% | 69.53% | 69.53% | +0.0008 | 0.996 |

Closure binned by fitted amplitude:

| Fitted-amplitude quartile | Median |A_fit| | Fits | Coverage | Closure width |
|---:|---:|---:|---:|---:|
| 1 | 0.0926 | 177 | 68.36% | 0.988 |
| 2 | 0.1140 | 176 | 67.05% | 0.994 |
| 3 | 0.2921 | 177 | 71.75% | 0.998 |
| 4 | 0.3128 | 177 | 67.23% | 0.992 |

Signals are injected onto untouched evaluation-half control curves. Each injection also draws the binomial transfer count and, where specified, independent lobe shot noise. This replaces the circular Gaussian-only Step 5 coverage claim.

Coverage is accepted only for fits that pass the frozen Step 6 detection threshold, have a two-sided interval, and are neither boundary-limited nor multimodal. Weak curves remain in the diagnostic table but their tau values are classified as non-characterizable and must not enter Step 10.

## Template regression guard

- Template pump projection delta-chi-square max/p95: 0.121/0.016.
- Conditional template |z| max/p95: 0.347/0.126.

## Off-diagonal correlation diagnosis

- Stored maximum absolute correlation: 0.969.
- Median absolute held-out correlation at each stored maximum pair: 0.538.
- Median full off-diagonal train/evaluation matrix correlation: 0.774.
- For stored |rho|>=0.8, same-sign held-out fraction: 1.0; held-out |rho|>=0.5 fraction: 0.9972602739726028.

A per-image scalar common mode is removed by the regional dwell template and cannot create covariance across control coordinates after centering. Split-coordinate reproducibility instead tests whether the large modes are persistent detector-coordinate/row-response structure.

## Gain provenance

- Selected images: 477.
- Sidecars missing: CSV=0, XML=0.
- Empty calibration CSVs: 5.
- Populated per-image gain fits: 0.
- XML default gains: [200.0].
- Pipeline global electronization scale: 400.0.
- Status: **REQUIRES_EXTERNAL_GLOBAL_CALIBRATION_CONFIRMATION**.

The sidecars exist but do not contain fitted per-image gains for this manifest; their XML value is a fallback, not an image-by-image measurement. The established MINOS global scale is therefore retained pending external calibration provenance.

## Acceptance gate

- PASS: independently scaled held-out null widths close.
- PASS: real-residual injection tau bias and coverage close across the characterization-eligible amplitude range.
- PASS: fitted-amplitude residual widths do not develop a signal trend.
- PASS: the null-template pump projection remains negligible.
