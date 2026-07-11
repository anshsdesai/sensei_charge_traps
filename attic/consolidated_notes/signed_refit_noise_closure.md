# Signed Refit Noise-Model Closure

- Closure version: `signed-refit-noise-closure-v2`
- Noise-model SHA-256: `d07dfec56bc8b5cad98282fe7a1c3c2fd3e5c157af660325338b7cd87535f39a`
- Held-out curves: 376832
- Whitened coordinates: 7815168
- Acceptance status: **PASS for covariance widths; nominal chi-square tails do not close**

## Global closure

- Whitened mean: -0.0008.
- Whitened width: 1.0265.
- Fraction `|z| > 3`: 0.4720%.
- Constant-model fraction `p < 0.05`: 7.7228%.
- Constant-model fraction `p < 0.01`: 2.2758%.
- Median reduced chi-square: 0.9925.
- P-value uniformity KS statistic: 0.0412 (`p=0`).

The KS p-value is reported as a sensitivity diagnostic; with this sample size, scientifically negligible deviations can be statistically significant.

## Preliminary pump-profile null statistic

- Trial grid: 256 log-spaced tau values from `min(dtph)/10` to `10*max(dtph)`.
- Fraction with maximum delta chi-square >= 11.83: 1.9417%.
- Empirical 95th/99th/99.9th percentiles: 9.205, 13.839, 24.136.

This directly demonstrates the look-elsewhere null distribution. It is diagnostic only; Step 6 will calibrate the final profile fitter and threshold.

## Temperature stratification

| T (K) | Width | |z|>3 | Max |corr| | p<0.05 | Trial >=11.83 | Trial p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 125 | 1.020 | 0.498% | 0.027 | 7.526% | 1.874% | 13.95 |
| 130 | 1.022 | 0.478% | 0.030 | 7.678% | 2.002% | 13.80 |
| 135 | 1.021 | 0.509% | 0.028 | 7.562% | 1.849% | 13.61 |
| 140 | 1.021 | 0.504% | 0.028 | 7.635% | 1.898% | 13.43 |
| 145 | 1.018 | 0.483% | 0.024 | 7.202% | 1.801% | 13.76 |
| 150 | 1.020 | 0.490% | 0.026 | 7.367% | 1.892% | 13.66 |
| 155 | 1.018 | 0.490% | 0.033 | 7.629% | 1.941% | 13.87 |
| 160 | 1.020 | 0.521% | 0.025 | 7.507% | 1.508% | 12.95 |
| 165 | 1.021 | 0.473% | 0.031 | 7.782% | 1.947% | 13.66 |
| 170 | 1.002 | 0.410% | 0.053 | 6.604% | 1.068% | 12.03 |
| 175 | 1.017 | 0.481% | 0.026 | 7.196% | 1.758% | 13.43 |
| 180 | 1.032 | 0.471% | 0.047 | 8.301% | 2.246% | 14.99 |
| 183 | 1.060 | 0.500% | 0.035 | 9.222% | 2.582% | 15.51 |
| 185 | 1.013 | 0.456% | 0.027 | 7.019% | 1.685% | 13.42 |
| 187 | 1.039 | 0.472% | 0.034 | 8.417% | 2.228% | 14.26 |
| 190 | 1.031 | 0.473% | 0.033 | 8.301% | 2.325% | 14.38 |
| 193 | 1.035 | 0.468% | 0.030 | 8.459% | 2.344% | 14.71 |
| 195 | 1.031 | 0.468% | 0.028 | 6.830% | 1.776% | 13.41 |
| 197 | 1.029 | 0.470% | 0.033 | 8.295% | 2.136% | 14.18 |
| 200 | 1.038 | 0.490% | 0.041 | 8.990% | 2.240% | 14.28 |
| 203 | 1.035 | 0.431% | 0.057 | 8.319% | 2.228% | 14.05 |
| 207 | 1.009 | 0.381% | 0.054 | 6.641% | 1.941% | 13.88 |
| 210 | 1.026 | 0.475% | 0.028 | 7.141% | 1.392% | 12.58 |

## Other strata

- Quadrant whitened widths: Q0=1.025, Q1=1.022, Q2=1.026, Q3=1.033.
- Region whitened widths: min 1.017, median 1.023, max 1.058.
- Individual covariance-cell width p05/median/p95: 0.986/1.019/1.063.
- Marginal `(temperature, dtph)` standardized-width p05/median/p95: 0.986/1.003/1.034.

Brightness quartiles:

| Quartile | Whitened width | p<0.05 | Trial >=11.83 |
|---:|---:|---:|---:|
| 1 | 1.025 | 7.490% | 1.881% |
| 2 | 1.021 | 7.428% | 1.811% |
| 3 | 1.024 | 7.688% | 1.871% |
| 4 | 1.035 | 8.282% | 2.204% |

## Warm scans

The 200, 203, 207, and 210 K rows are included explicitly in the table above. No separate warm-scan correction was applied.

## Superseded v1 diagnosis

The first closure attempt used an 8-pixel candidate halo and failed at 200, 203, 207, and 210 K. The excess correlation was localized to a small number of repeatable pump-like control curves 9-20 pixels from cataloged candidates, rather than to broad covariance miscalibration. The controls were regenerated with the independently meaningful 20-pixel deferred-charge scale, and the final v2 validation subset was not used to fit the covariance model or alter this acceptance gate.

## Tail and threshold policy

- No covariance inflation was fitted from this Step 4 validation data.
- The later R6 remediation uses a disjoint 64/64 split within these held-out
  controls to calibrate and evaluate explicit temperature scales.
- Any nonuniform analytical chi-square tails are retained as measured.
- The empirical trial-pump null distribution, stratified by scan, must be used by Step 6 rather than interpreting 11.83 as a universal 3-sigma cut.
- Step 4 does not alter the frozen Step 3 covariance matrices.

## Acceptance gate

- PASS: whitened residual widths meet the predefined practical ranges.
- PASS: temperature-level residual correlations meet the predefined limit.
- PASS: major-stratum tail rates are stable within the predefined limits.
- NON-CLOSURE: nominal 5%/1% analytical tails are too large (7.7%/2.3%
  globally); empirical Step 6 calibration is mandatory.

## Figures

- `figures/signed_refit_noise/closure_global_distributions.png`
- `figures/signed_refit_noise/closure_by_temperature.png`
