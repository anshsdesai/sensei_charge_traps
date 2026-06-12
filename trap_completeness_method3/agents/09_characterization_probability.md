# 09 Characterization Probability

## Objective

Convert per-temperature detection probabilities into
`P(characterized | tau_135, E)` using the Poisson-binomial `>= n_good` tail and amplitude
marginalization.

## Why This Matters

The experiment does not need a trap to be detected once; it needs enough good temperature points to
fit an Arrhenius model. This stage implements the multi-temperature selection effect that Method 1
and early Method 2 handled incorrectly.

## Inputs

- `agents/08_full_pdet_grid.md`
- `agents/05_amplitude_prior.md`
- `agents/01_hdf5_records_audit.md`
- `../dipole.py`

## Procedure

1. Define the final `tau_135` grid and observed-`E` grid/range.
2. For each `(tau_135, E)`, infer the implied `log_sigma` using
   `log_energy_cross_section(135, E, 0) - log(tau_135)`.
3. Compute `tau(T_i)` over all measured temperatures.
4. Interpolate `p_det(tau(T_i), A, T_i)` from the Stage 08 grid.
5. Compute the Poisson-binomial tail for `sum detections >= 4`.
6. Marginalize over the Stage 05 amplitude prior.
7. Save maps for `n_good = 4`; optionally repeat `n_good = 3` as a sensitivity variant.

## Required Checks

- Poisson-binomial implementation matches brute-force enumeration on small test vectors.
- Known characterized traps mostly lie in high-characterization-probability regions.
- The map clearly identifies the genuinely unbounded all-temperatures-out-of-band regime.

## Outputs

- `cache/09_characterization_probability_v1.h5`
- `cache/09_characterization_probability_summary.json`
- Optional plots under `cache/figures/`

## Stop Conditions

- Stop if `p_det` interpolation extrapolates over important regions without explicit handling.
- Stop if Poisson-binomial tests fail.
- Stop if amplitude marginalization is ambiguous after Stage 05.

## Results

Completed 2026-05-23T11:00:34-07:00.

Smoke command used before scaling:

```bash
wsl -e /home/ansh/miniforge3/envs/sensei_charge_traps/bin/python \
  /mnt/c/Users/Ansh/Projects/sensei_charge_traps/trap_completeness_method3/src/characterization_probability.py \
  --smoke
```

Production command used:

```bash
wsl -e /home/ansh/miniforge3/envs/sensei_charge_traps/bin/python \
  /mnt/c/Users/Ansh/Projects/sensei_charge_traps/trap_completeness_method3/src/characterization_probability.py
```

Input artifacts:

- `cache/08_pdet_grid_v1.h5`
- `cache/08_pdet_grid_summary.json`
- `cache/05_amplitude_prior_v1.npz`
- `cache/05_amplitude_prior_summary.json`
- `cache/01_records_ngood4.csv`
- `cache/01_records_ngood3.csv`
- `../dipole.py`

Output artifacts:

- `cache/09_characterization_probability_v1.h5`
- `cache/09_characterization_probability_summary.json`
- `cache/09_characterization_probability_smoke_summary.json`
- `cache/figures/09_characterization_probability_n4.png`
- `cache/figures/09_tau_oob_fraction.png`
- `src/characterization_probability.py`

Short numerical summary:

- Production map shape is `161 tau_135 x 121 E`, with `tau_135 = 1e-4` to `1e8 s` log-spaced
  and `E = 0.04` to `0.70 eV` linearly spaced.
- Stage 08 production grid consumed as `23 T x 55 tau x 35 A`; no Stage 08 pilot or smoke
  artifacts were used.
- Default Stage 05 prior used `2135` empirical depth samples and the stored `P_c(T)` factors,
  preserving the conditioning on observed high-confidence traps.
- Primary `n_good = 4` map has median `P = 0.9979`, mean `0.7348`, min/max `0.0`/`1.0`.
- Optional `n_good = 3` sensitivity map has median `P = 0.9996`, mean `0.7731`, min/max
  `0.0`/`1.0`.
- Runtime was `30.91 s` for production after a `6.13 s` smoke run.

Interpolation and out-of-grid policy:

- `p_det(T, tau, A)` interpolation is bilinear over `log(tau)` and linear amplitude at the exact
  measured Stage 08 temperatures.
- Tau values outside the Stage 08 tau grid (`2e-5` to `20 s`) are assigned `p_det = 0` for that
  temperature; no tau extrapolation is performed.
- Amplitudes below the Stage 08 grid are assigned `p_det = 0`; amplitudes above the grid would be
  edge-clipped. For the default prior, both amplitude below-grid and above-grid fractions were
  `0.0`.
- Across the broad production map, `52.57%` of temperature-level tau queries are outside the
  Stage 08 tau grid (`32.02%` above, `20.55%` below). This is expected because the map intentionally
  spans far beyond the observed high-probability band.

Required checks:

- `log_sigma = log_energy_cross_section(135, E, 0) - log(tau_135)` algebra check: PASS. Maximum
  `log(tau_135)` reconstruction error was `7.11e-15`.
- Poisson-binomial dynamic-programming tail vs brute-force enumeration: PASS. Maximum absolute
  difference over small test vectors was `1.39e-16`.
- Known characterized traps from `cache/01_records_ngood4.csv`: PASS. Of `2135` traps, `99.95%`
  have `P_4 >= 0.5`, `99.86%` have `P_4 >= 0.8`, median `P_4 = 0.99999996`; only one trap has
  `P_4 < 0.5`.
- `n_good = 3` sensitivity traps from `cache/01_records_ngood3.csv`: PASS. Of `2517` traps,
  `99.96%` have `P_3 >= 0.8`, median `P_3 = 0.999999997`.
- All-temperatures-out-of-band regime identified: PASS. `16.13%` of production grid points have
  all measured temperatures outside the Stage 08 tau band; under the explicit zero-tau-extrapolation
  policy their maximum `P_4` is `0.0`. The flagged all-out-of-band box spans `tau_135 >= 167.9 s`
  for `E = 0.04` to `0.4745 eV` within this grid.

Stop conditions encountered:

- None. Interpolation out-of-range handling is explicit, Poisson-binomial tests passed, and the
  Stage 05 default amplitude marginalization was unambiguous.

## Open Questions

- Stage 10 should decide whether to summarize final coverage as the full map only, representative
  `E` slices, or an additional average over an empirical observed `E` distribution.
- Stage 10 should propagate the Stage 05 faint-amplitude sensitivity variants if the completeness
  statement needs to move beyond the observed high-confidence amplitude prior.
