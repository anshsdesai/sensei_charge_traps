# Inverse-probability (Horvitz–Thompson) completeness correction — design & migration plan

**Status:** approved direction, not yet implemented. Written 2026-07-10 after the
axis-dependence investigation (probes in [probes/](probes/); evidence commit
`a34bbbc` is the pre-change checkpoint). Companion doc-of-record page:
[notebook/completeness_efficiency.qmd](../notebook/completeness_efficiency.qmd)
(to be rewritten in Stage 11d below).

This document is written to Method-3 documentation standard: a fresh reader
should be able to (a) understand why the current correction is invalid,
(b) understand the replacement estimator, and (c) execute the migration from
the file/line specs alone.

---

## 1. What is wrong with the current correction (measured, not argued)

The shipped correction (`figure_utils.py:919`) collapses the Method-3 2D map
`P(characterized | τ₁₃₅, E)` into a 1D efficiency by averaging over the
**global** catalog energy distribution, identically in every τ slice:

```python
default_curve = np.array([np.interp(observed_E, E_grid, row).mean() for row in p4_map])
```

then divides the τ₁₃₅ histogram (and its Poisson 90% UL) by that curve
(`_completeness_correction`, `figure_utils.py:1274`). Three defects, all
measured on the live `minimal_caldet` v1 artifacts
(3,798 characterized traps; probes 1–3, outputs reproduced in §6):

**D1 — the factorization assumption is contradicted by the catalog itself.**
"Every vertical slice has the global E distribution" requires E ⊥ τ₁₃₅.
Measured: corr(E, log₁₀ τ₁₃₅) = 0.58, ridge slope 0.0298 eV/decade ≈ the
constant-σ SRH slope kT·ln10 = 0.0268 eV/decade at 135 K. The catalog lies
along constant-σ lines; the per-slice E distribution slides up with τ.

**D2 — the correction is axis-dependent (the assumption does the work).**
Conditioning instead on "hidden traps share the observed σ distribution"
(= sliding each dot along its own constant-σ line into the slice, exactly the
correction one would write in (E, σ) axes) changes ε(3×10⁵ s) from 0.054 to
0.91 and the driver-band (τ ∈ [3×10⁵, 5×10⁷] s) UL population from 13,466 to
189 — a factor ~70. A correction whose answer moves 70× under an equally
plausible re-parameterization is reporting the prior, not the data.

**D3 — the simulated UL population is excluded by the catalog (~350×).**
`CCD.__init__` (`ccd_simulation.py:1205-1218`) assigns each seeded trap a σ
resampled from the *measured* pairs nearest in log τ. Replicating that
assignment, inverting SRH for the implied E, and reading the Method-3 grid
gives mean P(characterized) = **0.76** for the as-simulated driver-band UL
population: 13,466 seeded traps ⇒ 10,234 predicted catalog entries; the
catalog contains **29**. The correction assumes the hidden traps are
undetectable (ε = 0.001–0.05) while the simulation builds them detectable
(P = 0.5–0.96). No single hypothesis about the hidden population supports the
published exposure-dependent band:

| hypothesis for hidden tail traps | consequence |
|---|---|
| same σ as observed (deep, E ≈ 0.45 eV) | would have been seen; ≤ ~190 at 90% CL |
| tiny σ (10⁻¹⁹–10⁻²¹ cm², low E) | consistent with hiding, but capture ∝ σ ⇒ inert in operation |
| measured σ but faint (amplitude) | bounded by faint-by-2/4 × same-σ ε ≈ 300–1,800 |

Baseline (characterized-only) simulation results are unaffected.

---

## 2. The replacement estimator

### 2.1 The object we already have: P(characterized | τ₁₃₅, E)

Nothing new is built. Method-3 stages 02–09 already produce, at every point of
the (τ₁₃₅, E) plane, the probability that a trap **with those physical
parameters** would have ended up in the catalog: SRH maps (τ₁₃₅, E) → τ(T) at
all 23 measured temperatures; the injection-recovery grid gives per-temperature
p_det(T, τ(T), A); a Poisson-binomial tail gives P(≥4 good fits); the observed
amplitude prior (D_t, P_c(T)) is marginalized; `ENERGY_FIT_SURVIVAL = 0.972`
is applied once in `load_method3`. E matters because it sets the Arrhenius
slope — i.e., *which* temperatures bring τ(T) into the measurable window. Two
traps with identical τ₁₃₅ can have P ≈ 0.95 (deep, normal σ) or P ≈ 0
(shallow, tiny σ).

### 2.2 Horvitz–Thompson weighting

Model the catalog as a thinned point process: the true population is N_true
traps at parameter points x_i = (τ_i, E_i); each enters the catalog
independently with known probability P(x_i). For any region R:

    N̂(R) = Σ_{observed i ∈ R} 1 / P(x_i)

is unbiased: each caught trap "stands for" the 1/P traps of its exact kind, of
which we caught on average P·(1/P) = 1. Formally
E[N̂(R)] = Σ_{true i ∈ R} P(x_i)·(1/P(x_i)) = N_true(R), **regardless of how
the true population is distributed inside R** — the estimator never needs the
hidden population's p(E|τ), because every weight is evaluated at an observed
trap's own coordinates. Variance: Var[N̂] = Σ (1−P)/P² (binomial thinning),
negligible here (min P over the catalog = 0.255, only 4/3,798 dots below 0.5).

Properties that fix D1–D3 by construction:

- **Coordinate-invariant** (fixes D2): the weight is a scalar attached to a
  trap, not to an axis. Any histogram built from the weighted dots — τ
  marginal, σ marginal, joint — is consistent; the "which axes for fig. 4"
  question dissolves.
- **No factorization assumption** (fixes D1): the observed per-dot (τ, E)
  values replace the assumed per-slice distribution.
- **Sim-consistent seeds** (fixes D3): the seed resamples *whole observed
  traps* — (τ_i, σ_i) jointly, probability ∝ w_i — so every simulated trap is
  a copy of a caught trap upweighted by its miss rate. A detection-
  inconsistent (τ, σ) pairing can no longer be constructed.

Its honest limitation: HT says nothing where P ≈ 0 (no dots exist to carry
weights). That region is handled by the explicit-domain upper limit (§4), not
smuggled into the point estimate.

### 2.3 Measured result on the current catalog (probe 3)

| region | raw | HT-corrected |
|---|---|---|
| total | 3,798 | **3,919** |
| τ > 10⁵ s | 37 | 43 |
| driver band [3×10⁵, 5×10⁷] s | 30 | **34** |

(vs. the shipped effcorr 7,447 total / 3,468 driver band.)

---

## 3. Stage 11a — point estimate (replaces `effcorr`)

1. **`figure_utils.py`**: add `ht_weights(m3) -> np.ndarray` — per-trap
   w_i = 1/P(τ_i, E_i), bilinear on the stage-09 grid in (log₁₀ τ, E), E
   clipped to grid edges, survival factor already in `p4_map` per the
   `load_method3` convention. Refuse (raise) if any P < 0.10 — currently none;
   report count in [0.10, 0.25). Add `weights`, `N_ht = weights.sum()` to the
   bundle. Reference implementation: [probes/ht_2d_correction.py](probes/ht_2d_correction.py).
2. **Seed builder**: extend the pairs file
   (`trap_tau135_sigma_pairs_minimal_caldet.npz`, from `make_trap_pairs.py`)
   with a `weight` column aligned per trap. ⚠ Open fact to verify first: row
   identity between `01_records_minimal_caldet_ngood4.csv` and the pairs file
   (same 3,798 traps? same order? join on (quadrant,row,col) if not).
3. **`ccd_simulation.py` `CCD.__init__` (lines 1194–1218)**: replace the
   two-step draw (τ-bin from histogram + nearest-in-τ σ) with one weighted
   categorical draw over the pairs (p ∝ w_i), keeping the per-trap log-jitter
   of τ within its histogram bin **or** its fit error (decision §7). Persist a
   `seed_mode` HDF5 attr (`legacy_hist` | `weighted_pairs`) and reject
   mixed-mode aggregation in the per-file idempotency guard (precedent:
   `v3_phase_fraction`).
4. **`run_ccd_simulation.py` / `run_campaign.py`**: new population label
   `htcorr` (do not reuse `effcorr` — old outputs must not mix):
   trap-count scale = Σw / n_detected_dipoles, seeds = weighted pairs.
   `baseline` unchanged.

## 4. Stage 11b — upper limit (replaces the UL fill)

1. **Domain**: the UL is a statement about traps with (τ, E) such that the
   implied σ lies in a declared domain D. Default proposal: observed σ range
   trimmed to physical values, log₁₀σ ∈ [−19, −14] (**open decision** —
   literature floor vs observed range; see §7).
2. **Per-bin UL**: ε_D(τ) = mean over dots of P(τ, E_i(τ)) with the same-σ
   transport E_i(τ) = E_i + kT·ln(τ/τ_i) restricted to D (reference:
   [probes/efficiency_axis_sensitivity.py](probes/efficiency_axis_sensitivity.py));
   UL(bin) = `gamma.ppf(0.90, n+1)` / ε_D, bins with ε_D ≥ 10⁻³ within the
   existing [6×10⁻⁵, 5×10⁷] s bounds. Expected: total ≈ 4,465, driver ≈ 190.
3. **Seed consistency**: UL traps are seeded at the same (τ, σ) the limit was
   computed for — σ from the transported dots, never from nearest-measured-τ.
4. **Faint variants**: recompute the stage-08 amplitude marginalization with
   the faint-by-2 / faint-by-4 priors → variant P grids → variant weights and
   ε_D. Reported as stated conditional variants (as the paper already does),
   expected driver band ≈ 300–1,800.
5. **Beyond-domain closure (optional but recommended)**: one campaign scenario
   seeding the σ < domain population *with its detection-consistent tiny σ*
   (capture α scales ∝ σ) to demonstrate inertness — converts the unbounded
   corner into a bounded sentence: "traps hidden from pumping by small σ are
   correspondingly inactive during operation."

## 5. Stage 11c — validation (acceptance criteria)

1. **Closure test** (extend `src/naive_efficiency_closure.py`): draw a
   synthetic truth (e.g., the HT-corrected population itself), thin each trap
   by P(x), apply HT to the survivors. Accept: total recovered within 2σ
   Poisson; weighted τ marginal vs truth KS p > 0.05; repeat 100 realizations.
2. **Detection-consistency audit**: rerun
   [probes/ul_population_detectability.py](probes/ul_population_detectability.py)
   against the **new** seeds. Accept: predicted catalog entries from every
   seeded population statistically compatible with observed counts per band
   (the current seeds fail this at ~350×).
3. **Sim A/B**: rerun the trap-only bracket (zedr) with `htcorr` and the new
   UL; document old-band → new-band in the notebook. Expected qualitative
   outcome: exposure-dependent excess consistent with the baseline null.

## 6. Stage 11d — documentation & paper

- Rewrite the efficiency section of `notebook/completeness_efficiency.qmd`
  around §2 of this doc (grid → weights → domain-UL), with the D1–D3 evidence
  tables and probe provenance.
- `paper/paper.tex` §Measurement Completeness + band figure + abstract/
  conclusion claims ("substantial population of long-lived traps") rewritten;
  affected numbers: efficiency figure, corrected/UL histograms, densities
  2.34×10⁻³ / 2.86×10⁻³, the exposure-dependent band. All edits in `\claude{}`
  markup.
- Probe outputs (2026-07-10, minimal_caldet v1): production reproduction
  7,447 / 17,978 (campaign bracket 7,238 / 17,474, Δ few %); HT totals §2.3;
  same-σ ε and UL §1-D2; detectability audit §1-D3.

## 7. Open decisions (Ansh)

1. UL σ-domain: observed range vs literature-informed log₁₀σ ∈ [−19, −14]?
2. Faint-by variants: stated (as now) or folded into the headline UL?
3. τ jitter in the weighted-pairs seed: within histogram bin (status quo
   visual) or within per-trap fit error (more honest)?
4. Legacy analysis flavor: migrate alongside minimal_caldet, or freeze?
5. Keep the old ε(τ) figure in the paper as a "naive marginal" comparison?

## 8. Execution order

| step | touches | gated by |
|---|---|---|
| verify records↔pairs row identity | probe only | — |
| 11a weights + bundle | figure_utils.py | — |
| 11a seed builder | make_trap_pairs.py | row identity |
| 11a CCD sampling + seed_mode attr | ccd_simulation.py, run_ccd_simulation.py | seed builder |
| 11b domain-UL + variants | figure_utils.py, src/ (stage 08 rerun for faint grids) | 11a |
| 11c closure + audits | src/naive_efficiency_closure.py, probes/ | 11a/b |
| 11c bracket rerun | run_campaign.py (cluster) | all above |
| 11d notebook + paper | notebook/, paper/ | 11c numbers |
