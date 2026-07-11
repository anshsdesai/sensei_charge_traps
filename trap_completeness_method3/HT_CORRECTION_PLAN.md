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

### Bootstrap for a fresh session (read first)

- **Environment**: conda env `sensei_charge_traps_new` (NOT the name inside
  `requirements.yaml`). Smoke test — from the repo root, run
  `conda run -n sensei_charge_traps_new python trap_completeness_method3/probes/ht_2d_correction.py`
  and confirm it prints `HT-corrected total population: 3919.3` (3,798 dots,
  min P 0.255). If that reproduces, every input artifact this plan depends on
  is present and readable.
- **Line references** in this doc are pinned to commit `a34bbbc`
  (`git show a34bbbc:<file>` if the working tree has drifted).
- **Conventions**: probes/one-off scripts go in gitignored `claude_scripts/`,
  not the repo; never regenerate seed NPZ/H5 artifacts with guessed arguments
  (verify read-only; Ansh reruns the producing notebook/stage himself); Ansh
  launches long simulations and cluster campaigns himself; all `paper/paper.tex`
  edits wrapped in `\claude{}`; git commits carry **no** Claude attribution or
  Co-Authored-By trailer.
- **Input artifacts** (all under `trap_completeness_method3/cache/` unless
  noted): stage-09 grid `09_characterization_probability_minimal_caldet_v1.h5`
  (`grid/tau_135_seconds` 161 pts [2e-5, 1e9] s, `grid/E_eV` 121 pts [0, 0.7] eV,
  `results/p_characterized_n_good_4`); catalog
  `01_records_minimal_caldet_ngood4.csv` (3,798 rows; columns used: `E_eV`,
  `tau_135_seconds`); repo-root `tau_at_135k_hist_minimal_caldet.npz` and
  `trap_tau135_sigma_pairs_minimal_caldet.npz` (keys `tau135`, `sigma`,
  `energy`; 3,798 entries). `ENERGY_FIT_SURVIVAL = 0.972` is applied to
  "characterized" maps exactly once, in `load_method3`
  (`figure_utils.py:172,909`) — never a second time downstream.

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

**How robust is D3 to the poorly constrained cross-sections?** (Raised by Ansh
2026-07-10; per-trap fitted σ is noise-dominated — the pairs file spans
6.8×10⁻²⁶–5.2×10⁻¹² cm², 14 decades, because σ is ~99.7% degenerate with E.)
Three layers: (i) the 0.76 detectability is a statement about the population
*as simulated* — the sim literally runs the fitted pairs, so it holds
regardless of whether fitted σ equals true σ; (ii) fit noise moves a trap
along its constant-τ₁₃₅ line, and the probe already averaged over the full
14-decade fitted scatter — per-bin P stayed 0.5–0.96 because even the smallest
fitted candidates (~10⁻¹⁹ cm²) imply E ≈ 0.36–0.40 eV, at or above the
detectability edge; flipping the conclusion requires *true* tail σ ≥2 orders
below the smallest fitted values; (iii) that possibility is not dismissed — it
is exactly the "tiny-σ branch," which the plan brackets with the domain UL
(§4.1) and the required inertness run (§4.5) instead of assuming it away.

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

### 2.4 What this does and does not estimate (the "clones" objection)

The completeness question splits into two regimes that demand different tools:

- **Partial visibility (P meaningfully between 0 and 1).** Traps of some kind
  were catchable but sometimes missed. A data-driven estimate exists: catch n,
  estimate n/P. HT formalizes exactly this — a trap caught at P = 0.3 stands
  for ~3.3 traps, so the corrected composition shifts toward the kinds we
  barely catch. This is the valid core of the paper's "traps near the
  low-probability border imply a hidden population."
- **Blindness (P ≈ 0).** Traps that would never have been caught (τ(T) outside
  the window at all 23 temperatures, or amplitude far below threshold). No
  data-driven estimate exists here **in principle** — under any method. Any
  finite number produced for this regime is the prior, not the data; the old
  correction's 13,466 was 29 observed traps × 1/ε levers of 20–1000 acting on
  an *assumed* low-E population of which the catalog contains zero examples.

So the point estimate is deliberately "characterized traps reweighted" — that
is all the data can support pointwise. The hidden-population question is then
answered in layers, each defensible: (1) unseen traps of *catchable* kinds are
bounded by the domain UL (§4) — e.g. ~190 allowed in the driver band against
an HT point estimate of 34, a genuine ~6× allowance, not clones; (2) hidden-
because-faint, model-following traps are covered by the faint-prior variant
grids (§4.4); (3) hidden-because-invisible (tiny σ) traps are unbounded in
*number*; their *effect* is expected suppressed because capture is ∝ σ in the
transfer windows (exact for single-e packets within the shipped model), but
large-packet filling saturates and only the **required** inertness run (§4.5)
turns "suppressed" into a number. HT does not claim blind-corner traps don't
exist; it refuses to count them and hands them to layers (2)–(3).

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
2. **Seed builder**: extend `make_trap_pairs.py` to emit a `weight` key in the
   pairs NPZ. **No records↔pairs join is needed** (verified: `make_pairs`,
   `make_trap_pairs.py:24-74`, re-derives (τ₁₃₅, σ, E) per trap from the fit
   HDF5 and saves no coordinates) — instead compute each weight from the
   pairs' **own** (τ₁₃₅, E) against the stage-09 grid (add a `--stage09-h5`
   argument). This makes the seed self-consistent by definition: the weight is
   evaluated at exactly the (τ, E) the simulated trap will carry.
   Consistency assert: Σweight from the pairs file vs Σweight from the
   records-CSV bundle path (step 1) must agree within ~1% (both describe the
   same 3,798-trap selection; the pairs E comes from the corrected-constant
   refit, the records E from the stored fit, so exact equality is not
   expected — investigate if the gap exceeds 1%).
3. **`ccd_simulation.py` `CCD.__init__` (lines 1194–1218)**: replace the
   two-step draw (τ-bin from histogram + nearest-in-τ σ) with one weighted
   categorical draw over the pairs (p ∝ w_i), keeping the per-trap log-jitter
   of τ within its histogram bin **or** its fit error (decision §7). Persist a
   `seed_mode` HDF5 attr (`legacy_hist` | `weighted_pairs`) and reject
   mixed-mode aggregation in the per-file idempotency guard (precedent:
   `v3_phase_fraction`).
4. **`run_ccd_simulation.py` / `run_campaign.py`**: new population label
   `htcorr` (do not reuse `effcorr` — old outputs must not mix), seeds =
   weighted pairs. `baseline` unchanged. ⚠ Normalization (advisor gap G6):
   the campaign convention is that a seed artifact's integral *is* the
   simulated count (`run_campaign.py` POP_HIST/POP_SCALE) — do NOT also apply
   a Σw/n_detected scale or the population double-scales. Spec: one
   authoritative `N_population` stored in the seed NPZ; `CCD.__init__` asserts
   the realized trap count against it; campaign scale factors derive from it
   and nothing else.

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
   Artifact: `ul_seed_pairs_minimal_caldet.npz` with per-trap `tau135`,
   `sigma`, `weight` (weight = that bin's UL count spread over its transported
   dots), replacing the `tau_at_135k_hist_minimal_caldet_upper_limit.npz`
   histogram seed; consumed through the same weighted-pairs path as §3.
4. **Faint variants**: the faint priors already exist — the stage-05 NPZ
   carries `faint_0p5_depth_electrons_at_pc135` (faint-by-2) and
   `faint_0p25_...` (faint-by-4), and
   `load_amplitude_prior(variant=...)`
   (`src/characterization_probability.py:175-189`) is plumbed to read them.
   What is missing: the stage-09 runner does not expose the variant, and
   stage 10 only computes curve-level faint sensitivity
   (`src/validation_sensitivity.py:601-615`), not full 2D maps. Task: thread a
   `--amplitude-prior-variant` flag through the stage-09 entry point, write
   variant grids as `09_characterization_probability_minimal_caldet_faint0p5_v1.h5`
   (and `_faint0p25_`), then recompute weights and ε_D from them. Reported as
   stated conditional variants (as the paper already does), expected driver
   band ≈ 300–1,800.
5. **Beyond-domain closure (REQUIRED, not optional)**: one campaign scenario
   seeding the σ-below-domain population *with its detection-consistent tiny σ*
   to measure — not assume — how suppressed it is. The suppression argument
   has a precise scope (established 2026-07-10 after Ansh challenged it):
   - *Exact within the shipped transport model*: exposure-phase capture is
     zero by phase geometry — `charge_trap_interaction`
     (`ccd_simulation.py:1292-1314`) is emission-only because exposure charge
     parks under V2 while measured traps are V1/V3. All capture happens in
     transfer windows with per-crossing probability 1−e^(−q·α), α =
     σ·v_th·t_dwell/V_packet ∝ σ. For single-e packets (q ≈ 1) suppression is
     therefore *linear* in σ.
   - *Sub-linear for large packets*: a q ~ 10³ event packet has q·α ≥ 1 down
     to σ ~ 10⁻¹⁸ cm², so event/bleed-driven filling saturates and only
     suppresses fully for yet smaller σ. Whether the tail's
     exposure-dependent slope survives at tiny σ is a quantitative question —
     hence this run is required before any "inert" wording reaches the paper.
   - *Model-scope caveat*: none of this covers hypothetical **V2 traps**
     (parked under the collection phase for the whole exposure). They are
     invisible to pumping for phase-geometry reasons regardless of σ or τ,
     are excluded from the transport model by design
     (`ccd_simulation.py:1225-1227`), and were equally outside the old
     correction's scope ("makes no statement about traps that do not follow
     our model"). This stays a stated domain limitation, unchanged.
6. **(E, σ) rendering of the completeness map**: produce the stage-09 grid,
   catalog dots, and the UL domain boundary re-plotted in (E, log σ) axes via
   the exact affine relabeling log σ = `log_energy_cross_section`(135, E, 0) −
   ln τ₁₃₅. Identical information to the (τ, E) map, but the blind region
   becomes "below a boundary curve σ_min(E)", which makes *where the
   efficiency is bounded* — and hence the §7 domain decision — directly
   legible. Figure feeds the notebook and possibly the paper.

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

## 7. Open questions for Ansh

These block or shape implementation. A fresh session should ask them
**verbatim** (with the context lines) before starting the affected stage, and
record the answer inline here (`Answered YYYY-MM-DD: ...`).

**Q1 — UL σ-domain (blocks §4.1/§4.2).** The upper limit is a statement about
trap kinds inside a declared σ range; outside it, only the inertness run
speaks. *"For the domain-conditional upper limit, do you want the σ domain set
by the observed catalog range (trimmed of fit-noise extremes), by a
literature-informed range like log₁₀σ ∈ [−19, −14], or something else? The
(E, σ) map of §4.6 is built to make this choice visible."*
Answered: —

**Q2 — faint variants in the headline (shapes §4.4/§6).** *"Should the
faint-by-2/4 populations stay stated conditional variants, as the paper does
today, or be folded into the headline UL band?"*
Answered: —

**Q3 — τ jitter in the weighted seed (shapes §3.3).** *"When the sim draws
weighted (τ, σ) pairs, should τ be jittered within its old histogram bin
(status-quo look), within its per-trap fit error (more honest), or not at
all?"*
Answered: —

**Q4 — legacy flavor (scopes everything).** *"Migrate the legacy analysis
flavor alongside minimal_caldet, or freeze legacy at the old correction and
mark it deprecated?"*
Answered: —

**Q5 — old ε(τ) figure (shapes §6/paper).** *"Keep the old global-E ε(τ)
curve in the paper as an explicit 'naive marginal' comparison, or drop it?"*
Answered: —

**Q6 — inertness-run acceptance (shapes §4.5).** The tiny-σ population's
suppression is linear for single-e capture but saturates for large-packet
filling, so the run's outcome is genuinely open. Advisor: acceptance must tie
to the paper's SER uncertainty, not the sim's ~8% run-to-run CV, and a single
scenario cannot bound an unbounded population — it should be a worst case
optimized over (E, σ, A, N) subject to the catalog non-observation and a
physical site-density bound. *"What residual excess from the worst-case
catalog-consistent tiny-σ population counts as 'inert' — a stated fraction of
the paper's SER claim? And what physical density bound on defect sites are
you willing to assert?"*
Answered: —

**Q7 — headline estimator (shapes everything; advisor verdict 2).** The
advisor endorses HT for the visible-support estimate and seed construction
but holds that a *headline paper claim* needs a forward-modeled joint
likelihood (selection, parameter errors, amplitude prior, population
template) — HT as diagnostic, likelihood as inference. *"Is the paper claim
to be carried by (a) the layered HT + domain-UL + inertness statements,
clearly presented as three conditional statements, or (b) a forward
likelihood/Bayesian population fit with HT as cross-check? (b) is more
defensible and substantially more work."*
Answered: —

**Q8 — full inclusion probability (blocks the 'unbiased' wording; advisor
gap G1).** Stage-09 P covers intensity-fit recovery (+0.972 energy-fit
survival) but NOT the dipole finder, spatial masking/eligibility, or
downstream catalog cuts — the stage-09 catalog-selection factor is explicitly
disabled in `src/characterization_probability.py` (~line 694). Missing
selection ⇒ P too high ⇒ HT weights too small (same limitation as the old
ε, but the new wording claims unbiasedness). *"Do we (a) run end-to-end
injections into FITS before dipole finding to measure the full inclusion
probability, or (b) rename the estimand to 'population conditional on dipole
discovery and spatial eligibility' and state it?"*
Answered: —

**Q9 — UL construction (shapes §4.2; advisor verdict 4).** Summing per-bin
Poisson-90% limits is not a 90% limit on the band total (multiplicity;
undefined within-domain composition). *"Replace with a joint Poisson
likelihood over bins — total population as parameter, declared σ-mixture
template + nuisances, simulation-verified coverage — or keep per-bin limits
and present them only as per-bin statements?"*
Answered: —

## 8. Execution order

| step | touches | gated by |
|---|---|---|
| verify records↔pairs row identity | probe only | — |
| 11a weights + bundle | figure_utils.py | — |
| 11a seed builder | make_trap_pairs.py | row identity |
| 11a CCD sampling + seed_mode attr | ccd_simulation.py, run_ccd_simulation.py | seed builder |
| 11b domain-UL + variants | figure_utils.py, src/ (stage-09 reruns for faint grids) | 11a |
| 11c closure + audits | src/naive_efficiency_closure.py, probes/ | 11a/b |
| 11c bracket rerun | run_campaign.py (cluster) | all above |
| 11d notebook + paper | notebook/, paper/ | 11c numbers |

## 9. External review record (codex-advisor, gpt-5.6-sol/xhigh, 2026-07-10)

Read-only adversarial review of this plan + probes + pipeline code, requested
by Ansh with emphasis on necessity/justifiability/correctness and
survivorship bias. Thread `019f4f68-4708-7940-b3f4-689d28a07c90`. Verdicts
(condensed; the numbers refer to the review questions, not plan sections):

1. **D1–D3 core SOUND; two overstatements.** D1's observed ridge does not by
   itself refute a factorized *true* population (selection can induce E–τ
   correlation); D2 shows extreme prior sensitivity, not mathematical
   invalidity. D3 is the strongest defect, but "350× excluded" is overstated
   because stage-09 P is not the full catalog-inclusion probability — though
   closing 350× would need an implausibly large omitted-selection loss.
2. **HT appropriate but not headline-ready alone** → Q7.
3. **Survivorship bias: the "exactly unbiased" wording is FLAWED.**
   (a) stage-09 P omits finder/masking/downstream selection → weights too
   small → Q8; (b) amplitude prior measured from survivors — faint-by-2/4 are
   sensitivity tests, not corrections; (c) Eddington/error-in-variables bias
   in 1/P(x̂) near the boundary — small for the total, potentially material
   for the ~30-trap driver tail → mitigation: posterior/multiple-imputation
   averaging of 1/P over each trap's fit likelihood; (d) zero-caught kinds
   correctly delegated to the UL/inertness layers.
4. **Per-bin Poisson90/ε summed is not a 90% band limit** → Q9.
5. **Variance**: quote the observed-sample design estimator Σ(1−P)/P² *plus*
   a two-stage parametric bootstrap (regenerate detection, refit parameters);
   survivor-only bootstrap is inadequate; the 1% pairs-vs-records check is an
   alarm only — reconcile trap-by-trap.
6. **Same-σ transport is a template, not a model-independent efficiency**:
   use HT-weighted (not raw) dots for ε_D; REJECT (don't clip) transports
   that leave the calibrated E grid; treat domain membership as uncertain
   given 14-decade fitted-σ noise; consider a regularized true-σ mixture.
7. **Inertness/V2**: single-scenario run cannot bound an unbounded
   population — worst-case optimization needed (folded into Q6); V2 exclusion
   is defensible only if the paper's claim is narrowed to pumped V1/V3
   model traps (current paper text makes broader SER claims).
8. **Missing validations**: closure test must not draw truth from the HT
   estimate itself (use independent synthetic populations, including
   misspecified ones); stress-test the assumed cross-temperature independence
   of detections; propagate selection-grid MC error, noise-map and
   amplitude-prior uncertainty; quadrant leave-one-out; spatial blocking.

**Adopted immediately** (spec edits in this doc): normalization single-source
(§3.4 ⚠), ε_D weighting + rejection-not-clipping (§4.2 via verdict 6),
closure-test independence (§5.1 amended below), variance quotation (verdict
5), Q6 reworded to worst-case form, new Q7–Q9.

**§5.1 amendment**: closure truths must include (i) populations independent
of the HT estimate, (ii) at least one deliberately misspecified selection
(e.g., P shifted by its MC error) to measure sensitivity, not just
implementation correctness.

**Referee-view summary** (what the rewritten paper section must survive):
(1) "your P is only curve-fit recovery, not catalog inclusion" → Q8;
(2) "your 90% band is built from an arbitrary noise-dominated σ template and
isn't a real 90% limit" → Q9 + declared template + coverage check;
(3) "invisible tiny-σ/V2 traps make the SER-fraction claim prior-driven" →
narrow the claim to characterized-model V1/V3 traps + worst-case inertness
envelope + explicit statement that non-pumped populations are unmeasured.
