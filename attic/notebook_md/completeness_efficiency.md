# Completeness and efficiency: is the trap catalog missing a dangerous hidden population?

> **In plain terms.** This file answers one question: *could we have missed a dangerous hidden batch of
> traps?* The dangerous traps hold charge the longest, and those are also the hardest to catch — so this
> matters. The key trick is to **not** estimate our catch-rate from the traps we already caught (that
> would be circular — we'd be blind to what we missed). Instead we plant *fake* traps of known
> properties into realistic simulated data and count what fraction come back — called
> "injection-recovery" (this is "Method 3"). The result is a completeness statement (with honest
> caveats) plus an inflated "what if there are many more" trap population that the simulation
> ([[physics]]) uses to bound the worst case. Two side-puzzles that looked alarming — a "dip" in an
> efficiency curve and a cluster of high-temperature fit failures — turned out to be mostly measurement
> artifacts, not real losses; §4 and §5 explain them. New words are in the [glossary](glossary.md).

**Thread status:** the detection model is settled (Method 3, analytic injection–recovery); the
`trap_completeness_method3/` subtree is the **live implementation** and is *not* archived. This file
consolidates the conceptual design, the Method-3 results, and the two efficiency puzzles (the naive
"dip" and the high-T failures) that were resolved along the way. The output feeds the simulation's
upper-limit population — the link back to [[physics]] §4.

Sources consolidated here: `trap_completeness_method.md` (design, 2026-05-20, now in `attic/`);
`trap_completeness_method3/` (live subtree — README + `cache/10_*` statements + `agents/` packets,
**left in place**); `NAIVE_EFFICIENCY_DIP_EXPLANATION.md` (2026-06-18, now in `attic/`);
`HIGH_T_FAILURE_SPLIT_PLAN.md` (2026-06-18, now in `attic/`).

---

## 1. The question

Long-lived traps (large τ135) are the dangerous ones in a dark-matter search: they release a
trapped electron *long after* the originating high-energy event and across many exposures, so
masking the original event does not catch the released 1e⁻. The completeness question:

> Could a large hidden population of (especially long-lived) traps have been systematically missed
> by the pocket-pumping measurement — e.g. 2×10⁶ traps at τ135 = 10³ s of which we caught 0.01%?

This is a **population bound**, not just a per-trap efficiency (§6 explains the difference and the
extra assumption needed to connect them). Everything hinges on how the measurement's selection
function behaves at the warm-T / long-τ band edge that no data directly populate.

---

## 2. Three methods (Method 3 is the answer)

### Method 1 — per-temperature efficiency curve (what the paper originally used)

For each temperature T and τ bin, among **characterized** traps whose τ(T) lands in that bin:
`eff(τ,T) = measured / (measured + extrapolated)`, where "measured" = τ_e(T) was directly fit and
"extrapolated" = it comes only from the Arrhenius fit. Pool across traps/temperatures, read as
detection efficiency vs τ, apply an inverse-efficiency correction with a Poisson 90% CL per bin.

**Fatal problems:** (1) **survivorship bias** — both numerator and denominator are drawn from
*characterized* traps, so it conditions on being characterizable and is structurally blind to the
hidden population it is meant to bound; (2) low warm-T statistics read "0 efficiency" from lack of
*data*, not lack of *sensitivity*; (3) bin/threshold dependence; (4) it ignores that each trap is
re-interrogated at *many* temperatures as its τ(T) slides through the band.

### Method 2 — empirical sensitivity-band coverage (stepping stone, not recommended)

Reframe (correct, kept): each temperature is a separate detection opportunity; ask whether a trap's
τ(T) trajectory passes through the detectable band at ≥ n_good temperatures. Better than Method 1
(uses positive evidence, makes the E-dependence explicit: high-E steep-dτ/dT traps sweep the band
fast and are harder to characterize at the same τ135). **But it does not escape the survivorship
bias — it relocates it to the band edge**, because the band is defined as `{τ : eff(τ,T) ≥ 0.5}` and
`eff` uses the biased denominator. The warm-T upper edge — which controls the long-τ answer — is
exactly where the data are starved. It also discards the probabilistic structure (a trap at the
eff=0.5 edge counts the same as one at eff=1.0 center).

### Method 3 — analytic per-temperature detection probability (recommended, implemented)

**Key realization: the per-temperature detection model is already written down in closed form in the
paper.** It does not need a Monte Carlo of the science readout (that simulation — [[physics]] — has
*no* notion of "detected": no t_ph sweep, no dipole-intensity equation, no 35×35 noise, no good-trap
cuts; it must not be repurposed as a forward detection model).

The intensity model (paper Eq. `I_fit`), with `A ≡ N_pumps·D_t·P_c`:
```
I(t_ph) = A (e^{−t_ph/τ_e} − e^{−8 t_ph/τ_e})
```
Two consequences that reframe the whole question:
- **Peak is at** `t_ph* = (τ_e/7)·ln 8 ≈ 0.30 τ_e`, with `I_peak ≈ 0.65 A` — **independent of τ_e**.
  A long-τ trap is *not* intrinsically faint.
- It only *looks* faint because we cannot reach its peak: `t_ph^max ≈ 1.03 s`. For τ_e = 10³ s the
  sampled delays sit on the rising linear part (`e^{−x} − e^{−8x} ≈ 7x`), so `I ≈ 7 A t_ph/τ ≈
  0.007 A`. **The long-τ cutoff is a t_ph-reach limitation, not physics.**

So detection probability is a **selection-function / injection–recovery** computation:
1. Build the true curve on the *actual* t_ph grid at T (min ~fixed, max grows with T — longer t_ph
   was deliberately used at higher T to chase longer τ).
2. Draw a synthetic measurement `Î = I + ε`, ε at the per-point intensity noise (≈Gaussian,
   dominated by shot noise on the injected ~2000 e⁻ flat field).
3. Fit and apply the good-trap cuts verbatim.
4. Repeat N times → `p_det(τ, A, T | σ)` = fraction recovered. For the dominant peak cut alone the
   edge is a soft error function:
   ```
   p_det ≈ ½[1 + erf((I_peak^true − 3σ) / (√2 σ))]
   ```
   with `I_peak^true = 0.65 A` if the peak is reachable, else the rising-edge value `≈ 7 A t_ph^max/τ`.

**Where the bias-escape comes from:** the detection threshold is set by σ measured from **trap-free
image patches** — *zero* reference to which traps were found. σ is a spatial map that must be
marginalized: `p_det(τ,A,T) = ∫ p_det(τ,A,T|σ) p_σ(σ|T) dσ`. The *only* place the trap population
re-enters is the amplitude prior `p_A(A|T)`, deliberately calibrated from the **brightest**
(least-biased) traps and isolated as a single, testable assumption.

Multi-temperature characterization probability (walk the τ(T) trajectory for a grid point (τ135, E)):
```
P(characterized | τ135, E, A) = P(Σ_i Bernoulli(p_det,i) ≥ n_good)     (Poisson-binomial tail)
P(characterized | τ135, E)     = ∫ P(… | A) p_A(A|T) dA
```
This is the principled replacement for "count band-hits ≥ n_good," naturally weighting near-edge
temperatures less.

**Load-bearing assumption:** `D_t` (hence A) is independent of E/τ — justified because in the model
τ enters only the shape factor and A only depth/capture, so the amplitude distribution from detected
traps should carry to the hidden long-τ population. Tested (§3, Stage 05): `fit_coeff` central width
is a factor 2.89 and trap-depth correlations are weak (`max |ρ| = 0.150`), so `p_A` is a minor
systematic, not the dominant one. The residual open physics question is whether `P_c(T)` is flat or
falling at high T — a fall is *real* warm-T incompleteness (Stage 05 measured `P_c(210 K) = 0.681`).

---

## 3. Method 3 implementation and results (`trap_completeness_method3/`)

The subtree is a staged, injection–recovery build (env `sensei_charge_traps_new`, per-flavor via
`run_chain.py --flavor {minimal_caldet, legacy}`). All 13 stages (00–12) are complete; the noise-
parity gate (Stage 02) is the load-bearing check — FITS-derived local patch σ must reproduce HDF5
`intensity_err` (it does, exactly, over 1428 sampled rows), while `image_sigma` is a separate
whole-image threshold field.

Stage highlights (from the README status table):
- **03 noise map:** `p_σ(σ|T,quadrant,dtph)` from 577,200 trap-free patches over 481 dwell FITS.
- **05 amplitude prior:** from 13,963 high-confidence records; `fit_coeff` width ×2.89, weak depth
  correlations, `P_c(T)` falls to 0.681 at 210 K; faint-by-2 / faint-by-4 variants cached.
- **08 full grid:** `p_det(T,τ,A)` on 23×55×35, 100 realizations/point (April-only 200 K).
- **09 characterization map:** `P(characterized | τ135, E)`, 161×121.
- **10 validation:** 99.86% of known n_good=4 traps at P₄ ≥ 0.8.

### The completeness statements — legacy vs minimal are dramatically different

Averaging `P(characterized | τ135, E)` over the observed n_good=4 energy distribution, the ≥95%-
complete τ135 reach depends strongly on which catalog seeds it — this is the completeness payoff of
the pedestal/signed-refit corrections:

| flavor | validation P₄≥0.8 | mean compl. @ 10³ s | ≥95% complete over τ135 | unbounded regime starts |
|---|---:|---:|---|---|
| **legacy** (`10_completeness_statement.md`) | 95.93% | 0.227 | 0.0048 – **1.67 s** | 193 s |
| **minimal_caldet** (`..._minimal_caldet.md`) | 99.79% | 0.992 | 5.7e-4 – **7.29e3 s** | 52.8 s |
| **tau1000 variant** (`..._tau1000.md`) | 99.86% | 0.936 | 5.6e-4 – **668 s** | 8.91e3 s |

The legacy catalog is ~95% complete only out to ~1.7 s; the pedestal-corrected minimal catalog
extends the ≥95% reach by ~3–4 orders of magnitude in τ135, because the high-T pedestal fix recovers
the high-T detections that anchor long-τ135 Arrhenius extrapolations. All statements are
**recoverable-completeness, conditional** on hidden traps sharing the observed E distribution and the
high-confidence amplitude prior; under fainter priors the minimal @10³ s number drops 0.992 → 0.920
(faint-by-2) → 0.744 (faint-by-4). The genuinely unbounded regime (every measured T maps outside the
grid → zero p_det) covers ~8–21% of grid points depending on flavor.

### Validation and fixes (WS1 / WS2 / WS3)

Three corrections behind the current Method-3 numbers (from the tabbed review doc):

- **WS1 — flavor-consistent injection (a real bug, found via a Codex 5.5 stress-test).** Stage 08 was
  injecting **spatial patch noise** (`_local_sigma = np.std(patch)`, median ~183 e⁻) for *both*
  flavors. Correct for legacy (which detected with patch σ) but **wrong for minimal**, which detected
  with temporal pair-noise (~36 e⁻) and a threshold calibrated on it — a ~5× over-noising that made
  the old minimal completeness (a ~0.44 peak) far too pessimistic. Fix: inject
  `σ = √(σ_base(T,q)² + |I_true|/4)` with σ_base ~36 e⁻ from `pair_noise_table_minimal.npz` (the `/4`
  not `/2` because the live `(S_a+S_b)/4` clamps the dark dipole pixel to 0); inject the physical
  pedestal into the truth for both flavors (drawn from the catalog signed `fit_offset` pool); and for
  legacy do a signed→abs reconstruction (`np.abs(true + noise)`, offset-free) so its real high-T
  pedestal-driven rejections are reproduced rather than hidden. **Both grids had to be rebuilt once**
  (`run_chain.py --flavor {legacy, minimal_caldet} --fresh-grid`; the ~24 h Stage-08 rebuild is the
  pole).
- **WS2 — shared SRH helper.** The across-temperature energy fit was extracted into
  `fit_energy_cross_section()` in both `dipole.py` and `dipole_new.py` so the live analysis and any
  forward model call an identical estimator (the modules legitimately diverge only in
  reduced-χ² < 5 / no-orientation vs reduced-χ² < 4 + orientation). Closure gate: re-fit 16 dipoles
  before/after → byte-identical.
- **WS3 retired — the energy fit costs only ~4%, so no full forward grid is needed.** A lean
  injection–recovery at the known catalog traps' fitted (E, lnσ) with WS1 noise + pedestal and the
  live fitters (300 traps × 20 realizations/flavor) gave, for **minimal**, `P(GoodEnergyFit | ≥4) =
  0.961` with **σ_τ pull RMS 1.007** (target 1.0, flat in amplitude/T → honest error bars). So
  `P(characterized) ≈ 0.96 × P(≥4 good intensity)` and completeness = the intensity reach × 0.96. For
  **legacy** the pull RMS is 1.686 (up to 2.66 cold) — its SRH survival (0.925) is propped only by the
  loose reduced-χ² < 5 gate (within-sample re-pass under fresh noise: minimal 0.93 vs legacy 0.38).
  The 0.70 → 0.96 gap (real catalog GoodEnergyFit 0.70 vs clean-trap 0.96) is **purity, not
  completeness** — the fit rejecting non-SRH objects (blends, dual-response, the high-T lean).

**The two "double peaks" — distinct objects, do not conflate.** (A) *Completeness vs τ135* showed a
dip only after multiplying the smooth intensity reach by an empirical `survival(k)` factor keyed on
the good-temperature count — a non-monotonic **artifact** that conflated completeness with purity. It
was **removed** (`characterization_probability.py`, `APPLY_SURVIVAL_K=False`); the completeness is now
just the intensity reach × 0.96. (B) *Naive efficiency vs τ(T)* is the real GOF-collapse dip treated
in §4. A third figure — the **caught-at-T completeness map**
(`figure_utils.plot_completeness_map_caught_at_T`) — resolves the apparent paradox that characterized
traps sit in low-probability regions: the long-τ135 right arm was **caught hot** (~200–250 K, in-window
at high T, not at 135 K); `corr(log τ135, mean good-T) = +0.62`.

---

## 4. The naive efficiency "dip" — closed (a GOF artifact, not overclaiming)

Method 1's naive efficiency curve **dips** at high T / short τ(T) (the "double hump"), yet Method 3
predicts high completeness there — apparent paradox: are we overclaiming? **No.** The two are
different metrics:
- **naive efficiency = per-(trap, temperature)** — the p-value GOF cut acts directly on it.
- **trap census = per-trap** — robust to recovering one more of a trap's temperatures.

They respond differently to the same cut: of 17,658 high-T intensity points recovered by swapping the
per-temperature GOF cut, **62% sit on already-characterized traps** — they lift the per-point dip
while adding **zero** traps to the census. So a large dip effect coexists with a near-zero census
effect; no contradiction.

**What drives the dip** (chain of evidence, `tools/` diagnostics): (1) of dip-window high-T failures,
~26% are genuine undetectable fading, ~74% recoverable/analysis-limited; (2) ~50% are "p-value-
binding casualties" — they pass amplitude/δχ²/τ pre-cuts and fail *only* the `p_value > 0.05` GOF cut
(the dof-bias against well-sampled traps, [[dipole_algorithm]] §3, memory `srh-gof-overrejects`);
decoys fit *better* (reduced_χ² ≈ 0.82) than real casualties (≈3.0), so reduced_χ² does **not**
separate trap from non-trap — the multi-temperature Arrhenius + orientation gate does; (3) visually,
recovered casualties are clean dipole bumps while decoys run *anti-Arrhenius* (τ rising with T —
physically impossible).

Quantitative closure (same 3798 characterized traps held fixed, only the per-point `measured` mask
swapped; baseline reproduces the reference dip 0.414 exactly):

| dip (min over τ_e ∈ 3e-3–3e-2 s) | value |
|---|---|
| observed, `p_value > 0.05` (current) | **0.414** |
| observed, `reduced_chi2 < 5` | 0.596 |
| observed, `reduced_chi2 < 10` | **0.642** |
| hybrid (Method 3 × empirical GOF survival) | 0.676 |
| pure Method 3 | 0.943 |

Swapping the GOF cut closes almost the entire gap from the observed dip to the hybrid (0.414 →
0.642 ≈ 0.676). **Interpretation:** the dip decomposes into ~0.23 recoverable (p-value GOF wrongly
rejecting good well-sampled traps — validated as clean Arrhenius recoveries) + ~0.30 residual to pure
Method 3 (genuine high-T fading + faintness). Under the deliverable's definition — *"if a trap is
good and well-behaved, would we have caught it?"* — neither the p-value losses (genuinely good) nor
the residual (genuinely faded → not good at that T) should count against good-trap completeness.
**So Method 3 is not meaningfully overclaiming;** the naive dip is simply the wrong estimator to read
as completeness. The fix is diagnostic, not operationally mandatory. If ever applied, the per-
temperature GOF should be **tight (X ≈ 3)** — that *grows* the census by +275 (+7.2%), whereas X=10
is net-negative (−27); it is a separate stage from the energy-fit's own `reduced_chi2 < 10` backstop.

---

## 5. The high-T failure split — bounding the genuine fading

**Why there is an overclaim to bound.** A leave-one-temperature-out holdout test (bright, in-window,
grid-easy cohort; amplitude estimated only from cold/intermediate good temps) showed the Stage-08
forward grid predicts ~0.95 detection at 175–200 K where the **observed** good-fit rate is ~0.45 —
the completeness is **~2× optimistic at high T** for genuinely good traps. The reason is structural:
the grid **injects data from its own model**, so it can never reproduce the real high-T *morphology
misfits* — passing fits have median reduced χ² ~0.9 (noise model calibrated) but failing ones ~3.2–3.5
(the real intensity curves genuinely deviate from bump+pedestal at the small pair-noise error level).
This overclaim sits precisely on the long-τ135 "hot-measured" right arm of the completeness map.

**Step-0 residual audit — the bright high-T failures are two distinct populations** (model-independent,
Codex 5.5-reviewed): **POP1 real signal loss** (~40–50%, rising with T) — the raw curve is flat
(max 1.7–1.8σ from a constant, fitted amp ~70 e⁻ vs ~1600 cold), a genuine non-detection where the cold
amplitude does *not* transfer to high T (not cut-recoverable); and **POP2 real bumps failing GOF on
shape** (~50–60%) — a strong 13–25σ bump with fitted τ matching SRH to 0.02–0.05 dex, cut-recoverable
by relaxing the GOF (relaxing lifts the high-T rate ~0.44 → 0.78). The direction is two separate
tracks: recover POP2 with a purity-calibrated `reduced_chi2 < X` (never SRH τ-consistency, which would
bake Arrhenius into selection), and *model/bound* POP1's high-T amplitude loss (it affects only the
long-τ135 arm). Step 0 covered only the *bright* cohort; the naive dip is *faint*-dominated, which is
what the split tool below measures.

To bound the completeness overclaim, `tools/high_t_failure_split.py` (designed in
`HIGH_T_FAILURE_SPLIT_PLAN.md`, built via Codex under Claude physics review) classifies each high-T
(≥170 K) intensity-fit failure on a characterized trap. It is **not** a clean partition; it is a
primary binary with non-exclusive mechanism tags and a leverage sub-split (all leakage-free — the
target T is excluded from the SRH τ, the PC reference, and the cold-coeff estimate):

Per-point quantities (model linear in coeff & offset, nonlinear only in τ):
- **Fresh free 3-param refit** → MLE (coeff, τ, offset); sanity `perr[0] ≈ stored fit_coeff_err`.
- **τ_SRH** from the trap's stored E/logσ.
- **Fisher counterfactuals** at the MLE: `F = JᵀWJ`, `W = diag(1/err²)`, analytic Jacobian columns
  `∂/∂coeff = g(s;τ)`, `∂/∂offset = 1`, `∂/∂τ = 3000 coeff((s/τ²)e^{−s/τ} − 8(s/τ²)e^{−8s/τ})`.
  `sig_fixoff` inverts the **2×2 (coeff,τ) sub-block of F** (not of Cov = F⁻¹ — the single easiest
  bug); `sig_fixboth = |coeff|·√F[coeff,coeff]`. Assert `sig_fixoff ≥ sig_full` and
  `sig_fixboth ≥ sig_fixoff` always (fixing a nuisance cannot increase error).
- **Forced-amplitude fit at τ = τ_SRH** (linear WLS, offset profiled): `X = [g(s;τ_SRH), 1]`,
  `β = (XᵀWX)⁻¹XᵀWy`. Signed projection `A_proj = sgn·coeff_forced` (sgn = cold-coeff sign;
  wrong-sign bumps are not evidence). One-sided UL95 = `A_proj + 1.645·coeff_forced_err`.
- **Sampled leverage / compression:** `g_perp = g − weighted-mean(g)`, `I = g_perpᵀ W g_perp`;
  expected `A_exp = cold_coeff · PC(T)/PC(T_cold_ref)`; `SNR_exp = |A_exp|·√I`.

Classification:
- **RECOVERABLE_OR_ANALYSIS_LIMITED** if any of: `sig_forced ≥ 3`, model-free contrast `sig_C ≥ 3`,
  `sig_full < 3` but `sig_fixoff ≥ 3` (tag `pedestal_cost`), or `sig_fixoff < 3` but `sig_fixboth ≥ 3`
  (tag `tau_cost`).
- **UNDETECTABLE_SAMPLED_CONTRAST** otherwise, then leverage-split:
  - `genuine_fading` if `SNR_exp ≥ 3` (the sampling **had** leverage to see the expected amplitude,
    but the bump is below the UL — real loss beyond the survivor PC trend). **← the completeness-
    overclaim bound.**
  - `design_compression_limited` if `SNR_exp < 3` (even the expected amplitude wouldn't clear 3σ).
- **unclassified** if Fisher near-singular (never `genuine_fading`).

Result: ~26% of dip-window high-T failures are genuine fading (the overclaim bound); ~74% are
recoverable/analysis-limited. The headline `genuine_fading_fraction` is PC-scaled (loss *beyond* the
model's already-discounted smooth decline); because Stage-05 PC(T) is survivor-biased (shallower
decline than the true population), the PC-scaled expectation is a *high* bar and the fraction is a
**conservative (upper) bound** on the un-modeled overclaim. A survivor-free PC re-measurement (Track
B, §10 of the plan) would de-bias it — flagged, not done. See memory `naive-dip-minimal-closure`.

---

## 6. From recoverable fraction to a population bound, and into the simulation

The methods produce a **recoverable fraction of the observed-E family** `f(τ135)`. Converting to a
population bound `N_true ≲ N_obs/f` requires assuming the hidden population shares the **observed
E/σ distribution.** A population could hide *precisely because* it sits at an E/σ the protocol rarely
characterizes — that population is **not** bounded by any of this. The honest claim is conditional:
"assuming hidden traps share the observed E distribution, the set is ≥ X% complete for τ135 ∈ [a,b],"
plus an explicit statement of the genuinely unbounded regime (τ above τ_max(T) at every T, and
unobserved E/σ).

**Feeding the simulation.** The efficiency-corrected upper-limit histogram integrates to ~17,475–
18,646 effective traps (per-bin 90% CL) vs ~3,798 characterized. Upper-limit scenarios scale the
trap **density** (`--trap-density-scale`, or the campaign's `--populations upper`), simulating "the
population is this big with this composition." Completeness covers τ-window and amplitude/dimness
selection but **cannot cover phase blindness** — pumping only sees traps in the pumpable (edge)
phases while readout charge passes through all phases; if 2 of 3 phases are probed, true density is
another ×1.5 (`--upper-density-scale`).

The τ-band of the resulting exposure-dependent upper limit is analyzed in [[physics]] §4.4: the UL
slope (~2.8e-6) lives at τ > 1e5 s, is anchored by ~20 real characterized detections at τ > 1e6, and
should be reported as **two numbers** — a NULL measured/characterized result and an efficiency-
corrected 90% CL allowance carrying a ~25% Poisson + Arrhenius-completeness error band — **not**
floored. The completeness systematic on that band is precisely the amplitude-prior / P_c(T) / high-T
fading uncertainty quantified in §§3–5.

---

## 7. Open items

- **Amplitude prior / P_c(T)** is the residual assumption in the long-τ regime; sensitivity to
  faint-by-2/4 priors is cached (Stage 05/10). A falling P_c(T) at high T is *real* incompleteness.
- **Survivor-free PC re-measurement** (Track B) to fully de-bias the genuine-fading fraction — not
  done.
- **Fundamental floor:** a trap outside the band at *every* temperature is invisible and unbounded by
  any of these methods; closing it needs an independent handle (colder measurement, longer t_ph, or a
  dark-current/leakage argument).
- **Phase blindness** (×1.5 density) is stated as an assumption/systematic, not measured.
- Energy-fit cost of the completeness selection is ~4% for minimal (memory
  `completeness-energy-fit-resolution`); legacy was ~1.7× optimistic. Completeness =
  WS1-corrected reach × 0.96.
