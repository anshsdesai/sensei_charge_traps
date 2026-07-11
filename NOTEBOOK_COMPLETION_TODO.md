# Notebook completion — to-do list

What stands between the Quarto lab notebook (`notebook/`) and being a *complete
scientific record*, not just a complete methods record. Six items, in priority order (items 3, 4
and 6 now closed). Items 3–6 are written out in plain language because they are the ones that are
easy to lose track of. Item 5 (2026-07-05 review) collects one-sided systematics on the
*upper-limit allowance* that aren't in the quoted band. Item 6 (raised and resolved 2026-07-06)
was an apparent phase-geometry contradiction that turned out to be an electron-vs-hole polarity
misreading of the sequencer state bits; the reconciliation is now documented in `physics.qmd` §2.4.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done

---

## 1. `[~]` Land the phase-split (`v3_phase_fraction = 0.5`) production rerun

**State:** running now (Ansh).

**Why it's here:** the current headline numbers in the notebook — the 2.8e-6 upper-limit slope in
`physics.qmd` §4.4, the seed-cap τ-band table, the ~17,475 effective-trap count in
`completeness_efficiency.qmd` §8 — were computed under the *old* readout kernel. The A/B test
(2026-07-02) showed the phase-split fix **redistributes** the masked excess from slope to intercept
(masked intercept **+45%**, masked slope **−3.2%**). So those numbers are self-declared superseded
(`physics.qmd` §6, item 9).

**When it finishes:**
- [ ] Refresh `physics.qmd` §4.4 (the τ-band seed-cap table) and `completeness_efficiency.qmd` §8
      **in lockstep** — they quote the same population and will go stale together.
- [ ] Resolve a doc-internal tension while doing so: `physics.qmd` §2.4 already says "all campaign
      numbers … were superseded and regenerated under Model C … the `campaign/` results below are
      Model C," but §6 item 9 says the 0.5 rerun is "in progress." Make the two sections agree on
      what phase fraction the live `campaign/` numbers actually reflect.

---

## 2. `[~]` Build a **results page** for the Quarto notebook (minimal pipeline only)

**Why it's here:** the notebook documents *how* the answer is produced across four threads, but the
answer itself — the `run_campaign.py` output — is never reported on its own page. The numbers
currently live scattered as diagnostics (a 30-trial table in `physics.qmd` §4.4, prose in
`completeness_efficiency.qmd` §8). A reader has to reconstruct the science conclusion from footnotes.

**Skeleton built 2026-07-03** (`notebook/results.qmd`, added to `_quarto.yml` as thread 5). Done and
pending items:
- [x] Source of truth: `charge_trap_figures.ipynb`, minimal pipeline only (legacy skipped).
- [x] New page `notebook/results.qmd`, added to `_quarto.yml` as "5 · Results" after `physics.qmd`.
- [x] Two headline numbers stated explicitly (null measured/characterized; 90% CL UL slope ≈ 2.8e-6,
      not floored) — §1. *(2026-07-06, TODO 5.2: the original "~25% band" wording replaced by ±22%
      Poisson + explicit amplitude-prior conditionality.)*
- [x] **The measured catalog** (Ansh's addition): energies + cross-sections + SRH-degeneracy joint
      density (`make_figures.results_energy_sigma`), τ135-vs-E scatter + census
      (`results_tau135_energy`), and the selection funnel (`catalog_funnel`) — §2. Smoke-tested in the
      conda env (3,798 traps, figures render).
- [x] Completeness → population bound: overlay + efficiency-corrected census + 90% CL UL, reusing the
      `make_figures` `_capture_fu` wrappers (`write=False`, no seed regen) — §3.
- [x] Systematics ledger table (V_p, σ, ½ phase, phase-blindness **rejected**, high-T lean, P_c(T),
      masking fidelity) — §5.
- [x] "What is/ isn't claimed" + open items — §6.
- [x] **Campaign figures (§4) built with current (pre-split) numbers** (2026-07-03). Four live cells:
      single-scenario null + upper (`results_scenario_null` / `results_scenario_upper`) and two forest
      plots (`results_forest_headline` baseline-vs-UL, `results_forest_clearmodes` baseline clear-mode
      robustness). No helper port was needed — `figure_utils` already had `scenario_dir` /
      `aggregate_scenario` / `plot_simulation_results` / `compare_scenarios`; the new `make_figures`
      wrappers just drive them via `_capture_fu`. Smoke-tested in the conda env (all four render,
      match the reference figures exactly).
- [ ] **Regenerate the §4 figures after the rerun** — they carry pre-split numbers (a page callout
      says so). Once item 1 lands they refresh automatically on the next `quarto render` (cells read
      `campaign/`); also update the §4.3 τ-band table alongside item 1's lockstep `physics.qmd` §4.4
      edit.
- [ ] Full `quarto render notebook` in the conda env to confirm the page builds end-to-end (heavy —
      the flatten invalidated the freeze cache, so all pages re-execute once). Ansh to run.

---

## 3. `[x]` High-T Arrhenius lean — **fork settled 2026-07-03; ledger entry landed 2026-07-06**

**Resolution (2026-07-03, full writeup in `claude_scripts/high_t_pedestal_bound/FINDINGS.md`):**
the t_ph-pedestal test was run on the live minimal catalog — as a data bound (pooled signal-free
tail residuals) plus an injection–recovery through the live 3-parameter fit. Three independent
kills: **(a)** the data allow a pedestal growth of at most ~2–11 e⁻ per second of t_ph at the
well-sampled hot temperatures, while producing a 0.15 dex τ shift requires ~390–3,400 e⁻/s — a
40–230× gap; **(b)** an injected growing pedestal biases τ **up**, and the lean is τ **down** —
wrong sign; **(c)** at the required size the curves fail the live χ²<10 gate (pass fraction
0–28%), so the contamination could not hide inside the good-fit catalog. Two 2026-06-12
diagnostics (`trap_completeness_method3/agents/12_high_temp_misfit.md` stage 12;
`pedestal_vs_tph_test.py`) had already found the pedestal t_ph-independent (readout CTI dipole,
monopole test negative), but those results were never folded back into this fork — this analysis
quantifies and closes it. Per the decision rule below: the lean is **not instrumental** → accept
it and quote it as a systematic on τ135 / the long-τ tail. σ(T) is the leading physical candidate;
the T⁻² SRH prefactor stays fixed.

**Bracket run on the live minimal catalog (2026-07-03, same day):** the lean was quantified
directly (first minimal-native measurement): a smooth profile from −0.026 dex @170 K to
−0.137 dex @210 K (2,766 reference traps; empirical one-power descriptor n ≈ 1.5, which is a
summary only). Mechanism identification by self-consistent shape comparison
(`signed_refit.qmd` §6b, @fig-lean-mechanisms): the profile matches **Henry–Lang thermally
activated multiphonon capture with effective barrier E_b ≈ 0.16 eV** (χ²/dof ≈ 5.8); power-law
σ∝T^n, Green-1990 mass T-dependence, and Varshni gap-pinning all fail by factors ≥30. The lean is
quadrant-uniform and τ-window-independent (not instrumental). Refitting every characterized trap with the live energy-fit call, hot points as-measured
(variant A, reproduces production τ135 exactly) vs hot points un-leaned (variant B), gives a
**small** impact: τ135 > 1 hr fraction **unchanged at 1.47%**, UL-slope driver band
(3e5–5e7 s) **30 → 28 traps (−7%)**, median mover shift −0.08 dex. The earlier "tail halves,
2.4% → 1.1%" expectation is **superseded** — that was a legacy-vs-full-refit inter-catalog
comparison (catalog churn), not the isolated lean effect. Full documentation with live figures now
in `notebook/signed_refit.qmd` §6b (`make_figures.arrhenius_lean_profile` / `_bracket` re-derive
from the catalog at render time).

**Remaining:**
- [x] Enter the lean bracket in the results-page systematics ledger (item 2) at its measured size:
      ≤7% on the driver-band population, 0% on the >1 hr tail fraction — far below the V_p band.
      No dedicated campaign arm needed. *(Done 2026-07-06: `results.qmd` §5 row updated with the
      settled bracket, mechanism, and the item-5.1 non-conservativeness caveat.)*
- [x] Run the t_ph-pedestal test. (Done via tail-residual bound + injection; the FITS-stage
      "add a t_ph term and refit" variant is superseded — the term is absent and cannot produce
      the lean, so refitting with it cannot remove the lean.)
- [x] Quantify the lean's τ135 impact on the minimal catalog (the bracket above).

Original plain-language description follows (kept for context).


This is the open item that most directly moves the headline upper limit, and it's worth
understanding in plain terms.

### What we measure, and what "the line" is
For each trap, at every temperature (125–210 K) we measure how long it holds a captured electron
before releasing it — its emission time **τ**. Thermal-emission (SRH) theory says τ follows an
**Arrhenius law**: plot `log τ` against `1/(kT)` and you should get a **straight line**. The line's
**slope** gives the trap's energy depth **E** (how deep the defect sits in the bandgap) and its
**intercept** gives the capture cross-section **σ**. Fitting that straight line across temperatures
is how we extrapolate every trap to a common reference (τ at 135 K, "τ135"), which is what the
whole downstream analysis is built on.

### The problem
At the **hottest** temperatures (above ~165 K), the measured τ sits **~0.1–0.2 dex below** the
straight line drawn through the cold points — that's a factor of ~1.3–1.6 too short. Instead of a
straight line, the data **curve** (a "curved Arrhenius"). Concretely: with the corrected (smaller)
error bars, a single straight SRH line then **fails** the goodness-of-fit test for **~650 of 2,135**
legacy traps.

### Why it matters to the science
When you fit a straight line through *all* points, the leaning hot points **flatten the slope**.
A flatter slope means a **shorter inferred τ135**, which **shrinks the long-lived tail** — the
fraction of traps that hold charge longer than 1 hour at 135 K drops from **2.4% → 1.1%**. That
long-lived tail is *exactly* the population that drives the exposure-dependent upper-limit slope
(`physics.qmd` §4.4: the excess lives at τ > 1e5 s, dominated by τ ∈ [1e6, 5e7]). So this bias
feeds straight into the headline number. It is **not** cosmetic.

### The fork — two possible causes
1. **Instrumental (fixable): a t_ph-dependent pedestal.** The "pedestal" is a baseline charge
   offset in each image. Dark current accumulates *during* the 3,000-cycle pump train used to make
   the trap visible; the longer the pump dwell time `t_ph`, and the hotter the sensor, the more dark
   current piles up. The current fit subtracts a **constant** offset, which does not capture an
   offset that **grows with t_ph**. At hot temperatures that un-modeled growing pedestal biases the
   fitted τ downward — producing exactly the observed lean. If this is the cause, it's an instrument
   artifact and can be removed by adding a t_ph-dependent pedestal term to the intensity fit.
2. **Physical (real, quote as systematic): temperature-dependent cross-section σ(T).** The capture
   cross-section may genuinely depend on temperature. If so the curvature is real physics and must
   be reported as a systematic. **Important guardrail:** the SRH law already carries a fixed `T^-2`
   prefactor — we must **not** fit an empirical power to force the line straight, because that would
   hide the real physics under a fudge factor.

### Status and how to settle it
- Unresolved. It blocks a clean regeneration of the probability-of-measuring and
  completeness-efficiency figures.
- **Test:** add/measure a t_ph-dependent pedestal term in the intensity fit (needs the FITS +
  `getDipoleSpectra2` stage) and re-fit the hot temperatures.
  - If the lean **disappears** → it was instrumental; fix it and regenerate the catalog.
  - If the lean **persists** → it's σ(T); accept it and quote it as a systematic on τ135 / the
    long-τ tail.
- [x] Run the t_ph-pedestal test. *(Done 2026-07-03 — see the resolution block at the top of this
      item; the lean is not instrumental.)*
- [ ] Whichever way it lands, record the long-τ-tail number as a bounded systematic in the
      results-page ledger (item 2). *(Still open, but the numbers changed: the minimal-native
      bracket gives tail unchanged / driver band −7%, superseding the inter-catalog 0.011-vs-0.024
      figure — see the resolution block above.)*

(Details in memory `high-t-arrhenius-lean` and `signed_refit.qmd` §6b.)

---

## 4. `[x]` The two carried-as-knob assumptions — **both judged not a big deal** (Ansh, 2026-07-03)

**Disposition:**
- **4A (½ phase fraction):** not a big deal — bounded by construction (run 0/0.5/1.0 if a number is
  ever needed), carried as a knob.
- **4B (phase-blindness ×1.5):** not a big deal, and the inflation is **not correct** anyway. We
  cannot probe traps under V2, but V2 traps also **do not behave the same way** — V2 is where charge
  is *held* during readout, so those traps operate in a different regime than the pumped edge-phase
  traps. Scaling the measured density up by ×1.5 as if the unprobed phase held equivalent traps is
  therefore the wrong model. Do **not** apply `--upper-density-scale` as a blanket ×1.5 on physical
  grounds. Leave the definitions below for reference.

Both were stated honestly as assumptions rather than measured quantities. Here's what each one *is*,
and whether it's problematic (superseded by the disposition above).

### 4A. The ½ phase fraction (`v3_phase_fraction = 0.5`)

**What it is.** A CCD pixel has three gates ("phases": V1, V2, V3). During readout, charge parks
under V2 between transfers and briefly crosses V1 and V3 during each row-to-row transfer. A trap
sits under one specific phase, and *which* phase changes whether an electron the trap just emitted
can be immediately re-captured:
- A **V3 trap** is crossed by the charge packet on the packet's way *out* of the row, *after* it has
  already picked up the trap's emission — so that emitted electron faces a same-step recapture roll
  (it can be re-caught), and in the 3-hour drain its escape is slowed.
- A **V1 trap** is crossed on the way *in*; capture is checked *before* the trap emits, and any
  electron it then emits leaves over V3 without recrossing the trap — so it **always escapes**.

**The assumption.** Pocket pumping cannot tell a V1 trap from a V3 trap — they look identical in the
measurement. So we don't know, per trap, which it is. We assign each trap a coin flip with
probability `v3_phase_fraction = 0.5` (half V3, half V1). With symmetric clocking, roughly half the
pumped traps being V1 and half V3 is a reasonable neutral default — but it is a **default, not a
measurement**.

**Is it problematic?** *Moderately, but bounded.* The A/B test proved the value **matters**: going
from all-V3 to half-V3 moved the masked excess by intercept **+45%**, slope **−3.2%** — i.e. it
controls how the effect splits between "prompt / per-image" (intercept) and "exposure-correlated"
(slope) release. But it is **fully bounded**: the two physical extremes are `0.0` (all V1) and `1.0`
(all V3), and we can simply run both ends. So it is not a showstopper — it's a systematic we can
*bracket by construction*.
- [ ] *(Deferred by the disposition above — run only if a quoted band is ever needed.)* Run the
      campaign at `v3_phase_fraction` = 0.0, 0.5, 1.0 and quote the slope/intercept spread as a
      systematic band in the results ledger. The 2026-07-02 A/B already brackets 1.0 vs 0.5
      (intercept +45%, slope −3.2%); only the 0.0 end is unmeasured.

### 4B. Phase-blindness density inflation (×1.5, `--upper-density-scale`)

**What it is.** Pocket pumping only makes a trap visible if the trap sits in a phase the pumping
actually **probes** (it shuffles charge between the edge phases). A trap sitting under a phase that
pumping never fills-and-empties produces **no dipole** — it is invisible to the measurement. But
during **readout**, charge passes through **all** phases, so a trap in an unprobed phase still
perturbs the single-electron background. If pumping probes 2 of the 3 phases, then the *true* trap
density is about **×1.5** (3/2) the measured density — there is roughly one unmeasured phase's worth
of traps that we never counted but that still matter for the science. This is applied as an optional
multiplier on the upper-limit population.

**Is it problematic?** *This is the more concerning of the two,* for three reasons:
1. It is **not a modeling choice** — it's a claim that the measurement is *structurally blind* to a
   real fraction of the traps that nonetheless affect the result.
2. It is **one-sided**: it can only make the hidden population *bigger*, so it inflates the upper
   limit directly (a 50% multiplicative bump on top of the efficiency correction).
3. The exact factor is **unverified**: whether pumping probes 2 of 3 (×1.5) or 1 of 3 (×3) phases,
   and whether the unprobed-phase traps share the measured τ/σ distribution, has not been confirmed.

**What to check.**
- [x] Read the pumping sequencer recipe and confirm how many of the 3 phases are actually pumped
      (settles ×1.5 vs ×3 vs 1.0). *(Done 2026-07-06 via item 6: with hole polarity the pump probes
      **V1 and V3** — 2 of 3 phases — and V2 traps cancel (pumping-blind), so the naive factor
      would be ×1.5; but the inflation stays rejected per the disposition, since the unprobed V2
      population is rest-well resident and behaves as saturated near-inert sinks, not like the
      measured transit-crossed traps.)*
- [x] ~~State whether the unprobed-phase traps are assumed to share the measured E/τ/σ
      distribution~~ *(superseded — the scale-up is not applied; the V2 population is carried as an
      explicit unbounded caveat in `completeness_efficiency.qmd` §8/§9 instead).*
- [x] ~~Carry the result as a one-sided systematic on the upper limit in the results ledger~~
      *(superseded — the ledger row records the **rejection** and its physical grounds, 2026-07-06).*

---

## 5. `[ ]` Is the *allowance's* error bar honest? — one-sided systematics on the upper limit (2026-07-05 review)

**Why it's here.** A read-through of the four notebook threads (physics, signed_refit,
completeness_efficiency, results) against the results-page ledger found that both halves of the
headline — the null *and* the 90% CL allowance — are carried by the same object: the **long-τ135 tail**
(τ135 ≳ 10⁵ s), which is caught *only* at hot temperatures, extrapolated *down* to 135 K, inflated by
1/ε, and given its occupancy by the deferred HEE model. Several of that tail's uncertainties are (a)
**not in the quoted ~25% UL band** and (b) **one-sided toward "more dangerous."** None overturn the
two-number story; all bear on whether the allowance's error bar is conservative. 5.1 and 5.2 push the
same way, so they compound.

**Code cross-checks done during the review (no action needed):** the sim seeds the **characterized**
count divided by four (`ccd_simulation.py:1178`, `(n_detected_traps/4)/(nrow·ncol)`) — the one-quadrant
geometry is correct, *not* a 4× over-density. Hygiene only: that constructor arg is still *named*
`n_detected_traps` with a stale "9333 detected" comment even though it is fed the 3798 characterized
count.

### 5.1 `[x]` The high-T lean bracket may not be conservative for the tail — **resolved 2026-07-06: it is; production is the conservative end**

`signed_refit.qmd` §6b identifies the lean's mechanism as real thermally-activated (multiphonon)
capture, σ(T) = σ_LT + σ_∞e^{−E_b/kT}, E_b ≈ 0.16 eV — then deliberately does **not** propagate it into
the τ135 extrapolation ("a diagnosis, not a pipeline change"), bracketing with variant A (straight SRH
fit through the leaning hot points = production) vs variant B (lean removed). But if activated capture
is *true*, then at 135 K σ → σ_LT (smaller than the hot σ the straight line fits), so the
physically-correct τ135 for a hot-caught trap is **longer** than variant A — *more* dangerous — while
variant B moves it *shorter*. So the [A, B] bracket may not contain the mechanism-implied value, and it
is one-sided in the *safe* direction for exactly the hot-only tail traps that drive the UL. The "≤7%,
tail unchanged" smallness is an A-vs-B statement, not a "what if the identified mechanism holds" one.
This is a **different question from item 3** (same lean): item 3 asks *how big*, this asks *is the
bracket the right sign for the tail*.
- [x] Add a sentence to `signed_refit.qmd` §6b / the results ledger noting the bracket is not
      guaranteed conservative for the tail. *(Done 2026-07-06 — caveat paragraph in §6b + ledger.)*
- [x] Add the third variant that extrapolates each hot-caught trap with the fitted activated σ(T)
      and quotes the τ135 shift (sign and size). *(Done 2026-07-06,
      `claude_scripts/high_t_pedestal_bound/lean_bracket_variantC.py` + render-time variant C in
      `make_figures.arrhenius_lean_bracket`. Result: the worry INVERTS — the intuition missed the
      slope renormalization. In the activated regime the apparent Arrhenius slope is E + E_b, so
      the straight production line extrapolates steeper to cold and OVER-lengthens τ135; analytic
      deep-regime overshoot ln(1+x)−ln(x), x = R e^{−E_b/k·135K} ≈ 1.4 → ≈ +0.24 dex, measured
      median for hot-only traps Δlog₁₀τ135(C−A) = −0.211 dex (97.8% move down, zero enter the
      >1 hr tail). Tail 1.47% under A/B/C; driver band 30/28/26. **Production (A) is the
      conservative, dangerous-side end; no bracket widening needed.** Documented in
      `signed_refit.qmd` §6b + results ledger.)*

### 5.2 `[x]` The quoted ~25% UL band appears to omit the amplitude-prior (faint-by-N) axis

`results.qmd` §1 and `completeness_efficiency.qmd` §8 state the band as "Poisson on ~20 detections +
Arrhenius τ(T) completeness." But `completeness_efficiency.qmd` §4 shows the faint-prior sensitivity is
large and one-sided: mean completeness @10³ s drops 0.992 → 0.920 → 0.744 (faint-by-2/4) → up to ~+34%
on 1/ε — *comparable to the entire quoted band*, and it only pushes the UL **up**.
- [x] State explicitly whether "Arrhenius τ(T) completeness" already subsumes the faint-prior axis. If
      not, fold the cached faint-by-2/4 sensitivity (Stage 05/10) into the headline band — cheap, the
      grids already exist. (memories `completeness-energy-fit-resolution`, `naive-dip-minimal-closure`)
      *(Resolved 2026-07-06. Audit answer: the "~25% band" was **never computed anywhere** — it was a
      recommendation from the 2026-06-24 τ-band session written into prose (Poisson 1/√20 ≈ 22%
      rounded up); no script produces it, no figure draws it, `paper.tex` doesn't quote it, and the
      "completeness systematic" half was named but never quantified. The faint-prior axis was
      therefore NOT in it. Driver-band sensitivity re-read from the Stage-10 cache: at
      τ = 10⁵/10⁶/10⁷ s, faint-by-2 (faint-by-4) inflates 1/ε ×1.4–2.8 (×2.7–9.3) — the @10³ s "+34%"
      badly understated it. Disposition (Ansh, 2026-07-06): **conditional headline, no fold-in** —
      the faint variants are unmeasured stress hypotheses (selection-truncated prior), so folding
      requires arbitrary weighting. Headline now quotes ±22% Poisson as the statistical error and
      carries the amplitude prior as an explicit conditionality with the driver-band stress numbers
      stated alongside. Prose fixed in `results.qmd` §1 + §5 ledger, `physics.qmd` §4.4 item 4,
      `completeness_efficiency.qmd` §8 + §8-end.)*

### 5.3 `[ ]` HEE occupancy is the master variable but is framed as a deferred nicety

Per memory `occupancy-controls-trap-sign`, the sign *and* magnitude of the whole trap effect track
occupancy, which tracks bright-event rate — supplied by the `physics.qmd` §7 HEE model: one 20 h MINOS
image's clusters, deterministically scaled, **variance-suppressed**, with MINOS-vs-SNOLAB as a pure 10×
normalization on the *same* library. Framing this as a "§7 deferred improvement" undersells it: it is
plausibly the dominant uncertainty on the effect *scale*, and its suppressed event-count variance means
the run-to-run spread — hence the significance claimable on the UL slope — is likely optimistic.
- [x] Promote HEE from `physics.qmd` §7 forward-TODO to a named line in the results-page systematics
      ledger (scale uncertainty + suppressed variance + shared MINOS/SNOLAB spectrum), even while the
      Poisson-bootstrap *implementation* stays deferred. *(Done 2026-07-06 — ledger row + §7 cross-ref.)*

### 5.4 `[ ]` Traps initialize empty — the long-τ reservoir never sees an operational history

`ccd_simulation.py` sets `trapped_charge_1d = np.zeros(...)`; the reservoir fills from empty over the
~500-image run. The τ_max-convergence finding (`physics.qmd` §4.4 #3) handles the τ > 5×10⁷ tail (never
emits → no deferral) and the dominant τ ∼ 10⁶ band equilibrates within the run, so this is *mostly*
closed — but a real 135 K detector has months of prior filling history the sim does not reproduce, so
the steady-state occupancy of the τ ∼ 10⁷ band is *trusted via convergence*, not demonstrated.
- [x] One line in `physics.qmd` §4.3/§6 noting empty initialization + that long-τ occupancy rests on
      the convergence argument, not a demonstrated equilibration. *(Done 2026-07-06 — §6 item 8b.)*

### 5.5 `[ ]` Term D (cluster→1e redistribution) is the one un-pinned positive budget term

`physics.qmd` §4.3 already says the missing positive count term *is* cluster→1e redistribution and that
it is "not proven by these counters." Fine to leave, but it is more "open" than the §4.2 budget table
implies (asserted by sign, not quantified).
- [x] Tag term D in the §4.2 table as "sign asserted, magnitude not instrumented" so the budget's one
      genuinely open term is visible. *(Done 2026-07-06.)*

---

## 6. `[x]` Phase-geometry "contradiction" — **resolved same day (2026-07-06): p-channel polarity; Model-C geometry confirmed and now documented**

**What was found, then resolved.** A 2026-07-06 review flagged that `daq/temp_scan_run1_imgseq.xml`
holds V1|V3 HIGH / V2 LOW in every readout rest state, apparently contradicting the Model-C premise
("readout parks charge under V2") — and matching the 2026-06-12 phase-population audit, which had
read the same files and concluded charge rests under gates 1&3. **Ansh supplied the missing piece:
these are p-channel CCDs — the carriers are holes, which collect under the most *negative*, i.e. the
NON-asserted, gate.** With hole polarity the sequencer reads: readout rest = holes **under V2**
(V1/V3 asserted = barriers), and the pump cycle's hole well steps
V2[t_ph] → V3[7t_ph] → V2[t_ph] → V1[7t_ph]. Re-deriving the dipole formation on that sequence:
V1/V3 traps load during their long park and emit into the countable window **[t_ph, 8t_ph]** — the
canonical peaked `e^{−t/τ} − e^{−8t/τ}` the data follow, unidirectional — while V2 traps give a
non-peaked `1 − e^{−7t_ph/τ}` response into alternating sides that cancels (pumping-blind). So
**pumped/measured traps = V1/V3 and readout rest = V2, self-consistently: the phase-limited kernel
geometry is confirmed.** The 2026-06-12 audit's flagged shape anomaly is explained: it read the
state bits with electron polarity, swapping every gate label — its physics was right, its labels
inverted. One corrected implication: the unprobed V2 population is *rest-well resident* (not
barrier-resident) — it fills early and sits as saturated, near-inert sinks, which is the mechanism
behind the item-4B ×1.5 rejection (Ansh's "V2 is where charge is held" rationale was exactly right).

- [x] Reconciliation derived and written into `physics.qmd` §2.4 (blockquote note; ⚠ flag removed);
      glossary "clock phases" entry updated with the polarity subtlety.
- [ ] (Residual, minor) Confirm the vertical clock rail voltages from the DAQ-side
      `voltage_skp_lta_v2_C_minos.sh` (not in this repo) — the "asserted = high rail = hole
      barrier" convention is standard LTA and independently confirmed by the pump-shape argument,
      so this is belt-and-braces only.

---

## Not on this list (deliberately deferred, ≤0.3% / hygiene)

- 170 K detection-threshold = 17.0 suspected calibration artifact (`signed_refit.qmd` §6a) — ≤0.3%
  of catalog, deliverable-neutral.
- Noiseless simulated images justification "stated nowhere else" (`physics.qmd` §6.7).
- `charge_trap_simulation.ipynb` predates the current `CCD(...)` signature (`physics.qmd` §6.6).
- Deferred HEE Poisson-bootstrap sampling (`physics.qmd` §7) — forward TODO, not blocking.
