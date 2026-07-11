# Trap-branch deviation budget (phase_limited_v1v3 campaign)

Goal: for a trap-branch image at exposure `t` (hours), account for **every** deviation
of the measured single-electron density from the injected truth, with a mathematical
form per term. Reviewed adversarially by Codex.

## 0. Setup and notation

Per quad: `N_pix = 512 × 3072 = 1.573e6`. Injected truth (recovered by the no-trap branch
to <0.3% on intercept, <1.3% on slope across all 53 complete scenarios):

    d_true(t) = d_EI + d_ED · t,   d_EI = 9.94e-5 e/pix/img,  d_ED = 4.36e-5/24 e/pix/hr.

The two branches share the **same injected realization** (common random numbers), so the
deviation is purely trap-induced:

    Δ(t) = C_trap(t)/U_trap(t) − C_notrap(t)/U_notrap(t)        [density]
         ≈ ΔC(t)/U   +   C_nt · (1/U_trap − 1/U_notrap)          (numerator + denominator)

where C = single-e counts, U = unmasked-pixel count. Measured split (minos/upper, 20h):
87% numerator, 13% denominator.

Per-image pipeline (`take_fake_image`, ccd_simulation.py:1305):
1. `simulate_clear()` — retains trap occupancy; **emitted charge during clear is zeroed (lost, not counted)**.
2. `charge_trap_interaction(ccd_state, dt=t·3600)` — **integration emission**: occupied traps
   emit with `1−e^{−t/τ}`; empty traps do NOT capture (charge under V2).
3. inject dark (`n∝t`) + spurious (`n=const`) + clusters (transplant; snolab = t/10).
4. `simulate_readout()` — **readout**: per row-step, occupied→emit `1−e^{−τ_pix/τ}`,
   empty & q≥1 → capture `1−e^{−q·α}`, `α = σ v_th t_phase / V`, `t_phase = 20µs`.

Timing (computed): `τ_pix = 47.13 s`, `T_read = 512·τ_pix = 6.70 h` (bin1) or `12.6 min` (bin32,
divides τ_pix only). Clear (sequencer) = 3.26 s.

Key unification: within an image, integration (t) and readout (T_read) are consecutive and
both counted. A trap occupied at integration start yields a counted emission with prob

    P_emit_counted(τ; t) = 1 − e^{−(t + T_read)/τ}.            (★)

## 1. The deviation terms

For each: mechanism, math, sign, t-dependence, binning, population/condition scaling, evidence.

### A+C. Reservoir emission (occupied traps emit into the counted image)  — DOMINANT
Mechanism: occupied traps emit during integration (A) or readout (C); both counted.
Math:  ΔA+C(t) = ⟨s⟩ · Σ_i o_i · (1 − e^{−(t+T_read)/τ_i})        [from (★)]
  - o_i ∈ {0,1} occupancy at integration start; ⟨s⟩ = survival (not recaptured/masked/merged) ≤1.
  - Sign +. 
  - t-dependence: rises, saturating to ⟨s⟩·N_occ as t→∞. This is THE slope driver and most of the +intercept.
  - Binning: enters only through T_read. At t≫T_read (20h) → ≈1, binning-INVARIANT.
    At t=0 → 1−e^{−T_read/τ}, binning-SENSITIVE.
  - Population: ∝ N_occ. Upper N_occ≈10–12k (55–69% of 17475); baseline ≈8 (0.2% of 3798) → null.
  - Condition: ~independent (reservoir built from dark+spurious+cluster captures; the dark/spurious
    part is identical minos/snolab) → the common floor.
Evidence (quantitative):
  - 20h raw Δcount binning-invariant: minos 35.7(bin1)/33.4(bin32); snolab 57.2/59.9. ⇒ confirms the
    t-term of (★) (binning enters only via T_read, negligible at 20h).
  - 0h raw Δcount binning-sensitive: minos +4.3→−18.9; snolab −22.3→−31.8. ⇒ confirms the T_read term
    (bin1 0h emission `1−e^{−6.7h/τ}` ≫ bin32 `1−e^{−0.21h/τ}`).
  - 2D regression: at fixed exposure, Δcount rises with occ (corr 0.6–0.93); exposure & occ
    uncorrelated (r≈0.003) ⇒ A (occ·t-factor) is separable and real.

### B. Readout capture of spurious charge (sink)  — sets the −intercept (pre only)
Mechanism: empty traps capture injected spurious 1e during readout; held → deferred to a later image.
Math:  ΔB = − N_spur · p̄_cap,   N_spur = d_EI·N_pix ≈ 156/img (exposure-INDEPENDENT),
  p̄_cap = P(a spurious e shares a column-path with an empty trap that fires) 
        ≈ (n_empty/N_pix)·(geom)·(1−e^{−1·α}).
  - Sign −. t-dependence: ~constant (intercept). Binning-invariant (α fixed). 
  - pre_readout ONLY: post injects spurious after readout ⇒ ΔB=0 ⇒ intercept less negative.
  - Population ∝ n_empty.
Evidence: snolab/upper 0h deficit at LOW occ tertile = −34 counts/img (many empty hungry traps);
  at HIGH occ = −15 (fewer empty). Baseline 0h ≈ 0 (n_empty tiny). Direct pre/post test BLOCKED
  (snolab_upper_post dirs incomplete).

### D. Readout smearing / multi-e splitting (CTI redistribution)  — +, mostly masked
Mechanism: empty trap captures from a multi-e (cluster) pixel and re-emits downstream into an empty
pixel ⇒ splits concentrated charge into extra 1e counts (charge-conserving).
Math:  ΔD(t) ≈ Σ_clusters (captured carriers)·P(re-emit into empty unmasked pixel).
  Cluster charge ∝ event rate × accumulation ∝ t (clusters accrue with integration). 
  - Sign +. t-dependence: grows with t. Binning: partly sensitive (re-emit uses readout emission).
  - Condition: ∝ cluster rate ⇒ minos ≫ snolab UNMASKED. But landed near clusters ⇒ removed by halo/bleed.
Evidence: UNMASKED Δslope minos/snolab = 2.86e-4 / 9.10e-5 = 3.1× (cluster part present) but NOT 10×
  (so D is a minority of the unmasked slope; A dominates). Masked: minos slope loses 64%, snolab 25%.

### E. Readout capture of dark current (sink)  — −, small
Math: ΔE(t) = − N_dark(t)·p̄_cap, N_dark(t)=d_ED·N_pix·t ≈ 2.86·t(h) e (≈57 e at 20h).
  - Sign −, t-dependent. Magnitude bounded by dark budget: ≤ p̄_cap·57 ≪ measured slope ⇒ negligible.

### F. Masking-denominator coupling (analysis term, not charge)
Math: extra term C_nt·(1/U_trap − 1/U_notrap). Trap-deferred charge/smearing creates extra
  hot-column/bleed flags ⇒ U_trap < U_notrap ⇒ inflates trap density even at fixed counts.
Evidence: 13% of minos/upper 20h density excess; this is why minos/upper looks 2× snolab in DENSITY
  while in raw COUNTS snolab ≥ minos. Report upper comparisons in counts.

### G. Clear-mode modulation of N_occ (scales A+C, B)
Mechanism: clear sets the reservoir entering each image.
  - sequencer: retains occupancy; clear emission `1−e^{−3.26s/τ}` (tiny for long τ) ⇒ full reservoir.
  - three_hour: 3h drain empties occupied traps via pure emission `e^{−3h/τ}` survival ⇒ N_occ small
    ⇒ A+C suppressed, n_empty large ⇒ B enhanced ⇒ biggest DEFICIT, flattest.
  - binned_0h: no hardware clear; 0h binned readout resets free charge, occupancy carries over.
Evidence: clear3h shows the most negative intercepts in the baseline rows; upper clear3h not yet
  isolated here.

### H. 2e channel coupling (second-order)
Capture 2e→1e (+1e count, −1 2e); emission 1e→2e. Δ2e measured POSITIVE and growing (minos/upper 2e
~3× with traps at 20h) ⇒ net charge piling (emission/redeposit), consistent with A+D, inconsistent
with a pure multi-e→1e sink.

## 2. Mapping to the two fitted channels

Exposure-INDEPENDENT deviation (intercept, t→0):
    Δ_EI = ⟨s⟩·Σ_i o_i(1−e^{−T_read/τ_i})   −  N_spur·p̄_cap   (+ small D,E,F)
         = [reservoir readout-emission, +, binning-sens]  −  [spurious capture, −, pre-only]
    Net sign = occupancy balance: hot/full reservoir → +, cold/empty + pre → − (deficit).

Exposure-DEPENDENT deviation (slope, d/dt):
    Δ_ED = ⟨s⟩·Σ_i o_i (1/τ_i) e^{−(t+T_read)/τ_i}  +  dΔD/dt  −  dΔE/dt  (+ denominator slope)
         = [integration-emission ramp, +, binning-INVARIANT]  +  [cluster splitting, +, masked]  − [dark capture, −, tiny]
    Dominated by the integration-emission ramp (A). Confirmed binning-invariant.

## 3. Per-scenario sign/magnitude predictions

| axis | effect on terms | predicted result |
|---|---|---|
| baseline vs upper | N_occ: 8 vs ~11k; τ short vs hours-day | baseline NULL both channels; upper LARGE |
| minos vs snolab | D,F ∝ clusters (10×); A,B,E common | counts: similar (A floor); density: minos inflated by F |
| pre vs post | B only in pre | post intercept less negative (B→0) |
| clearseq vs clear3h | G: 3h drains N_occ | clear3h → smaller A+C, larger B → deficit, flat |
| bin1 vs bin32 | T_read 6.7h→12.6min in (★) | 20h Δ ~unchanged; 0h Δ more negative; fitted slope steeper |
| vp1 vs vp3 | α ∝ 1/V → capture | weak; baseline negligible, upper mild |

## 4. Open / unverified (flag honestly)
- ⟨s⟩ survival factor not measured; A is an upper bound without it.
- Per-τ occupancy o(τ) not stored; Σ_i o_i(…) not yet evaluated against the seeded τ histogram
  (closed-form steady-state occupancy model needed). The hours-to-day band is argued, not summed.
- B magnitude (p̄_cap) not computed from trap geometry; only sign+ordering shown.
- D vs A split in the unmasked slope is bounded (3.1× vs 10×) but not cleanly separated.
- pre/post and clear3h-upper isolation blocked by incomplete dirs.

## 5. Codex adversarial review — corrections (2026-06-23)

What SURVIVED: pipeline ordering (clear/integration/injection/readout), clear charge loss
(ccd_state zeroed, :1300), no static capture during integration (B pre-only), post-readout
removal of B (:1380-1392), denominator coupling F, and the POSITIVE sign of integration emission A.

What was WRONG / overstated:
1. **(★) is too clean.** Readout-counted emission is NOT over the full T_read. In
   fast_readout_numba the row is OUTPUT before trap interaction (:802 vs :809), so an emission by a
   trap at row `tr` on shift `t` is read out this frame only if `t < tr` (lands in a still-to-be-read
   packet); for `t ≥ tr` it is copied back into `image` (:838-840) and DISCARDED by the next clear.
   So the counted readout window is row-position-dependent (~half the rows on average), not full
   T_read. (★) should read `P_counted ≈ (1−e^{−t/τ})_integration + [position-weighted readout term]`.
   The QUALITATIVE binning behavior (20h invariant via the integration term; 0h sensitive via the
   readout term) still holds; the closed form does not.
2. **⟨s⟩ is NOT ≈1.** An integration-emitted electron can be (a) immediately RE-captured by the same
   trap during readout (:817-827 tests capture right after emit), (b) lost to 1e-classification if it
   merges with a neighbor/cluster (:1402-1406), (c) masked/hot-column removed (:1496-1582). Survival
   is a free, un-measured factor — A is only an upper bound without it.
3. **MAGNITUDE reframing (most important).** Net Δcount ≈ 36–57 per ~11000 occupied traps =
   **0.3–0.5% per occupied trap**, not ≈1. So the deviation is a small RESIDUAL of large opposing
   flows (A+C emission MINUS B+recapture+E sinks), not a clean "every occupied trap emits a count."
   The naive (★)≈1-at-20h overpredicts the per-trap NET by ~200×. Either survival is ~0.5%, or the
   counted-emission τ-band is far longer than "hours-to-day." Cannot claim the band without summing
   o(τ)·survival explicitly.
4. **A/C/D/E are NOT mutually exclusive.** One shared capture/emission state machine; carriers flow
   across images (a dark/spurious/cluster electron captured in one image becomes reservoir emission
   in a later one). The terms are separable only by tagging each carrier's origin+transitions.
5. **binned_0h fits EXCLUDE the 0h point** (figure_utils.py:1541-1544). The "lower 0h intercept
   steepens the fitted slope" reading applies only where 0h is INCLUDED (clearseq bin32, which is
   what §1 evidence used) — NOT for binned_0h scenarios.

BIGGEST HOLE: the counted-emission **survival kernel** (row-dependent readout visibility +
immediate/downstream recapture + 1e/mask survival) is unmodeled; it sets ⟨s⟩ and the true τ-band.

CLOSING CALCULATION: instrument ONE run with per-carrier counters — for initially-occupied traps
binned by τ and row, tally emissions that are integration-counted / readout-counted / recaptured /
merged-non-1e / masked / lost-to-clear. That replaces (★) with the measured kernel response and
fixes the magnitude/τ-band attribution.

## 6. Instrumented closure — measured kernel response (2026-06-23)

Ran `_instrument_run.py` (origin-tagged readout + integration emission, minos/upper/clearseq/pre,
60 cycles, equilibrated occ 62% = 12.4k of 20k traps). Fate funnel per image (mean), 20h:

| quantity | bin1 | bin32 | binning |
|---|---|---|---|
| readout emissions Eread | 19121 | 10268 | SCALES ~1.9× |
| self-recaptured (same step) | 17366 (91%) | 8686 (85%) | scales |
| escaped (Eread−selfrec) | 1754 | 1582 | ~invariant |
| integration emission E_int | 254 | 294 | invariant (small!) |
| net e⁻ reaching readout (NETcnt) | 1917 | 1803 | ~invariant |
| injected-1e SINK | 2022 | 1851 | ~invariant |
| **net Δ1e unmasked** | 402 | 834 | not invariant |
| **net Δ1e halo+bleed-masked** | **61** | **68** | **INVARIANT** |

VERDICT — three budget claims OVERTURNED by the measurement:
1. **"A integration emission dominates the slope" is WRONG.** E_int is only ~254/img (the occupied
   reservoir is very-long-τ, P_emit(t)≈0). The bulk of NETcnt is escaped readout emission.
2. **Readout emission is ~91% futile** (immediate self-recapture; capture saturates for the
   occupied large-σ traps). So emission flux ≠ effect.
3. **The masked observable is CAPTURE-LIMITED, not emission-limited.** Eread is binning-dependent
   (19k→10k) but the masked net Δ1e is INVARIANT (61→68), as are the sink and NETcnt. The capture
   rate α=σ·v_th·t_phase/V is fixed by the 20µs phase window ⇒ binning-invariant. The effect is
   driven by CAPTURE of exposure-accumulated image charge during readout (∝ exposure ⇒ slope;
   α-fixed ⇒ binning-invariant) + its redistribution; the huge emission flux is mostly self-
   recapture cycling that cancels. This RESTORES the original "capture/splitting during readout"
   reading (recanted after the naive binning argument) — now with instrumented proof — and RETIRES
   the integration-emission story.
4. **Survival ⟨s⟩ ≈ 61/19121 ≈ 0.3%** (masked), ≈2% unmasked. Net is a tiny residual of two
   ~2000-electron opposing flows (NETcnt ≈ sink). Confirms Codex's magnitude correction exactly.

STILL OPEN: precise scalar reconciliation of net Δ1e (+402 unmasked) against NETcnt(1917) − sink(2022)
is loose (multi-e/cluster captures don't cost a 1e count; splitting adds 1e; masking removes ~85%);
and τ-resolved counters are saved (_instr_counters.npz) but not yet reduced to the contributing band.

## 7. Final Codex review — conclusions TEMPERED (2026-06-23)

Codex full review of §6 + _instrument_run.py vs the production kernels. Instrumentation is
*faithful* (readout order/indexing, post-emission capture test, cti all match production) but the
counters do NOT CLOSE the observable. Corrections:

- **The flow counters are not the observable.** NETcnt/sink/recap are carrier-flow tallies BEFORE
  cluster classification + halo/bleed masking; Δ1e is a nonlinear count of *isolated* 1e clusters.
  So NETcnt−sink (=−105) is NOT expected to equal Δ1e (+402); the sign disagreement just means the
  flow model is incomplete. The missing POSITIVE term is **cluster/packet redistribution: multi-e
  (cluster) charge split into isolated 1e by capture/re-emission** — this creates 1e counts without
  a net positive emission, and the emission/sink counters can't see it. THIS is the +402 driver.
- **Origin-tagging is biased in mixed packets.** `por>0 ⇒ recap` is wrong when q>por (a capture
  from a pixel holding both a trap electron AND injected/cluster charge may have taken the injected
  one; should attribute ~por/q). Overcounts selfrecap/recap_other, undercounts sink. So the **91%
  self-recapture is biased high** (real claim: "mostly futile", not literally 91%).
- **occ_before is mislabeled** — recorded AFTER readout in the script (line ~139), so it is
  occ_after; occupancy-trend statements need relabeling. (Occupancy ~stable so conclusions hold.)
- **Equilibration NOT demonstrated.** 300 images ≈ 200 days sim-time; τ≥1e7 s (months-yr, the bulk
  of the upper-limit tail) cannot equilibrate and occ was still rising. So "E_int small / reservoir
  very-long-τ" is a finite-run result, not asymptotic.

VERDICTS: C1 (0.3% survival) OVERSTATED — qualitative "tiny residual of large flows" holds, exact
ratio not (denominator=gross emission≠all observable charge). C2 (integration emission not dominant)
SUPPORTED for this run state, with equilibrium caveat. C3 (readout emission mostly futile) SUPPORTED
qualitatively, 91% biased high. C4 (capture-limited/binning-invariant) OVERSTATED — the MASKED
residual is ~binning-invariant but the UNMASKED Δ1e DOUBLES (402→834) ⇒ binning-DEPENDENT; masking
may simply remove the binning-dependent smearing. Capture-limited is plausible, NOT proven.

DEFENSIBLE HEADLINE (what actually survives): "Under minos/upper/clearseq/pre, the halo+bleed-masked
trap excess is a small residual after large readout capture/emission/recapture flows, and is
~invariant bin1↔bin32; gross readout emission is mostly cancelled by immediate recapture; integration
emission is small in this run. The UNMASKED deviation is binning-DEPENDENT, the +ve count term is
cluster→1e redistribution, and the capture-limited interpretation is plausible but unproven by these
counters."

TO TRULY CLOSE IT (Codex's prescription, not yet done): (a) track per-carrier/probabilistic origin
THROUGH cluster-classification + mask fate (so the counters reproduce Δ1e), and (b) demonstrate
long-τ equilibration via a longer run or an analytic stationary occupancy calc.

## 8. τ-band of the slope, measured by seed-capping (three_hour/upper, 2026-06-24)

Closes part of the §4 open item "Σ_i o_i(…) not yet evaluated… the hours-to-day band is argued,
not summed." Method: cap the minimal_caldet **upper-limit** seed at successive τ_max and re-run
(MINOS, **three_hour** clear, 30 trials/cap, only τ_max varied), reading the fitted slope/intercept
from `plot_simulation_results`. This empirically *sums* Σ o_i(…) by removing τ bands one at a time.

| seed τ_max | N_traps | slope (exp-dep) | intercept (exp-indep) |
|---|---:|---:|---:|
| 1e5 | 4179 | 1.842e-6 (≈ no-trap truth 1.82e-6) | 9.91e-5 |
| 3e5 | 4386 | 1.846e-6 | 9.96e-5 |
| 1e6 | 5529 | 2.110e-6 | 1.006e-4 |
| 5e7 ("full" prod cap) | 17475 | 2.816e-6 | 9.96e-5 |
| 1e8 | 29967 | 2.853e-6 | 9.90e-5 |
| 3e8 (unbounded UL) | 66087 | 2.827e-6 | 9.74e-5 |
| **baseline (characterized, no 1/ε)** | **3798** | **1.818e-6 (= truth)** | **9.93e-5 (= truth)** |

No-trap branch recovers truth every cap (slope 1.80-1.83e-6, intercept ~9.92e-5) ⇒ harness faithful.

FINDINGS (what this corrects in §§1-3,G):
1. **The slope band is τ > 1e5 s (days-to-months), NOT "hours-to-day."** With τ<1e5 traps only
   (cap 1e5) the slope ≈ the no-trap truth — traps with τ < 1e5 s add ~ZERO exposure-dependent
   excess. This **contradicts the "τ short vs hours-day" tag in §1.A+C (line ~51) and the §3 table
   (line ~124)**, and confirms (sharpens) §6's "the occupied reservoir is very-long-τ." The entire
   trap slope excess (+1.0e-6 over truth at the 5e7 cap) lives at τ>1e5, with τ∈[1e6,5e7] (11946
   traps) contributing the largest chunk (+0.71e-6) and τ∈[3e5,1e6] the next (+0.26e-6).
2. **For the UPPER population, three_hour does NOT flatten/suppress A+C** — directly fills the
   §G/§3 gap "upper clear3h not yet isolated." §G's "three_hour → N_occ small → A+C suppressed →
   flattest, biggest deficit" holds only for drainable τ. The upper reservoir is τ≫3h, where the 3h
   drain survival e^{−3h/τ}≈1 (e.g. τ=1e6 → 0.989), so it survives the clear OCCUPIED and produces a
   LARGE positive slope (2.82e-6, +55% over truth) under three_hour. The flatten-on-3h prediction is
   a BASELINE (short-τ) result; it inverts for upper.
3. **The slope-driving band is τ ∈ [3e5, 5e7] s and the slope CONVERGES in τ_max** (revised from the
   first-draft "does not converge" — the extended runs refute that). Slope excess over truth, by band:
   τ<1e5 ≈0; [1e5,3e5] ≈0; [3e5,1e6] +0.26e-6; [1e6,5e7] +0.71e-6 (dominant); [5e7,1e8] +0.04e-6;
   [1e8,3e8] ~0. Per-trap contribution collapses ~300× across the turnover (5.9e-11 in [1e6,5e7] →
   ~2e-13 in [5e7,3e8]): leakage-per-trap → 0 as τ→∞ (a trap that never emits within the run defers
   no charge), so even the 48612 monster 1/ε-inflated traps at τ>5e7 add <1%. **The production
   correction_tau_max=5e7 sits past the turnover — extending to an unbounded 3e8 UL leaves the slope
   unchanged (2.82e-6). So the UL is NOT sensitive to correction_tau_max above ~5e7.**
4. **The entire UL slope is 1/ε-inflation headroom, not a measured effect — the baseline is NULL.**
   The real characterized population (3798 traps) gives slope 1.818e-6 = truth (excess ≈0), confirming
   the §1 line ~51 / §3 line ~124 "baseline NULL both channels" prediction. The 2.8e-6 UL slope is the
   90% CL efficiency-corrected *allowance* for hidden long-τ traps (τ~3e5-1e7, ε~0.01-0.07), where ~40
   measured traps are amplified ~300× by 1/ε. So the exposure-dependent UL is a genuine upper-limit
   construct on an unobserved population. **Treatment (corrected after review): do NOT floor it.** The
   ~21 characterized (decoy-rejected) traps at τ>1e6 are real evidence the long-τ region is populated;
   n/ε is the legitimate hidden-population estimate and an ε-floor would assert the null by fiat. The
   deliverable is TWO numbers — measured/characterized population = NULL (finding 4 above), and the
   efficiency-corrected 90% CL UL = ~2.8e-6 (slope) — and 2.8e-6 should carry an ERROR BAND from
   Poisson on the ~20 anchoring detections (~25%) + the Arrhenius τ(T) completeness systematic
   (cf. high-T lean), NOT a cut. The only principled exclusion is ZERO-detection bins where 2.3/ε is
   pure Poisson-floor amplification (e.g. the empty τ=2.7e-5 cold bin, leakage-dead) — distinct from
   the populated low-ε bins, which stay. (Supersedes the first-draft "floor at the calibration
   boundary ε≈0.1" reading of this section.)

Note vs §6/§7: §6 (capture-limited, emission futile) was a **clearseq** instrumented run; this is
**three_hour**, where the surviving reservoir is specifically the undrainable long-τ tail. The two
are not in conflict on mechanism — both point away from the original "hours-to-day integration
emission" — but the clear modes select different occupied τ-bands. See memory
`exposure-dependent-slope-is-low-eps-tail`.
