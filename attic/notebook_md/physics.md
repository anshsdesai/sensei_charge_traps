# Simulation physics: emission, capture, recapture, and the deviation budget

> **In plain terms.** This file is the *simulation* half of the project — the fake dark-matter
> exposures we run to see how much traps shift the single-electron count. The hardest thing to get
> right was *how a trap grabs charge as it is clocked past during readout*. We tried three models; the
> first two were wrong. The current one, `phase_limited_v1v3`, says a trap can only grab charge during
> a brief 20-microsecond window as the charge crosses it. §4 is the most important and most up-to-date
> part: a careful accounting (checked three times, including by directly instrumenting the code) showing
> that the trap effect surviving the normal data-cleaning is a *tiny leftover* (~0.3% per trap) of much
> larger grab-and-release flows that mostly cancel out. **If short on time:** §1 and §3 are stable
> background, §2 is the model history (only Model C is live), §4 is where things stand. New words are in
> the [glossary](glossary.md).

**Thread status:** current model = `phase_limited_v1v3` (readout capture in a ~20 µs transfer
window, full-dwell emission, no full-row recapture). This file folds in the full history of how
the trap-interaction model reached that point, because two earlier models are still referenced in
the code comments and in older notes.

**Reading order.** §1 (emission physics) and §3 (packet volume / σ) are stable and were never
overturned. §2 is the model's history — read it as a narrative of three successive models, only
the last of which is live. §4 is the current deviation accounting (the newest and most reliable
statement). §5–§7 are operating-condition decisions and a deferred improvement.

Sources consolidated here (now in `attic/`): `TRAP_SIMULATION_PHYSICS.md` (2026-06-11),
`RECAPTURE_PUMPING_HANDOFF.md` (2026-06-19), `DEVIATION_BUDGET.md` (2026-06-24),
`HIGH_ENERGY_EVENT_SAMPLING_NOTE.md` (2026-06-12).

---

## 1. Emission-time physics (verified against literature — never overturned)

**What this section is:** the formula linking a trap's depth (E) and size (σ) to how long it holds
charge (τ) at a given temperature. It is standard textbook physics, we checked it matches the
literature, and it never changed. It's used two ways: to fit E and σ from a trap's measured
τ-vs-temperature points, and to extrapolate τ to the 135 K operating temperature.

The per-trap emission time follows Shockley–Read–Hall detailed balance, as used throughout the
CCD trap-pumping literature (Hall 1952; Hall et al. IEEE TNS 61, 1826 (2014); Wood et al.; Oscura
arXiv:2406.18502; Brusco arXiv:2510.23336):

```
tau_e(T) = exp(E_t / kT) / (sigma * v_th * N_v)
v_th = sqrt(3 kT / m_cond)               (thermal velocity, holes)
N_v  = 2 (2 pi m_dens kT / h^2)^(3/2)    (valence-band effective DOS)
```

*In words:* holding time τ rises steeply (exponentially) with trap depth E and falls as the sensor
warms (larger kT), while a bigger cross-section σ shortens it. `v_th` and `N_v` are bundled
temperature-dependent constants (see [glossary](glossary.md)); everything not explicitly modeled is
absorbed into the fitted effective E and σ.

SENSEI CCDs are **p-channel; carriers are holes.** Effective masses for 100–200 K (Green 1990, as
adopted by arXiv:2406.18502): `m_cond = 0.41 m_e`, `m_dens = 0.94 m_e`. Note
`tau_e ∝ T^-2 exp(E/kT)` — the standard DLTS form. The degeneracy/entropy factor is taken as 1 and
temperature-independent σ is assumed; both are conventional and absorbed into the fitted effective
(E, σ).

The pumping dipole-intensity model
`I = N_pumps · D_t · P_c · (exp(-t_ph/tau) - exp(-8 t_ph/tau))` matches the Oscura derivation; the
emission window is `[t_ph, 8 t_ph]` for this clocking scheme (the factor 8 is the pump duty ratio,
`delay_Tph_long = 7 × delay_Tph`).

### Constants bug (fixed 2026-06-11 in `dipole.log_energy_cross_section`)

The original implementation had three constant errors:

| quantity | was | should be | effect on fitted value |
|---|---|---|---|
| k_B | 8.6717333262e-5 (typo) | 8.617333262e-5 eV/K | fitted E biased +0.63% |
| h | 4.1135e-15 (typo) | 4.135667696e-15 eV·s | h³ off by −1.6% |
| mass prefactor | bare m_e | m_dens^{3/2}/m_cond^{1/2} = 1.4233 m_e | fitted σ biased ×1.42 |

**τ_e(135 K) predictions are exactly invariant under these fixes.** The fit family
`log tau = const − log sigma − 2 log(kT) + E/kT` is unchanged, so the constant errors were
absorbed entirely into the fitted (E, σ). Verified bit-for-bit: rehistogramming the refit τ135
values reproduces `tau_at_135k_hist.npz` with zero bins changed. Net effect of the fix: quoted
σ → ×0.722, quoted E → ×0.9937, all τ-based inputs unchanged. All `tau_at_<T>k_hist.npz` files
therefore remained valid.

---

## 2. The readout trap-interaction model — three successive versions

The simulation clocks charge row-by-row through each trap. How a trap captures and releases charge
during that transit went through **three models**. Only the third is live. This section explains
each and why the transition happened, because the failure of each motivated the next.

### 2.1 Model A — pure emission, no recapture (original; WRONG)

Original `fast_readout_numba`: captured deterministically (P_c = 1 whenever a packet with ≥1
carrier passed an empty trap) and released with `P = 1 − exp(−t_dwell/tau_e)` per row transfer, the
released carrier rejoining the packet above the trap **and never being recaptured.**

Why it was wrong: instant capture taken seriously implies an emitted carrier sitting over the same
trap is immediately recaptured. With the fitted cross-sections (median σ ≈ 7.5e-17 cm²), the
capture time of a single carrier over its trap is ~ms–s, far shorter than the ~49 s row dwell.
Consequences:

- 97% of traps had `tau_e` < one row dwell (median `tau_e(135 K) ≈ 4.4 s` vs 49.1 s dwell) and
  re-emitted into their own packet within the capture dwell → **completely inert.**
- The entire simulated trap effect came from the ~2–3% long-τ tail.

### 2.2 Model B — full-row two-state SRH with recapture (2026-06-11; superseded)

*Idea:* fix Model A by letting an emitted carrier be **recaptured** by the same trap. A trap is
either empty or occupied and flips between the two at an emission rate (`lambda_e`, how fast it lets
go) and a capture rate (`lambda_c`, how fast it grabs). Over each small time-step we use the exact
probability of ending up in each state.

For each trap, at each row-transfer dwell Δt, with `q` carriers in the packet above:

```
lambda_e = 1 / tau_e                          (emission rate)
lambda_c = q_band * k_c,  k_c = sigma * v_th / V_p   (capture rate per carrier over the trap)
```

`q_band = q` for an empty trap and `q + 1` for an occupied one (the emitted carrier joins the packet
and can be recaptured). End-of-dwell state drawn from the **exact two-state Markov transition
probabilities**:

```
P(release | occupied) = [lambda_e / (lambda_e + lambda_c)] * (1 − exp(−(lambda_e+lambda_c) Δt))
P(capture | empty)    = [lambda_c / (lambda_e + lambda_c)] * (1 − exp(−(lambda_e+lambda_c) Δt))
```

Same expressions with (Δt = exposure time, q = charge in the trap's own pixel) drive
`charge_trap_interaction` for the static exposure phase. Limits: `k_c → 0` recovers Model A;
`k_c → ∞` gives a sticky trap that drags carriers indefinitely. Detailed-balance check: equilibrium
occupancy ratio = `n / (N_v e^{−E/kT})`, the standard SRH result.

In the strong-recapture regime the per-shift escape probability reduces to
`p_esc ≈ lambda_e/lambda_c = N_v V_p e^{−E/kT} / q_band` — **independent of σ** — so a captured
carrier is dragged multiple pixels or held across images rather than re-emitted in place. Model B
made ~97.5% of trapped charge retained through a readout (traps as near-permanent sinks; mean
shifts-to-escape `≈ k_c·τ ≈ 10⁴`).

**Validation tests that passed for Model B** (reproduce with small grids + `seed_numba`): exact
charge conservation; correct CTI geometry (deferred charge lands in trailing rows); `k_c = 0`
reproduces the geometric law `P(k) = p(1−p)^k`; capture/escape frequencies match the analytic
two-state formulas; padded-buffer bookkeeping independently verified.

### 2.3 Why Model B was abandoned — the pumping-consistency tension

Model B assumed capture happens over the **full ~49 s row dwell**. The pocket-pumping data
contradict that:

- Pumping uses a **flat field** (~2000 e/pixel, roughly constant image-to-image), so there is
  *always* charge over the trap — `n` is never 0 and recapture is never geometrically excluded.
- Pumping therefore runs at high density (`n ≈ 2000/V_packet`), ~2000× stronger recapture than
  readout single-e (`n ≈ 1/V_packet`). With Model B's `k_c`, recapture during pumping would be
  ~10⁶× faster than the dwell → the dipole would be suppressed to ~1e-4 e (invisible). **Yet
  dipoles of hundreds–thousands of e are observed.** Taken at face value, Model B's σ-based
  recapture rate is orders of magnitude too strong to be consistent with the dipoles existing.

The investigation (`RECAPTURE_PUMPING_HANDOFF.md`) also confirmed, before the fix, that:
- The magnitude ratio between the minimal and legacy catalogs (~2×) is driven by **trap count**
  (9333 vs 5171 detected), not the 5× σ difference, because capture **saturates**:
  `k_c·tpix_vertical ≈ 8×10⁴ ≫ 1` → capture prob ≈ 1 for ~97% of traps (σ_sat ≈ 5e-20 cm²).
- σ is 99.7% E-degenerate in the Arrhenius fit (`logσ = 37.3·E + const` at fixed τ135) and ~5×
  below literature (~1e-15). Under Model B this σ offset did not propagate because capture
  saturated. (See [[dipole_algorithm]] for the σ/E fit.)

### 2.4 Model C — `phase_limited_v1v3` (current, live)

Resolution: **pumping probes V1/V3 traps; readout parks charge under V2.** Capture during readout
is not a full-row-dwell process — it happens only in the brief inter-pixel transfer window when the
packet actually crosses the trap's phase. The model was switched accordingly (audited and signed
off; see memory `recapture-pumping-test-open`):

- **Capture** happens in a `t_phase = 20 µs` transfer window: empty trap with `q ≥ 1` captures with
  `1 − e^{−q·α}`, `α = σ v_th t_phase / V`. Because the window is short and fixed, capture no longer
  saturates for single electrons and is **binning-invariant** (α does not scale with readout speed).
- **Emission** still runs over the full per-row dwell: occupied trap emits with `1 − e^{−τ_pix/τ}`.
- **No full-row recapture.** An emitted carrier is not re-captured over the whole dwell.
- **σ matters again.** Single-e capture probability `∝ σ` (unsaturated), inverting the Model-B
  conclusion that σ was irrelevant. See memory `sigma-degeneracy-and-literature`.

Consequence for the science: **all campaign numbers computed under Model B were superseded** and
regenerated under Model C. The `campaign/` results and the deviation budget below are Model C.

---

## 3. Packet volume V_p and why poorly-known σ is not a separate problem

`V_p` is the effective volume a **single carrier** explores in a pixel well (local density a trap
sees is `q / V_p`). It is forced on the model by capture (which needs a density) and is not
constrained by the pumping measurement.

Determinants:
- **Lateral:** collecting-phase area = (15 µm pixel − channel stops) × (5 µm × #phases held high).
  Set by geometry/clocking, **not** clock swing (barriers only need to exceed kT = 11.6 meV at
  135 K; the ~2.25 V operating swing is ~200 kT).
- **Vertical:** thermal spread in the buried-channel implant well,
  `sigma_z = sqrt(eps_Si kT / (q² N_channel))` ≈ 9–87 nm for N = 1e17–1e15 cm⁻³.

Electrostatic estimate: **V_p ≈ 1–8 µm³, baseline 3 µm³** (code default `--packet-volume-um3`).
Quote the V_p = 1–10 µm³ band as the systematic.

Under Model B (saturated), the **masked** trap excess varied only ~±25% over V_p = 0.3–30 µm³.
Under Model C (unsaturated single-e capture), V_p enters through `α ∝ 1/V`, a mild effect on the
upper population and negligible on baseline (see the deviation budget table, axis `vp1 vs vp3`).

Why global σ error is absorbed: everywhere σ appears only as `σ/V_p` (verified: σ ×10 ≡ V_p ÷10
exactly), so any global σ bias folds into the V_p systematic band; per-trap random σ errors average
over ~1300 traps/quadrant, and the empirical (τ, σ) pair resampling
(`trap_tau135_sigma_pairs.npz`, from `make_trap_pairs.py`) carries the measured scatter in. τ_e —
the non-degenerate directly measured quantity — is what the histogram supplies.

---

## 4. Deviation budget under `phase_limited_v1v3` (current, 2026-06-24)

Goal: for a trap-branch image at exposure `t` (hours), account for **every** deviation of the
measured single-electron density from injected truth. Reviewed adversarially by Codex over three
rounds; §4.4 records what survived and what was overturned by direct instrumentation.

### 4.1 Setup

Per quad: `N_pix = 512 × 3072 = 1.573e6`. Injected truth (recovered by the no-trap branch to
<0.3% intercept, <1.3% slope across all 53 complete scenarios):

```
d_true(t) = d_EI + d_ED · t,   d_EI = 9.94e-5 e/pix/img,  d_ED = 4.36e-5/24 e/pix/hr.
```

Both branches share the **same injected realization** (common random numbers), so the deviation is
purely trap-induced:

```
Δ(t) = C_trap/U_trap − C_notrap/U_notrap ≈ ΔC/U + C_nt·(1/U_trap − 1/U_notrap)   (numerator + denominator)
```

C = single-e counts, U = unmasked-pixel count. Measured split (minos/upper, 20 h): 87% numerator,
13% denominator.

Per-image pipeline (`take_fake_image`, `ccd_simulation.py`): (1) `simulate_clear()` retains trap
occupancy, emitted charge during clear is zeroed/lost; (2) `charge_trap_interaction(dt=t·3600)` —
integration emission `1−e^{−t/τ}`, empty traps do NOT capture (charge under V2); (3) inject dark
(`∝t`) + spurious (const) + clusters (snolab = t/10); (4) `simulate_readout()` — per row-step,
occupied→emit `1−e^{−τ_pix/τ}`, empty & q≥1 → capture `1−e^{−q·α}`, `α = σ v_th t_phase / V`,
`t_phase = 20 µs`.

Timing (computed): `τ_pix = 47.13 s`, `T_read = 512·τ_pix = 6.70 h` (bin1) or 12.6 min (bin32,
which divides τ_pix only). Clear (sequencer) = 3.26 s.

### 4.2 The deviation terms

| term | mechanism | sign | t-dependence | key property |
|---|---|---|---|---|
| **A+C** reservoir emission | occupied traps emit during integration (A) or readout (C); both counted | + | rises, saturating | DOMINANT; slope driver + most of +intercept; `∝ N_occ` |
| **B** readout capture of spurious | empty traps capture injected spurious 1e during readout; deferred | − | ~const (intercept) | pre_readout only; `∝ n_empty` |
| **D** readout smearing / multi-e splitting | empty trap captures from a cluster pixel, re-emits downstream → extra 1e | + | grows with t | mostly masked (lands near clusters) |
| **E** readout capture of dark current | empty traps sink dark 1e | − | ∝ t | negligible (bounded by dark budget) |
| **F** masking-denominator coupling | trap-deferred charge → more hot-column/bleed flags → U_trap < U_notrap | + | — | analysis term; 13% of minos/upper density excess; report upper in COUNTS not density |
| **G** clear-mode modulation of N_occ | clear sets the reservoir entering each image | ± | — | three_hour drains drainable τ; scales A+C and B |
| **H** 2e-channel coupling | capture 2e→1e (+1e count), emission 1e→2e | ± | — | Δ2e measured positive & growing → net charge piling |

Mapping to the two fitted channels:
```
Δ_EI (intercept) = ⟨s⟩·Σ_i o_i(1−e^{−T_read/τ_i})  −  N_spur·p̄_cap   (+ small D,E,F)
Δ_ED (slope)     = ⟨s⟩·Σ_i o_i (1/τ_i) e^{−(t+T_read)/τ_i}  +  dΔD/dt  −  dΔE/dt
```
`o_i` = occupancy at integration start; `⟨s⟩` = survival (not recaptured/masked/merged) ≤ 1.

Per-scenario sign/magnitude predictions:

| axis | effect | predicted result |
|---|---|---|
| baseline vs upper | N_occ: 8 vs ~11k | baseline NULL both channels; upper LARGE |
| minos vs snolab | D,F ∝ clusters (10×); A,B,E common | counts: similar (A floor); density: minos inflated by F |
| pre vs post | B only in pre | post intercept less negative |
| clearseq vs clear3h | G drains N_occ | clear3h → smaller A+C, larger B → deficit, flat (drainable τ only) |
| bin1 vs bin32 | T_read 6.7h→12.6min | 20 h Δ ~unchanged; 0 h Δ more negative |
| vp1 vs vp3 | α ∝ 1/V | weak; baseline negligible, upper mild |

### 4.3 Instrumented closure — the measured kernel response

The clean closed form `P_emit_counted(τ;t) = 1 − e^{−(t+T_read)/τ}` (originally called (★)) was
**overturned by direct instrumentation** (`_instrument_run.py`, origin-tagged carriers,
minos/upper/clearseq/pre, equilibrated occ 62% = 12.4k of 20k traps). Fate funnel per image, 20 h:

| quantity | bin1 | bin32 | binning |
|---|---|---|---|
| readout emissions E_read | 19121 | 10268 | scales ~1.9× |
| self-recaptured (same step) | 17366 (91%) | 8686 (85%) | scales |
| escaped (E_read−selfrec) | 1754 | 1582 | ~invariant |
| integration emission E_int | 254 | 294 | invariant (small!) |
| injected-1e SINK | 2022 | 1851 | ~invariant |
| net Δ1e unmasked | 402 | 834 | not invariant |
| **net Δ1e halo+bleed-masked** | **61** | **68** | **INVARIANT** |

What the measurement established (three budget claims overturned):
1. **"Integration emission A dominates the slope" is WRONG.** `E_int ≈ 254/img` — the occupied
   reservoir is very-long-τ, so `P_emit(t) ≈ 0`. The bulk of the net count is escaped readout
   emission.
2. **Readout emission is ~91% futile** (immediate self-recapture; capture saturates for the
   occupied large-σ traps). Emission flux ≠ effect.
3. **The masked observable is CAPTURE-LIMITED, not emission-limited.** `E_read` is binning-dependent
   (19k→10k) but the masked net Δ1e is invariant (61→68), as are sink and net-count. Capture rate
   `α = σ·v_th·t_phase/V` is fixed by the 20 µs phase window → binning-invariant. The effect is
   driven by **capture of exposure-accumulated image charge during readout** (∝ exposure → slope;
   α-fixed → binning-invariant) plus its redistribution; the huge emission flux is mostly
   self-recapture cycling that cancels.
4. **Survival ⟨s⟩ ≈ 61/19121 ≈ 0.3%** (masked), ≈2% unmasked. The net is a tiny residual of two
   ~2000-electron opposing flows. The naive `P_emit_counted ≈ 1` overpredicts per-trap net by ~200×.

Codex's final tempering (do not overstate): the flow counters are **not** the observable (they
precede cluster classification + halo/bleed masking, which is a nonlinear count of isolated 1e
clusters); the missing positive term is **cluster→1e redistribution** (D); origin-tagging is biased
in mixed packets so "91% self-recapture" is qualitative ("mostly futile"), not literal; long-τ
equilibration is not demonstrated (300 images ≈ 200 days, τ≥1e7 s cannot equilibrate).

**Defensible headline:** under minos/upper/clearseq/pre, the halo+bleed-masked trap excess is a
small residual after large readout capture/emission/recapture flows, ~invariant bin1↔bin32; gross
readout emission is mostly cancelled by immediate recapture; integration emission is small in this
run state; the unmasked deviation is binning-dependent; the positive count term is cluster→1e
redistribution; capture-limited is plausible but not proven by these counters.

### 4.4 τ-band of the slope (three_hour/upper, seed-capping)

Capping the minimal_caldet upper-limit seed at successive `τ_max` and re-running (MINOS, three_hour,
30 trials/cap) empirically sums `Σ o_i(…)`:

| seed τ_max | N_traps | slope (exp-dep) | intercept (exp-indep) |
|---|---:|---:|---:|
| 1e5 | 4179 | 1.842e-6 (≈ no-trap truth 1.82e-6) | 9.91e-5 |
| 1e6 | 5529 | 2.110e-6 | 1.006e-4 |
| 5e7 (prod cap) | 17475 | 2.816e-6 | 9.96e-5 |
| 3e8 (unbounded) | 66087 | 2.827e-6 | 9.74e-5 |
| **baseline (characterized)** | **3798** | **1.818e-6 (= truth)** | **9.93e-5 (= truth)** |

Findings (these correct the earlier "hours-to-day" tag):
1. **The slope band is τ > 1e5 s (days-to-months), NOT "hours-to-day."** τ<1e5 traps add ~zero
   exposure-dependent excess. The slope excess (+1.0e-6 over truth) lives at τ>1e5, dominated by
   τ∈[1e6, 5e7] (+0.71e-6) then τ∈[3e5, 1e6] (+0.26e-6).
2. **For the UPPER population, three_hour does NOT flatten A+C.** The 3 h drain survival
   `e^{−3h/τ} ≈ 1` for τ≫3h, so the long-τ reservoir survives the clear occupied and produces a
   LARGE positive slope (2.82e-6, +55% over truth). The "flatten-on-3h" prediction is a **baseline
   (short-τ) result; it inverts for upper.**
3. **The slope converges in τ_max** (production cap 5e7 sits past the turnover; extending to
   unbounded 3e8 leaves the slope unchanged). Per-trap contribution collapses ~300× across the
   turnover — a trap that never emits within the run defers no charge — so even the 48612 monster
   1/ε-inflated traps at τ>5e7 add <1%.
4. **The baseline is NULL; the UL slope is 1/ε-inflation headroom, not a measured effect.** The real
   characterized population (3798 traps) gives slope = truth (excess ≈0). The 2.8e-6 UL slope is the
   90% CL efficiency-corrected *allowance* for hidden long-τ traps (τ~3e5–1e7, ε~0.01–0.07), where
   ~40 measured traps are amplified ~300× by 1/ε. **Do NOT floor it** — the ~21 characterized
   (decoy-rejected) traps at τ>1e6 are real evidence the region is populated; an ε-floor would
   assert the null by fiat. The deliverable is **two numbers**: measured/characterized = NULL, and
   efficiency-corrected 90% CL UL ≈ 2.8e-6 (slope) carrying an error band (~25% Poisson on the ~20
   anchoring detections + the Arrhenius τ(T) completeness systematic). The only principled exclusion
   is zero-detection bins where 2.3/ε is pure Poisson-floor amplification (e.g. the empty τ=2.7e-5
   cold bin). See memory `exposure-dependent-slope-is-low-eps-tail`.

---

## 5. Operating-condition decisions (confirmed by Ansh, 2026-06-11)

- **SNOLAB = 10× fewer high-energy events** than MINOS (code `exp/10`; the paper's "factor of 2"
  was corrected).
- **Rates are per pixel:** exposure-independent 9.94e-5 e/pix/image, exposure-dependent
  4.36e-5 e/pix/day (UR quadrant, SENSEI:2024yyt). Old "superpix" comments were wrong.
- **Inter-image dead time ≈ 0.** The default `sequencer` clear follows
  `temp_scan_run1_clearseq.xml`: 1500 fast + 10 slow vertical shifts, retaining trap occupancy;
  free surface charge at the clear boundary is discarded. The former no-interaction behavior remains
  as `--clear-mode instantaneous`. Clear modes and exposure orders are documented in `CLAUDE.md`.

### Masking fidelity vs the real SENSEI pipeline (`hotcol.py`, 2026-06-11)

Goal: reproduce SENSEI's masking as faithfully as possible, not optimize it. **Verified
verbatim-equivalent:** windowSum/expandWindow; calculatePvals incl. `sf(hits−0.5)`; pCut=0.5 with
p×nCells scaling; window sizes 1–5; addNeighbors; iteration to stability; hot-pixel→column merge;
halo (60 pix around >100e) and bleed (full column above >100e). **Deliberate exclusions:**
very-hot pre-cuts and the 16-chunk cut (target real detector defects the sim lacks). **Kept-as-is
differences:** the sim uses a flat rate model (real pipeline uses `max(uniform, linX/linY)`
envelope) and feeds only 1e counts (real uses 1e + 2+e). Net effect: simulated masks are somewhat
*stricter* than real, so the quoted masked excess errs low — acceptable and documented.

---

## 6. Known remaining simplifications (assessed minor)

1. Charge resident in a trap's own pixel during exposure is never captured (interaction runs on the
   cleared state before injection). Second-order.
2. Exposure-independent charge injection point is `--exp-indep-charge-mode` (default `pre_readout`
   traverses active-area traps; `post_readout` does not — preserves common-random-number
   cancellation but makes spurious charge un-trappable). See `CLAUDE.md`.
3. No serial-register traps.
4. V_p is a single uniform-density number (absorbed by the V_p band).
5. q changes by ±1 within a dwell are ignored in the two-state rates (fixed-q approximation).
6. `charge_trap_simulation.ipynb` predates the current `CCD(...)` signature — update before use.

---

## 7. Deferred improvement — high-energy-event (HEE) sampling

**Not implemented; kept as a forward TODO** (`HIGH_ENERGY_EVENT_SAMPLING_NOTE.md`). Current method:
extract connected clusters from one fixed 20 h MINOS image, scale count deterministically with
exposure, select without replacement, transplant to random positions; SNOLAB reuses the same
population at 10× lower rate. Adequate for a first estimate but suppresses event-count variance,
reuses one detector realization, and assumes MINOS/SNOLAB differ only in normalization.

Proposed backward-compatible `hee_sampling_mode = "legacy" | "poisson_bootstrap"`:
1. Build a cluster library from **all** approved reference images (store charge cutout, bbox, total
   & peak charge, area, source image/exposure, condition).
2. Separate HEE core from surrounding deferred/trap-tail pixels (store wide context for validation
   only).
3. Draw `N_HEE ~ Poisson(rate_condition · exposure · area)` independently per image.
4. Sample clusters with replacement from the condition-matched library.
5. Place at random valid active-area locations; permit physical overlap.
6. Use the same injected realization for trap and no-trap branches (preserves common-random-number
   cancellation).
7. Separate MINOS/SNOLAB libraries when data allow; until then expose the 10× scaling and
   shared-spectrum assumption as metadata/systematics.

Validation before production use: linear mean-count scaling with exposure/area; `Var(N)/Mean(N) ≈ 1`
unless overdispersion is significant; compare charge/peak/area/aspect/NN-distance distributions;
compare halo/bleed/masked fractions vs held-out real images; confirm cutouts are not mutated;
bit-for-bit identical injected HEE images in the paired branches; fixed-seed regression tests for
both modes. Watch the masked single-electron excess (can respond nonlinearly to HEE fluctuations).

---

## Data flow and inputs (current)

```
proc/*.fits → run_charge_traps.py → fit_dipole_spectra_*_err_4.h5 (per-trap tau(T), E, sigma)
                                  → tau_at_135k_hist.npz          (tau sampling histogram)
                                  → trap_tau135_sigma_pairs.npz   (per-trap tau135/sigma/E pairs)
charge_trap_figures.ipynb         → tau_at_135k_hist_upper_limit.npz (efficiency-corrected 90% CL)
run_ccd_simulation.py / run_campaign.py → per-trial HDF5 in campaign/<scenario>/
```

Each simulated trap: τ sampled log-uniformly within a histogram bin; σ resampled from the ~20
measured (τ135, σ) pairs nearest in log τ (preserves the empirical τ–σ correlation);
`k_c = σ·v_th(135 K)/V_p`. `make_trap_pairs.py` refits (E, log σ) from the stored per-temperature τ
values, so it is correct regardless of whether the HDF5 predates the constants fix. The simulation
now seeds the **characterized** trap count (`n_baseline_traps = round(tau_weights.sum())` ≈ 3798),
not the detected count — decoy control showed detection is a poor false-positive filter
(random/horizontal decoys reach "well-behaved" at 22–47%) while characterization rejects ~97–99%.
See [[completeness_efficiency]] for the upper-limit population construction.
