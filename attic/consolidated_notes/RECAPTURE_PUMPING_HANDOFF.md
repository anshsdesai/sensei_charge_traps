# Recapture model investigation & pump forward-model test — session handoff

**Status:** open. Last session ended at the design of a "pump forward model" to test whether
the simulation's SRH recapture rate is consistent with observed pocket-pumping dipoles.
**Env:** conda env `sensei_charge_traps_new` (NOT the requirements.yaml name). Run python via
`conda run -n sensei_charge_traps_new python <script>` (note: `conda run` can't take `-c` with
newlines — write a script file).

---

## How we got here (the thread)

Started from: "the minimal_caldet campaign results don't make sense." Investigated and resolved a
chain of questions; the live open item is the **recapture-vs-pumping consistency test**.

### 1. The minimal_caldet "weird features" are occupancy physics, not a bug
Masked single-e excess (trap−notrap rate) per exposure, computed from `campaign/*/aggregated_results_*.json`
(`total_counts_*/total_pix_*`). Features: sign flips across clear-mode/condition/exp-indep; clear3h flat
~−26e-6 deficit; minimal ≈ 2× legacy everywhere; excess declines with exposure. All explained by the
occupancy-controlled SRH model (see memory [[occupancy-controls-trap-sign]]). NOT a readout bug
(charge-conserving: `image-=1; trapped+=1`).

### 2. Capture SATURATES → magnitude is driven by trap COUNT, not σ
Key fact: `tpix_vertical ≈ 49 s` (= ncol×pixel_time at NSAMP=300; vertical shift waits for full serial
readout). Capture rate `kc = σ·v_th/V_packet` (ccd_simulation.py:1180); median `kc≈1620/s`, so
`kc·tpix_vertical ≈ 8×10⁴ ≫ 1` → capture prob saturates at `≈1` for ~97% of traps (σ_sat≈5e-20 cm²).
So the minimal/legacy magnitude ratio (~2×) = **trap count** (9333 vs 5171 detected dipoles, ×1.10 from
σ-tail saturation), NOT the 5× σ difference. Closed-form deficit prediction validated to ~1.4×
(tools/_predict_deficit.py: minimal 3.53e-5 / legacy 1.78e-5 vs observed 2.60 / 1.17e-5).

### 3. Switched sim to seed CHARACTERIZED count (CODE CHANGED, NOT YET RE-RUN)
Decoy control (decoy_control_log.txt) shows DETECTION is a poor FP filter (random/horiz-null decoys
reach "well-behaved" at 22-47%) but CHARACTERIZATION rejects ~97-99% (decoys characterize at 0.8-2.9%).
Characterized = 3798/2135 = ~41% of detected for both flavors. **Decision: seed from characterized count.**
- `run_ccd_simulation.py`: now `n_baseline_traps = int(round(tau_weights.sum()))`; `--coord-list` removed.
- `run_campaign.py`: upper_scale divisor → `baseline_hist.sum()` (preserves upper-limit abs count ~17470);
  `--vp-scan` flag added (default = central V_p=3 only; band {1,3,10} opt-in, since saturation makes V_p a
  non-effect). Default campaign now 24 scenarios, --vp-scan 48.
- **STALE OUTPUTS:** existing `campaign/*minimal_caldet/` were computed with the OLD detected count (9333)
  and are skipped as "complete" — must delete (or new --out_base) to regenerate at 3798. Expect ~0.41×
  rescale of all SER numbers; ratio + shapes unchanged. NOT yet re-run.

### 4. σ is E-degenerate and ~5× below literature (tools/_sigma_lit_check.py, _sigma_constraint.py)
σ and E are 99.7% correlated in the Arrhenius fit (ρ from `energy_CovarianceMatrix` in
fit_dipole_spectra_minimal_caldet_err_4.h5); marginal σ uncertainty small (0.08 dec) but σ is pinned to E
(logσ = 37.3·E + const at fixed τ135). "Small σ" = "small E". At E≈0.30 our σ≈2-3e-16 vs literature
~1e-15 (5×, ≈0.02 eV E-bias, consistent with high-T lean [[high-t-arrhenius-lean]]); formula
(dipole.py:428 log_energy_cross_section) lacks trap-degeneracy g + entropy factors (uniform prefactor).
Because capture saturates, this σ offset doesn't propagate to the deficit.

### 5. Recapture physics (the live thread)
Model uses the EXACT two-state SRH transition prob, NOT the textbook one-shot. Empty trap capture
(fast_readout_numba:824-832): `p_occ = (λc/tot)(1−e^{−Δt·tot})`, `λc=q·kc`, `λe=1/τ`, `tot=λc+λe`.
Derived from the 2-state master equation `dP_O/dt = λc − tot·P_O`. n in the SRH `τc=1/(σ v_th n)` is
**`n = q/V_packet`** (q carriers in the packet ÷ effective well volume; `--packet-volume-um3` default 3µm³).

- The analysis FIT (dipole.py:409 `intensity_function = npumps·coeff·(e^{−tph/τ} − e^{−8·tph/τ})`) is
  emission-only, NO recapture, NO σ — capture folded into `coeff`. The "8" = pump duty ratio
  (delay_Tph_long = 7×delay_Tph). This is the user's remembered `1−exp(−shift/τ)` (= eq2.1 with t1=0),
  which is the λc→0 limit; the live readout sim uses the coupled form (the old simple form is commented
  out at ccd_simulation.py:1217).
- In readout the 49s dwell + recapture (`p_free ≈ λe/kc ≈ 1e-4`/dwell) means ~97.5% of trapped charge is
  RETAINED through a readout (traps act as near-permanent sinks); mean shifts-to-escape ≈ kc·τ ≈ 10⁴.

### 6. The pumping-consistency question → the OPEN TEST
Pumping (sequencer `temp_scan_run1_pumpseq.xml`, recipe `vpump` lines 197-206) produces dipoles, fit
cleanly by the no-recapture `intensity_function`. We tried to reconcile via geometry (charge displaced
during emission → no recapture) BUT **the user correctly refuted this**: pocket pumping uses a FLAT FIELD
(~2000 e/pixel, approx constant image-to-image), so there is ALWAYS charge over the trap (the neighbor's
packet) — `n` is never 0, recapture is never geometrically excluded.

**This SHARPENS the tension against the sim:** pumping runs at high density (`n≈2000/V_packet`), so
recapture there is ~2000× STRONGER than readout single-e (`n≈1/V_packet`). With the sim's kc, recapture
during pumping is ~10⁶× faster than the dwell → a naive estimate gives the dipole suppressed to ~1e-4 e
(invisible). Yet dipoles of hundreds–thousands of e are OBSERVED. Taken at face value, the sim's
recapture rate (σ-based) is orders of magnitude too strong to be consistent with the dipoles existing —
and by extension the readout "near-permanent sink" (97.5% retention) is suspect.
**Loophole:** recapture might only rescale the dipole AMPLITUDE (into `coeff`) while preserving the
`e^{−t1/τ}−e^{−t2/τ}` SHAPE → τ still clean, sim survives. Whether recapture distorts shape or just
amplitude is the open question.

---

## THE NEXT STEP: pump forward model (agreed plan, not yet built)

Build a small dedicated driver (NOT run_ccd_simulation, which only does readout) that reuses the **same
SRH kernel** (`λc=q·kc`, `λe=1/τ`, the two-state p_occ/p_free, `kc=σ·v_th/V_packet`) but drives it with
the `vpump` clock sequence on a FLAT FIELD:
1. Flat field `N₀` (electrons) across a few pixels + one trap.
2. Step the 8-state vpump cycle with real dwells (`delay_Tph=dtph`, `delay_Tph_long=7·dtph`, shorts=5
   ticks @15MHz), applying the SRH transition to the trap each dwell vs the charge over it.
3. Repeat NPUMPS times, read out, measure dipole amplitude (± deviation from N₀ in adjacent pixels).
4. Scan dtph → intensity-vs-dtph; compare amplitude to observed dipoles AND shape to intensity_function.

**Care point:** mapping vpump phase states (STATE_11/13/15…) to charge POSITION (which well/pixel relative
to the trap) so inter-well charge transport is modeled right. SRH kinetics are trivial to reuse; the
"where is the charge each step" bookkeeping is the work.

**Discriminating outcome:**
- If sim recapture crushes the dipole to ~1e-4 e vs observed hundreds–thousands → recapture rate (σ /
  `n=q/V_packet` mapping) is FALSIFIED; readout near-sink behavior too strong.
- If recapture only rescales amplitude but preserves the dtph peak shape → τ clean, sim recapture survives.

**Inputs needed (mostly self-extractable):**
- `N₀` in e⁻: Gaussian fit of bulk-pixel histogram of a `cal/` pumping frame (use electronized frame, or
  skipper single-e peak spacing as gain). User confirmed flat field ≈ consistent image-to-image, ~2000 e.
- dtph scan values + NPUMPS: from `cal/` filenames (dtph 1200→2.5M ticks = 80µs→167ms; NPUMPS3000).
- Representative τ, σ: from catalog at 160 K (cal scan temp).
- Observed dipole amplitudes: measure from the same cal frames.
- Pump frame example: `cal/cal_skp_dp_scan1_160k_binned_..._dtph<N>_NPUMPS3000_*.xml` (also .fits in cal/?).

---

## Artifacts created this session (tools/, untracked, _ prefix)
`_ser_probe.py` (SER excess per scenario), `_pop_probe.py` (tau/sigma populations), `_sigma_diag.py`
(σ vs E), `_attrs_probe.py` (HDF5 run attrs), `_timing_probe.py` (tpix_vertical from FITS),
`_predict_deficit.py` (closed-form deficit), `_sigma_lit_check.py` (σ vs literature), `_sigma_constraint.py`
(σ fit degeneracy), `_count_probe.py` (detected vs characterized), `_sigma_fit_err.py` (HDF5 fit struct).

## Key constants (135 K unless noted)
v_th(135K)=1.22e7 cm/s; V_packet=3e-12 cm³; kc_median(minimal)≈1620/s (legacy≈274); recapture time ~0.6ms;
tpix_vertical≈49s; τ_median≈6s (legacy 4.3); σ_median minimal 4e-16 / legacy 6.7e-17; σ_sat≈5e-20 cm².
detected 9333/5171; characterized 3798/2135. Pump: delay_Tph_long=7×delay_Tph; flat field ~2000e @160K.
