# Charge-Trap Simulation: Physics, Decisions, and Assumptions

Reference document from the physics verification and model upgrade of 2026-06-11.
Read this before modifying `ccd_simulation.py` trap physics, `dipole.log_energy_cross_section`,
or re-running the simulation campaign.

---

## 1. Emission-time physics (verified against literature)

The per-trap emission time follows standard Shockley–Read–Hall detailed balance, as used
throughout the CCD trap-pumping literature (Hall 1952; Hall et al. IEEE TNS 61, 1826 (2014);
Wood et al.; Oscura arXiv:2406.18502; Brusco arXiv:2510.23336):

```
tau_e(T) = exp(E_t / kT) / (sigma * v_th * N_v)
v_th = sqrt(3 kT / m_cond)                      (thermal velocity, holes)
N_v  = 2 (2 pi m_dens kT / h^2)^(3/2)           (valence-band effective DOS)
```

SENSEI CCDs are p-channel; carriers are **holes**. Effective masses for 100–200 K
(Green 1990, as adopted by arXiv:2406.18502): `m_cond = 0.41 m_e`, `m_dens = 0.94 m_e`.
Note `tau_e ∝ T^-2 exp(E/kT)` — the standard DLTS form. The degeneracy/entropy factor is
taken as 1 and temperature-independent sigma is assumed; both are conventional and are
absorbed into the fitted (effective) E and sigma.

The pumping dipole intensity model `I = N_pumps * D_t * P_c * (exp(-t_ph/tau) - exp(-8 t_ph/tau))`
matches the Oscura derivation (emission window [t_ph, 8 t_ph] for this clocking scheme).

### Constants bug (fixed 2026-06-11 in `dipole.log_energy_cross_section`)

The original implementation had three errors:

| quantity | was | should be | effect |
|---|---|---|---|
| k_B | 8.6717333262e-5 (typo) | 8.617333262e-5 eV/K | fitted E biased +0.63% |
| h | 4.1135e-15 (typo) | 4.135667696e-15 eV·s | h³ off by −1.6% |
| mass prefactor | bare m_e | m_dens^{3/2}/m_cond^{1/2} = 1.4233 m_e | fitted sigma biased ×1.42 |

**Crucially, tau_e(135 K) predictions are exactly invariant** under these fixes: the fit family
`log tau = const − log sigma − 2 log(kT) + E/kT` is unchanged, so the constant errors were
absorbed entirely into the fitted (E, sigma). Verified bit-for-bit: rehistogramming the refit
tau135 values reproduces `tau_at_135k_hist.npz` with zero bins changed. Net effect of the fix:
quoted sigma → ×0.722, quoted E → ×0.9937, all tau-based inputs unchanged.
All `tau_at_<T>k_hist.npz` files therefore remained valid.

---

## 2. Trap interaction model in the readout simulation

### 2.1 The old model and why it was wrong

The original `fast_readout_numba` captured deterministically (P_c = 1 whenever a packet with
≥1 carrier passed an empty trap) and released with `P = 1 − exp(−t_dwell/tau_e)` per row
transfer, with the released carrier rejoining the packet above the trap — **never to be
recaptured**. This is internally inconsistent: instant capture taken seriously implies an
emitted carrier sitting over the same trap is immediately recaptured. Quantitatively, with the
fitted cross-sections (median sigma ≈ 7.5e-17 cm²), the capture time of a single carrier over
its trap is ~ms–s, far shorter than the 49 s row dwell. Consequences of the old model:

- 97% of traps had tau_e < one row dwell (median tau_e(135 K) ≈ 4.4 s vs 49.1 s dwell) and
  re-emitted into their own packet within the capture dwell → **completely inert**.
- The entire simulated trap effect came from the ~2–3% long-tau tail.

### 2.2 The correct model (implemented): two-state SRH kinetics per dwell

For each trap, at each row-transfer dwell of duration Δt, with q carriers in the packet above:

```
lambda_e = 1 / tau_e                          (emission rate; tau_e sampled from histogram)
lambda_c = q_band * k_c,  k_c = sigma * v_th / V_p   (capture rate per carrier over the trap)
```

where `q_band = q` for an empty trap and `q + 1` for an occupied one (the emitted carrier
joins the packet and can be recaptured). The end-of-dwell state is drawn from the **exact
two-state Markov transition probabilities**, which include any number of capture/emission
toggles within the dwell:

```
P(release | occupied) = [lambda_e / (lambda_e + lambda_c)] * (1 − exp(−(lambda_e+lambda_c) Δt))
P(capture | empty)    = [lambda_c / (lambda_e + lambda_c)] * (1 − exp(−(lambda_e+lambda_c) Δt))
```

The same expressions (with Δt = exposure time, q = charge in the trap's own pixel) are used in
`charge_trap_interaction` for the static exposure phase. Limits: `k_c → 0` recovers the old
pure-emission law; `k_c → ∞` gives a sticky trap that drags carriers indefinitely. Detailed
balance check: equilibrium occupancy ratio = n / (N_v e^{−E/kT}), the standard SRH result.

In the strong-recapture regime the per-shift escape probability reduces to
`p_esc ≈ lambda_e/lambda_c = N_v V_p e^{−E/kT} / q_band` — **independent of sigma** — so a
captured carrier is typically dragged multiple pixels (or held across images) rather than
re-emitted in place. Population classification at 135 K (V_p = 1–37.5 µm³): only ~10–20% of
traps release "immediately", ~25–45% drag the carrier 2–512 pixels, ~35–70% hold longer than a
full readout and defer charge across images. (Old model: 97% inert / 2% / 2%.)

### 2.3 Validation tests (all passed; reproduce with small grids + `seed_numba`)

1. Charge conservation exact: input = read out + still trapped + stranded in never-read rows.
2. Geometry: packets interact with every trap below their starting row in their own column,
   and only those; deferred charge lands in trailing rows (read later) — correct CTI direction.
3. `k_c = 0` reproduces the geometric law `P(k) = p(1−p)^k`, `p = 1 − exp(−Δt/tau)`.
4. Capture and recapture-suppressed escape frequencies match the analytic two-state formulas.
5. Vectorized exposure-phase version matches the same analytics.
6. The stationary padded-buffer bookkeeping was independently verified: trap row tr at
   iteration t interacts with padded row `tr + rows − 1 − t`; charge emitted after the image
   window passes is stranded in never-read rows and then discarded by the clear (correct).

---

## 3. The packet-volume parameter V_p

`V_p` is the effective volume a **single carrier** explores in a pixel well (so the local
density a trap sees is `q / V_p`). It is forced on the model by recapture: capture rate
requires a density, and the pumping measurement does not constrain it. Determinants:

- Lateral: collecting-phase area = (15 µm pixel − channel stops) × (5 µm × #phases held high).
  Set by geometry/clocking scheme, **not** clock swing: barriers only need to exceed kT
  (11.6 meV at 135 K), and even the small operating swing of ~2.25 V is ~200 kT.
- Vertical: thermal spread in the buried-channel implant well,
  `sigma_z = sqrt(eps_Si kT / (q² N_channel))` ≈ 9–87 nm for N = 1e17–1e15 cm⁻³.

Electrostatic estimate: **V_p ≈ 1–8 µm³, baseline 3 µm³** (code default). Quote the
V_p = 1–10 µm³ band as the systematic. Single-trial scan result: the **masked** trap excess
varies only ~±25% over V_p = 0.3–30 µm³ (two decades), because capture is saturated for most
traps and the escape boundary moves only ∝ kT·ln V_p through the E distribution. The unmasked
excess grows monotonically with V_p.

### Why poorly-known sigma is not a separate problem

- Initial capture saturates: `P = 1 − exp(−q sigma v Δt / V_p)` > 0.95 per dwell for 91–95% of
  the fitted population at V_p = 1–10 µm³. Order-of-magnitude sigma errors don't matter there.
- Elsewhere sigma appears only as sigma/V_p (verified: sigma ×10 ≡ V_p ÷10 exactly), so any
  **global** sigma bias is absorbed by the V_p systematic band. Per-trap random errors average
  over ~1300 traps/quadrant; the empirical (tau, sigma) pair resampling carries the measured
  scatter into the simulation.
- tau_e — the non-degenerate, directly measured quantity — is what the histogram supplies.

---

## 4. Data flow and inputs

```
proc/*.fits → run_charge_traps.py → fit_dipole_spectra_err_4.h5 (per-trap tau(T), E, sigma)
                                  → tau_at_135k_hist.npz          (tau sampling histogram)
                                  → trap_tau135_sigma_pairs.npz   (per-trap tau135/sigma/E pairs,
                                                                   via make_trap_pairs.make_pairs)
charge_trap_figures.ipynb         → tau_at_135k_hist_upper_limit.npz (efficiency-corrected 90% CL,
                                                                   richer keys; notebook-generated)
run_ccd_simulation.py / run_campaign.py → per-trial HDF5 in campaign/<scenario>/
```

- Each simulated trap: tau sampled log-uniformly within a histogram bin; sigma resampled from
  the ~20 measured (tau135, sigma) pairs nearest in log tau (preserves the empirical tau–sigma
  correlation); `k_c = sigma * v_th(135 K) / V_p`.
- `make_trap_pairs.py` refits (E, log sigma) from the **stored per-temperature tau values**
  (`energy_temperatures`/`energy_taus`/`energy_tau_errs`), so it is correct regardless of
  whether the HDF5 cache predates the constants fix.
- Bug fixed in `run_charge_traps.py` step 6 (2026-06-11): it read `EnergyFitInfo/...` keys that
  `fitTrapIntensity` never wrote (it writes `energy_BestFitCrossSection` etc. directly), so a
  pipeline rerun silently produced an **empty** tau histogram. Now fixed + hard error on empty.
  Binning standardized to `geomspace(1e-7, 1e8, 100)`.

### Upper-limit runs

The UL histogram integrates to 18,646 effective traps (efficiency-corrected per-bin 90% CL)
vs 5,171 detected. UL scenarios therefore scale the trap **density** by
`hist.sum()/5171 = 3.61` (via `--trap-density-scale`), not just the tau shape — simulating the
hypothesis "the population is this big with this composition". Decision confirmed 2026-06-11.
The completeness study (`trap_completeness_method3/`, Method 3 injection-recovery giving
P(characterized | tau_135, E) with a measured amplitude prior) covers tau-window and
amplitude/dimness selection effects, but **cannot cover phase blindness**: pumping only sees
traps in the pumpable (edge) phases, while readout charge passes through all phases. If 2 of 3
phases are probed, true density is another ×1.5 (use `--upper-density-scale 5.4` for the fully
conservative variant, or state the per-pumpable-phase assumption in the paper; effects scale
linearly with density).

---

## 5. Operating-condition decisions (confirmed by Ansh, 2026-06-11)

- **SNOLAB = 10× fewer high-energy events** than MINOS (code `exp/10` is right; the paper text
  said "factor of 2" and was corrected).
- **Rates are per pixel**: exposure-independent 9.94e-5 e/pix/image, exposure-dependent
  4.36e-5 e/pix/day (UR quadrant, SENSEI:2024yyt). Old "superpix" comments were wrong.
- **Inter-image dead time ≈ 0** in real operations → the simulation's instantaneous clear with
  no trap interaction is acceptable. (Traps do not release unobserved between images.)

## 6. Known remaining simplifications (assessed as minor, in order of relevance)

1. Charge resident in a trap's own pixel during exposure is never captured (interaction runs on
   the cleared state before injection). Second-order: column transits during readout load the
   relevant traps with near-certainty anyway.
2. Exposure-independent (readout-generated) charge is injected before readout, so it sees the
   full column of traps instead of the remaining transfer distance — mildly conservative.
3. No serial-register traps (tpix_horizontal unused for trap physics).
4. V_p is a single uniform-density number; in reality trap position within the well modulates
   the local density (traps outside the carrier cloud capture less). Absorbed by the V_p band.
5. q changes by ±1 within a dwell are ignored in the two-state rates (fixed-q approximation).
6. `charge_trap_simulation.ipynb` predates the API change (`CCD(...)` signature) — update before use.

## 7. Masking fidelity vs the real SENSEI pipeline (hotcol.py comparison, 2026-06-11)

Goal (per Ansh): reproduce the SENSEI analysis masking as faithfully as possible — not optimize
it. The A/B split-sample is the accepted stand-in for SENSEI's "derive masks on commissioning,
apply to blinded data" (masks always derived on data independent of the data they cut).
Reference implementation: `hotcol.py` (copied into the repo root).

**Verified verbatim-equivalent**: windowSum/expandWindow; calculatePvals incl. `sf(hits−0.5)`;
pCut=0.5 with p×nCells trials scaling; window sizes 1–5; addNeighbors (unscaled p, 0.05,
seeded from all bad cols); iteration to stability; hot-pixel→column merge thresholds;
hot columns zeroed from pixel data before the pixel pass; halo (60 pix around >100e) and
bleed (full column above >100e) confirmed to match the real pipeline.

**Deliberate exclusions (confirmed by Ansh)**: the very-hot column/pixel pre-cuts and the
16-chunk cut exist in the translation but are disabled — they target real detector defects
(dark spikes, persistent clusters) that the simulation does not contain.

**Known differences — reviewed with Ansh 2026-06-11 and deliberately kept as-is**
(documentation only; do not "fix" these without a new decision):

1. **Rate model.** hotcol.py uses the worst-case envelope `max(uniform, linX 0→2)` for columns
   and `max(uniform, linX, linY)` for pixels (linX = serial direction / spurious-charge
   gradient; linY = dark-current gradient). The simulation currently uses a flat model
   (`rateModel(mode="uniform")`). Fidelity argues for the envelope: it makes the real cut
   up to ×2 *more lenient* at high column/row numbers, so the uniform model overestimates how
   much trap background SENSEI's masks remove. Switching would likely *increase* the surviving
   trap excess. The sim's "linear" mode currently implements only the X ramp — the 2-D
   pixel-stage version (max with linY) would need to be added.
2. **Joint 1e + 2+e flagging.** hotcol.py flags columns/pixels using both the 1e and the 2+e
   datasets (pScales = [nCells, nCells·nHDUs] per file pair, combined by min). The simulation
   feeds only 1e counts to the finder (2e_counts are recorded but unused for masking). Also
   note nHDUs = 4 in the real analysis; a single simulated quadrant should use nHDUs = 4 for
   the 2e scale to match the real p-cut.
3. (minor) hotcol.py's chunk cut is on by default in the real pipeline; excluded here per the
   deliberate-exclusions note above.

Net effect of keeping the uniform rate model and 1e-only flagging: the simulated masks are
somewhat *stricter* than the real pipeline's, so the quoted masked trap excess errs on the low
side — acceptable and documented. Under the upper-limit population (~1.5 traps/column) the
hot-column finder largely absorbs the quasi-uniform trap excess into the rate estimate
regardless, so the rate-model choice matters little there.

## 8. Headline quantitative impact of the recapture fix (single-trial, MINOS, baseline pop.)

| model | unmasked excess | fully-masked excess |
|---|---|---|
| old (no recapture) | ~2e-5 | ~1.6e-7 (paper value) |
| new, V_p = 0.3–30 µm³ | 2.4–4.5e-5 | **4.4–7.0e-6** |

The masked trap effect is ~35× the old estimate — ≈5–7% of the measured exposure-independent
rate, no longer "below 1%". Final numbers require the full campaign (`run_campaign.py`,
12 scenarios: {baseline, UL} × {minos, snolab} × V_p {1, 3, 10}, priority-ordered, resumable)
including the exposure-dep/indep rate-decomposition fits.

An independent data-driven cross-check of recapture strength (no V_p assumption): the model
predicts deferred 1e events spatially correlated with cluster **columns** in subsequent images —
measurable in existing MINOS data.
