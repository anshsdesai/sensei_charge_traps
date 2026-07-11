# Plan: split the high-T intensity-fit failures — fading vs shape-fail vs pedestal-cost

**Status:** ready to execute (fresh session, morning of 2026-06-18+).
**Workflow:** delegate the mechanical build to **Codex (5.5, high)**, then **Claude does a physics review** of the diff before acceptance (per the user's delegation policy in `~/.claude/CLAUDE.md`). Show the user this spec before sending; review and report the diff before accepting.
**Deliverable:** `tools/high_t_failure_split.py` + `figures/high_t_failure_split.png` + `trap_completeness_method3/cache/high_t_failure_split_summary.json`.

---

## 0. How to run this in the morning
1. Read this whole file. Read `MEASUREMENT_COMPLETENESS_REVIEW.html` §9 (anchor `v-hight`, the "Naive-efficiency closure re-run … 2026-06-18" + "Open work" subsections) for context.
2. Send the **Codex handoff** in §7 to `mcp__codex5-5__codex` (sandbox `workspace-write`, cwd repo root). Show the user the spec first.
3. When Codex returns, run the **physics review checklist** in §8. Fix or bounce back.
4. Run the tool, sanity-check against §6 acceptance criteria, report the headline number (genuine-fading share = completeness-overclaim bound) to the user.

## 1. Why (background — one paragraph)
The naive measurement-efficiency curve `measured/(measured+extrapolated)` vs τ_e(T) dips at high T (170–210 K, short τ(T)). Confirmed 2026-06-18 on the minimal catalog: amplitude decline alone never makes the dip (PC-only prediction 0.943 both catalogs); the legacy dip was the harsh `p_value` GOF cut on a through-zero model, **already fixed** by the minimal pedestal model (dip 0.13→0.41). The **residual** minimal dip is dominated by `amplitude_significance < 3` (faintness, ~55% of high-T failures). The stage-11 hybrid now **over-predicts** (0.68 vs observed 0.41) because it rescales by the *bright* p-value survival and can't see the faint failures. That residual is the §9 "grid ~2× optimistic" and it makes `plot_completeness_overlay(m3)` **overclaim completeness at the high-T / long-τ₁₃₅ arm**. We do not yet know how much of the residual is real signal loss vs recoverable. This tool measures that split. See memory `naive-dip-minimal-closure`.

## 2. Goal
For the **high-T failures that drive the dip** — characterized traps (in the minimal catalog) that **failed** GoodIntensityFit at a high T whose τ(T) lands in the dip window — assign each point a mechanism, then report the composition count-weighted into the dip τ-window.

Per codex5.5's design review, this is **NOT** a clean (a)/(b)/(c) partition. It is:
- a **primary binary**: `recoverable_or_analysis_limited` vs `undetectable_sampled_contrast`;
- **mechanism tags** attached (not mutually exclusive): `shape_fail`, `pedestal_cost`, `tau_cost`, `compression`, `low_contrast`;
- the undetectable bucket is **leverage-split**: `genuine_fading` (had leverage, saw nothing below expectation → **the overclaim bound**) vs `design_compression_limited` (even expected amplitude wouldn't show).

`(a)` (= `undetectable_sampled_contrast` ∧ `genuine_fading`) is an **upper-limit category, never a residual bucket.** This is the one consequential boundary; the b/c distinction is secondary (both recoverable).

## 3. Data sources (exact)
- **Catalog (minimal):** `fit_dipole_spectra_minimal_caldet_err_4.h5` (repo root). Structure: `quad_<q>/dp_<row>_<col>/` group per trap (group name prefix is **`dp_`**, not `dipole_`; filter on the `energy_BestFitEnergy` attr as the loader does, don't match on name) with attrs `energy_BestFitEnergy`, `energy_BestFitCrossSection` (use `xs>0`), and per-temperature subgroups `temp_<T>` with datasets `seconds`, `intensities`, `intensity_err` and attrs `fit_coeff`, `fit_tau`, `fit_offset`, `fit_coeff_err`, `image_sigma`, `GoodIntensityFit`, `fit_p_value`. (Loader pattern: `tools/residual_audit_step0.py:60-87`.) **Verified 2026-06-18** (`quad_0/dp_100_25`): `fit_coeff_err` present per point.
  - **The catalog ALSO already caches the §4.1 free-fit quantities per `temp_<T>`:** `amplitude_significance`, `delta_chi2_vs_constant`, `delta_chi2_threshold` (the per-T δχ² threshold, **per point** — read this directly for the `shape_fail` tag instead of calling `load_delta_chi2_thresholds`), `fit_covariance_matrix`, `fit_tau_err`, `fit_offset_err`, `fit_reduced_chi_squared`. Use these as the **ground truth for the §6 self-consistency asserts** (fresh-refit `perr[0]` vs stored `fit_coeff_err`; fresh `amplitude_significance` vs stored). The fresh refit is still required (Fisher counterfactuals need the MLE + Jacobian), but it should reproduce these stored values — divergence is a bug signal.
- **Model:** `dipole_new.intensity_function_offset(tph, coeff, tau, offset)` (`dipole_new.py:485`) = `3000*coeff*(exp(-tph/tau) - exp(-8*tph/tau)) + offset`. **Linear in coeff and offset**; nonlinear only in τ. Shape template `g(s;τ) = 3000*(exp(-s/τ) - exp(-8*s/τ))`. Peak factor 0.650 at x=ln8/7 (`INTENSITY_SHAPE_PEAK`, `dipole_new.py:481`). Amplitude in electrons `A_e = 3000*coeff`.
- **SRH τ(T):** `dipole.log_energy_cross_section(T, E, logsigma)` returns **ln(τ_e)** → `tau_pred = exp(log_energy_cross_section(T, E, logsig))` (usage: `tools/residual_audit_step0.py:104`). **Verify** it returns the log (the name + this usage imply yes).
- **Fit settings to replicate** (for fresh refit + Fisher): `scipy.optimize.curve_fit(intensity_function_offset, s, y, sigma=err, p0=[...], bounds=([-inf,1e-8,-inf],[inf,1000,inf]), absolute_sigma=True, maxfev=20000)` — see `single_curve_recovery.py:256-277`. `amplitude_significance = |popt[0]|/perr[0]`, `perr=sqrt(diag(pcov))` (`:296,302`). `delta_chi2_vs_constant = chi2(best const) - chi2(model)` (`:305-308`).
- **PC(T) amplitude factor (Stage 5, minimal):** `trap_completeness_method3/cache/05_amplitude_prior_minimal_caldet_v1.npz`. **Load via `from trap_completeness_method3.src.validation_sensitivity import load_stage05`** (returns a dict). The keys `pc` (per-T factor) and `temperatures` are the **dict keys the loader produces**, NOT raw npz keys — the raw npz keys are `pc_temperature_factor` / `temperatures_K` (`validation_sensitivity.py:147-158`). Do **not** `np.load` the npz and index `pc`/`temperatures` directly; use the loader. (Verified 2026-06-18.)
- **Detection grid (optional, minimal):** `trap_completeness_method3/cache/08_pdet_grid_minimal_caldet_v1.h5` (τ 2e-5..20 s). Only needed if you cross-check "UL below detection threshold" against the grid; the primary undetectable test is forced-amplitude significance < 3 (self-contained).
- **Dip τ-window:** 3e-3–3e-2 s (matches `trap_completeness_method3/src/naive_efficiency_closure.py` `feature_windows["dip_3e-3_3e-2"]`).
- **Cold reference temps for amplitude expectation:** 140–165 K (`AMP_T_RANGE` in `residual_audit_step0.py:26` / `holdout_high_t_test.py:23`).
- **High-T targets:** all T ≥ 170 K present in the catalog (the dip arm). Also report the bright-cohort subset (grid p_det>0.9) for cross-check vs Step 0.

## 4. Per-point computed quantities (all leakage-free: target T is excluded from SRH τ, PC ref, and cold_coeff)
For each **failed** (trap, T) point at high T (GoodIntensityFit False, T≥170 K, trap characterized):

1. **Fresh free refit** (3-param, same settings as §3) → MLE (coeff, τ, offset), `perr`, `p_value`, `reduced_chi2`, `delta_chi2_const`, `fit_tau`. Sanity: `perr[0]` ≈ stored `fit_coeff_err`.
2. **τ_SRH** = exp(log_energy_cross_section(T, E, logσ)), E/logσ from the trap's stored energy fit. (Leakage-free: target T was not a good temp for a failure.)
3. **Free-fit Fisher counterfactuals** at the MLE. Build `J` (n×3), columns the analytic partials of `intensity_function_offset` wrt (coeff, τ, offset):
   - ∂/∂coeff = `g(s;τ) = 3000*(exp(-s/τ)-exp(-8 s/τ))`
   - ∂/∂offset = 1
   - ∂/∂τ = `3000*coeff*( (s/τ²)exp(-s/τ) - 8(s/τ²)exp(-8 s/τ) )`
   `W = diag(1/err²)`, `F = JᵀWJ` (3×3), `Cov = F⁻¹`.
   - `sig_full = |coeff| / sqrt(Cov[coeff,coeff])` (should ≈ amplitude_significance).
   - **Fix offset:** invert the **2×2 (coeff,τ) sub-block of F** (drop the offset row/col), `sig_fixoff = |coeff| / sqrt(F2x2_inv[coeff,coeff])`. (NOT the 2×2 block of `Cov` — that's wrong; see §8.)
   - **Fix offset & τ:** `sig_fixboth = |coeff| * sqrt(F[coeff,coeff])`.
   - Conditioning: if `cond(F)` (or the 2×2) exceeds a threshold (e.g. 1e12) → tag `unclassified`, do not assign (a).
4. **Forced-amplitude fit at τ=τ_SRH** (linear WLS, offset profiled): design `X=[g(s;τ_SRH), 1]`, `β=(XᵀWX)⁻¹XᵀWy`, `C=(XᵀWX)⁻¹`. `coeff_forced=β[0]`, `coeff_forced_err=sqrt(C[0,0])`. Orientation sign `sgn` = sign(cold_coeff). Projected `A_proj = sgn*coeff_forced`. `sig_forced = A_proj / coeff_forced_err` (signed; a wrong-sign bump is not evidence). **UL95** (one-sided) on magnitude = `A_proj + 1.645*coeff_forced_err`.
5. **Sampled leverage / compression:** `g = g(s;τ_SRH)`; `g_perp = g - (Σ w g/Σ w)` with `w=1/err²`; `I = g_perpᵀ W g_perp`. Expected coeff `A_exp = cold_coeff * PC(T)/PC(T_cold_ref)` where `cold_coeff` = median over the trap's good temps in 140–165 K of `fit_coeff`, `T_cold_ref` = the median of those temps (or 150 K). `SNR_exp = |A_exp| * sqrt(I)`.
6. **Raw pre-registered contrast:** define early/late delay sets from the **shape** (not the data): early = delays where `|g_perp|` is in its top tertile, late = bottom tertile. `C = wmean(y_early) - wmean(y_late)`, `σ_C` from propagated errors; project with `sgn`. `sig_C = sgn*C/σ_C`.
7. **τ match:** `tau_match_dex = |log10(fit_tau/τ_SRH)|`.

## 5. Classification (per point)
Primary binary + tags (tags are non-exclusive flags; the binary is the deliverable axis):

- **RECOVERABLE_OR_ANALYSIS_LIMITED** if ANY of:
  - `sig_forced ≥ 3` (a real bump at the SRH τ is significant when the pedestal is profiled), OR
  - `sig_C ≥ 3` (model-free contrast), OR
  - `sig_full < 3` but `sig_fixoff ≥ 3` → tag `pedestal_cost`, OR
  - `sig_fixoff < 3` but `sig_fixboth ≥ 3` → tag `tau_cost`.
  - tag `shape_fail` if `sig_full ≥ 3` AND fails `p_value` AND `delta_chi2_const ≥ per-T threshold` AND `tau_match_dex` small (≤0.3).
- **UNDETECTABLE_SAMPLED_CONTRAST** otherwise (forced insignificant AND contrast null AND no pedestal/τ recovery). Then leverage-split:
  - `genuine_fading` if `SNR_exp ≥ 3` (the sampling **had** the leverage to see the expected amplitude, but the bump is below the UL → real loss beyond the survivor PC trend). **← completeness-overclaim bound.**
  - `design_compression_limited` if `SNR_exp < 3` (even the expected amplitude would not clear 3σ given the delay grid/noise → tag `compression` or `low_contrast`).
- **unclassified** if Fisher near-singular (report separately; exclude from fractions or report as its own slice).

## 6. Output + acceptance criteria
**Outputs:**
- Per-point CSV: trap id, q, T, τ(T), all §4 quantities, primary class, tags, leverage class.
- Summary JSON: composition fractions (counts) — overall high-T, **dip-window-weighted** (τ(T)∈3e-3–3e-2), and per-T; the headline `genuine_fading_fraction` (PC-scaled A_exp) with a binomial error **AND** `genuine_fading_fraction_rawcold` (cold_coeff, no PC) as the bracketing context number (§9); the bright-cohort (grid p>0.9) subset split for the Step-0 cross-check.
- Figure: stacked composition vs T (and/or vs τ).

**Acceptance / self-consistency asserts:**
- Fresh-refit `perr[0]` within ~5% of stored `fit_coeff_err` for the bulk (flag outliers).
- `sig_fixoff ≥ sig_full` and `sig_fixboth ≥ sig_fixoff` **always** (fixing a nuisance cannot increase error) — assert; violation ⇒ a Fisher/sub-block bug.
- Buckets sum to the number of failed high-T points.
- On the **bright cohort**, recoverable vs undetectable should roughly track Step 0's POP2 (~50–60%) / POP1 (~40–50%) (`MEASUREMENT_COMPLETENESS_REVIEW.html` §9 Step-0 table). Large divergence ⇒ investigate before trusting the faint-population numbers.
- Report `genuine_fading_fraction` separately for "vs survivor-PC expectation" (A_exp with PC scaling) — see open decision in §9.

## 7. Codex handoff (send to mcp__codex5-5__codex, sandbox workspace-write)
> Implement `tools/high_t_failure_split.py` per the spec in `HIGH_T_FAILURE_SPLIT_PLAN.md` §§3–6 (read that file). It classifies high-T intensity-fit failures in the minimal catalog into a recoverable-vs-undetectable binary with mechanism tags and a leverage sub-split, to bound the completeness overclaim. Reuse the catalog/SRH loader pattern from `tools/residual_audit_step0.py`. Key requirements, do not deviate: (1) forced-amplitude fit at τ=τ_SRH is a **linear** weighted LS (model is linear in coeff & offset); (2) the pedestal-cost test inverts the **2×2 (coeff,τ) sub-block of the Fisher matrix F=JᵀWJ**, NOT the 2×2 block of Cov=F⁻¹; (3) analytic Jacobian columns exactly as in §4.3; (4) all SRH τ / PC ref / cold_coeff exclude the target T (leakage-free); (5) no post-hoc threshold tuning; (6) near-singular Fisher ⇒ `unclassified`, never `genuine_fading`; (7) emit the asserts in §6. Produce the CSV, summary JSON, and figure named in the header. Print the dip-window-weighted composition and the headline `genuine_fading_fraction`. Use env `sensei_charge_traps_new`.

## 8. Physics review checklist (Claude, after Codex returns — do NOT skip)
- [ ] Forced fit: design matrix `[g(s;τ_SRH), 1]`, weights `1/err²`, offset truly profiled (free). Closed-form matches a `curve_fit` with τ fixed.
- [ ] Jacobian ∂/∂τ correct (sign and the `s/τ²` factors); verify numerically against finite differences on a sample curve.
- [ ] **Fix-offset error = 2×2 sub-block of F inverted**, not the sub-block of Cov. (This is the single easiest bug; it would understate pedestal_cost.) Confirm `sig_fixoff ≥ sig_full` holds.
- [ ] Sign/orientation: cold sign applied consistently to forced amplitude AND contrast; wrong-sign "bumps" not counted as evidence.
- [ ] Leakage: τ_SRH from E/logσ (fit on good temps excl. target T); `cold_coeff` and PC ref both exclude target T.
- [ ] `A_exp` PC scaling uses the **minimal** stage-5 npz; document that PC is survivor-biased so `genuine_fading` = loss *beyond* the survivor trend (conservative for the overclaim — see §9).
- [ ] Near-singular handling present; `unclassified` reported, excluded from (a).
- [ ] Dip-window weighting reproduces τ(T) the same way as `naive_efficiency_closure.py` (`tau_at_temperature`).
- [ ] Acceptance asserts in §6 all fire and pass; bright-cohort cross-check vs Step 0 sane.

## 9. Open decisions (flag to user) vs missing facts (verify, not decisions)
**Open decisions (have a default, confirm with user):**
- `A_exp` scaling — **DECIDED 2026-06-18 (user): emit BOTH, PC-scaled is the headline.**
  - **Primary / headline:** `A_exp = cold_coeff × PC(T)/PC(ref)` (PC-scaled). `genuine_fading` then means loss *beyond* the model's already-discounted smooth decline = the *extra, un-modeled* overclaim. **Caveat to carry through the output:** stage-5 `PC(T)` is fit on **good (surviving) fits only**, so it is survivor-biased — the modeled decline is shallower than the true population decline, which makes the PC-scaled expectation a *high* bar and the resulting `genuine_fading_fraction` a **conservative (upper) bound** on the un-modeled overclaim. De-biasing this requires the survivor-free PC re-measurement in §10.
  - **Also report:** `A_exp = cold_coeff` (raw cold, no PC) as a second `genuine_fading_fraction_rawcold`. This brackets the other extreme (counts the entire decline, including the smooth PC part already in the grid → an over-count, not the overclaim). Headline stays PC-scaled; raw-cold is context only.
- 95% one-sided UL via 1.645σ (Gaussian). DEFAULT fine.
- `SNR_exp ≥ 3` as the "had leverage" threshold (matches 3σ detection). DEFAULT fine.

**Missing facts to verify before/while building (not decisions):**
- `log_energy_cross_section` returns ln(τ) (usage implies yes — confirm).
- Stage-5 npz keys `pc`, `temperatures` (and that PC is anchored such that `PC(T)/PC(ref)` is the right ratio).
- Catalog stores `fit_coeff_err` per `temp_<T>` point.
- The per-T `delta_chi2` threshold source for the `shape_fail` tag (`load_delta_chi2_thresholds(flavor)`, used in `holdout_high_t_test.py:19,42`).

## 10. Follow-on (not in this tool — the other half of Track B)
Separately, feed a **survivor-free PC(T)** (re-measured by forced photometry at fixed cold coords incl. failed points) through the minimal grid and re-run stage 11; if the 0.26 over-prediction gap closes, the residual dip is fully accounted for. This tool quantifies the mechanism split; that closure quantifies whether the un-biased amplitude alone explains the gap. Do after this split lands.
