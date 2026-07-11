# Signed Refit — Adversarial Physics Review of Steps 1–5

Date: 2026-06-13
Reviewer role: adversarial physicist
Scope: `SIGNED_REFIT_RUNBOOK.md` Steps 1–5 and their code/artifacts
(`signed_refit_manifest.py`, `signed_refit_controls.py`,
`signed_refit_noise_model.py`, `signed_refit_noise_closure.py`,
`signed_refit_profile_fitter.py`, `signed_refit_profile_fitter_validation.py`).

## Bottom line

The methodology is a real improvement over the legacy absolute-value/zero-offset
fit, and the reproducibility plumbing (frozen manifest, SHA-pinned artifacts,
held-out splits, deferral of detection thresholds to Step 6) is sound. But Steps
2–5 build a self-consistent **null** machine and then validate it largely
**against itself**. The central physics risk — that a real pumped dipole has
variance the empty control pairs cannot see — is not measured, has **no
implemented plan**, and is already visible in the Step 4 closure numbers. Steps
1–5 should not be treated as scientifically closed until item 1 below is
addressed.

Severity legend: **BLOCKING** / **HIGH** / **MEDIUM** / **CLEARED** (raised,
checked, found benign).

---

## 1. BLOCKING — Signal-dependent (pumping/shot) variance is never measured

Everything in Steps 2–5 is calibrated on control pairs that are empty by
construction (no pumped trap; static-intensity outliers trimmed at 8σ;
residuals winsorized at 5σ). The per-dwell σ ≈ 35 e⁻ is the Poisson shot-noise
floor of a typical **background** pixel.

A real candidate is a charge *transfer* and sits on more charge. For
`I=(a−b)/2`, a transferred charge `X` contributes `Var(X)` to the variance, and
over 3000 pump cycles `Var(X) ≠ E[X]` (see `SIGNED_REFIT_PHYSICS_AUDIT.md`,
"the current per-point error model is not yet physical"). The covariance applied
to candidates is therefore a **lower bound** on their true variance:

- candidate χ² inflated → p-values too small;
- amplitude / τ significances overstated;
- Step 5 Δχ²=1 τ intervals **under-cover on real (bright) traps**.

Status confirmed by the owner: *there is not yet an adequate implemented plan.*
The audit's recommendation (derive a pumping-variance term; close residuals
binned by fitted amplitude) appears only as a vague clause in Step 11 and is
never derived. The τ uncertainties feeding the SRH fit (Step 10) will not
contain it.

**Direct evidence it is already biting (from the Step 4 closure, on clean
nulls):** whitened width and tail rate both rise with control-pair brightness —
quartile 1→4: width 1.025→1.035, `p<0.05` 7.49%→8.28%, trial-rate
1.88%→2.20%. Within the *tiny* brightness range of null controls the variance is
already underestimated for brighter pairs; real candidates are orders of
magnitude brighter.

## 2. HIGH — The Step 4 closure does not close at the nominal level; "PASS" overstates it

On held-out nulls the constant-model tail rates are systematically wrong:
`p<0.05` = **7.7%** (target 5%), `p<0.01` = **2.3%** (target 1%), KS-vs-uniform
p = 0. It is worst at warm/high-charge scans (183 K → 9.2% and width 1.060;
200 K → 9.0%; 180/187/190/193/197 K all 8.3–8.5%). Width ≈ 1.03 *together with*
a 7.7% tail rate means the residuals are heavier-tailed than Gaussian and the
variance is mildly underestimated, more so where there is more charge — the same
signature as item 1.

The gate passes only because thresholds are loose (width ≤ 1.15, `p05` ≤ 0.10,
`|z|>3` ≤ 0.02). The honest prose ("analytical tails non-uniform; defer to
empirical Step 6") contradicts the headline "PASS". Recommend the verdict say
"covariance whitens widths to ≈1.03 but does **not** close the 5%/1% tails;
empirical Step 6 calibration is therefore mandatory, not optional."

The genuinely strong result here: the trial-pump statistic gives 1.94% of *null*
curves above Δχ²=11.83 (p99.9 = 24.1), quantitatively demolishing the
"11.83 = 3σ" claim. Keep and rely on this.

## 3. HIGH — Step 5 coverage validation is circular

`signed_refit_profile_fitter_validation.py` draws synthetic data with
`rng.multivariate_normal(0, calibration.covariance)` — the **same** covariance
the fitter uses. The 67.4% coverage / 0.007-dex bias prove only that (a) the GLS
algebra is correct and (b) Δχ²=1 ≈ 68% **when noise is exactly the assumed
Gaussian**. Given items 1–2 we know it is not. This validation cannot detect the
under-coverage it most needs to.

Fix (cheap, data already exists): inject the known signal onto **real held-out
control residual curves** (residual/bootstrap injection) and refit, so the test
sees the true noise PDF including heavy tails. Until then, state the result as
"coverage *under the assumed covariance*," not as empirical coverage. Also note
the gates are weak: per-scenario coverage band 0.58–0.78 for a 68% interval;
bias gate 0.03 dex with only 300 realizations (bias SE ≈ the measured value).

## 4. HIGH — Step 1 freezes an acquisition-family confound onto the Arrhenius lever arm

From the manifest summary: **160 K and 170 K are the only points from
`dp_scan1`** (February, 300 000 charge shifts); every other temperature is
`temp_scan_run1` (March–April, 200 000 charge shifts). At the cold end — which
carries most of the leverage on trap energy `E` — temperature is aliased with
acquisition family, 1.5× illumination, and a ~2-week-earlier detector state.

"Independent amplitude per temperature" protects amplitude, **not τ or the
cross-temperature SRH fit**. A family-dependent dark-current/pedestal/filling
difference shifts the curve *shape* (τ) and biases `E`. This is a plausible
driver of the −8.91 median pull and "acquisition-family offsets" the audit
flagged, now frozen in as a fixed assumption.

Owner decision: test/drop in Step 10. Acceptable **if pre-registered now** —
stratify SRH residuals by family and commit, before looking, to dropping
160/170 K as a systematic if their residuals are anomalous. Related: the scan
was acquired non-monotonically over ~2 months with run-ID reuse; with "filename
temperature exact, no slow-control logs," setpoint drift is unquantified
(accepted limitation, but the family aliasing is the sharp version).

## 5. CLEARED — Unintended dwell-dependent `null_template` subtraction does not bias τ

The new pipeline subtracts a regional, dwell-dependent `null_template` from
every candidate curve in addition to fitting a constant offset — effectively a
background-model change the owner confirms was **not** intended as one. Because
the fit is linear in amplitude/offset at fixed τ, subtracting a fixed vector `T`
shifts each candidate's fitted amplitude by exactly the amplitude one would fit
to `T` alone. I measured that directly over all 2944 regions (treating each
`null_template` as data, no auto-subtraction):

- template Δχ²-vs-constant: **max 0.12**, p95 0.02, p50 0.00;
- conditional |z| of the template's pump amplitude: **max 0.35σ**, p95 0.13;
- 0.000% of regions exceed Δχ² 11.83 or even 4.

So the template carries essentially **no** pump-shaped component; subtracting it
is equivalent to a constant/common-mode removal the offset term would absorb.
**Verified benign for τ.** Recommendation: keep this projection as a cheap
regression guard (add to the noise-model report), since the template is
empirically derived and could acquire a pump-shaped component if controls or
masking change.

## 6. MEDIUM — Robustification biases the variance estimate low

The stored covariance is OAS on 5σ-winsorized residuals. Winsorization pulls in
tails → **reduces** the variance estimate; OAS shrinkage toward the scaled
identity does too. This is consistent with the closure width sitting *above* 1.0
everywhere (1.002–1.060). For a catalog whose headline risk is over-acceptance,
a robust estimator that shrinks σ is the wrong-direction default. Either justify
it with a measured inflation factor or fold the >1 closure width into an
explicit covariance scale-up.

## Smaller items

- **Off-diagonal correlations up to 0.969 in a null covariance across
  independent images** need a physical explanation, not just "retained." Likely
  a per-image common-mode (bias/gain) the regional template doesn't fully
  remove. GLS handles it numerically, but if image-specific it can also
  down-weight real candidate signal in those images.
- **Lobe-order / sign contract:** the noise model defines
  `I=(image[row]−image[row−1])/2`. Step 9 candidate extraction must use the
  identical lobe order or every sign flips. Pin this now.
- **Gain `400` e⁻/ADU** (`ELECTRONIZE_SCALE`) sets the absolute σ scale and hence
  every χ²; confirm it is the right calibration.
- Reproducibility/versioning/hashing chain is clean; closure curve count
  `23×4×32×128 = 376,832` checks out.

## Recommended status changes

- Step 3 / Step 4: change the "PASS" headline to "PASS with documented
  non-closure of nominal χ² tails; empirical calibration mandatory."
- Step 5: relabel coverage as model-conditional; add a real-residual injection
  test before Step 6 relies on the intervals.
- Add a blocking item for the pumping-variance term (item 1) with its own
  acceptance gate, ahead of using any τ uncertainty in the SRH fit.
