  Adversarial review — Signed Refit Runbook, Steps 1–5

  Overall judgment

  The methodology is a genuine improvement over the legacy fit, and the engineering discipline (frozen manifest,
  SHA-pinned artifacts, held-out splits, deferral of detection thresholds to Step 6) is good. The work is internally
  honest — most of my objections are foreshadowed somewhere in SIGNED_REFIT_PHYSICS_AUDIT.md. But the audit recommended
  these things be calibrated and closed, and what Steps 2–5 actually do is build a self-consistent null machine and then
  validate it against itself. The central physics risk the audit raised — that a real pumped dipole has variance the
  null controls cannot see — is never measured, and the closure data already shows its fingerprint. I would not let
  Steps 2–5 be called "PASS" without addressing the items below.

  ---
  1. The through-line problem: everything is calibrated and validated on charge-free nulls

  This is the most important issue and it touches Steps 2, 3, 4, and 5 simultaneously.

  The covariance is built from control pairs that are, by selection, empty — no pumped trap, low static intensity,
  outliers trimmed at 8σ (PAIR_Z_THRESHOLD), winsorized at 5σ. The per-dwell σ ≈ 35 e⁻ you measure is dominated by
  Poisson shot noise on the deferred/dark charge at a typical background pixel. That is the right floor for a null
  pixel.

  But a real candidate is a charge transfer. The audit said it plainly (SIGNED_REFIT_PHYSICS_AUDIT.md:79–86): for
  I=(a−b)/2, a transferred charge X contributes Var(X) to the variance, and over 3000 pump cycles Var(X) ≠ E[X]. A real
  dipole also sits on more charge → more shot noise. None of Steps 2–5 measure or bound this excess. The covariance
  applied to candidates is therefore a lower bound on their true variance, which means:

  - candidate χ² is inflated → p-values too small,
  - amplitude/τ significances are overstated,
  - the Δχ²=1 τ intervals from Step 5 will under-cover on real (bright) traps.

  This isn't hypothetical — your own closure shows it (next point).

  It is also not scheduled anywhere I can find. The audit's recommendation #4 ("treat signal-dependent pumping variance
  as a separate model uncertainty… checked using residual closure binned by fitted amplitude") appears only as a vague
  clause in Step 11 ("any pumping-variance model"). It is never derived. The τ uncertainties that feed the SRH fit (Step
  10) will not contain it.

  2. The closure (Step 4) does not actually close — and fails in the physically diagnostic direction

  The runbook reports Step 4 as PASS, but look at what the numbers say on held-out nulls:

  - Constant-model p<0.05 rate = 7.7% globally (should be 5%), p<0.01 = 2.3% (should be 1%). KS p-value vs uniform = 0.
  - It is worst at warm/high-charge scans: 183 K → 9.2%, 200 K → 9.0%, 180/187/190/193/197 all 8.3–8.5%.
  - It scales with brightness: quartile 1→4 gives width 1.025→1.035, p05 7.49%→8.28%, trial-rate 1.88%→2.20% (closure
  report, brightness table).

  Width ≈ 1.026 alone looks fine, but width≈1.03 together with a 7.7% tail rate means the residuals are heavier-tailed
  than Gaussian and the variance is mildly underestimated, increasingly so where there is more charge. That is exactly
  the signature of the missing signal-dependent term in #1, seen even within the tiny brightness range of null controls.
  Extrapolating to real candidates (orders of magnitude brighter) the under-coverage will be worse, not 1.5×.

  The gate passed only because the thresholds are loose: width up to 1.15, p05 up to 0.10, |z|>3 up to 0.02. A genuinely
  miscalibrated-at-the-tails model clears those. I'd argue Step 4 demonstrates the model does not close at the 5%/1%
  level and should say so, rather than asserting PASS. The honest framing — "analytical χ² tails are non-uniform; defer
  to empirical Step 6" — is in the prose, but the gate verdict and the runbook headline contradict it.

  The one unambiguously excellent result here: the trial-pump statistic empirically gives 1.94% of null curves exceeding
  Δχ²=11.83, p99.9=24.1. That kills the "11.83 = 3σ" claim cleanly and quantitatively. Keep that.

  3. Step 5's coverage validation is circular

  signed_refit_profile_fitter_validation.py:131 draws synthetic data with rng.multivariate_normal(0,
  calibration.covariance) — the same covariance the fitter then uses. So the 67.4% coverage and 0.007-dex bias prove
  only that (a) the GLS algebra is correct and (b) Δχ²=1 maps to ~68% when the noise is exactly the assumed multivariate
  Gaussian. Given #2, we already know the real noise is not that. This validation therefore cannot detect the
  under-coverage it most needs to detect.

  A non-circular version is cheap and you have the data for it: inject the known signal onto real held-out control
  residual curves (bootstrap/residual injection), then refit. That tests the fitter against the actual noise PDF,
  including the heavy tails. As written, the coverage claim should be stated as "coverage under the assumed covariance,"
  not as empirical coverage.

  Secondary: the gates are loose (per-scenario coverage 0.58–0.78 for a 68% interval; bias 0.03 dex with only 300
  realizations → bias SE is comparable to the 0.0068 dex result). Fine as a smoke test, weak as a guarantee.

  4. Step 1 freezes an acquisition-family confound onto the Arrhenius lever arm

  The manifest summary makes this concrete: 160 K and 170 K are the only two points from dp_scan1 (February, 300 000
  charge shifts), while every other temperature is temp_scan_run1 (March–April, 200 000 charge shifts). So at the cold
  end of the range — which carries most of the leverage on the trap energy E — temperature is aliased with acquisition
  family, illumination (1.5× charge shifts), and a 2-week-earlier detector state.

  The runbook's defense ("intensity model fits an independent amplitude per temperature") protects the amplitude but not
  τ or the SRH fit. A family-dependent change in dark current / pedestal / trap-filling shifts the curve shape (τ) and,
  more importantly, the cross-temperature SRH fit weights all T equally. This is a plausible contributor to the −8.91
  median pull and "acquisition-family offsets" the audit already flagged (SIGNED_REFIT_PHYSICS_AUDIT.md:280–285), and it
  is now frozen in as a fixed assumption. At minimum Step 10 must stratify residuals by family (it's in the task list)
  and pre-register dropping 160/170 K as a systematic. I'd want this called out as an open risk in Step 1, not buried as
  an "intentional illumination difference."

  Related, smaller: the scan was taken non-monotonically over ~2 months (185/195/210 K are early March; 197/200/203/207
  K are April), with run-IDs reused across families. Combined with "filename temperature is exact, no slow-control
  logs," any setpoint drift over that period is unquantified. Acknowledged as a limitation — fine — but the family
  aliasing above is the sharp version of it.

  5. The background model quietly changed, and the new term is not orthogonalized against the signal

  The legacy model was constant pedestal I0. The new pipeline subtracts a regional, dwell-dependent null_template
  (signed_refit_noise_model.py:130) and fits a constant offset. So the effective background is now null_template(dtph) +
  I0. This is potentially a real improvement — if the high-T lean was a t_ph-dependent pedestal (the open question in
  your memory high-t-arrhenius-lean and HIGH_ENERGY_EVENT_SAMPLING_NOTE), this template subtraction may be silently
  absorbing it. But:

  - it is nowhere flagged as a change to the physical background model; and
  - the fitter subtracts the template as a fixed vector without orthogonalizing it against pump_shape(τ). If
  null_template has any component along the pump shape (very plausible — dark current accumulated during the pump train
  grows with dwell, just like the rising exponential), that component biases the fitted amplitude and τ systematically.

  Concrete ask: regress each region's null_template onto the shape_grid and report the projection. If it's
  non-negligible, the template subtraction is partly cancelling real signal (or injecting bias) and needs to be handled
  inside the fit, not before it.

  6. Robustification biases the variance low — the wrong direction for error bars

  The stored covariance is OAS on 5σ-winsorized residuals (calibrate_region). Winsorizing pulls in the tails, which
  reduces the variance estimate; OAS shrinkage toward the scaled-identity mean does too. That is consistent with the
  closure width sitting above 1.0 everywhere (1.002–1.060). So the "robust" choice is making the error bars slightly
  anti-conservative across the board, compounding #1/#2. For a catalog whose headline risk is over-acceptance,
  robustification that shrinks σ is the wrong default — or at least should be corrected by a measured inflation factor
  rather than left as a >1 width that "passes."

  ---
  Smaller items worth a line

  - Off-diagonal correlations up to 0.969 in a null covariance across independent images deserve a physical explanation,
  not just "retained." Among independent exposures, large dwell-to-dwell correlation after template removal suggests a
  per-image common-mode (bias/gain) that the regional template didn't fully remove. GLS down-weighting handles it
  numerically, but if that common-mode is image-specific it could also down-weight real candidate signal in those
  images.
  - Sign/lobe-order consistency: the noise model defines I=(image[row]−image[row−1])/2. Step 9's candidate extraction
  must use the identical lobe order or every sign flips. Pin this in the manifest/contract now.
  - np.rint(image/400) quantizes to integer e⁻ before differencing; negligible at σ≈35 e⁻, but confirm the gain 400
  e⁻/ADU is the right calibration, since it sets the absolute σ scale (and hence every χ²).
  - The closure curve count 23×4×32×128 = 376,832 checks out; the artifact hashing/versioning chain is clean and I have
  no complaints about reproducibility plumbing.

  ---
  What I'd want answered before signing off