# Trap finding, fitting, and grading: legacy vs. minimal

This note explains, in plain language, how the pipeline (1) **finds** charge traps,
(2) **fits** them, and (3) decides which fitted traps are **good**. For each stage it
compares the original code in `dipole.py` ("legacy") with the rewritten code in
`dipole_new.py` ("minimal").

The two versions live side by side and never overwrite each other's output files.
Which one runs is controlled by one switch in `run_charge_traps.py`:

- `--pipeline legacy` (default) → uses `dipole.py` with its original settings.
- `--pipeline minimal` → uses `dipole_new.py` **and turns on all of the new
  options at once**: the sturdier noise estimate and relaxed symmetry check when
  finding, the physical error bars when building the curves, and the
  pedestal-aware signed fit with absolute error bars when fitting.

So the "new defaults" people refer to are really the minimal-pipeline settings;
in the code each one still defaults to the old behaviour unless minimal turns it on.

There is a second switch that matters for the "is the signal real" decision:

- `--detection calibrated` → **this is what the final analysis uses.** On top of
  minimal, it builds a per-temperature detection threshold measured from the data
  itself (explained in the grading section below) instead of using a fixed guessed
  value. The headline result is produced with `--pipeline minimal --detection
  calibrated`.

---

## 1. Finding the traps

### What we're looking for

A charge trap grabs an electron as charge is clocked past it, holds it briefly,
then drops it one row later. In the image this shows up as a **dipole**: two
stacked pixels where one is darker than its surroundings and the one right below
it is brighter by about the same amount (charge stolen from one spot, dropped one
row down). The finder hunts the image for these stacked dark-then-bright pairs.

### The shared recipe (same in both versions)

1. **Estimate the noise level.** Measure the typical random pixel-to-pixel
   wiggle ("sigma"). Anything much bigger than sigma is a real feature, not noise.
2. **Flatten the background.** Subtract each row's median, so a row that happens
   to be brighter overall doesn't fool the finder. A normal pixel now sits near zero.
3. **Set a threshold at "3 sigma."** A real dipole is one pixel well above zero
   sitting directly on top of one pixel well below zero.
4. **Find the pairs with a multiply trick.** For every pixel, multiply its value
   by the pixel directly above it. Bright (positive) on top of dark (negative)
   gives a negative product; the more extreme both are, the more negative. Flag
   every spot where the product is more negative than the squared 3-sigma
   threshold — that's a strong up-down swing, exactly the dipole signature.
5. **Symmetry check.** A real trap moves the *same* charge from one pixel to the
   next, so the two lobes should be roughly equal. Keep a pair only if the bright
   and dark lobes are within a set percentage of each other; throw out lopsided
   pairs (cosmic rays, hot defects, stars).
6. **Repeat-appearance filter** (in `getDipoleList2`, the wrapper that runs the
   finder over many images). A real trap reappears image after image; random
   noise does not. Keep a coordinate only if it shows up in **more than one
   image** (taken with different delay settings) at the same temperature. This is
   the main false-alarm filter, and it is **unchanged** between versions.

### What changed in minimal

Two physics-motivated knobs were added to step 1 and step 5, plus a plumbing change.

**(a) A sturdier noise estimate (`robust_sigma`).**

This changes *how* sigma in step 1 is measured. There are two ways to measure the
"typical pixel wiggle," and they disagree only when there are a few weird pixels.

- **Legacy — standard deviation.** Take every pixel, measure how far it sits from
  the average, **square** that distance, average the squares, take the square
  root. The squaring is the catch: a pixel 10× farther out than normal counts
  **100×** as much. So a handful of hot defects, cosmic-ray hits, or the uneven
  dark-current blotches you get at high temperature can single-handedly drag the
  estimate upward. The number stops describing the *typical* pixel and starts
  describing the *worst* ones. Sigma comes out too big → the 3-sigma bar is set
  too high → genuine dipoles fall under the bar and are never flagged. The finder
  goes quietly blind, and nothing in the output tells you.

- **Minimal — MAD (median absolute deviation).** Read the one line of code
  inside-out:
  1. Take the **median** pixel value (line everything up smallest-to-largest, take
     the middle one). The median ignores extremes by construction — make the top
     10% of pixels a million times brighter and the middle one doesn't move.
  2. For each pixel, measure its distance from that median (distance only, ignore
     direction).
  3. Take the **median of those distances** — the typical distance from the
     middle. Again a median, so a few crazy pixels can't inflate it.
  4. Multiply by **1.4826**, a fixed conversion constant that makes the MAD line
     up with what the standard deviation *would* report for ordinary bell-curve
     noise. So for clean data the two methods agree; the new one only diverges
     when there are outliers to ignore.

  The key difference: **no squaring anywhere, and medians instead of averages.**
  Outliers can't dominate, because a median only cares about the middle of the
  pack, not how far the stragglers reach. This matters specifically at **high
  temperature**, where uneven dark current sprinkles in abnormal pixels: the
  standard deviation eats them and reports a falsely large jitter, while the MAD
  steps over them and reports the true jitter — so the detection bar stays where
  it should and the finder keeps catching real traps when it's hot.

  *(Side note on the squared threshold: the cutoff is written as `(3·sigma)²`
  with a minus sign. That isn't a second statistical step — it's bookkeeping for
  the multiply trick. Two lobes each clearing 3 sigma means their product clears
  `(3·sigma)×(3·sigma)`, and the minus sign enforces "one up, one down." Both
  versions square the threshold for the same reason; only the sigma feeding into
  it changed.)*

**(b) An adjustable / disableable symmetry check (`symmetry_perc`).**

Legacy hard-wires the lobe-matching requirement at 30%. Minimal lets you loosen
it or switch it off entirely, and the minimal pipeline switches it **off**. Why:
at high temperature a readout pedestal (explained in the fitting section below)
makes a *genuine* trap's two lobes unequal, so the strict 30% rule would wrongly
reject real traps. Turning the lobe check off here is safe because the repeat
appearance filter and the later fitting cuts do the real rejection.

**(c) Reproducible file lists (`image_files`).**

Legacy always re-scans the folder with a wildcard, so the exact set of images
processed depends on whatever happens to be on disk. Minimal lets you hand it an
explicit, frozen, sorted list of files, so the same run gives the same result —
useful for audit trails and fair run-to-run comparisons.

---

## 2. Fitting the traps

Once the finder has a list of trap locations, the pipeline measures, for each
trap, **how strongly it pumps** and **how fast it releases charge**, then watches
how the release speed changes with temperature to extract the trap's depth
(energy) and grab-radius (cross-section).

This happens in two layers:

- **Build the curves** (`getDipoleSpectra2`): for each trap, at each temperature,
  measure the dipole strength ("intensity") versus the delay setting, with an
  error bar on each point.
- **Fit per temperature** (`fitTrapIntensity`): fit each intensity-vs-delay curve
  to get the release time τ at that temperature, then fit all the τ's across
  temperature to get energy and cross-section.

### Building the curves: how the error bars are measured

The error bar on each point decides how much that point pulls on the fit, so this
is part of "how fitting is done."

- **Legacy — patch spread (`error_model='patch'`).** The error bar on a point is
  the spread of pixel values in the little 35×35 patch around the trap. But that
  spread mixes together two different things: real point-to-point random
  fluctuation *and* the fact that neighbouring pixels are genuinely different from
  each other (nonuniformity). It therefore **overestimates** the true random
  noise by roughly 2.5×, making every point look noisier than it is.

- **Minimal — physical noise (`error_model='physical'`).** The error bar is built
  from what actually fluctuates: a measured baseline noise for that
  temperature-and-quadrant (looked up from a noise table built from trap-free
  pixel pairs) combined with each pixel's own shot noise (brighter pixels carry
  more random scatter, which matters at high temperature). This is the genuine
  point-to-point uncertainty. The old patch spread is still recorded alongside,
  for reference, but it no longer drives the fit.

There's also a **sign** change here. Legacy takes the absolute value of each
intensity (`absolute=True`), throwing away whether the dipole points up or down.
Minimal keeps the sign (`absolute=False`), which the grading stage later uses to
check that a trap always pumps in the same direction.

### The model that gets fitted

- **Legacy — two-knob model.** Intensity versus delay is fit with a curve that
  rises then falls, controlled by two numbers: an overall **strength** and the
  **release time τ**. The strength is forced to be positive.

- **Minimal — three-knob model with a pedestal (`fit_offset=True`).** Same rise-
  and-fall shape, plus a third number: a constant **offset** (pedestal) that sits
  under the whole curve. The reason is physical: as the dark-current background is
  clocked out through the trap's pixel during readout, the trap defers some of
  that charge too, adding a roughly constant dipole that is present in *every*
  image regardless of the delay setting. The old two-knob model has nowhere to put
  that constant, so it leaks into the strength and τ and biases them — most badly
  at high temperature where the pedestal is largest. The third knob absorbs it.
  In the minimal model the strength is also allowed to be **either sign**, because
  the pumped dipole and the pedestal dipole can point in different directions.

### Bookkeeping that affects the answer

- **Absolute error bars (`errors_are_absolute=True`).** This tells the fitter to
  trust the supplied error bars as real, measured uncertainties. Legacy leaves
  this off, which makes the fitter quietly **rescale** the error bars to whatever
  makes the fit look average-quality — that changes the parameter uncertainties
  and therefore changes the later "is this significant?" cuts. Since the minimal
  error bars are already physically calibrated, minimal turns this on so they're
  used as-is.

---

## 3. Grading the traps (which fits count as "good")

A trap has to clear two sets of gates: per-temperature gates (is *this*
temperature's τ trustworthy?) and per-trap gates (does the trap behave like a
real, single physical trap across all temperatures?).

### Per-temperature gates (inside `fitTrapIntensity`)

Both versions require the fit to converge, the curve to actually describe the data
(a goodness-of-fit check), and the release time τ to be measured to better than
50%. They differ in how they decide the **signal is real**:

- **Legacy — "is the peak tall enough?"** The trap passes if the tallest point on
  the curve sticks up above 3× the noise (checked two ways: 3× the average error
  bar and 3× the image noise). With the absolute-value intensities and no
  pedestal, "tallest point" was a reasonable stand-in for "how strong is the
  pump."

- **Minimal — "is the pump strength significantly non-zero?"** Once there's a
  pedestal in the model, the tallest point is mostly measuring the pedestal, not
  the pump, so the old test is meaningless. Instead minimal asks whether the
  fitted **pump strength** is at least 3× its own uncertainty
  (`amplitude_significance ≥ 3`) — a direct "is the pumping real" test. It adds a
  second backstop (`delta_chi2_vs_constant`): the pumped curve must beat a flat
  line by some margin, which guards against cases where the fit latches onto a
  single spike. **How big that margin has to be is set by the calibrated detection
  table — see below.**

#### The calibrated detection table (what the final analysis uses)

The backstop above needs a number: *how much* better than a flat line is "better
enough" to count as a real trap? There are two ways to pick it.

- **The fixed guess (not used for the headline result).** Early code used a single
  fixed margin (11.83) for every temperature, meant to represent a strict
  "3-sigma" bar. The problem is that this fit has a hidden free knob — the release
  time τ — that the fit is free to slide around even when there's no real trap. A
  noise-only patch can therefore get a deceptively good "beat the flat line"
  margin just by parking τ wherever the noise happens to wiggle. Because of that,
  the usual textbook statistics don't apply, and the fixed 11.83 is **not** really
  a 3-sigma cut: in practice about 1.9% of trap-free locations sneak past it
  (closer to 2.3-sigma than the intended 0.27%). So the true false-alarm rate was
  several times higher than the number implied.

- **The calibrated table (`--detection calibrated`, used for the final
  analysis).** Instead of guessing, the pipeline **measures** the false-alarm rate
  directly from the data:
  1. Pick thousands of **trap-free control locations** — random pixel pairs that
     are not at any found trap (and not touching one). About 2,000 per quadrant.
  2. Run the **exact same signed pedestal fit** on each of these noise-only
     locations, so their "beat the flat line" margin is computed the identical way
     a real candidate's is.
  3. For each temperature, look at the spread of those noise-only margins and set
     the threshold at the point where only a small target fraction of pure-noise
     locations would exceed it. That target is **0.1%** (one in a thousand) per
     temperature.

  The result is a **separate threshold for each temperature**, set so that, by
  construction, only ~0.1% of genuinely empty locations would be mistaken for a
  trap at that temperature. This is a real, measured false-positive rate rather
  than a guess, and it adapts to how noisy each temperature actually is. A trap
  has to clear its temperature's measured threshold to count as detected there.

  *(The controls are drawn with a fixed random seed so the table is reproducible,
  and it's cached to a file so it isn't rebuilt every run.)*

### Per-trap gates (inside `fit_energy_cross_section`)

After collecting the good-temperature τ's, both versions require **at least 4
good temperatures** (`wellBehavedThreshold`, the "well-behaved trap" flag) and
then fit the depth/cross-section curve across temperature. Two things changed:

**(a) A new orientation-consistency gate (minimal only).**

A single real trap pumps in **one** direction, so all of its significant
temperatures should share the same sign. Minimal records how many temperatures
came out positive vs. negative and classifies the trap:

- all one sign → `single_positive` / `single_negative` (consistent, allowed
  through);
- a real mix of both → `ambiguous_sign_conflict` or `dual_response`, which is
  flagged **inconsistent and rejected**.

This catches locations that look like a trap at some temperatures and an
anti-trap at others — i.e. two overlapping defects or contamination, not one
clean trap. Legacy has no concept of sign at all (it threw the sign away when
building the curves), so it cannot make this distinction.

**(b) A different goodness-of-fit rule for the depth fit.**

Both versions ultimately judge the across-temperature fit by a reduced-chi-square
("how far off is the curve, per data point"), and both also reject unphysical
depth or cross-section values at the boundaries. The threshold changed:

- **Legacy — reduced-chi-square < 5.**
- **Minimal — reduced-chi-square < 10** (looser).

The reason is subtle and important. An earlier attempt graded this fit with a
p-value cut (`p > 0.05`), which **gets stricter the more temperatures you have**:
with ~15 precise τ points it has enough statistical power to reject a trap over a
~1% real-but-unmodelled lean in the high-temperature data, so the pass rate fell
from ~65% down to ~21% as the number of temperatures grew — meaning the
**best-measured traps were preferentially thrown away**. A reduced-chi-square cut
does **not** get harsher with more points, so it keeps the well-measured traps
while still removing genuinely non-trap-like blends (whose reduced-chi-square is
far above the threshold). The chosen value of 10 is deliberately loose for this
reason. *(Project notes flag that a tighter value around 3 may actually grow the
final trap census; 10 is the in-code default, not necessarily the last word.)*

Also removed in minimal: legacy had extra machinery for modelling intrinsic
scatter and auto-rejecting outliers in the depth fit; the minimal version
deliberately drops both, keeping the fit simple and predictable.

---

## Quick reference

| Stage | Legacy (`dipole.py`) | Minimal (`dipole_new.py`, via `--pipeline minimal`) |
|---|---|---|
| Noise estimate for finding | Standard deviation (inflated by outliers / hot pixels) | MAD, outlier-resistant (`robust_sigma=True`) |
| Lobe-symmetry check | Hard-wired 30% | Off (`symmetry_perc=None`); later cuts do the rejecting |
| Image list | Wildcard rescan of folder | Optional frozen, sorted list (`image_files`) |
| Error bars on curve points | Patch spread (~2.5× too big) | Physical temporal + shot noise (`error_model='physical'`) |
| Intensity sign | Absolute value (discarded) | Kept (`absolute=False`) |
| Curve model | 2 knobs: strength (≥0) + τ | 3 knobs: signed strength + τ + pedestal (`fit_offset=True`) |
| Error-bar handling | Rescaled by fitter | Trusted as-is (`errors_are_absolute=True`) |
| "Signal is real" gate | Tallest point > 3× noise | Fitted pump strength > 3× its error + Δχ² vs. flat line above a **calibrated per-temperature threshold** (`--detection calibrated`) |
| Δχ² threshold source | n/a | Final analysis: measured from trap-free control pixels at a 0.1% false-positive rate per temperature (not the fixed 11.83 guess) |
| Orientation gate | None | Reject sign-inconsistent traps (single physical direction) |
| Across-temperature GOF | reduced-χ² < 5 | reduced-χ² < 10 (avoids over-rejecting well-measured traps) |
| Min. good temperatures | 4 | 4 (unchanged) |
| Repeat-appearance filter | >1 image at same temperature | unchanged |
