# Amplitude vs Temperature Across Traps

## Suggested code cell

```python
from IPython.display import Image, display

display(Image("coeff_vs_temperature_summary.png", width=1200))
display(Image("coeff_vs_temperature_followups.png", width=1200))
```

## Suggested markdown cell

### Question

Do all traps have the same amplitude-versus-temperature relation, where the amplitude is the fitted `fit_coeff` from the intensity model in `dipole.py`?

### Methodology

I used the saved per-trap intensity-fit results in `fit_dipole_spectra_err_4.h5`. For each trap and each temperature, I extracted the fitted amplitude `fit_coeff` only when the temperature point had `GoodIntensityFit=True`. To focus on traps with reasonably complete and stable behavior, I restricted the main normalized study to traps that were marked `WellBehavedTrap=True`, had `EnergyFitFailed=False`, and had at least 4 good temperature points. This left 2,514 usable traps.

I studied the question in two complementary ways:

1. **Absolute amplitude**: compare the raw fitted `fit_coeff(T)` values across traps.
2. **Normalized amplitude**: divide each trap's amplitude curve by its own typical scale and compare the resulting shapes.

For the normalization, I used each trap's **geometric-mean amplitude** across temperature, rather than normalizing by the maximum point. The goal was to test whether the curves share a common **shape** up to an overall multiplicative scale, without artificially forcing every trap to equal 1 at its highest point.

### How to read the main summary figure

- **Top left: Absolute Amplitude**
  This panel shows the median raw `fit_coeff` across traps at each temperature, with the 16th to 84th percentile band. It answers whether all traps have the same absolute amplitude scale. They do not.

- **Top right: Trap-Normalized Amplitude**
  This panel shows the same quantity after dividing each trap by its own geometric-mean amplitude. This isolates the temperature dependence of the amplitude from the trap-to-trap scale differences.

- **Bottom left: Normalized Curves Across Traps**
  Gray curves are individual normalized trap curves, and the red curve is the population median with the 16th to 84th percentile band. This makes it easy to see both the common trend and the remaining trap-to-trap scatter.

- **Bottom right: Normalized Curves by Trap Energy Quartile**
  This checks whether the normalized amplitude shape depends on fitted trap energy.

### Main result

The amplitude-versus-temperature relation is **not identical for all traps**, but it is also **not completely unrelated from trap to trap**.

After removing the overall scale of each trap, there is a clear shared average trend:

- the normalized amplitude is low near 135 K,
- rises toward a broad maximum around 160 to 170 K,
- then falls again toward 210 K.

Numerically, the median normalized amplitude is:

- about **0.58** at 135 K,
- about **1.01** at 145 K,
- about **1.20** at 160 K,
- about **1.20** at 170 K,
- about **0.97** at 190 K,
- about **0.73** at 210 K.

So the simplest summary is:

> The traps appear to share a broad common amplitude-versus-temperature shape once each trap is rescaled by its own characteristic amplitude, but there is still noticeable trap-to-trap scatter around that shared trend.

## Follow-up checks

### 1. Does the answer depend strongly on how the curves are normalized?

I compared three normalizations:

- geometric mean of each trap,
- median of each trap,
- maximum of each trap.

The **geometric-mean** and **median** normalizations give very similar temperature trends. The **max-normalized** curves look systematically flatter and are forced to peak near 1 by construction, which makes the traps look more similar than they really are. For that reason, I treat the geometric-mean normalization as the most honest version of the “same shape up to scale?” question.

### 2. Are some temperatures too sparse to trust?

Yes, especially the lowest temperatures.

The usable-trap counts by temperature are:

- 125 K: 25 traps
- 130 K: 41 traps
- 135 K: 389 traps
- 140 K: 1,444 traps
- 145 K: 2,238 traps
- 150 K: 2,354 traps

and then gradually decreasing again at the highest temperatures.

This means the broad trend from roughly **140 K to 210 K** is much better supported than the behavior at **125 K to 130 K**. The dip at 135 K is already based on a much healthier sample than 125 K or 130 K, but the very earliest points should still be interpreted cautiously.

### 3. Do different CCD quadrants behave differently?

Not in a major way.

All four quadrants show the same broad normalized pattern: rise toward about 160 to 170 K and decline by 210 K. There is some modest spread at the highest temperatures. For example, the median normalized amplitude at 210 K is:

- Quad 0: 0.69
- Quad 1: 0.72
- Quad 2: 0.80
- Quad 3: 0.83

So there may be some quadrant dependence in the high-temperature tail, but quadrant-to-quadrant differences are smaller than the overall shared temperature trend.

### 4. Does the trend depend on fitted trap energy?

Somewhat, but not enough to erase the shared shape.

When the traps are split into fitted-energy quartiles, all quartiles still show the same broad behavior: a dip around 135 K, a rise toward 160 to 170 K, and a decline at high temperature. The lowest-energy quartile shows the largest deviations from the overall median shape, especially at high temperature, so it would be fair to say:

> there is evidence for a common trend, but the detailed shape is not perfectly universal across trap-energy populations.

### 5. Could amplitude-tau covariance in the fit be driving the trend?

Possibly in part, but probably not all of it.

I checked the correlation between `log10(coeff)` and `log10(tau)` across traps at fixed temperature. The correlation varies with temperature:

- near zero at several temperatures,
- moderately positive around 140 K,
- mildly positive again at the highest temperatures.

So amplitude-tau covariance is real and should be kept in mind, but it does **not** look like a single simple degeneracy that explains the whole temperature pattern. A stronger follow-up would be to repeat the study with `tau(T)` fixed from each trap's energy-fit model and refit only the amplitude.

## Conclusion

The data do **not** support the statement that the amplitude-versus-temperature relation is exactly the same for all traps.

A better statement is:

> Most traps appear to follow a similar average amplitude-versus-temperature curve after removing their overall amplitude scale, but there is still meaningful trap-to-trap scatter, plus some dependence on trap energy and possibly on fit covariance at specific temperatures.

## Reasonable next steps

1. Repeat the study with `tau(T)` fixed from the energy fit, to reduce amplitude-tau degeneracy.
2. Quantify the residual scatter around a shared temperature curve and compare it to the reported `fit_coeff_err`.
3. Fit a simple shared-shape model, for example `log coeff_trap(T) = alpha_trap + f(T) + residual`, and test whether the residual structure depends on energy or quadrant.
4. If the low-energy quartile remains distinct, treat that as evidence for multiple trap populations rather than one universal amplitude shape.
