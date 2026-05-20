# Trap-population completeness: from efficiency to band coverage to an analytic detection model

Working notes (2026-05-20) on how we assess whether our characterized-trap set is
**complete** — in particular whether a large hidden population of (especially long-lived)
traps could have been systematically missed by the measurement.

> **Status / reading order.** Method 1 (§3) is what currently lives in the paper. Method 2
> (§4) was a first attempt to fix it and is kept here as a stepping stone, but it is *not*
> the recommended approach. Method 3 (§5) — an analytic per-temperature detection model
> built directly from the paper's intensity equation — is what we now think is the right
> way forward. §6 records a dead end (the DM-impact simulation is **not** a detection model)
> so we don't relitigate it. Read §5, §7, §8 for the actual plan.

---

## 0. Background & glossary (read first if you're new to this)

**Why traps matter (the background mechanism).** A charge trap is a lattice defect that can
capture an electron and re-emit it later. In a dark-matter search the danger is: charge deposited
by a high-energy event gets trapped, then released *much later* as one or a few lone electrons —
mimicking the single-electron ($1e^-$) signal that light-dark-matter searches like SENSEI look
for. The release timescale is the emission time constant $\tau_e(T)$. **Long-lived traps (large
$\tau_{135}$) are the dangerous ones**, because they release long after the originating event and
across many exposures, so masking the original event doesn't catch the released electron.

**Pocket pumping & dipoles.** To find and characterize traps we uniformly fill the CCD with charge
and then "pocket-pump" — repeatedly shuffle charge back and forth between pixel phases
($N_{\text{pumps}}=3000$ cycles, dwell time $t_{ph}$ per phase). When a trap sits at a phase edge
it captures charge from one pixel and releases it into the neighbor, producing a **dipole**: an
adjacent +/− pair (one pixel above background, the neighbor below) standing out against the uniform
fill. The **intensity** $I$ of a dipole is *half the charge difference between the two pixels* —
i.e. how many carriers were trapped-and-released. $I$ peaks when $t_{ph}$ is comparable to $\tau_e$
(see §2.2), which is how scanning $t_{ph}$ and $T$ measures $\tau_e(T)$. In code the dipole finder
is `findDipoles2` (`dipole.py:55`): after subtracting the per-row median it flags adjacent vertical
pixel pairs whose product is below $-(3\sigma)^2$ (large and opposite-sign) and whose magnitudes are
comparable (`comparable_perc`, within 30%). The full analysis (identify → build spectra → fit
$\tau(T)$ → fit $E/\sigma$) lives in `dipole.py` and is orchestrated by `run_charge_traps.py`; the
code map is in the appendix.

**Glossary.**
- $\tau_e(T)$ — emission time constant; decreases with $T$ (warmer → faster release). Modeled by
  Eq. `tau(T)`, $\tau_e=(\sigma v_{th}N_c)^{-1}e^{E/k_BT}$, giving each trap an energy $E$ and
  capture cross-section $\sigma$.
- $\tau_{135}$ — $\tau_e$ extrapolated to the 135 K SENSEI@SNOLAB operating temperature; the axis
  the completeness question lives on.
- $t_{ph}$ — pumping dwell time per phase (the swept "delay"); sets which $\tau_e$ a measurement is
  sensitive to. Swept $\sim 5\times10^{-5}$–$1.03$ s, longer at higher $T$.
- $P_c$ — capture probability (temperature-dependent amplitude scaling).
- $D_t$ — per-trap amplitude factor (trap "depth" / number of trappable carriers); together
  $A = N_{\text{pumps}}D_t P_c$ is the dipole amplitude in Eq. §2.2.
- **good** trap — passed the per-temperature dipole/τ-fit quality cuts (§2.3) at a given $T$.
- **well-behaved / characterized** trap — has $\ge n_{good}$ good temperature fits, so it gets an
  Arrhenius $E/\sigma$ fit. Paper uses $n_{good}=4$ → 2121 traps.

**The paper's headline result (the context for this study).** Using the characterized traps (and a
90% CL upper-limit population from the efficiency correction) as input to an end-to-end CCD
simulation, the paper finds traps shift the SENSEI $1e^-$ rate by $<1\%$, and that standard masking
removes the effect — i.e. **traps do not significantly impact the single-electron background**.
This completeness study is the remaining qualifier on that conclusion: *could a hidden population of
long-lived traps, missed by the measurement, be large enough to overturn it?* That is the question
§1 onward addresses.

## 1. What we are trying to answer

We characterize single-electron charge traps in skipper-CCDs by their emission time constant
$\tau$, measured across temperatures 125–210 K using the **pocket-pumping** technique
(see paper §II–III). Each trap's energy $E$ and cross-section $\sigma$ are fit from its
$\tau(T)$ points via the thermal-emission model (Eq. `tau(T)` in the paper); we then project
$\tau$ to the 135 K operating temperature. The traps that matter for the dark-matter search
are the **long-lived** ones (large $\tau_{135}$).

**Core question:** in the $\tau_{135}$ regions that matter, are we confident the measured set is
complete? Concretely — *could there be, say, $2\times10^6$ traps at $\tau_{135}=10^3$ s that we
missed 99.99% of because of a measurement-efficiency effect?*

This is a question about a **population bound**, not just a per-trap efficiency. Keep that
distinction in view — §8 explains why the two are not the same and what extra assumption is
needed to get from one to the other.

## 2. What we know about the measurement (the detection physics is analytic)

The crucial realization since the first draft: **the per-temperature detection model is already
written down in closed form in the paper.** It is not something we need a Monte-Carlo for.

### 2.1 The measurement protocol (paper §II–III)
- Charge is generated uniformly (~$2000\,e^-$/pix) via spurious-charge clock swings (10 V,
  100 000 shifts) — no LED.
- Pocket pumping with $N_{\text{pumps}} = 3000$. The delay per clock state $t_{ph}$ is swept
  from $\sim 5\times10^{-5}$ s to $1.03$ s. **Only one skipper sample** (fast readout).
- Temperature scanned over 125–210 K, 18–25 images per temperature. **Longer $t_{ph}$ values
  were deliberately used at higher temperatures** to chase longer emission times — so the
  $t_{ph}$ grid, and therefore the sensitivity band, is **temperature-dependent by design**.

### 2.2 The dipole-intensity model (paper Eq. `I_fit`)
A trap's dipole intensity as a function of delay is
$$ I(t_{ph}) = N_{\text{pumps}}\, D_t\, P_c \left(e^{-t_{ph}/\tau_e} - e^{-8\,t_{ph}/\tau_e}\right), $$
where $P_c$ is the (temperature-dependent) capture probability and $D_t$ a per-trap amplitude
factor. Write the amplitude as $A \equiv N_{\text{pumps}}\, D_t\, P_c$.

Two consequences that reframe the whole completeness question:

- **The peak over $t_{ph}$ is at** $t_{ph}^\star = (\tau_e/7)\ln 8 \approx 0.30\,\tau_e$, and at
  that delay $I_{\text{peak}} \approx 0.65\,A$ — **independent of $\tau_e$**. A long-$\tau$ trap
  is *not* intrinsically faint.
- It only *looks* faint because we cannot reach its peak: $t_{ph}^{\max} \approx 1.03$ s. For
  $\tau_e = 10^3$ s the sampled delays sit on the rising linear part,
  $I \approx 7 A\, t_{ph}/\tau_e \approx 0.007\,A$. **The long-$\tau$ cutoff is a $t_{ph}$-reach
  limitation, not a physics one.** (Small-$x$ expansion: $e^{-x}-e^{-8x}\approx 7x$.)

So the upper $\tau$-edge of the detectable band at temperature $T$ is, to first order,
$$ \tau_{\max}(T) \;\approx\; \frac{7\,A\,t_{ph}^{\max}(T)}{\text{(detection threshold)}}, $$
and the short-$\tau$ edge is where the curve empties before the first sampled $t_{ph}$
($e^{-t_{ph}^{\min}/\tau}\to 0$). **Every ingredient in $\tau_{\max}(T)$ is measurable or
calibratable** (see §5.3), not a free parameter.

### 2.3 The "good trap" / "well-behaved" cuts (paper §III–IV)
A trap is logged as **good** at a temperature if:
1. max intensity $> 3\sigma$, where $\sigma = \sqrt{\mathrm{Var}}$ of the charge in the
   surrounding $35\times35$ pixels (the noise floor — **trap-independent and measurable from
   the images**, and it rises with $T$ as dark current grows);
2. relative error on the fitted $\tau$ is $< 50\%$;
3. max intensity $> 3\times$ the intensity error;
4. $\chi^2$ goodness-of-fit $p$-value $> 0.05$.

These cuts are implemented verbatim in `fitTrapIntensity` (`dipole.py:449`): see the
`GoodIntensityFit` flag being set/cleared at `dipole.py:530–540`. The noise $\sigma$ enters as
`image_sigma` (computed per dipole per temperature in `getDipoleSpectra2`, `dipole.py:324`), the
fitted-$\tau$ relative-error cut is `perr[1]/popt[1] > 0.5`, and the single-temperature fit bounds
$\tau$ to $[10^{-8}, 1000]$ s — so no single-$T$ fit can return $\tau > 1000$ s regardless of how
long-lived the trap is. (`fitTrapIntensity_cutflow`, `dipole.py:639`, is the same logic with a
cutflow tally.)

A trap is **well-behaved** (gets an $E/\sigma$ Arrhenius fit) only if it has $\ge n_{good}$
good temperature fits — the paper uses $n_{good}=4$, giving **2121 well-behaved traps** out of
5171 identified / 3379 "good". This $\ge n_{good}$ requirement is the central selection effect
and was under-weighted in the first pass.

### 2.4 Why $\tau(T)$ sweeps through the band
A physical trap has **fixed** $E,\sigma$; its $\tau(T)$ follows the thermal-emission model and
**decreases monotonically as $T$ rises** (warmer → faster release). As $T$ scans, a trap's
$\tau(T)$ trajectory slides downward across orders of magnitude. The right question is whether
that trajectory lands inside the (T-dependent) detectable band at $\ge n_{good}$ temperatures.

### 2.5 Observed sample
Records from `fit_dipole_spectra_err_3.h5`: **2517 characterized traps** at $n_{good}\ge3$
(the paper's headline 2121 uses $n_{good}\ge4$; see `_err_4.h5`). $\tau_{135}$ distribution:
median ~4.4 s, 95% below ~11 s, tail to ~$10^8$ s; ~2.3% have $\tau_{135} > 10^3$ s.

## 3. Method 1 — the per-temperature efficiency curve (what's in the paper now)

For each temperature $T$ and $\tau$ bin, define
$$\mathrm{eff}(\tau, T) = \frac{\text{measured}}{\text{measured} + \text{extrapolated}},$$
where, among **characterized** traps whose $\tau(T)$ lands in that bin, "measured" = $\tau_e(T)$
was directly fit and "extrapolated" = $\tau_e(T)$ comes only from the Arrhenius fit. The paper
pools this across all traps/temperatures, reads it as detection efficiency vs $\tau$ (near unity
at $\tau\sim0.3$–1 s, →0 below $\sim10^{-5}$ s and above $\sim10^3$ s), and applies an
**inverse-efficiency correction** with a Poisson 90% CL upper limit per bin to bound the missed
population.

### Problems with the efficiency method
1. **Survivorship bias in the denominator.** Both numerator and denominator are drawn from
   *characterized* traps, so the efficiency conditions on being characterizable. It is
   structurally **blind to traps never characterized at any temperature** — exactly the hidden
   population we care about. It cannot bound that population from within itself.
2. **Low statistics at warm $T$.** Only tens of characterized traps exist above ~160 K (Group A
   is ~50 traps), so most warm-$T$ $\tau$ bins are empty and read "0 efficiency" from lack of
   *data*, not lack of *sensitivity*. The product $1-\prod_T(1-\mathrm{eff}_T)$ then badly
   *under*-estimates coverage there. (Observed directly: $\mathrm{eff}=0$ at 175 K for
   $\tau=0.86$ s, even though that $\tau$ is well within the demonstrated detectable range.)
3. **Bin / threshold dependence.** Window edges depend on binning and threshold; the unphysical
   plateau below $\tau\sim10^{-2}$ s is a binning/contamination artifact, not a real response.
4. **It answers the wrong question.** Efficiency at a *single* temperature ignores that each trap
   is re-interrogated at *many* temperatures as its $\tau$ slides through the band.

## 4. Method 2 — empirical sensitivity-band coverage (stepping stone, not recommended)

**Reframe (correct and worth keeping):** each temperature is a separate detection opportunity;
ask whether a trap's $\tau(T)$ trajectory passes through the band at enough temperatures to be
characterized. Procedure as first tried:

1. Per temperature, define the **detectable band** as the $\tau$ range where measured
   efficiency $\geq 0.5$ (a "BAND_EFF" threshold).
2. For a grid of $(\tau_{135}, E)$: reconstruct $\log\sigma = A(E) - \log\tau_{135}$ with
   $A(E)=\texttt{log\_energy\_cross\_section}(135,E,0)$, walk $\tau(T)$ across all temperatures,
   and **count** how many temperatures' bands it lands in.
3. **Coverage criterion:** recoverable iff in-band at $\ge n_{good}$ temperatures (see §7).
4. Recoverable fraction of the observed-$E$ family vs $\tau_{135}$ = conservative completeness.

### Why Method 2 is *better than Method 1 but still not enough*
- (+) Uses **positive evidence** (where traps were actually detected), is conservative w.r.t.
  band width, and models the multi-temperature structure explicitly.
- (+) Makes the **$E$-dependence explicit**: high-$E$ (steep $d\tau/dT$) traps sweep through the
  band fast, hit fewer in-band temperatures, and are harder to characterize *at the same*
  $\tau_{135}$.
- (−) **It does not actually escape the survivorship bias — it relocates it to the band edge.**
  The band is `{τ : eff(τ,T) ≥ 0.5}`, and `eff` uses the biased denominator. The band *interior*
  (cold $T$, good stats) is genuine positive evidence, but the band *extent* — especially the
  **warm-$T$ upper edge that controls the long-$\tau_{135}$ answer** — is set by where eff crosses
  0.5 and is data-starved. So Method 2 is most uncertain in exactly the regime that matters.
- (−) Using `eff ≥ 0.5` re-imports the denominator we claimed to avoid; a pure positive-evidence
  band would be "$\tau$ range where $\ge k$ traps were *detected* at $T$" (numerator only).
- (−) **Threshold-then-count discards the probabilistic structure.** A trap at the eff=0.5 edge
  and one at eff=1.0 center count identically. The rigorous object is
  $P(\text{characterized}) = P(\ge n_{good}\text{ detections})$, a combinatorial sum over
  temperature subsets of per-$T$ detection *probabilities*.

## 5. Method 3 — analytic per-temperature detection probability (recommended)

The fix to all of Method 2's "(−)" points is to **stop reading the band off the characterized
sample and compute the per-temperature detection probability from the physics** (§2.2) plus the
*measured noise* and the *known $t_{ph}$ grid*. Method 2's empirical band is then just a coarse,
data-starved proxy for this $p_{\text{det}}(\tau, T)$; Method 3 replaces the proxy with the model
and confines all residual assumptions to a single, calibratable amplitude term.

### 5.1 Per-temperature detection probability (injection–recovery)
This is a **selection-function / injection–recovery** computation, not a yes/no on real data. For
traps we *did* characterize there is no probability to compute — we have them. We ask the
**counterfactual**: *if a trap of brightness $A$ and emission time $\tau$ existed at a given
location, what fraction of the time would our pipeline have flagged it "good" at temperature $T$?*
Because the measurement is noisy, that fraction is $p_{\text{det}}$ — and that is exactly the
detection efficiency needed to bound a hidden population. We evaluate it over a **grid of fake
$(\tau, A)$ spanning the whole probed space**, not over the detected traps.

Procedure (reuses `fitTrapIntensity`'s cut logic verbatim):
1. Build the true curve $I(t_{ph};\tau,A)$ on the **actual $t_{ph}$ grid at $T$** (`seconds`; min
   ~fixed, max grows with $T$ — the engineered long-$\tau$ reach).
2. Draw a synthetic measurement $\hat I_i = I(t_{ph,i}) + \varepsilon_i$ with $\varepsilon_i$ at the
   **per-point intensity noise** (scaled from the local $\sigma$; in data this is `intensity_err`,
   not `image_sigma` directly — confirm the scaling). Noise is ≈Gaussian (dominated by shot noise
   on the injected charge, §5.3).
3. Fit $\hat I$ with `intensity_function` and apply the good-trap cuts (§2.3): peak $>3\sigma$,
   peak $>3\times$ intensity error, fitted-$\tau$ rel. error $<0.5$, $\chi^2\,p>0.05$.
4. Repeat $N$ times; $p_{\text{det}}(\tau, A, T\mid\sigma) = $ fraction recovered.

**Where the probability comes from:** entirely the noise the measurement would have had — the cuts
are deterministic but the data they act on fluctuates. Bright traps pass in ~every realization
($p\to1$); faint ones almost never ($p\to0$); a near-threshold trap (true peak $\approx 3\sigma$)
passes ~half the time. The shape cuts ($\chi^2$, $\tau$-error) suppress pure-noise spikes, so
$p_{\text{det}}\to0$ cleanly rather than floating at a false-positive rate. For the dominant peak
cut alone the edge is a soft error function:
$$ p_{\text{det}} \approx \tfrac12\Big[1+\mathrm{erf}\big((I^{\text{true}}_{\text{peak}}-3\sigma)/\sqrt2\,\sigma\big)\Big], $$
with $I^{\text{true}}_{\text{peak}}=0.65A$ if the peak ($t_{ph}^\star\approx0.30\tau$) is reachable,
else the rising-edge value $\approx 7A\,t_{ph}^{\max}(T)/\tau$.

**Noise is not a scalar — it is a distribution in two senses** (§5.3): it grows with $T$ (dark
current) and **varies spatially across the CCD** at fixed $T$ (columns, regions). A fake trap in a
quiet region has higher $p_{\text{det}}$ than the same trap in a noisy one, so we must marginalize
over the local-$\sigma$ distribution:
$$ p_{\text{det}}(\tau, A, T) = \int p_{\text{det}}(\tau, A, T\mid\sigma)\, p_\sigma(\sigma\mid T)\, d\sigma. $$
This is bias-free in the $\tau$-shape: it uses only the closed-form curve, the logged delays, and a
**trap-independent** noise measurement.

### 5.2 Multi-temperature characterization probability
Walk the trajectory $\tau(T)$ for a grid point $(\tau_{135}, E)$ (the $\log\sigma$ inversion in
the appendix), get $p_{\text{det}}(\tau(T_i), T_i)$ at each measured $T_i$, and compute
$$ P(\text{characterized} \mid \tau_{135}, E, A) = P\!\left(\textstyle\sum_i \text{Bernoulli}(p_{\text{det},i}) \ge n_{good}\right), $$
the Poisson-binomial tail. This is the principled replacement for "count band-hits $\ge n_{good}$"
and naturally weights near-edge temperatures less. Then marginalize over the amplitude
distribution (§5.3):
$$ P(\text{characterized} \mid \tau_{135}, E) = \int P(\text{characterized}\mid\tau_{135},E,A)\, p_A(A\mid T)\, dA. $$

### 5.3 Calibrating the two inputs: the noise map $p_\sigma(\sigma\mid T)$ and the amplitude prior $p_A(A\mid T)$

**$N_{\text{pumps}} = 3000$** (known; hard-coded in `intensity_function`, `dipole.py:409`).

**What $\sigma$ physically is.** It is the fluctuation of (pixel charge $-$ local background), *not*
the charge level. The flat-field **level** (~$2000\,e^-$ of injected spurious charge) is subtracted
(per-row median in `findDipoles2`, `dipole.py:101`), but its **shot noise** survives subtraction:
$\sqrt{2000}\approx 45\,e^-$, which likely *dominates* at low $T$ and is roughly $T$-independent.
Adding in quadrature:
- **dark-current shot noise** — grows with $T$; this is the *real* physical warm-$T$ degradation;
- **readout noise** — only **one skipper sample** was taken, so it is *not* beaten down by
  multi-sampling; a few-$e^-$ floor;
- **residual fixed-pattern** non-uniformity within the region.

(Verify these magnitudes against data — they set whether a dipole's accumulated intensity over 3000
pumps clears $3\sigma\approx 135\,e^-$.)

**$\sigma$ is a spatial map, and must be sampled trap-free.** Build $p_\sigma(\sigma\mid T)$ from
the $35\times35$ statistic on a **grid of trap-free patches across the CCD** at each $T$. Do **not**
use the `image_sigma` of *detected* traps as the injection input: that sample is selection-biased
*low* (traps in noisy regions were preferentially missed), so it would *overstate* sensitivity.
Detected-trap `image_sigma` is fine only as a **cross-check**. Because $\sigma$ is a background
statistic, it is measurable with **zero reference to the trap population** — this is the crux of
how Method 3 escapes the survivorship bias. **It requires the raw FITS images, not just the `.h5`**
(see §10).

**Amplitude $A = 3000\cdot\texttt{fit\_coeff}$** is already fit per trap per $T$
(`intensity_function` sets $d_t=p_c=1$, so `fit_coeff` $\equiv D_t P_c$). Build $p_A(A\mid T)$ from
the bright, well-behaved traps (near-peak detection is high-efficiency → least biased); factor the
common per-$T$ scaling $P_c(T)$ from the $A$-vs-$T$ trend, leaving the per-trap $D_t$ spread.

- **Load-bearing assumption: $D_t$ (hence $A$) is independent of $E/\tau$.** Justification: in
  Eq. §2.2, $\tau$ enters *only* the shape factor and $A$ *only* depth/capture — a long-$\tau$ trap
  is hard to detect because its peak is unreachable, **not** because it is faint. So the amplitude
  distribution measured from detected traps should carry over to the hidden long-$\tau$ population.
  **Test it** before relying on it: (i) how *broad* is `fit_coeff`? If it clusters within a factor
  of a few, $A$ is effectively a known function of $T$ and the whole amplitude worry is minor; if it
  spans decades, $p_A$ is the dominant systematic. (ii) Does `fit_coeff` correlate with fitted
  $\tau$ or $E$? A strong correlation breaks the assumption.
- **Caveat — observed $A$ is truncated at the faint end.** We only have $A$ for traps that passed
  the cuts, so using the observed $p_A$ is mildly *optimistic*. Sensitivity-scan toward fainter
  $p_A$ (§11.6). A hidden population with anomalously tiny $D_t$ is **unbounded** — a faintness floor
  analogous to the all-$T$-out-of-band floor (§9). State completeness *conditional* on $p_A$.
- **Open physics question:** is $P_c(T)$ flat over 125–210 K or falling at high $T$? If falling,
  that is a *real* warm-$T$ narrowing, which Method 3 will (correctly) report as genuine
  incompleteness rather than a correctable inefficiency.

### 5.4 Why this is the right tool
- Fills the warm-$T$ / long-$\tau$ band edge with **physics + measured noise** instead of leaving
  it undefined where there is no data — the one regime that controls the answer.
- **The bias-escape hinges on the noise being measurable without the trap population.** $\sigma$
  from trap-free image patches sets the detection threshold with zero reference to which traps were
  found (contrast Method 2, whose band edge is read from the characterized sample). The *only* place
  the trap population re-enters is the amplitude prior $p_A$ — deliberately calibrated from the
  least-biased (brightest) traps, and isolated as a single, testable assumption.
- Confines all survivorship risk to the amplitude prior $p_A(A\mid T)$, which is small, explicit,
  and calibrated from the brightest (least-biased) traps.
- Respects the fundamental floor (§9): a trap whose peak-reachable intensity is below threshold at
  *every* $T$ is genuinely invisible — and now we can **state $\tau_{\max}(T)$ quantitatively** and
  say exactly where the unbounded regime begins.

## 6. Dead end: the DM-impact simulation is NOT a detection model

For the record, so we don't repeat this: the simulation in `ccd_simulation.py` / paper §V models
the **science readout** under DM-search conditions (inject single-$e$ + high-energy events, clock
row-by-row, capture/release at traps with $P=1-e^{-dt/\tau}$, count integer electrons to get a
$1e^-$ rate). It has **no notion of "detected"**: no $t_{ph}$ sweep, no dipole-intensity equation,
no $35\times35$ noise, no good-trap cuts, and it uses noiseless integer electron counts. It cannot
produce a detection efficiency and must not be repurposed as a forward detection model. The
detection model is the analytic Eq. §2.2 + measured noise (Method 3), not this simulation.

## 7. The $\ge n_{good}$ correction (applies to Methods 2 and 3 alike)

The first band-coverage pass used a **$\ge 1$ temperature** criterion ("did $\tau(T)$ ever enter
the band"). That is too weak — one in-band point cannot support an Arrhenius fit, which needs
$\ge n_{good}$ ($=4$ in the paper). Preliminary $\ge 1$ numbers (now superseded):

| $\tau_{135}$ | covered fraction (≥1 criterion) |
|---|---|
| 1–100 s | 1.00 |
| $10^3$ s | 0.98 |
| $10^4$ s | 0.62 |
| $10^5$ s | 0.31 |

Fully covered (≥1) was $\tau_{135}\in[0.4, 800]$ s, containing 96.6% of observed traps. These are
**optimistic**; the correct $\ge n_{good}$ criterion shrinks the covered region, most sharply at
high $E$ and at the long-$\tau$ edge. In Method 3 this becomes the Poisson-binomial $\ge n_{good}$
tail (§5.2) rather than a hard count.

## 8. What can actually be claimed: fraction recoverable vs population bound

The methods above produce a **recoverable fraction of the observed-$E$ family** as a function of
$\tau_{135}$. The §1 question asks for a **population bound**. These are not the same:

- To convert recoverable-fraction $f(\tau_{135})$ into a population bound
  ($N_{\text{true}} \lesssim N_{\text{observed}}/f$), you must assume the hidden population shares
  the **observed $E/\sigma$ distribution**.
- A population could hide *precisely because it sits at an $E/\sigma$ the protocol rarely
  characterizes* — i.e. outside the observed $E$ range *because* it is hard to see. That
  population is **not** bounded by any of this.
- So the honest claim is conditional: *"assuming hidden traps share the observed
  $E$-distribution, the set is $\ge X\%$ complete for $\tau_{135}\in[a,b]$,"* plus a separate,
  explicit statement of the regime ($\tau$ above $\tau_{\max}(T)$ for all $T$, and unobserved
  $E/\sigma$) that is genuinely unbounded.

## 9. Caveats / hard limits

- **Amplitude prior is the residual assumption.** Method 3's answer in the long-$\tau$ regime
  depends on $p_A(A\mid T)$ and on $P_c(T)$; show sensitivity to both.
- **Capture/occupancy.** Detection needs the trap *filled* then released in transit; $P_c$
  captures the temperature scaling but check whether occupancy (electron-density dependence) is a
  separate effect at our charge level (~$2000\,e^-$/pix should keep traps well-filled, but
  confirm).
- **Fundamental floor.** A trap outside the band at *every* temperature is invisible and
  **unbounded by any of these methods**. Closing that gap needs an independent handle (a colder
  measurement, longer $t_{ph}$, or a dark-current/leakage argument).
- **$E/\sigma$ family at fixed $\tau_{135}$** is restricted to the observed $E$ range; values
  outside it are speculative (see §8).

## 10. Open questions / feasibility checks

**Already in the `.h5`** (`save_spectra_hdf5` recurses the full nested dict, so per-temperature
sub-dicts persist):
- ~~**$t_{ph}$ grid**~~ — `seconds` per dipole per $T$ (`dipole.py:398`); `intensities` /
  `intensity_err` alongside. ✓
- ~~**per-trap amplitude**~~ — `fit_coeff` ($A=3000\cdot$`fit_coeff`) per dipole per $T$. ✓
- **detected-trap local noise** — `image_sigma` per dipole per $T$ (`dipole.py:324`), **but only at
  trap sites** → use as a cross-check, *not* the injection input.

**Needs the raw data, not just the `.h5`:**

1. **Unbiased noise map $p_\sigma(\sigma\mid T)$** — must be built from **trap-free $35\times35$
   patches across the CCD** at each $T$ (detected-trap `image_sigma` is selection-biased low, §5.3).
   ⇒ **Move the `proc/` FITS images to the analysis machine, not only the fit `.h5` files.** While
   there, confirm the per-point intensity-noise scaling (`intensity_err` vs `image_sigma`) used in
   the injection (§5.1 step 2).

**Judgement calls / physics:**

2. **Amplitude prior $p_A$** — comfortable calibrating $A=3000\cdot$`fit_coeff` and $P_c(T)$ from
   bright traps? First check the *breadth* of `fit_coeff` and its $\tau$/$E$ correlation (§5.3) —
   that decides whether $p_A$ is a minor or dominant systematic.
3. **$P_c(T)$ behavior** — flat vs falling over 125–210 K (from the $A$-vs-$T$ trend of bright
   traps)? A fall is real warm-$T$ incompleteness, not a correctable inefficiency.
4. **$D_t \perp E/\tau$ assumption** — the one that lets the observed amplitude prior carry to
   hidden long-$\tau$ traps; test via the `fit_coeff`–$\tau$/$E$ correlation (§5.3).

## 11. Plan / TODO

0. **Pre-flight checks (cheap, do first):** breadth of `fit_coeff` and its $\tau$/$E$ correlation
   (decides if $p_A$ is minor or dominant, §5.3/§10.2,4); rough magnitude of $\sigma(T)$ vs the
   $\sqrt{2000}\approx45\,e^-$ shot-noise expectation; `intensity_err`-vs-`image_sigma` scaling.
1. **Build $p_{\text{det}}(\tau, A, T)$ by injection–recovery** (Method 3 §5.1): synthetic curves
   from Eq. §2.2 on the logged $t_{ph}$ grid, noise drawn at the per-point level, run through the
   real good-trap cuts; $N$ realizations → recovery fraction. Marginalize over the **noise map**
   $p_\sigma(\sigma\mid T)$ and the amplitude prior $p_A(A\mid T)$.
2. **Calibrate inputs** (§5.3): noise map $p_\sigma(\sigma\mid T)$ from **trap-free** image patches
   (needs FITS, §10.1); $p_A(A\mid T)$ and $P_c(T)$ from bright traps.
3. **Validate** against the empirical band (Method 2) and against where characterized traps
   actually sit — demonstrated detections should fall where $p_{\text{det}}$ is high. Use Method 2
   only as a cross-check, not the primary instrument.
4. **Recompute coverage** as the Poisson-binomial $\ge n_{good}$ tail (§5.2) over a
   $(\tau_{135}, E)$ grid; report the corrected covered window and the fraction of observed traps
   inside it.
5. **Maps:** $(\tau_{135}, E)$ recoverability map + coverage-vs-$\tau_{135}$ curve, observed
   distribution overlaid; band-hit / $p_{\text{det}}$-weighted count vs $E$ at fixed $\tau_{135}$
   to expose the $E$-disadvantage.
6. **Sensitivity:** vary the cuts ($3\sigma$, $p$-value), $p_A$, $P_c(T)$, and $n_{good}$ (3 vs 4);
   show how the covered window moves.
7. **State completeness quantitatively and conditionally** (§8): "$\ge X\%$ complete for
   $\tau_{135}\in[a,b]$, assuming hidden traps share the observed $E$-distribution," and delimit
   the genuinely-unbounded regime ($\tau > \tau_{\max}(T)\,\forall T$, and unobserved $E/\sigma$).

---

## Appendix — how to read & handle the data

### Environment

The project `.venv/` is empty; the interpreter that has the deps (numpy, scipy, h5py, astropy,
iminuit) is **pyenv 3.12.2**:

```bash
~/.pyenv/versions/3.12.2/bin/python <script>.py
```

(`.python-version` pins 3.12.2, but `python` on PATH resolves to the empty venv — call the pyenv
binary explicitly, or activate the right env first.)

### Input files

Per-trap fits live in HDF5 files named `fit_dipole_spectra_err_<thr>.h5`, where `_err` =
`useIntensityErr=True` and `<thr>` = `wellBehavedThreshold` (the minimum number of good temperature
fits required, i.e. $n_{good}$). We mostly use:

- `fit_dipole_spectra_err_3.h5` → `records_3` ($n_{good}\geq 3$, 2517 characterized traps)
- `fit_dipole_spectra_err_4.h5` → `records_4` ($n_{good}\geq 4$, the paper's 2121 well-behaved set)

These are large (~0.5 GB) and untracked; do not commit them. They are produced by the analysis
pipeline (`run_charge_traps.py`) and saved via `save_spectra_hdf5` in `utils.py`.

### Where the analysis lives (code map)

The whole identification → characterization chain is in `dipole.py`, driven by
`run_charge_traps.py` (which applies the cached-stage pattern — see `CLAUDE.md`). The stages:

| stage | function (`dipole.py`) | output |
|---|---|---|
| find dipoles in one image | `findDipoles2` (`:55`) | list of `(row,col)` dipole coords |
| per-quadrant dipole list across temps | `getDipoleList2` (`:183`) | `dipole_coord_list.npz` |
| build intensity-vs-$t_{ph}$ spectra | `getDipoleSpectra2` (`:261`) | `dipole_spectra.h5` (sets `seconds`, `intensities`, `image_sigma`) |
| per-$T$ $\tau$ fit + Arrhenius $E/\sigma$ fit | `fitTrapIntensity` (`:449`) | `fit_dipole_spectra_err_<thr>.h5` (adds `fit_*`, `energy_*`, `GoodIntensityFit`, `WellBehavedTrap`) |
| intensity model $I(t_{ph})$ | `intensity_function` (`:409`) | $N_{\text{pumps}}\cdot$`coeff`$\cdot(e^{-t/\tau}-e^{-8t/\tau})$ |
| thermal-emission $\log\tau(T)$ | `log_energy_cross_section` (`:416`) | used for $\tau(T)$ + the $\log\sigma$ inversion |

`fitTrapIntensity_cutflow` (`:639`) mirrors `fitTrapIntensity` while tallying which cut each trap
fails — useful for understanding *why* traps drop out (relevant to the completeness question).

### Loading and structure

```python
import numpy as np
from utils import load_spectra_hdf5          # utils.py
from dipole import log_energy_cross_section   # dipole.py — thermal-emission tau(T) model

fit_spectra = load_spectra_hdf5('fit_dipole_spectra_err_3.h5')
```

The result is a nested dict: `fit_spectra[q]` for quadrants `q in {0,1,2,3}`. Within a quadrant the
keys are **either** a dipole coordinate `(row, col)` **tuple** (a real trap) **or** metadata
(non-tuple) — so always filter on `type(dp) == tuple`. Each trap dict (`testdp`) carries flat keys:

| key | meaning |
|---|---|
| `WellBehavedTrap` (bool) | passed the per-temperature τ-fit quality cut |
| `EnergyFitFailed` (bool) | the Arrhenius $E/\sigma$ fit failed |
| `GoodEnergyFit` (bool) | the $E/\sigma$ fit passed the goodness cut |
| `energy_BestFitEnergy` (float) | fitted $E$ [eV] |
| `energy_BestFitCrossSection` (float) | fitted $\sigma$ (linear; take `np.log` for $\log\sigma$) |
| `energy_temperatures` (array) | the temperatures with a good τ measurement — `len(...)` is $n_{good}$ |
| `energy_taus`, `energy_tau_errs` | the per-temperature τ points used in the fit |

For Method 3, everything is in the same file: each trap dict *also* has **integer-temperature
sub-keys** (`d[T]` for `T` in the measured temperatures), each a dict carrying the per-temperature
spectrum and fit. `save_spectra_hdf5` (`utils.py:417`) recurses the whole nested structure, so
`load_spectra_hdf5` returns it intact. Per-temperature keys we need:

| key (`d[T][...]`) | meaning | set in |
|---|---|---|
| `seconds` | the $t_{ph}$ grid sampled at $T$ | `dipole.py:398` |
| `intensities`, `intensity_err` | dipole intensity + error vs `seconds` | `getDipoleSpectra2` |
| `image_sigma` | $35\times35$-pixel noise $\sigma(T)$ (the $3\sigma$ cut) | `dipole.py:324` |
| `fit_coeff`, `fit_coeff_err` | fitted amplitude `coeff`; $A = 3000\cdot$`fit_coeff` | `dipole.py:554` |
| `fit_tau`, `fit_tau_err` | per-$T$ fitted $\tau$ (bounded $\le 1000$ s) | `dipole.py:555` |
| `GoodIntensityFit` (bool) | passed the good-trap cuts at this $T$ | `dipole.py:530` |
| `fit_p_value`, `fit_reduced_chi_squared` | fit goodness | `dipole.py:546` |

> Note: `run_charge_traps.py` reads a *nested* `EnergyFitInfo` dict (`BestFitEnergy`,
> `BestFitCrossSection`); the HDF5 loaded here uses the **flat** `energy_*` keys above. Use the flat
> keys with `load_spectra_hdf5`.

### Characterized-trap filter and record build

```python
def is_characterized(d):
    return (d.get('WellBehavedTrap', False)
            and not d.get('EnergyFitFailed', True)
            and d.get('GoodEnergyFit', False))

TARGET_TEMP = 135.0
records = []
for q in [0, 1, 2, 3]:
    for dp, d in fit_spectra[q].items():
        if type(dp) != tuple or not is_characterized(d):
            continue
        E    = d['energy_BestFitEnergy']
        logS = np.log(d['energy_BestFitCrossSection'])
        records.append({
            'E': E, 'logS': logS,
            'mt':   np.asarray(d['energy_temperatures'], float),  # measured temps; len = n_good
            't135': np.exp(log_energy_cross_section(TARGET_TEMP, E, logS)),
        })
```

### The τ(T) model and the (E, τ_target) ⇄ logσ inversion

`log_energy_cross_section(T, E, logσ)` returns $\log\tau$ and is **linear in $\log\sigma$ with unit
slope**: $\log\tau(T) = A(E,T) - \log\sigma$, where $A(E,T) =$ `log_energy_cross_section(T, E, 0.0)`.
(This is just Eq. `tau(T)`: $\tau = (\sigma v_{th} N_c)^{-1} e^{E/k_B T}$, so $\ln\tau$ shifts by
$-\Delta\ln\sigma$.) So:

```python
tau_T = np.exp(log_energy_cross_section(T, E, logS))          # tau at temperature T (T may be an array)
# fix a target tau at T0 and solve for the cross-section it implies:
logS  = log_energy_cross_section(T0, E, 0.0) - np.log(tau_target)
```

This inversion is what lets us sweep a $(\tau_{135}, E)$ grid and recover the implied $\log\sigma$
for the coverage map.

### Intensity-model quick reference (Method 3)

```python
# I(t_ph) = A * (exp(-t_ph/tau) - exp(-8*t_ph/tau)),  A = N_pumps * D_t * P_c
# peak delay:      t_ph_star = (tau/7)*ln(8) ~= 0.297*tau
# peak intensity:  I_peak ~= 0.65*A  (independent of tau)
# long-tau (t_ph << tau): I ~= 7*A*t_ph/tau   ->  faintness is a t_ph-reach limit, not physics
def intensity(t_ph, tau, A):
    return A * (np.exp(-t_ph / tau) - np.exp(-8.0 * t_ph / tau))
```

### Measurement temperatures

```python
measurement_temperatures = np.array([
    125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175,
    180, 183, 185, 187, 190, 193, 195, 197, 200, 203, 207, 210])
```

$t_{ph}$ was swept $\sim 5\times10^{-5}$–$1.03$ s, with **longer $t_{ph}$ at higher $T$** — recover
the exact per-temperature grid from the run log / FITS headers (open question §10.2).

### Per-temperature efficiency (Method 1 input; band proxy for Method 2)

For a temperature `T` and bin edges `edges`, split characterized traps (with `len(mt) >= ngood`)
into **measured** (`T` in `mt`) vs **extrapolated** (not), histogram each, and take
`measured / (measured + extrapolated)` per bin (with Wilson errors). The Method-2 **detectable
band** is the τ range where that efficiency $\geq$ `BAND_EFF` (0.5). Bin choice used elsewhere:
`np.geomspace(5e-2, 1e2, 25)`, ignoring bins with center $\leq 0.1$ s. In Method 3 this band is
only a validation cross-check (§11.3); the operative quantity is the analytic $p_{\text{det}}$.
