# Agent 04: User Rerun Protocol

## Goal
Define the exact reruns the user should perform on the raw FITS outside this workspace, and specify the exact summaries and result artifacts they should paste back here for the next review cycle.

## Constraints
- Raw FITS are not available in this workspace.
- This protocol must work even if the user has to run parts of the analysis manually or from notebooks.
- The output format must be simple enough to paste back into chat if full files are inconvenient to share.

## Rerun Variants To Request

### Variant A: Baseline Current Pipeline
Purpose:
- Reproduce the current analysis as closely as possible from raw data.

What to run:
- The standard detection, spectra, intensity-fit, and energy-fit workflow with the current code and current thresholds.

Deliverables:
- total identified traps
- traps with any good temperature fit
- traps with `>= 4` good temperature fits
- traps with `GoodEnergyFit`
- per-quadrant identified trap counts

### Variant B: Baseline With Per-Image Electronization Calibration
Purpose:
- Check whether using per-image calibration instead of fixed `eval = 400` changes counts materially.

What to change:
- Replace fixed electronization where practical with the per-image calibration already available in the user's environment.
- Do not otherwise loosen selection unless necessary to isolate this effect.

Deliverables:
- same count block as Variant A
- one short note describing how the per-image calibration was obtained
- one short comparison note versus Variant A

### Variant C: Cross-Temperature Rescue Bookkeeping
Purpose:
- Quantify candidate traps that are currently excluded because they only appear at one `dtph` per temperature but recur across temperatures.

What to change:
- Keep the baseline selection as the official count.
- Add bookkeeping that records coordinates with:
  - exactly one observed `dtph` within a temperature
  - recurrence across multiple temperatures

Deliverables:
- baseline official counts
- number of coordinates that would be eligible for cross-temperature rescue
- if implemented, number of additional traps created by allowing rescue
- 5 to 10 example coordinates with their temperature recurrence pattern

### Variant D: Fit-Quality Sensitivity
Purpose:
- Test whether current recall is being dominated by the p-value gate.

Pick one of these if feasible:
- a p-value threshold grid such as `0.001`, `0.01`, `0.05`, `0.1`
- or a mode where p-value is computed and stored but not enforced for selection

Deliverables:
- trap counts with any good temperature fit
- trap counts with `>= 4` good temperature fits
- trap counts with `GoodEnergyFit`
- compact temperature-fit cutflow

## Exact Summary Block To Paste Back
For each variant, paste back one block in this format:

```text
Variant:
Code branch or notebook:
Electronization mode:
Detection persistence rule:
Fit-quality rule:

Identified traps total:
Identified traps by quadrant:
Traps with any good temperature fit:
Traps with >= 4 good temperature fits:
Traps with GoodEnergyFit:

Temperature-fit cutflow:
- p-value failures:
- low-SNR failures by mean(intensity_err):
- low-SNR failures by image_sigma:
- tau_err / tau failures:

Notes:
```

## Example Trap Records To Request
For at least one rerun, ask the user to paste back `10` to `20` example trap records for borderline diagnosis.

Each example should include:
- quadrant
- coordinate
- temperature
- retained or rejected
- fitted `tau`
- fitted `tau_err`
- `p_value`
- `reduced_chi2`
- peak intensity
- local noise estimate
- whether it would be rescued by cross-temperature logic, if that bookkeeping exists

## Preferred File Artifacts
If the user can share files, request these in order of usefulness:
1. updated `.h5` outputs for each rerun variant
2. compact CSV summaries of trap-level and temperature-fit-level quantities
3. text or Markdown summary files if binary outputs are inconvenient

Useful compact CSV columns:
- `quad`
- `coord_row`
- `coord_col`
- `temperature`
- `good_intensity_fit`
- `fit_tau`
- `fit_tau_err`
- `fit_p_value`
- `fit_reduced_chi_squared`
- `fit_coeff`
- `fit_coeff_err`
- `well_behaved_trap`
- `good_energy_fit`

## Minimum Acceptable Return If Time Is Tight
If the user cannot generate files, ask for:
- one summary block for Variant A
- one summary block for Variant C or D
- 5 borderline examples

## Handoff Notes For Later Agents
- Agent 01 will use Variant C to judge whether cross-temperature rescue is worth implementing.
- Agent 02 will use Variant D to judge whether p-value should stay a hard cut.
- Agent 03 will use Variant A and any energy-fit rerun outputs to reconcile cache and manuscript counts.
- Agent 05 will use all returned blocks to produce the final recommendation.
