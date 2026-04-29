# Agent 01: Detection Audit

## Goal
Audit the trap detection logic in `findDipoles2` and `getDipoleList2`, document what the current code is doing, and define the exact rerun comparisons needed to test recall without raw data access in this workspace.

## Primary Questions
- Is the current dipole-finding logic aligned with the manuscript description and the intended pocket-pumping geometry?
- Are any detection-stage thresholds clearly mistaken, or are they conservative but defensible?
- How many plausible traps are being excluded by requiring the same coordinate to appear in more than one `dtph` image within a single temperature?
- How much additional recall might come from cross-temperature rescue of coordinates that appear only once per temperature but recur across temperatures?

## Files To Read
- `dipole.py`
- `utils.py`
- `dipole_coord_list.npz`
- `paper/paper.tex`

## Current Repo Facts
- Current saved detection count from `dipole_coord_list.npz`: `5171` traps total.
- Quadrant counts from the current cache:

| Quadrant | Identified traps |
| --- | ---: |
| 0 | 1324 |
| 1 | 1602 |
| 2 | 1068 |
| 3 | 1177 |
| Total | 5171 |

- The manuscript text also states `5171` identified traps, so the detection-stage cache matches the paper-level count.

## Detection Logic To Audit
1. Global histogram width estimate in `findDipoles2`.
   - Check how `sigma` is computed from the image-wide histogram.
   - Confirm whether the product cut `image[1:, :] * image[:-1, :] < -(3*sigma)^2` behaves as intended for opposite-sign vertical dipoles.
2. Row-median subtraction.
   - Verify whether subtracting `median_charge_per_row` is the intended correction for non-uniform illumination and row structure.
   - Flag whether this could suppress real structure or help reject broad gradients.
3. Charge-balance requirement.
   - Review `comparable_perc(a, b, perc=0.3)`.
   - Assess whether the 30% balance cut is likely rejecting asymmetric but still physical dipoles.
4. Temperature-local persistence gate in `getDipoleList2`.
   - Current logic keeps a coordinate only when it appears in more than one `dtph` image at the same temperature.
   - Document the consequence: a coordinate seen once per temperature across many temperatures never becomes a trap candidate at all.

## Required Output 1: Current Detection-Stage Counts
Produce a short note with:
- the current per-quadrant counts from `dipole_coord_list.npz`
- confirmation that the total matches the manuscript count of `5171`
- whether any code-path drift is evident between the saved cache and the manuscript description

## Required Output 2: Alternate Detection Logic Table
Prepare a table with these rows:

| Variant | Description | Expected effect | Needs raw rerun? |
| --- | --- | --- | --- |
| Baseline | Current `len(dtphs) > 1` within a temperature | Reference count | No, cache exists |
| Cross-temperature rescue bookkeeping | Track coordinates that appear once per temperature but recur across temperatures | Higher recall candidate | Yes |
| Looser balance cut | Test `perc > 0.3` if needed | Possible modest recall gain | Yes |
| No change to row-median subtraction | Preserve current background handling | Control | No |

Fill the first two rows at minimum. Only include the latter rows if justified by code review.

## Required Output 3: Confirmed Issues vs Judgment Calls
Split findings into two lists:

### Confirmed Issues
Use this only for things that are unambiguously wrong or inconsistent.

### Judgment Calls
Use this for choices that may be conservative or aggressive but are not clearly bugs.

## Required Output 4: Exact User Rerun Request
Since raw FITS are not available here, prepare the exact detection-stage rerun request for the user.

Request these deliverables:
1. Baseline rerun with current code and current thresholds.
2. Rerun with cross-temperature rescue bookkeeping enabled.
   - This variant should not automatically change selection unless the user chooses to; it must at least report how many coordinates would be rescued.
3. If practical, one sensitivity rerun that changes the balance cut from `0.3` to a slightly looser value.

For each rerun, ask the user to paste back:
- total identified traps
- per-quadrant trap counts
- number of coordinates seen at exactly one `dtph` per temperature but recurring across multiple temperatures
- if cross-temperature rescue is enabled, number of additional candidate traps created by it

## Suggested Command/Artifact Expectations For The User
- If the user can rerun the existing pipeline directly, ask for the saved `dipole_coord_list` equivalent and a plain-text summary block.
- If they instrument a custom script or notebook, ask them to preserve:
  - exact code branch or notebook cell changes
  - temperature list used
  - quadrant list used
  - any changes to the persistence criterion

## Handoff Notes For Later Agents
- Agent 02 will need to know whether cross-temperature rescue changes the candidate population.
- Agent 05 should not treat cross-temperature rescue as the new default unless the rerun evidence shows a meaningful gain without obvious contamination.
