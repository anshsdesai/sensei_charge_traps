# Agent 05: Synthesis Report

## Goal
Combine the outputs of Agents 01 through 04 into the final recommendation on whether the trap-finding and fitting criteria are sound, whether recall is too low, and which changes are true fixes versus optional analysis choices.

## Inputs
- `plans/trap_review_agents/01_detection_audit.md`
- `plans/trap_review_agents/02_intensity_fit_audit.md`
- `plans/trap_review_agents/03_energy_fit_audit.md`
- `plans/trap_review_agents/04_user_rerun_protocol.md`
- user-pasted rerun summaries
- any updated `.h5` or CSV outputs from the user

## Questions To Answer
- Are the current fit criteria defensible as a baseline analysis?
- Is the current pipeline clearly missing a substantial number of plausible traps?
- Which findings are confirmed bugs or inconsistencies?
- Which findings are conservative policy choices rather than mistakes?
- What should be the recommended default pipeline after the review?
- What optional higher-recall variant is worth testing further?

## Output Structure

### 1. Ranked Findings
Rank findings from highest confidence and highest impact to lowest.

Use these categories:
- Confirmed bug
- Code/manuscript mismatch
- Conservative but defensible selection choice
- Recall-enhancing option needing validation

### 2. Recommended Default Pipeline
State the recommended default clearly.

This section must answer:
- whether to keep or revise the detection-stage persistence rule
- whether to keep or revise the p-value requirement
- whether to keep or revise the `tau_err / tau` requirement
- whether to adopt per-image electronization calibration by default
- whether to update the energy-fit prefactor implementation

### 3. Optional Higher-Recall Variant
Provide one alternative pipeline that prioritizes recall more than the default.

This variant should be concrete and limited.
Examples:
- preserve baseline detection but downgrade p-value to a review flag
- preserve baseline fitting but allow cross-temperature rescue

Do not recommend multiple optional variants unless the rerun evidence clearly requires it.

### 4. Short Follow-Up List
List only the smallest necessary next actions if ambiguity remains.

Examples:
- rerun one more p-value variant
- inspect 10 additional borderline traps
- confirm cross-temperature rescue on raw FITS before adopting it

## Required Decision Boundaries
- Treat `>= 4` good temperature fits as the main characterization threshold unless the user explicitly changes that definition later.
- Report `GoodEnergyFit` separately rather than using it as a synonym for well-behaved unless rerun evidence shows that was the actual intended definition.
- Do not call a criterion "wrong" unless the evidence supports that claim strongly.
- Do call out manuscript drift explicitly if counts or formulas no longer match the implementation.

## Current Baseline Facts To Carry Forward
- Current saved identified-trap count: `5171`
- Current saved traps with any good temperature fit: `3365`
- Current saved traps with `>= 4` good temperature fits: `2514`
- Current saved traps with `GoodEnergyFit`: `2135`
- Current saved fit-stage review indicates the p-value cut is the dominant recall limiter.
- Current saved review indicates the thermal prefactor in `log_energy_cross_section` likely does not match the manuscript formula and may shift fitted cross-sections by about a factor of `1.71`.

## Final Deliverable Requirements
The final synthesis must include:
- one ranked findings list
- one recommended default pipeline
- one optional higher-recall variant
- one short rerun follow-up list if still needed

The final synthesis should be written so it can directly guide a later implementation pass in `dipole.py` and the manuscript if the user chooses to make changes.
