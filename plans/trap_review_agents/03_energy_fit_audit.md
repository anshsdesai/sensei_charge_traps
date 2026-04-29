# Agent 03: Energy Fit Audit

## Goal
Audit `log_energy_cross_section` and the final `tau(T)` fit stage, reconcile the code with the manuscript formula, and separate the count definitions so later synthesis is not confused by mismatched labels.

## Primary Questions
- Does the implementation of the thermal prefactor match the formula written in the manuscript?
- Does the current implementation mainly shift fitted cross-sections, or does it materially change the inferred trap energies as well?
- Are the manuscript counts using the same definition as the cached outputs?
- Should the paper text be updated even if the code is left unchanged for now?

## Files To Read
- `dipole.py`
- `fit_dipole_spectra_err_4.h5`
- `paper/paper.tex`

## Current Repo Facts
- Current cached stage counts:

| Stage | Count |
| --- | ---: |
| Identified traps | 5171 |
| Traps with any good temperature fit | 3365 |
| Traps with `>= 4` good temperature fits | 2514 |
| Traps with `GoodEnergyFit` | 2135 |

- Manuscript statements currently in tension with the cache:
  - The paper states `3379` traps are identified as "good".
  - The paper states `2121` traps are "well-behaved".
  - The current cache instead shows `3365` traps with any good temperature fit, `2514` traps with `>= 4` good temperature fits, and `2135` traps with `GoodEnergyFit`.

## Code/Formula Consistency To Audit
1. Review `log_energy_cross_section` in `dipole.py`.
   - Current code uses a compact prefactor based on a single `m_e`-scaled term.
2. Review the manuscript formula in `paper/paper.tex`.
   - The paper states that:
     - `v_th = sqrt(3 k_B T / m_cond)`
     - `N_c = 2 [2 pi m_dens k_B T / h^2]^(3/2)`
     - with `m_cond ~= 0.41 m_e`
     - and `m_dens ~= 0.94 m_e`
3. Quantify the difference.
   - Based on the current audit, the prefactor difference corresponds to an approximate cross-section shift of:
     - multiplicative factor: `~1.71`
     - `Delta log sigma ~= 0.54`
   - Treat the first-order expectation as:
     - cross-sections shift appreciably
     - energies likely change much less, but this should be checked on rerun if code is updated

## Required Output 1: Code-vs-Paper Consistency Note
Produce a short memo with:
- exact statement of the current code implementation
- exact statement of the manuscript formula
- whether the discrepancy is a true bug, a convention mismatch, or an unresolved ambiguity
- whether the current manuscript wording should be revised if the code remains unchanged

## Required Output 2: Count Reconciliation Note
Separate the following definitions explicitly:

| Definition | Meaning |
| --- | --- |
| Any good temperature fit | Trap has at least one temperature with `GoodIntensityFit=True` |
| Well-behaved by threshold | Trap has at least four good temperature fits |
| Final energy-fit pass | Trap additionally has `GoodEnergyFit=True` |

Then reconcile these with the manuscript text:
- `3379` "good" traps in the paper
- `2121` "well-behaved" traps in the paper
- `3365`, `2514`, and `2135` in the current cache

The likely explanation to test is that the manuscript may be mixing threshold-based well-behaved traps with final energy-fit-passing traps.

## Required Output 3: Recommendation On Paper Updates
End with one of these recommendations:
- Code and paper should both be updated
- Code should be updated, paper text can remain after rerun
- Paper text should be clarified even if code stays unchanged for now

This recommendation must mention:
- the count-label mismatch
- the prefactor mismatch

## Exact User Rerun Request
If the user can rerun a modified energy-fit stage locally, ask for:
1. Baseline final energy-fit counts
2. If practical, a rerun using the manuscript-consistent prefactor

Ask the user to paste back:
- number of traps with `>= 4` good temperatures
- number of traps with `GoodEnergyFit`
- summary statistics comparing fitted energies and cross-sections between baseline and prefactor-corrected runs
- if possible, a small CSV of `(quad, coord, energy_old, energy_new, sigma_old, sigma_new)`

## Handoff Notes For Later Agents
- Agent 05 should treat count labeling as a communication problem even if no code bug is ultimately confirmed.
- If the prefactor mismatch is verified on rerun, later implementation work should update both the code and the manuscript wording in the same change set.
