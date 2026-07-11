# Lab notebook index

This is the entry point to the consolidated scientific documentation for the SENSEI charge-trap
analysis. It replaces ~30 loose `.md` working notes that had accumulated in the repo root. The
notes were classified as relevant / superseded, their content folded into four thematic notebook
files under `notebook/`, and the superseded originals moved to `attic/` (nothing was deleted;
`attic/` is the archive of record).

For code-level orientation (how to run the pipeline, cache layout, campaign axes, clear modes) see
`CLAUDE.md` — that is living configuration, not notebook material, and is unchanged.

---

## Start here (the whole project in plain words)

Silicon sensors (CCDs) used to search for dark matter have tiny crystal defects called **charge
traps**. A trap grabs a passing electron and releases it a while later — and a late-released electron
can be mistaken for the faint single-electron signal a dark-matter search is trying to see. So we need
to know: **how much do traps distort that single-electron count?**

Answering it takes three steps, and this notebook is the record of all three:

1. **Find and measure the traps.** We shine charge through the sensor, "pump" it back and forth, and
   watch which spots steal-and-release charge (each shows up as a **dipole** — a dark pixel stacked on
   a bright one). From how a trap behaves across temperatures we get its depth and how long it holds
   charge. → *the catalog.*
2. **Check we didn't miss a dangerous hidden batch.** Long-holding traps are the risky ones and also
   the hardest to catch. We estimate what fraction we would have caught, and build a deliberately
   pessimistic "what if there are many more" population. → *the completeness study.*
3. **Simulate the impact.** We run fake dark-matter-search exposures with these traps and see how much
   they shift the single-electron count, before and after the standard data cleanup ("masking"). →
   *the simulation.*

**Where it landed:** the traps we actually measured shift the single-electron count by essentially
nothing once the normal masking is applied. The only remaining caveat is an *upper limit* on a
possible hidden population of very long-holding traps — reported honestly as a bounded allowance with
an uncertainty range, not as a confirmed effect. Getting there required fixing a subtle flaw in the
measurement (the "pedestal," step 1) and correcting the simulation's model of how traps grab charge
(step 3).

## How to read this notebook

- **Read this INDEX first**, then open whichever of the four thread files you need. Each file opens
  with an **"In plain terms"** box summarizing what it covers and how it ended, so you can re-orient in
  30 seconds after a long time away.
- **Hit an unfamiliar word, symbol, or piece of shorthand?** Everything is defined in plain words in
  the **[glossary](notebook/glossary.md)** — including project coinages like *the pedestal*, *V_p*,
  *caldet*, *WS1/WS2/WS3*, *POP1/POP2*, and which of several models is the current one.
- **Cross-references** between files are written as `[[filename]]` (e.g. `[[physics]]` means
  `notebook/physics.md`).
- **When a file and an archived original disagree, the file wins** — it folds in later corrections.
- **Figures** are committed SVGs under [notebook/figures/](notebook/figures/), embedded inline in the
  thread files. They are regenerated from the real analysis caches by
  [notebook/figures/make_figures.py](notebook/figures/make_figures.py) — see that folder's README to
  add or refresh one.

---

## The story in one page (thread map)

The project has two coupled deliverables: a **measured trap catalog** (energies, cross-sections,
lifetimes from pocket-pumping across 125–210 K) and a **CCD simulation** that projects how those
traps perturb the SENSEI single-electron dark-matter background. Four investigation threads
dominate the notebook, and they connect end to end:

1. **[The dipole algorithm](notebook/dipole_algorithm.md)** — how traps are found, fit, and graded.
   The legacy analysis failed above ~165 K. A pedestal-aware, signed, physically-calibrated
   rewrite (`dipole_new.py`, the "minimal" pipeline) fixes it. *This produces the catalog.*

2. **[The signed-refit investigation](notebook/signed_refit.md)** — the multi-month exploration
   behind that rewrite. An ambitious "full refit" (regional covariance, fitted overdispersion,
   Wilks thresholds, acquisition-family exclusion) was built, then found under adversarial review to
   claim more precision than the data support. The chosen path is a **minimal synthesis**: keep the
   physically required corrections, drop the overbuilt machinery. *This is "how the correct path was
   chosen."*

3. **[Completeness & efficiency](notebook/completeness_efficiency.md)** — is the catalog missing a
   dangerous hidden population of long-lived traps? An analytic injection–recovery detection model
   (Method 3) escapes the survivorship bias of the paper's original efficiency method and yields a
   conditional completeness statement. The pedestal fix from thread 1 extends the ≥95%-complete
   reach by ~3–4 decades in τ135. *This bounds the catalog and builds the upper-limit population.*

4. **[Simulation physics](notebook/physics.md)** — the SRH capture/emission/recapture model in the
   readout simulation. It went through three versions; the recapture rate was over-strong until a
   pumping-consistency argument forced a switch to `phase_limited_v1v3` (capture in a 20 µs transfer
   window). A three-round instrumented deviation budget then showed the masked trap excess is a tiny
   (~0.3%) residual of large opposing capture/emission flows, driven by readout capture of
   exposure-accumulated charge. *This turns the catalog + upper-limit population into the science
   result.*

The through-line: **thread 2 chooses the algorithm → thread 1 documents it → it produces a catalog
→ thread 3 bounds its completeness and builds the upper-limit population → thread 4 propagates both
to the masked single-electron residual.** The headline finding survives: the *measured* trap impact
is null after masking; the residual risk is an efficiency-corrected 90% CL upper limit on a hidden
long-τ population, reported with an error band rather than floored.

---

## Notebook files

| file | thread | what it covers |
|---|---|---|
| [notebook/dipole_algorithm.md](notebook/dipole_algorithm.md) | catalog | finder / fit / grading, legacy vs minimal, calibrated detection table |
| [notebook/signed_refit.md](notebook/signed_refit.md) | catalog method | the full investigation: pedestal discovery → full machinery → adversarial reviews → minimal decision |
| [notebook/completeness_efficiency.md](notebook/completeness_efficiency.md) | completeness | Methods 1/2/3, Method-3 results, the naive-dip closure, high-T fading split, upper-limit seeding |
| [notebook/physics.md](notebook/physics.md) | simulation | emission physics, three recapture models, phase-limited deviation budget, operating conditions, deferred HEE sampling |
| [notebook/glossary.md](notebook/glossary.md) | — | plain-language definitions of every term, symbol, and piece of project shorthand used above |

Each notebook file is self-contained (derivations, equations, and numeric tables folded in), opens
with an "In plain terms" summary, and cross-links the others with `[[name]]` references. Any term you
don't recognize is in the [glossary](notebook/glossary.md).

---

## Classification of the original working notes

**Superseded → moved to `attic/`** (content folded into the notebook file shown):

| original file | folded into | why superseded |
|---|---|---|
| `TRAP_SIMULATION_PHYSICS.md` | physics | §2 recapture model superseded by `phase_limited_v1v3`; rest folded |
| `RECAPTURE_PUMPING_HANDOFF.md` | physics | its open pumping-consistency test was resolved by the model switch |
| `DEVIATION_BUDGET.md` | physics (+ completeness §6) | current content preserved; it *is* the current physics state |
| `HIGH_ENERGY_EVENT_SAMPLING_NOTE.md` | physics §7 | still-valid deferred TODO, preserved as such |
| `DIPOLE_ALGORITHM_LEGACY_VS_MINIMAL.md` | dipole_algorithm | current; folded verbatim-in-spirit + detection table |
| `SIGNED_REFIT_AUDIT.md` | signed_refit §1 | first change audit (pedestal discovery) |
| `SIGNED_REFIT_PHYSICS_AUDIT.md` | signed_refit §2 | physics audit (4 problems + refit sequence) |
| `SIGNED_REFIT_RUNBOOK.md` | signed_refit §3 | the full Step 1–10 machinery (not adopted as production) |
| `SIGNED_REFIT_RUNBOOK_S01_S05_AUDIT.md` | signed_refit §4 | **redundant** — earlier draft of the Steps 1–5 review |
| `SIGNED_REFIT_STEP1_5_REVIEW.md` | signed_refit §4 | adversarial review Steps 1–5 |
| `SIGNED_REFIT_STEP6_10_REVIEW.md` | signed_refit §4 | adversarial review Steps 6–10 |
| `MINIMAL_SIGNED_REFIT_RECOMMENDATION.md` | signed_refit §5 | **the decision** |
| `signed_refit_manifest_summary.md` | signed_refit §3 (Step 1) | per-step report |
| `signed_refit_control_pair_summary.md` | signed_refit §3 (Step 2) | per-step report |
| `signed_refit_noise_model_report.md` | signed_refit §3 (Steps 3–4) | per-step report |
| `signed_refit_noise_closure.md` | signed_refit §3 (Step 4) | per-step report (non-closure numbers) |
| `signed_refit_profile_fitter_validation.md` | signed_refit §3 (Step 5) | per-step report |
| `signed_refit_detection_calibration.md` | signed_refit §3 / dipole_algorithm §3.1 | per-step report (threshold table) |
| `signed_refit_finder_calibration.md` | signed_refit §3 (Step 7) | per-step report (finder tradeoff) |
| `signed_refit_orientation_policy.md` | signed_refit §3 (Step 8) | per-step report |
| `signed_refit_intensity_cutflow.md` | signed_refit §3 (Step 9) | per-step report (cutflow) |
| `signed_refit_candidate_variance_closure.md` | signed_refit §3/§4 (R1 v1) | per-step report (φ knob, superseded by v2) |
| `signed_refit_candidate_variance_closure_v2.md` | signed_refit §3/§4 (R1) | per-step report (φ = 8–30) |
| `signed_refit_variance_validation.md` | signed_refit §4 | per-step report |
| `signed_refit_srh_validation.md` | signed_refit §3 (Step 10) | per-step report (SRH catalog numbers) |
| `trap_completeness_method.md` | completeness_efficiency §§1–2 | conceptual design (Methods 1/2/3) |
| `NAIVE_EFFICIENCY_DIP_EXPLANATION.md` | completeness_efficiency §4 | closed result (dip = GOF artifact) |
| `HIGH_T_FAILURE_SPLIT_PLAN.md` | completeness_efficiency §5 | executed plan (fading split) |
| `MEASUREMENT_COMPLETENESS_REVIEW.html` | dipole_algorithm + completeness_efficiency | rendered 4-tab synthesis (2026-06-17/18); tab 1 → dipole_algorithm, tabs 2+4 → completeness_efficiency (Method-3 + validation/WS1-3), tab 3 (figure reference) → completeness_efficiency §3b |

> **Note on the HTML.** It was a polished tabbed dashboard and, being later than most notes, carried
> corrected numbers that superseded a couple I had folded from the older `.md` files — in particular
> the minimal∩legacy catalog overlap is **1,061** (I had briefly mis-stated 711, which is the
> unrelated *full-refit* overlap) and the minimal SRH GOF threshold history is **p-value → χ²<4 →
> χ²<10**. Those are now reconciled in the notebook. Its "Figure Reference" tab (per-trap example
> figures, `figures/dropped_catalog/`, the completeness map and caught-at-T map) is summarized in
> `dipole_algorithm.md` §4 and `completeness_efficiency.md` §3b rather than reproduced image-by-image.

**Left in place** (not archived):

- `CLAUDE.md` — current project/agent configuration. Living document.
- `README.md` — trivial repo stub.
- `AGENTS.md` — the Codex-facing sibling of `CLAUDE.md`. **Resynced to `CLAUDE.md` (2026-07-01)**: it
  was previously stale (said "SNOLAB ~2× fewer high-energy events" — now 10× — and lacked the
  recapture, clear-mode, campaign, and exp-indep-charge-mode sections). It is now a faithful
  Codex-headed copy with the same content; keep the two in sync when either changes. Config, not a
  note, so not folded into the notebook.
- `trap_completeness_method3/` — **live analysis workspace** (code in `src/`, stage packets in
  `agents/`, generated `cache/`). Its results are summarized in
  [completeness_efficiency.md](notebook/completeness_efficiency.md), but the subtree itself is kept
  intact because it is runnable infrastructure, not a note. `agents/11_naive_efficiency_closure.md`
  and `agents/12_high_temp_misfit.md` hold the stage-level detail behind §4/§5 of that notebook file.
- `paper/deferred.md`, `figures/dropped_catalog/_index.md` — out of scope (manuscript / figure
  index).

---

## Provenance

Consolidated 2026-07-01. The `attic/` copies are the verbatim originals (git history preserved for
the two tracked files, `TRAP_SIMULATION_PHYSICS.md` and `trap_completeness_method.md`). If a
notebook file and an archived original disagree, the notebook is authoritative — it folds in later
corrections that postdate the original note.
