# Agent Protocol For Method 3

This protocol keeps the Method 3 completeness study transparent, resumable, and cheap in context.

## Context Budget

For any stage, read only:

1. `README.md`
2. this file
3. the current stage packet in `agents/`
4. previous stage packets explicitly listed as inputs
5. source files explicitly named by the current packet

Do not load large notebooks, HDF5 files, FITS files, or broad project history unless the packet
requires it. Prefer small scripts or narrow notebook cells that print compact summaries.

## Runtime Environment

Use the WSL conda environment `sensei_charge_traps` for all Method 3 commands, notebooks, and
scripts. This environment is available at:

```bash
/home/ansh/miniforge3/envs/sensei_charge_traps/bin/python
```

The most reliable command form from this WSL workspace is:

```bash
/home/ansh/miniforge3/bin/conda run -n sensei_charge_traps python <script>.py
```

or, for short one-off checks:

```bash
/home/ansh/miniforge3/envs/sensei_charge_traps/bin/python <script>.py
```

The Windows-side environment `sensei_charge_traps_new` is not available from this WSL shell and
should not be used by agents working in this workspace. If the WSL environment above is not
available or a required dependency is missing, stop the stage, record the failure in the current
packet's `Results` section, and do not silently fall back to another interpreter.

## Output Discipline

Each stage must fill its packet's `Results` section with:

- date/time of completion,
- exact command or notebook used,
- input artifact paths,
- output artifact paths,
- short numerical summary,
- required checks and pass/fail status,
- stop conditions encountered,
- open questions for the next stage.

Do not paste large arrays, tables, plots, FITS headers, or notebook output into chat. Save them
under `cache/` or `notebooks/` and reference the paths.

## Correctness Rules

- Keep every transformation inspectable. A final number without an intermediate diagnostic is not
  acceptable.
- Every cached artifact must include metadata: producing stage, date, input files, grid definitions,
  cuts, random seed if any, and code/notebook path.
- Every detection-probability artifact must include cutflow fractions, not just final `p_det`.
- Stage 02 is a hard gate: do not build the trap-free noise map until FITS-derived local patch
  noise matches stored HDF5 `intensity_err` within an understood tolerance, and the separate
  semantics of HDF5 `image_sigma` are documented. `image_sigma` is not the local per-point noise
  field unless Stage 02 explicitly overturns that finding.
- Use deterministic seeds for injection-recovery. Record the seed and repeat a small subset with at
  least one alternate seed when the packet asks for it.
- If a result is surprising, stop and write the surprise into `Open Questions` before scaling up.

## Notebook Policy

Notebooks are encouraged for inspection and plots. Once a notebook cell becomes a reusable operation,
move that operation into `src/` and call it from the notebook. This avoids hidden cell-state bugs
while preserving interactive exploration.

## Recommended Artifact Naming

Use stage-prefixed names:

```text
cache/00_data_inventory.json
cache/01_hdf5_records_summary.csv
cache/02_noise_parity_sample.csv
cache/03_noise_map_v1.h5
cache/04_intensity_error_scaling.json
cache/05_amplitude_prior_v1.npz
cache/07_single_temperature_pdet_160K_v1.h5
cache/08_pdet_grid_v1.h5
cache/09_characterization_probability_v1.h5
```

Version artifacts with `_v2`, `_v3`, etc. Do not overwrite a result that was used downstream unless
the downstream packets are explicitly invalidated.
