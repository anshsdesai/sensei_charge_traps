# notebook/figures

Static figures embedded in the `notebook/*.md` files, plus the one script that
makes them.

- **`make_figures.py`** — one function per figure. Each reads a real analysis
  cache (named in the function's docstring), draws the figure, and writes an SVG
  here. The script is the provenance record: the cache path and the numbers each
  figure illustrates are stated inline.
- **`*.svg`** — the committed outputs. They are checked in so the notebook reads
  correctly a year from now *without* rerunning anything or having the data
  caches present. SVG for line/vector plots; PNG for pixel-image examples where the raster
  pixels are the point of the figure.

## Regenerate

From the repo root, in the project conda env:

```bash
conda run -n sensei_charge_traps_new python notebook/figures/make_figures.py
```

Re-run after any analysis change that moves the numbers, then commit the
refreshed SVGs.

## Current figures

| Figure | Made by | Source cache | Used in |
|---|---|---|---|
| `simulation_fake_image.png` | `simulation_fake_image()` | deterministic synthetic event field | `physics.qmd` code map |
| `simulation_source_image.png` | `simulation_source_image()` | `minos_image/proc_corr_proc_skp_72000secs_exp_run10_NSAMP_300_36.fits` | `physics.qmd` code map |
| `simulation_condition_grid.png` | `simulation_condition_grid()` | deterministic MINOS/SNOLAB visual examples | `physics.qmd` code map |
| `simulation_trap_effect.png` | `simulation_trap_effect()` | HEE-filled selected-trap proof image + `fast_readout_numba` | `physics.qmd` code map |
| `pedestal.svg` | `pedestal()` | `fit_dipole_spectra_minimal_caldet_err_4.h5` | `dipole_algorithm.md` §2 |

## Adding a figure

Add a function to `make_figures.py` following the `pedestal()` template (docstring
names the cache + what it shows; write an SVG or PNG into `HERE`), call it from
`__main__`, run the script, embed with `![caption](figures/<name>.svg)` or the corresponding PNG path in the relevant `.md`, and add a row to the table above.

If we later promote the notebook to Quarto (`.qmd`, executable), these same
functions become the code cells — nothing here is throwaway.
