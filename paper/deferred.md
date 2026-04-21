# Deferred Items

Items that need to be resolved before submission but are not yet ready to write.

## Physics / open questions

- **E vs σ correlation**: We observe a strong positive correlation between trap energy $E_t$ and capture cross-section $\sigma$ across all groups (Fig. `characterized_traps`). The physical origin of this correlation is not yet understood. Need to investigate and decide whether to offer an interpretation or simply note it as an observation consistent with prior work.

## Missing calculations / numbers

- **No-mask exposure-dependent rate benchmark**: The simulation section currently lacks a number for the increase in the exposure-dependent single-electron rate when *no* masking is applied. This needs to be calculated from the simulation output and inserted in §5 (see `\ansh{Also missing benchmark...}` comment in paper.tex).

## Open writing decisions

- **Bad traps**: Whether to discuss the 1783 traps that were rejected for not fitting the model at any temperature (see `\ansh{maybe we talk about some of the bad traps...}` in §3).
- **Same MC treatment as Brusco:2025**: Whether to reproduce the same Monte-Carlo cross-check done in the previous paper using the new τ distribution (see `\ansh{Think I need to do the same MC thing...}` in §5).
- **Future work section**: Whether to expand the outlook in §6 into a dedicated subsection.
