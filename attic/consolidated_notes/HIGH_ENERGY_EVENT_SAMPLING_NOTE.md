# Deferred improvement: high-energy-event sampling

## Scope

This note records a future replacement for the current high-energy-event (HEE)
transplantation in `ccd_simulation.py`. It is intentionally not implemented yet.

The current method extracts connected clusters from one fixed 20-hour MINOS
image, scales the requested number deterministically with exposure, selects
clusters without replacement, and transplants them to random positions. SNOLAB
uses the same cluster population with a factor-of-10 lower rate.

This is adequate for a first estimate of HEE masking, but it suppresses event-count
variance, repeatedly reuses one detector realization across trials, and assumes
that MINOS and SNOLAB differ only in normalization.

## Proposed implementation

Add a backward-compatible sampling mode:

```text
hee_sampling_mode = "legacy" | "poisson_bootstrap"
```

Keep `"legacy"` available for reproducing existing campaigns. The proposed
`"poisson_bootstrap"` mode should do the following:

1. Build a cluster library from all approved reference images rather than one
   image. For each connected HEE, store its charge cutout, bounding box, total
   charge, peak charge, area, source image, source exposure, and run condition.
2. Separate the HEE core from any surrounding deferred-charge or trap-tail
   pixels. Store a wider context cutout only for validation; do not automatically
   transplant that context as primary deposited charge.
3. Draw the number of HEEs independently for each fake image:

   ```text
   N_HEE ~ Poisson(rate_condition * exposure * simulated_area)
   ```

   Estimate `rate_condition` from the summed source exposure and active area.
4. Sample clusters with replacement from the appropriate run-condition library.
   Weight source images by their live time or normalize each extracted cluster by
   the exposure represented by its source.
5. Place clusters at random valid active-area locations. Permit physical overlap
   unless a documented detector or reconstruction constraint requires otherwise.
6. Use the exact same injected HEE realization for the trap and no-trap branches.
   This preserves the current common-random-number cancellation.
7. Use separate MINOS and SNOLAB libraries when sufficient data exist. Until
   then, expose the factor-of-10 rate scaling and the shared-spectrum assumption
   as explicit metadata/systematics.

## Data products and metadata

Record at least the following in each output file:

- HEE sampling mode and library version
- source files and total source exposure
- assumed HEE rate and run condition
- sampled HEE count per fake exposure
- RNG seed or seed sequence
- whether context/tail pixels were included

## Validation

Before using the new mode in a production campaign:

1. Verify that the mean HEE count scales linearly with exposure and simulated
   area.
2. Verify `Var(N_HEE) / Mean(N_HEE)` is consistent with one unless source data
   show statistically significant overdispersion.
3. Compare simulated and source distributions of total charge, peak charge,
   cluster area, aspect ratio, and nearest-neighbor distance.
4. Compare halo, bleed, and final masked fractions against held-out real images.
5. Confirm that source cutouts are not mutated during transplantation.
6. Confirm bit-for-bit identical injected HEE images in the paired trap/no-trap
   branches before trap transport is applied.
7. Add a fixed-seed regression test for both legacy and Poisson-bootstrap modes.

The main physics quantity to watch is the final masked single-electron excess.
Unmasked trap transport should be largely insensitive to this change, while
masking can respond nonlinearly to HEE count fluctuations, overlap, and the
cluster charge spectrum.
