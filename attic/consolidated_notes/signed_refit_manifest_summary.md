# Signed Refit Input Manifest Summary

- Manifest version: `signed-refit-input-v1`
- Manifest file: `signed_refit_manifest.csv`
- Manifest SHA-256: `477cca5d74a2dcf953aaeb6e8f614f3f5a807f87f09a992024609acb73c07b67`
- Candidate CCD2 FITS files: 481
- Selected files: 477
- Excluded files: 4
- Quadrants available in every file: 0, 1, 2, 3

## Selection rule

- Include processed CCD2 files matching `proc*dtph*_2_*.fits`.
- At 200 K, include only the recent acquisition, run IDs 160-184.
- At other temperatures, include the unique available image for each `dtph`.

## Selected scans

| Temperature (K) | Images | Unique dtph | Family | Run IDs | Charge shifts | Date range |
|---:|---:|---:|---|---|---:|---|
| 125 | 18 | 18 | temp_scan_run1 | 194-211 | 200000 | 2025-03-10T14:00:08 to 2025-03-10T23:20:29 |
| 130 | 18 | 18 | temp_scan_run1 | 175-192 | 200000 | 2025-03-10T00:46:22 to 2025-03-10T10:06:47 |
| 135 | 18 | 18 | temp_scan_run1 | 156-173 | 200000 | 2025-03-09T11:32:32 to 2025-03-09T20:53:01 |
| 140 | 18 | 18 | temp_scan_run1 | 137-154 | 200000 | 2025-03-08T22:18:49 to 2025-03-09T07:39:12 |
| 145 | 18 | 18 | temp_scan_run1 | 118-135 | 200000 | 2025-03-08T09:04:56 to 2025-03-08T18:25:29 |
| 150 | 18 | 18 | temp_scan_run1 | 99-116 | 200000 | 2025-03-07T19:51:15 to 2025-03-08T05:11:34 |
| 155 | 18 | 18 | temp_scan_run1 | 80-97 | 200000 | 2025-03-07T06:37:33 to 2025-03-07T15:57:55 |
| 160 | 18 | 18 | dp_scan1 | 3-20 | 300000 | 2025-02-21T23:57:40 to 2025-02-22T09:31:34 |
| 165 | 18 | 18 | temp_scan_run1 | 61-78 | 200000 | 2025-03-06T17:23:50 to 2025-03-07T02:44:12 |
| 170 | 18 | 18 | dp_scan1 | 21-38 | 300000 | 2025-02-22T18:40:50 to 2025-02-23T04:14:45 |
| 175 | 18 | 18 | temp_scan_run1 | 42-59 | 200000 | 2025-03-06T04:10:07 to 2025-03-06T13:30:29 |
| 180 | 25 | 25 | temp_scan_run1 | 4-28 | 200000 | 2025-03-18T23:40:48 to 2025-03-21T05:19:28 |
| 183 | 25 | 25 | temp_scan_run1 | 30-54 | 200000 | 2025-03-21T09:12:49 to 2025-03-23T14:51:29 |
| 185 | 18 | 18 | temp_scan_run1 | 23-40 | 200000 | 2025-03-05T14:56:24 to 2025-03-06T00:16:47 |
| 187 | 25 | 25 | temp_scan_run1 | 56-80 | 200000 | 2025-03-23T18:44:50 to 2025-03-26T00:23:30 |
| 190 | 25 | 25 | temp_scan_run1 | 82-106 | 200000 | 2025-03-26T04:16:51 to 2025-03-28T09:55:31 |
| 193 | 25 | 25 | temp_scan_run1 | 108-132 | 200000 | 2025-03-28T13:48:52 to 2025-03-30T19:27:31 |
| 195 | 18 | 18 | temp_scan_run1 | 4-21 | 200000 | 2025-03-05T01:42:40 to 2025-03-05T11:03:03 |
| 197 | 25 | 25 | temp_scan_run1 | 134-158 | 200000 | 2025-03-30T23:20:52 to 2025-04-02T04:59:33 |
| 200 | 25 | 25 | temp_scan_run1 | 160-184 | 200000 | 2025-04-02T08:52:53 to 2025-04-04T14:31:34 |
| 203 | 25 | 25 | temp_scan_run1 | 186-210 | 200000 | 2025-04-04T18:24:55 to 2025-04-07T00:03:33 |
| 207 | 25 | 25 | temp_scan_run1 | 212-236 | 200000 | 2025-04-07T03:56:54 to 2025-04-09T09:35:34 |
| 210 | 18 | 18 | temp_scan_run1 | 3-20 | 200000 | 2025-03-04T07:44:49 to 2025-03-04T17:05:12 |

## Compatibility checks

- All selected files use `NPUMPS=3000`.
- All selected files use `vl=-2.75`, `vh=7.5`, `NROW=580`, `NCOL=3600`, and unit row/column binning.
- All selected files contain four image HDUs with shape `580x3600`.
- Filename geometry agrees with FITS headers.
- Readout delay headers are identical across selected files.
- `dp_scan1` uses 300000 charge-generating shifts at 160 K and 170 K; `temp_scan_run1` uses 200000 elsewhere. This is an intentional per-scan illumination setting, and the intensity model fits an independent signed amplitude at every temperature.

## Excluded files

| Temperature (K) | dtph | Run ID | File | Reason |
|---:|---:|---:|---|---|
| 200 | 750 | 21 | `proc/proc_skp_temp_scan_run1_200k_binned_NROW580_NBINROW1_NCOL3600_NBINCOL1_SC200000_vl-2.75_vh7.5_dtph750_NPUMPS3000_2_21.fits` | superseded 200 K acquisition; use recent run IDs 160-184 |
| 200 | 1200 | 22 | `proc/proc_skp_temp_scan_run1_200k_binned_NROW580_NBINROW1_NCOL3600_NBINCOL1_SC200000_vl-2.75_vh7.5_dtph1200_NPUMPS3000_2_22.fits` | superseded 200 K acquisition; use recent run IDs 160-184 |
| 200 | 2000 | 23 | `proc/proc_skp_temp_scan_run1_200k_binned_NROW580_NBINROW1_NCOL3600_NBINCOL1_SC200000_vl-2.75_vh7.5_dtph2000_NPUMPS3000_2_23.fits` | superseded 200 K acquisition; use recent run IDs 160-184 |
| 200 | 3000 | 24 | `proc/proc_skp_temp_scan_run1_200k_binned_NROW580_NBINROW1_NCOL3600_NBINCOL1_SC200000_vl-2.75_vh7.5_dtph3000_NPUMPS3000_2_24.fits` | superseded 200 K acquisition; use recent run IDs 160-184 |

## Acceptance gate

- PASS: every selected `(temperature, dtph)` is unique.
- PASS: every exclusion has a recorded reason.
- PASS: selected pumping, voltage, geometry, binning, and readout-delay settings are compatible.
- PASS: `load_selected_image_files()` provides the frozen input list, so downstream analysis does not use `glob` as its scientific selection rule.
