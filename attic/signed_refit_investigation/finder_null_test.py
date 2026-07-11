"""Quantify finder false positives: chance coincidences and a horizontal-pair null.

For selected temperatures and quadrants:
 1. Run the new finder (robust sigma, no symmetry cut) per image; count candidates.
 2. Observed persistent dipoles = coordinates seen at >= 2 distinct dtphs.
 3. Chance-coincidence expectation: permute each image's candidates to random
    coordinates (preserving counts) and recompute persistence (100 shuffles).
 4. Horizontal-pair null: same detection statistic on horizontally adjacent
    pixels, which vertical pocket pumping cannot produce; persistent horizontal
    "dipoles" measure the rate of non-pumping structure passing the finder.
"""

import glob
import re
import sys
from collections import defaultdict

import numpy as np

from utils import get_qdata, crop_qdata, approximate_electronize

ELECTRONIZE_EVAL = 400


def find_pairs(image, axis):
    """Coordinates whose adjacent-pixel product (along axis) < -(3 sigma_MAD)^2."""
    med = np.median(image, axis=1)
    image = (image.T - med).T
    sigma = 1.4826 * np.nanmedian(np.abs(image - np.nanmedian(image)))
    cutoff = -1 * (3 * sigma) ** 2
    if axis == 0:  # vertical (pumping direction)
        product = image[1:, :] * image[:-1, :]
        rows, cols = np.where(product < cutoff)
        return set(zip((rows + 1).tolist(), cols.tolist()))
    product = image[:, 1:] * image[:, :-1]
    rows, cols = np.where(product < cutoff)
    return set(zip(rows.tolist(), (cols + 1).tolist()))


def persistence(per_image_sets):
    counts = defaultdict(set)
    for dtph, coords in per_image_sets:
        for c in coords:
            counts[c].add(dtph)
    return sum(1 for v in counts.values() if len(v) > 1)


def chance_coincidences(per_image_sets, shape, n_shuffle=100, seed=0):
    rng = np.random.default_rng(seed)
    n_pix = shape[0] * shape[1]
    results = []
    for _ in range(n_shuffle):
        fake = []
        for dtph, coords in per_image_sets:
            flat = rng.choice(n_pix, size=len(coords), replace=False)
            fake.append((dtph, set(zip((flat // shape[1]).tolist(), (flat % shape[1]).tolist()))))
        results.append(persistence(fake))
    return float(np.mean(results)), float(np.std(results))


def main():
    temps = [int(t) for t in (sys.argv[1:] or [145, 190])]
    for temp in temps:
        files = sorted(
            f for f in glob.glob('proc/proc*dtph*_2_*') if re.search(f'_{temp}k', f)
        )
        for quad in [0, 2]:
            vert, horiz = [], []
            shape = None
            for f in files:
                dtph = int(re.findall(r'dtph(\d+)_', f)[0])
                img = approximate_electronize(crop_qdata(get_qdata(f, quad)), ELECTRONIZE_EVAL).astype(float)
                shape = img.shape
                vert.append((dtph, find_pairs(img, axis=0)))
                horiz.append((dtph, find_pairs(img, axis=1)))
            obs_v = persistence(vert)
            obs_h = persistence(horiz)
            exp_mean, exp_std = chance_coincidences(vert, shape)
            n_per_img = [len(c) for _, c in vert]
            print(
                f"{temp} K quad {quad}: per-image candidates med {int(np.median(n_per_img))} "
                f"(max {max(n_per_img)}) | persistent vertical {obs_v} | "
                f"chance expectation {exp_mean:.1f}+-{exp_std:.1f} | "
                f"persistent horizontal (null) {obs_h}"
            )


if __name__ == '__main__':
    main()
