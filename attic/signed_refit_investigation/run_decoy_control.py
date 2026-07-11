"""End-to-end false-characterization control for the signed pipeline.

Builds a decoy coordinate list per quadrant:
  - 400 random coordinates at least 3 pixels away from any real candidate
    (pure-noise control), and
  - persistent horizontal-pair sites from 145/175/190 K (structure that triggers
    the detection statistic but cannot be a vertical pumped dipole),
then runs the identical spectra extraction + 3-parameter offset fit + selection
chain and reports how many decoys get 'good' fits / become 'characterized'.
"""

import glob
import re
from collections import defaultdict

import numpy as np

from utils import get_qdata, crop_qdata, approximate_electronize, save_spectra_hdf5
from dipole_new import getDipoleSpectra2, fitTrapIntensity
from finder_null_test import find_pairs

GOOD_QUADS = [0, 1, 2, 3]
N_RANDOM = 400
SEED = 20260613
NULL_TEMPS = [145, 175, 190]
SPECTRA_FILE = 'decoy_spectra_signed.h5'
FITS_FILE = 'decoy_fit_signed.h5'


def horizontal_persistent(temp, quad):
    files = sorted(f for f in glob.glob('proc/proc*dtph*_2_*') if re.search(f'_{temp}k', f))
    counts = defaultdict(set)
    for f in files:
        dtph = int(re.findall(r'dtph(\d+)_', f)[0])
        img = approximate_electronize(crop_qdata(get_qdata(f, quad)), 400).astype(float)
        for c in find_pairs(img, axis=1):
            counts[c].add(dtph)
    return {c for c, v in counts.items() if len(v) > 1}


def main():
    rng = np.random.default_rng(SEED)
    real = np.load('dipole_coord_list_signed.npz')
    noise_npz = np.load('pair_noise_table.npz')
    noise_table = {
        (int(t), int(q)): float(s)
        for t, q, s in zip(noise_npz['temperature_K'], noise_npz['quadrant'], noise_npz['sigma_base_e'])
    }

    decoy_list = []
    decoy_kind = {}
    for quad in GOOD_QUADS:
        real_coords = {tuple(c) for c in real[f'quad_idx_{quad}']}
        blocked = {
            (r + dr, c + dc) for (r, c) in real_coords for dr in (-3, -2, -1, 0, 1, 2, 3)
            for dc in (-3, -2, -1, 0, 1, 2, 3)
        }
        coords = []
        while len(coords) < N_RANDOM:
            r = int(rng.integers(2, 508))
            c = int(rng.integers(2, 3070))
            if (r, c) not in blocked:
                coords.append((r, c))
                decoy_kind[(quad, (r, c))] = 'random'
        horiz = set()
        for temp in NULL_TEMPS:
            horiz |= horizontal_persistent(temp, quad)
        horiz = [h for h in horiz if h not in blocked and h[0] >= 2]
        rng.shuffle(horiz)
        for h in horiz[:400]:
            coords.append(tuple(h))
            decoy_kind[(quad, tuple(h))] = 'horizontal_null'
        decoy_list.append(coords)
        print(f'quad {quad}: {N_RANDOM} random + {len(coords) - N_RANDOM} horizontal-null decoys')

    print('Extracting decoy spectra...')
    decoy_dict = getDipoleSpectra2(
        'proc/', GOOD_QUADS, decoy_list,
        absolute=False, error_model='physical', noise_table=noise_table,
    )
    save_spectra_hdf5(decoy_dict, SPECTRA_FILE)

    print('Fitting decoys...')
    fitTrapIntensity(
        decoy_dict,
        useIntensityErr=True,
        wellBehavedThreshold=4,
        fit_offset=True,
        errors_are_absolute=True,
    )
    save_spectra_hdf5(decoy_dict, FITS_FILE)

    stats = defaultdict(lambda: [0, 0, 0, 0])  # n, n_any_good, n_wellbehaved, n_characterized
    good_temp_count = defaultdict(list)
    for quad in GOOD_QUADS:
        for dp, trap in decoy_dict[quad].items():
            if not isinstance(dp, tuple):
                continue
            kind = decoy_kind[(quad, dp)]
            n_good = sum(
                1 for t, v in trap.items()
                if isinstance(t, int) and isinstance(v, dict) and v.get('GoodIntensityFit', False)
            )
            characterized = (
                trap.get('WellBehavedTrap', False)
                and not trap.get('EnergyFitFailed', True)
                and trap.get('GoodEnergyFit', False)
            )
            s = stats[kind]
            s[0] += 1
            s[1] += int(n_good > 0)
            s[2] += int(trap.get('WellBehavedTrap', False))
            s[3] += int(characterized)
            good_temp_count[kind].append(n_good)

    for kind, (n, any_good, wb, char) in stats.items():
        counts = np.array(good_temp_count[kind])
        print(f'{kind}: n={n}  >=1 good temp: {any_good} ({any_good/n:.3%})  '
              f'well-behaved (>=4): {wb} ({wb/n:.3%})  characterized: {char} ({char/n:.3%})  '
              f'mean good temps {counts.mean():.2f}')


if __name__ == '__main__':
    main()
