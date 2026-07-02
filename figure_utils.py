"""Figure helpers for charge_trap_figures.ipynb.

All reusable logic for the paper figure notebook lives here so the notebook
itself stays a thin, toggle-driven driver. Three data sources are made
explicit and switchable from the notebook's top config block:

1. Trap-fit catalog  -> ``load_trap_fits(FIT_SOURCE)`` (an ``fit_dipole_spectra*.h5``).
2. Method-3 completeness -> ``load_method3(version, variant)`` (versioned
   artifacts under ``trap_completeness_method3/cache/``).
3. Simulation campaign -> ``scenario_dir(...)`` / ``compare_scenarios(...)``
   (the labelled dirs under ``campaign/`` produced by run_campaign.py, plus
   the legacy flat dirs registered as aliases).

The plotting functions are direct ports of the original notebook cells; they
preserve the figures written into ``figures/``.
"""

import csv
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple, HandlerBase
from lxml import etree
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from scipy.stats import gamma

import hist
from hist import Hist

from utils import *        # noqa: F401,F403  (load_spectra_hdf5, ...)
from dipole import *       # noqa: F401,F403  (log_energy_cross_section, intensity_function)
from run_campaign import label_for, VP_ORDER, VP_BASELINE  # reuse the campaign's own label scheme + V_p sweep


# ---------------------------------------------------------------------------
# Style / palette
# ---------------------------------------------------------------------------
# Font sizes used across the figures.
smaller = 20
small = 24
medium = 30
large = 36

# Consistent paper palette: one color per semantic role.
C_RED = '#be0031'    # trap-affected / Monte Carlo / 'with traps'
C_BLUE = '#00429d'   # observed / baseline / 'without traps'
C_REF = 'slategrey'  # reference lines (operating temp, 1h/1day/1yr, etc.)
C_SEQ = ['#be0031', '#00429d', '#FFB000', 'seagreen', 'darkviolet']  # multi-series


def setup_style(use_tex=True):
    """Apply the paper matplotlib rcParams. Call once at the top of the notebook."""
    params = {
        'text.usetex': use_tex,
        'font.size': small,
        'font.family': 'serif',
        'figure.autolayout': True,
    }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.labelsize'] = small
    plt.rcParams['figure.figsize'] = (5, 5)
    plt.rcParams['axes.formatter.use_mathtext'] = True


def load_scivis_cmap(xml_file='2-redy3.xml'):
    """Load a ParaView SciVisColor XML colormap into a LinearSegmentedColormap."""
    tree = etree.parse(xml_file)
    nodes = tree.findall('.//Point')
    colors_ = [(float(n.get('r')), float(n.get('g')), float(n.get('b'))) for n in nodes]
    positions = [float(n.get('x')) for n in nodes]
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(positions, colors_):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    return mpl.colors.LinearSegmentedColormap('SciVisColor', cdict)


try:
    my_cmap = load_scivis_cmap('2-redy3.xml')
except (OSError, etree.XMLSyntaxError):
    my_cmap = plt.cm.plasma


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def format_sci(number):
    """Format a float in scientific notation with 3 decimals, e.g. '1.235e+04'."""
    return "{:.3e}".format(number)


def linear_func(x, m, b):
    return m * x + b


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


# Clean-trap energy-fit survival: a genuine SRH trap with >=4 good intensity
# fits still fails the across-temperature energy fit ~3% of the time from
# noise on its tau(T) points alone (tools/energy_fit_recovery_check.py).
# WS3 measured 0.961 (300x20) under the then reduced-chi2 < 4 gate; the
# 2026-07-02 re-run under the live chi2 < 10 gate (400x25) gives 0.972 with
# sigma_tau pull RMS 1.014, which is the value matching the shipped cuts.
# It is deliberately NOT baked into the stage-09 HDF5 (whose "characterized"
# datasets contain only the intensity reach since survival(k) was removed --
# characterization_probability.APPLY_SURVIVAL_K = False). Instead it is
# applied exactly once, in load_method3, to every bundle entry named
# "characterized" (p4_map, p3_map, known_p4 and the default_curve* built from
# them), so completeness figures and the seed builders all share the single
# convention P(characterized) = intensity reach x ENERGY_FIT_SURVIVAL. The
# *_intensity_* bundle entries stay raw reach on purpose.
ENERGY_FIT_SURVIVAL = 0.972


def calculate_corrected_upper_limits(raw_counts, efficiency, confidence_level=0.90):
    """Efficiency-corrected Poisson upper limits on a trap histogram."""
    raw_upper_limits = gamma.ppf(confidence_level, raw_counts + 1)
    corrected = np.zeros_like(raw_upper_limits, dtype=float)
    valid_mask = efficiency > 0
    corrected[valid_mask] = raw_upper_limits[valid_mask] / efficiency[valid_mask]
    corrected[~valid_mask] = np.inf
    return corrected


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays / scalars."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        return super().default(obj)


def find_repo_root(start=None):
    """Walk up from ``start`` until the trap_completeness_method3 workspace is found."""
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "trap_completeness_method3" / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find repo root from current working directory")


# Figure output directory, auto-selected by analysis flavor. The legacy
# (old-catalog) pipeline writes to 'figures_legacy'; the minimal/new pipeline
# (the current deliverable) writes to 'figures'. Set via use_flavor() (called
# from the notebook config cell and from load_method3) and read via
# figures_dir() by every save in this module.
_FIGURE_DIR = 'figures'


def figure_dir_for_flavor(flavor):
    """'figures_legacy' for the legacy (old-catalog) pipeline, else 'figures'."""
    return 'figures_legacy' if flavor == 'legacy' else 'figures'


def flavor_from_rundir(rundir):
    """Infer the analysis flavor that seeded a simulation run.

    Prefers the ``flavor`` attr recorded in the run's HDF5 (written by
    run_single_trial for runs generated after that change). Falls back to the
    rundir path: campaign dirs carry the flavor tag appended by
    ``run_campaign.label_for`` (``_minimal_caldet`` for the minimal pipeline;
    legacy is untagged), and since minimal always carries the explicit tag and
    legacy is the only other flavor, a missing tag means legacy. Used to
    self-route simulation figure saves from the data being plotted rather than
    the global ``use_flavor`` toggle."""
    try:
        probe = os.path.join(rundir, 'ccd_traps_run0.h5')
        if os.path.exists(probe):
            with h5py.File(probe, 'r') as f:  # noqa: F405
                flavor = f.attrs.get('flavor')
                if flavor is not None:
                    if isinstance(flavor, bytes):
                        flavor = flavor.decode()
                    return str(flavor)
    except (OSError, KeyError):
        pass
    return 'minimal_caldet' if 'minimal_caldet' in str(rundir).replace('\\', '/').lower() else 'legacy'


def use_flavor(flavor):
    """Route subsequent figure saves to the directory for this analysis flavor
    and ensure it exists. Accepts 'legacy' / 'minimal' / 'minimal_caldet'.
    Call once after selecting the pipeline (e.g. fu.use_flavor(PIPELINE))."""
    global _FIGURE_DIR
    _FIGURE_DIR = figure_dir_for_flavor(flavor)
    os.makedirs(_FIGURE_DIR, exist_ok=True)
    return _FIGURE_DIR


def figures_dir():
    """Current figure output directory (set by use_flavor / load_method3)."""
    return _FIGURE_DIR


def ensure_figures_dir():
    os.makedirs(figures_dir(), exist_ok=True)


MEASUREMENT_TEMPERATURES = [
    125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180,
    183, 185, 187, 190, 193, 195, 197, 200, 203, 207, 210,
]


# ---------------------------------------------------------------------------
# 1. Trap-fit catalog loading + aggregation
# ---------------------------------------------------------------------------
def load_trap_fits(fit_source):
    """Load a per-trap fit catalog (an ``fit_dipole_spectra*.h5``) into a dict."""
    fit_dipole_spectra = load_spectra_hdf5(fit_source)  # noqa: F405
    print(f"Loaded {fit_source}")
    return fit_dipole_spectra


def aggregate_trap_fits(fit_dipole_spectra,
                        measurement_temperatures=None,
                        quads=(0, 1, 2, 3)):
    """Reduce the fit catalog to the arrays the figures consume.

    Returns a dict with:
      - ``energy_crossSections``: list of (sigma, E, sigma_err, E_err, avg_good_temp)
      - ``energy_covariances``: parallel list of the 2x2 [E, logsigma] fit
        covariance per good trap (or None if unavailable); same order/length as
        ``energy_crossSections``. Used to draw the correlated (E, sigma)
        uncertainty and the covariance-propagated tau(T) band.
      - ``tau_temp_fits``: list of [temps, taus, tau_errs] per good trap
      - ``tau_temperatures``: {T: {'measured': [...], 'extrapolated': [...]}}
      - ``cross_sections`` / ``energies``: arrays from energy_crossSections
      - ``maxtaus``: per-trap max measured tau
    """
    if measurement_temperatures is None:
        measurement_temperatures = MEASUREMENT_TEMPERATURES

    energy_crossSections = []
    energy_covariances = []
    tau_temp_fits = []
    maxtaus = []
    tau_temperatures = {t: {'measured': [], 'extrapolated': []}
                        for t in measurement_temperatures}

    for q in quads:
        if q not in fit_dipole_spectra:
            continue
        for dp in list(fit_dipole_spectra[q]):
            if not isinstance(dp, tuple):
                continue
            testdp = fit_dipole_spectra[q][dp]
            # Minimal pipeline (dipole_new.py) only writes 'EnergyFitFailed' when
            # WellBehavedTrap AND single_orientation; a missing key means no energy
            # fit was attempted, which is equivalent to a failed fit.
            if not (testdp['WellBehavedTrap'] and not testdp.get('EnergyFitFailed', True)):
                continue

            maxtaus.append(np.max(testdp['energy_taus']))

            if not testdp["GoodEnergyFit"]:
                continue

            cs = testdp['energy_BestFitCrossSection']
            cserr = testdp['energy_BestFitCrossSectionErr']
            e = testdp['energy_BestFitEnergy']
            e_err = testdp['energy_BestFitEnergyErr']
            avg_good_temp = np.mean(testdp['energy_temperatures'])

            for t in measurement_temperatures:
                tau = np.exp(log_energy_cross_section(t, e, np.log(cs)))  # noqa: F405
                if t in testdp['energy_temperatures']:
                    tau_temperatures[t]['measured'].append(tau)
                else:
                    tau_temperatures[t]['extrapolated'].append(tau)

            energy_crossSections.append((cs, e, cserr, e_err, avg_good_temp))
            # 2x2 [E, logsigma] covariance from the Arrhenius fit (popt order),
            # as stored by dipole(_new).py. None if this catalog predates it.
            cov = testdp.get('energy_CovarianceMatrix')
            energy_covariances.append(np.array(cov, dtype=float)
                                      if cov is not None else None)
            tau_temp_fits.append([testdp['energy_temperatures'],
                                  testdp['energy_taus'],
                                  testdp['energy_tau_errs']])

    energy_crossSections_arr = np.array(energy_crossSections)
    cross_sections = energy_crossSections_arr[:, 0] if len(energy_crossSections) else np.array([])
    energies = energy_crossSections_arr[:, 1] if len(energy_crossSections) else np.array([])

    print(f"Number of good energy fits: {len(energy_crossSections)}")
    return {
        'energy_crossSections': energy_crossSections,
        'energy_covariances': energy_covariances,
        'tau_temp_fits': tau_temp_fits,
        'tau_temperatures': tau_temperatures,
        'cross_sections': cross_sections,
        'energies': energies,
        'maxtaus': maxtaus,
    }


# ---------------------------------------------------------------------------
# 2. Trap-fit plots
# ---------------------------------------------------------------------------
def plot_example_traps(fit_dipole_spectra, err_field='intensity_err', max_plots=5,
                       min_points=8, cmap=None, vmin=100, vmax=250, save=True,
                       paper_mode=True):
    """Per-trap intensity(t_ph) and tau(T) example panels (figures/example_trap_*.pdf).

    ``paper_mode`` controls the top-panel presentation:
      - ``True`` (default, paper) subtracts the fitted t_ph-independent pedestal
        ('fit_offset') from both data and curve and flips the orientation sign so
        the pumped response points up (canonical rise-peak-fall). The pedestal is
        a fitted nuisance parameter (readout-direction charge deferral) and the
        orientation sign carries no tau information, so removing both isolates the
        tau-bearing signal. State the subtraction/flip in the figure caption.
        Legacy catalogs (no 'fit_offset') are unaffected by the pedestal subtraction.
      - ``False`` (diagnostic) draws the faithful fit: raw signed intensities and
        the full model including the pedestal. This is the honest fit-quality check.

    ``err_field`` selects which stored per-point error array is drawn as the
    intensity error bars:
      - 'intensity_err' : the error the fit used (the new physical temporal noise
                          under the minimal pipeline; the patch spatial sigma in
                          the legacy pipeline),
      - 'patch_sigma'   : the legacy 34x34 patch spatial sigma (minimal catalogs only),
      - 'poisson_err'   : sqrt(|patch mean|) Poisson estimate.
    If a catalog lacks the requested field (e.g. legacy catalogs have no
    'patch_sigma') it falls back to 'intensity_err' with a one-time warning."""
    valid_fields = {'intensity_err', 'patch_sigma', 'poisson_err'}
    if err_field not in valid_fields:
        raise ValueError(f"err_field must be one of {sorted(valid_fields)}, got {err_field!r}")
    _warned = {'missing': False}

    def _yerr(dipole):
        if err_field in dipole:
            return dipole[err_field]
        if not _warned['missing']:
            print(f"  [plot_example_traps] '{err_field}' not in catalog; "
                  f"falling back to 'intensity_err'.")
            _warned['missing'] = True
        return dipole['intensity_err']

    if cmap is None:
        cmap = my_cmap
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    ensure_figures_dir()
    num_plotted = 0

    for q in (0, 1, 2, 3):
        if q not in fit_dipole_spectra:
            continue
        for dp in list(fit_dipole_spectra[q]):
            if num_plotted >= max_plots:
                break
            if not isinstance(dp, tuple):
                continue
            testdp = fit_dipole_spectra[q][dp]
            # Minimal pipeline (dipole_new.py) only writes 'EnergyFitFailed' when
            # WellBehavedTrap AND single_orientation; a missing key means no energy
            # fit was attempted, which is equivalent to a failed fit.
            if not (testdp['WellBehavedTrap'] and not testdp.get('EnergyFitFailed', True)):
                continue
            if not testdp["GoodEnergyFit"]:
                continue
            if len(testdp['energy_taus']) <= min_points:
                continue

            num_plotted += 1
            fig, ax = plt.subplots(2, 1, figsize=(12, 10))
            ax[0].set_xlabel('$t_{ph}$ [s]', fontsize=small)
            ax[0].set_ylabel('Intensity [$e^-$]', fontsize=small)
            ax[1].set_xlabel('Temperature [K]', fontsize=small)
            ax[1].set_ylabel('$\\tau_e$ [s]', fontsize=small)
            ax[1].set_xticks([120, 130, 140, 150, 160, 170, 180, 190, 200, 210])
            ax[0].set_xscale('log')
            ax[1].set_yscale('log')
            # Paper mode flips the orientation sign so the pumped response points
            # up; use the trap's overall amplitude sign (consistent across temps).
            sign = 1.0
            if paper_mode:
                coeffs = [testdp[t]['fit_coeff'] for t in testdp.keys()
                          if isinstance(t, int) and testdp[t].get('GoodIntensityFit')]
                if coeffs and np.mean(coeffs) < 0:
                    sign = -1.0

            has_offset = False
            data_lo, data_hi = 0.0, 0.0
            for temp in testdp.keys():
                if not isinstance(temp, int):
                    continue
                color = cmap(norm(temp))
                dipole = testdp[temp]
                if dipole['GoodIntensityFit']:
                    yerr = _yerr(dipole)
                    offset = dipole.get('fit_offset', 0.0)
                    if 'fit_offset' in dipole:
                        has_offset = True
                    # Paper mode: subtract the fitted pedestal and flip the sign so
                    # the tau-bearing pumped response reads as a positive peak.
                    pedestal = offset if paper_mode else 0.0
                    data_ints = sign * (np.asarray(dipole['intensities']) - pedestal)
                    line, caps, bars = ax[0].errorbar(
                        dipole['seconds'], data_ints, yerr=yerr,
                        color=color, ls='None', capsize=3, marker='o', markersize=3)
                    for cap in caps:
                        cap.set_alpha(0.3)
                    for bar in bars:
                        bar.set_alpha(0.3)
                    seconds = np.geomspace(np.min(dipole['seconds']),
                                           np.max(dipole['seconds']), 100)
                    fit_ints = intensity_function(seconds, dipole['fit_coeff'], dipole['fit_tau'])  # noqa: F405
                    # Diagnostic mode keeps the pedestal so the curve matches the raw
                    # (signed) data; paper mode already removed it from the data.
                    fit_ints = sign * (fit_ints + offset - pedestal)
                    ax[0].plot(seconds, fit_ints, ls='-', color=cmap(norm(temp)))
                    data_lo = min(data_lo, float(np.min(data_ints)), float(np.min(fit_ints)))
                    data_hi = max(data_hi, float(np.max(data_ints)), float(np.max(fit_ints)))

            if paper_mode:
                # Pedestal removed + sign flipped: peaks rise from ~0.
                pad = 0.05 * (data_hi - data_lo) if data_hi > data_lo else 10.0
                ax[0].set_ylim(min(data_lo - pad, -10.0), data_hi + pad)
            elif has_offset:
                # Signed pedestal model: size the window to the actual data extent.
                pad = 0.1 * (data_hi - data_lo) if data_hi > data_lo else 10.0
                ax[0].set_ylim(data_lo - pad, data_hi + pad)
            else:
                ax[0].set_ylim(-10, 2500)

            temperatures = np.linspace(120, 210, 50)
            fit_taus = np.exp(log_energy_cross_section(  # noqa: F405
                temperatures, testdp['energy_BestFitEnergy'],
                np.log(testdp['energy_BestFitCrossSection'])))
            ax[1].plot(temperatures, fit_taus, color='black', ls='--')

            for t_idx in range(len(testdp['energy_temperatures'])):
                temp_val = testdp['energy_temperatures'][t_idx]
                color = cmap(norm(temp_val))
                ax[1].errorbar(temp_val, testdp['energy_taus'][t_idx],
                               yerr=testdp['energy_tau_errs'][t_idx], ls='None',
                               color=color, marker='o', markersize=5, capsize=5)

            ax[0].grid()
            ax[1].grid()
            if save:
                plt.savefig(f'{figures_dir()}/example_trap_{num_plotted - 1}.pdf', dpi=300)
            plt.show()
            plt.close()


def plot_energy_sigma_hists(agg,log=True):
    """1D histograms of fitted energy and cross-section (cell 14)."""
    energies = agg['energies']
    cross_sections = agg['cross_sections']

    plt.figure(figsize=(8, 6))
    plt.hist(energies, bins=np.linspace(0, 0.8, 50))
    plt.xticks(np.linspace(0, 0.8, 5))
    plt.xlabel("Fitted Trap Energy $E$ [eV]")
    plt.ylabel("Counts")
    if log:
        plt.yscale('log')
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(cross_sections, bins=np.geomspace(1e-25, 1e-10, 50))
    plt.xscale('log')
    plt.xticks(np.geomspace(1e-25, 1e-10, 6))
    plt.xlabel("Fitted Capture Cross-Section $\\sigma$ [cm$^2$]")
    plt.ylabel("Counts")
    if log:
        plt.yscale('log')
    plt.show()
    plt.close()


def plot_energy_sigma_hist2d(agg):
    """2D histogram of fitted cross-section vs energy (cell 15)."""
    cross_sections = agg['cross_sections']
    energies = agg['energies']
    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.hist2d(cross_sections, energies,
                  bins=[np.geomspace(1e-25, 1e-10, 40), np.linspace(0, 0.7, 40)],
                  cmap='viridis', norm=mpl.colors.LogNorm())
    ax.set_xscale('log')
    ax.set_xlabel(r"Fitted Capture Cross-Section $\sigma$ [cm$^2$]")
    ax.set_ylabel(r"Fitted Trap Energy $E$ [eV]")
    fig.colorbar(h[3], ax=ax, label='Counts')
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_characterized_traps(agg, save=True, show_sigma_xerr=True,
                             show_cov_ellipses=False, show_tau_band=False,
                             n_sigma=1.0):
    """tau(T) extrapolation band and the sigma-E scatter split into populations (cell 20).

    The bottom panel's plain marginal E error bar is mathematically correct but
    misleading: E and log(sigma) are ~0.997 anticorrelated along the SRH line, so
    the *marginal* bars overstate the tau(135 K) uncertainty by ~4x and hide the
    correlated structure. Display controls:

      - ``show_sigma_xerr`` (default True, the fix): also draw the marginal sigma
        error bar (computed but previously dropped), as a log-symmetric x error.
      - ``show_cov_ellipses`` (default False): draw the correlated n_sigma (E, sigma)
        covariance ellipse per trap instead of independent +-bars, showing the thin
        tilted degeneracy direction. Requires ``agg['energy_covariances']``.
      - ``show_tau_band`` (default False): replace the top-panel tau(T) spaghetti
        with each trap's covariance-propagated +-n_sigma tau(T) band, which is
        hair-thin near the measured range and fans out only on extrapolation.
        Requires ``agg['energy_covariances']``.

    ``n_sigma`` scales the ellipses and the band.
    """
    energy_crossSections = agg['energy_crossSections']
    tau_temp_fits = agg['tau_temp_fits']
    covs = agg.get('energy_covariances')
    if (show_cov_ellipses or show_tau_band) and covs is None:
        raise ValueError("show_cov_ellipses/show_tau_band need agg['energy_covariances']; "
                         "re-run aggregate_trap_fits on a catalog that stores "
                         "energy_CovarianceMatrix.")

    max_tau = 0
    for t in range(len(tau_temp_fits)):
        max_tau = max(max_tau, np.max(tau_temp_fits[t][1]))

    fig, ax = plt.subplots(2, figsize=(10, 12))
    fit_temp = np.linspace(120, 220, 100)
    kb_phys = 8.617333262e-5  # eV/K, matches log_energy_cross_section

    def _cov_ellipse(cov_E_logsig, e0, logsig0, nsig):
        """n_sigma ellipse boundary in (sigma, E) for the log-x scatter panel."""
        # cov is [E, logsigma]; reorder to (logsigma, E) for the (x=sigma, y=E) plane
        C = np.array([[cov_E_logsig[1, 1], cov_E_logsig[1, 0]],
                      [cov_E_logsig[0, 1], cov_E_logsig[0, 0]]])
        vals, vecs = np.linalg.eigh(C)
        vals = np.clip(vals, 0.0, None)
        th = np.linspace(0, 2 * np.pi, 60)
        pts = (vecs @ (np.sqrt(vals)[:, None] * np.array([np.cos(th), np.sin(th)]))) * nsig
        return np.exp(logsig0 + pts[0]), e0 + pts[1]

    def _tau_band(cov_E_logsig, e0, logsig0, temps, nsig):
        """Covariance-propagated +-n_sigma tau(T) band (J = [1/kT, -1])."""
        inv_kbT = 1.0 / (kb_phys * temps)
        var = (inv_kbT ** 2 * cov_E_logsig[0, 0]
               - 2.0 * inv_kbT * cov_E_logsig[0, 1]
               + cov_E_logsig[1, 1])
        sd = np.sqrt(np.clip(var, 0.0, None))
        logtau = log_energy_cross_section(temps, e0, logsig0)  # noqa: F405
        return np.exp(logtau - nsig * sd), np.exp(logtau + nsig * sd)

    cross_sections = np.array(energy_crossSections)[:, 0]

    # Population decision boundary (high-energy vs standard).
    T = 170
    kb = 8.6717333262e-5
    m = (kb * T)
    tau_test = 1
    b = m * (np.log(tau_test) - (-68.267) + 2 * np.log(m))

    avg_A, avg_B = [], []
    for d, data in enumerate(energy_crossSections):
        energy = data[1]
        cs = data[0]
        cserr = data[2]
        avgtemp = data[4]
        cov = covs[d] if covs is not None else None
        if energy > (m * np.log(cs) + b):
            color = C_RED
            avg_A.append(avgtemp)
        else:
            color = C_BLUE
            avg_B.append(avgtemp)

        # --- top panel: tau(T) line, or covariance-propagated band (option c) ---
        tau_fit = np.exp(log_energy_cross_section(fit_temp, data[1], np.log(data[0])))  # noqa: F405
        if show_tau_band and cov is not None:
            lo, hi = _tau_band(cov, energy, np.log(cs), fit_temp, n_sigma)
            ax[0].fill_between(fit_temp, lo, hi, color=color, alpha=0.05, lw=0)
            ax[0].plot(fit_temp, tau_fit, color=color, alpha=0.15, lw=0.5)
        else:
            ax[0].plot(fit_temp, tau_fit, color=color, alpha=0.1)

        # --- bottom panel: marginal bars (with the sigma xerr fix), or ellipse ---
        if show_cov_ellipses and cov is not None:
            ex, ey = _cov_ellipse(cov, energy, np.log(cs), n_sigma)
            ax[1].plot(ex, ey, color=color, alpha=0.1, lw=0.6)
            ax[1].plot(cs, energy, color=color, marker='o', markersize=2, ls='None')
        else:
            xerr = None
            if show_sigma_xerr and cs > 0:
                s = cserr / cs  # natural-log sigma sd (cserr = s * sigma)
                xerr = np.array([[cs - cs * np.exp(-s)], [cs * np.exp(s) - cs]])
            line, caps, bars = ax[1].errorbar(cs, energy, yerr=data[3], xerr=xerr,
                                              ls='None', color=color, capsize=3,
                                              marker='o', markersize=2)
            for bar in bars:
                bar.set_alpha(0.1)
            for cap in caps:
                cap.set_alpha(0.1)

    ax[0].text(219, 3.16e1, "Extrapolated", fontsize=small, ha='right')
    ax[0].text(219, 3.16, "Measured", fontsize=small, ha='right')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[0].fill_between(fit_temp, np.ones_like(fit_temp) * max_tau,
                       np.ones_like(fit_temp) * 1e10, color='grey', alpha=0.4)
    ax[0].set_ylim(1e-7, 1e10)
    ax[0].set_xlim(120, 220)
    ax[0].set_xlabel("Temperature [K]", fontsize=small)
    ax[0].set_ylabel("$\\tau_e$ [s]", fontsize=small)
    ax[1].set_xlabel("$\\sigma$ [cm$^2$]", fontsize=small)
    ax[1].set_ylabel("Energy [eV]", fontsize=small)
    ax[1].set_ylim(-0.1, 0.8)
    ax[0].axvline(135, ls='--', label='Operating Temperature', color=C_REF)
    ax[0].minorticks_on()
    ax[1].minorticks_on()

    custom_handles = [
        Line2D([0], [0], marker='o', color=C_RED, markerfacecolor=C_RED, markersize=8,
               label='High-Energy Population'),
        Line2D([0], [0], marker='o', color=C_BLUE, markerfacecolor=C_BLUE, markersize=8,
               label='Standard Population'),
        Line2D([0], [0], linestyle='--', color=C_REF, lw=2, label='Operating Temperature'),
    ]
    custom_handles2 = custom_handles[:2]
    ax[0].legend(handles=custom_handles, loc='best', fontsize=smaller, frameon=False)
    ax[1].legend(handles=custom_handles2, loc='best', fontsize=small, frameon=False)

    if save:
        ensure_figures_dir()
        plt.savefig(f'{figures_dir()}/characterized_traps.pdf', dpi=300)
    plt.show()
    plt.close()


def plot_amplitude_vs_temperature(fit_dipole_spectra, num_traps_to_plot=500, seed=None):
    """Raw and normalized fit-amplitude vs temperature for well-behaved traps (cell 50)."""
    if seed is not None:
        random.seed(seed)
    all_well_behaved = []
    for q in (0, 1, 2, 3):
        if q not in fit_dipole_spectra:
            continue
        for dp, trap in fit_dipole_spectra[q].items():
            if isinstance(dp, tuple) and trap.get('WellBehavedTrap', False):
                all_well_behaved.append(trap)

    traps_subset = random.sample(all_well_behaved,
                                 min(num_traps_to_plot, len(all_well_behaved)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for trap in traps_subset:
        temps, coeffs, coeff_errs = [], [], []
        for key, data in trap.items():
            if isinstance(key, int) and data.get('GoodIntensityFit', False):
                temps.append(key)
                coeffs.append(data['fit_coeff'])
                coeff_errs.append(data['fit_coeff_err'])
        if len(temps) > 3:
            sort_idx = np.argsort(temps)
            temps = np.array(temps)[sort_idx]
            coeffs = np.array(coeffs)[sort_idx]
            coeff_errs = np.array(coeff_errs)[sort_idx]
            axes[0].errorbar(temps, coeffs, yerr=np.abs(coeff_errs), fmt='-o', alpha=0.3, markersize=3)
            # Minimal pipeline retains negative-orientation dipoles (negative fit_coeff),
            # so normalize by the peak magnitude and keep error bars non-negative.
            max_coeff = coeffs[np.argmax(np.abs(coeffs))]
            axes[1].errorbar(temps, coeffs / max_coeff, yerr=np.abs(coeff_errs / max_coeff),
                             fmt='-o', alpha=0.3, markersize=3)

    axes[0].set_xlabel('Temperature [K]', fontsize=12)
    axes[0].set_ylabel('Amplitude (fit_coeff) [$e^-$]', fontsize=12)
    axes[0].set_title(f'Raw Amplitude vs Temperature ({len(traps_subset)} Traps)', fontsize=14)
    axes[0].grid(True, alpha=0.5)
    axes[1].set_xlabel('Temperature [K]', fontsize=12)
    axes[1].set_ylabel('Normalized Amplitude (coeff / max_coeff)', fontsize=12)
    axes[1].set_title(f'Normalized Amplitude vs Temperature ({len(traps_subset)} Traps)', fontsize=14)
    axes[1].grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_spatial_distribution(full_dipole_coord_list, mc_bin_centers, mc_mean,
                              ci_lower, ci_upper, quads=(0, 1, 2, 3), save=True):
    """Observed dipole pair-distance distribution vs Monte-Carlo null (cell 6)."""
    plt.figure(figsize=(10, 5))
    all_distances = []
    for q in quads:
        quad_dipoles = np.array(full_dipole_coord_list[q])
        all_distances.append(pdist(quad_dipoles))
    all_distances = np.concatenate(all_distances)

    bins_ = np.linspace(0, 3072, 50)
    plt.hist(all_distances, bins_, color=C_BLUE, edgecolor='black', alpha=0.8,
             label='Observed', density=True)
    plt.plot(mc_bin_centers, mc_mean, color=C_RED)
    plt.fill_between(mc_bin_centers, ci_lower, ci_upper, color=C_RED, alpha=0.4)
    mc_proxy = (Patch(facecolor=C_RED, alpha=0.4), Line2D([0], [0], color=C_RED))
    obs_proxy = plt.Rectangle((0, 0), 1, 1, facecolor=C_BLUE, edgecolor='black', alpha=0.8)

    class _HandlerOverlay(HandlerBase):
        """Draw all artists in a tuple stacked on top of each other (overlapping)
        in a single legend slot, instead of side by side as HandlerTuple does."""
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                            width, height, fontsize, trans):
            artists = []
            for sub in orig_handle:
                handler = legend.get_legend_handler(legend.get_legend_handler_map(), sub)
                artists.extend(handler.create_artists(
                    legend, sub, xdescent, ydescent, width, height, fontsize, trans))
            return artists

    plt.legend(
        [obs_proxy, mc_proxy],
        ['Observed', 'MC Mean + 90\% CI'],
        handler_map={tuple: _HandlerOverlay()},
        fontsize=small, frameon=False,
    )
    plt.xlabel('Distance [pixels]', fontsize=small)
    plt.ylabel('Probability Density', fontsize=small)
    if save:
        ensure_figures_dir()
        plt.savefig(f'{figures_dir()}/trap_distribution.pdf', dpi=300)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 3. Method-3 completeness loading
# ---------------------------------------------------------------------------
def resolve_method3(version='v1', variant='default', repo=None,
                    pipeline='legacy', detection='fixed',
                    analysis_flavor=None):
    """Resolve the Method-3 cache file paths for a given version/variant.

    ``variant='tau1000'`` selects the extended-tau artifacts. Raises a clear
    error if a requested stage file is missing so version bumps fail loudly.
    """
    repo = find_repo_root() if repo is None else Path(repo)
    cache = repo / "trap_completeness_method3" / "cache"
    tag = '' if variant == 'default' else f'_{variant}'
    if analysis_flavor is None:
        if pipeline == 'minimal' and detection == 'calibrated':
            analysis_flavor = 'minimal_caldet'
        else:
            analysis_flavor = 'legacy'

    if analysis_flavor in ('minimal', 'minimal_caldet'):
        mtag = 'minimal_caldet'
        stage09_name = f"09_characterization_probability_{mtag}{tag}_{version}.h5"
        summary_name = f"10_validation_sensitivity_{mtag}{tag}_summary.json"
        statement_name = f"10_completeness_statement_{mtag}{tag}.md"
        records4 = cache / f"01_records_{mtag}_ngood4.csv"
        tau_hist = repo / f"tau_at_135k_hist_{mtag}.npz"
        catalog_h5 = repo / f"fit_dipole_spectra_{mtag}_err_4.h5"
    else:
        stage09_name = f"09_characterization_probability{tag}_{version}.h5"
        summary_name = f"10_validation_sensitivity{tag}_summary.json"
        statement_name = f"10_completeness_statement{tag}.md"
        records4 = cache / "01_records_ngood4.csv"
        tau_hist = repo / "tau_at_135k_hist.npz"
        catalog_h5 = repo / "fit_dipole_spectra_err_4.h5"

    # The efficiency-corrected upper-limit histogram is built from THIS flavor's
    # completeness curve, so its saved name must track the flavor (otherwise the
    # simulation could ingest a seed built from a different Method-3 selection).
    # Mirror the regular-hist naming: tau_at_135k_hist[_<flavor>]_upper_limit.npz.
    tau_hist_upper = tau_hist.with_name(tau_hist.stem + "_upper_limit.npz")

    # Detected-dipole coordinate list for this flavor (run_charge_traps.py:242).
    # Only the minimal pipeline changes the raw dipole finding; the `_caldet`
    # detection tag does not, so it is dropped from the coord-list suffix. This
    # is the same count run_campaign / run_ccd_simulation use as the baseline
    # trap population, so the "scale of total traps" reported here matches it.
    coord_suffix = '_minimal' if analysis_flavor in ('minimal', 'minimal_caldet') else ''
    dipole_coord_list = repo / f"dipole_coord_list{coord_suffix}.npz"

    paths = {
        'repo': repo,
        'cache': cache,
        'analysis_flavor': analysis_flavor,
        'stage09_h5': cache / stage09_name,
        'summary': cache / summary_name,
        'statement': cache / statement_name,
        'records4': records4,
        'catalog_h5': catalog_h5,
        'tau_hist_npz': tau_hist,
        'tau_hist_upper_npz': tau_hist_upper,
        'dipole_coord_list_npz': dipole_coord_list,
        'figures_dir': figure_dir_for_flavor(analysis_flavor),
    }
    missing = [str(p) for k, p in paths.items()
               if k in ('stage09_h5', 'summary', 'records4') and not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Method-3 version='{version}', variant='{variant}' is missing:\n  "
            + "\n  ".join(missing))
    return paths


def load_method3(version='v1', variant='default', repo=None,
                 pipeline='legacy', detection='fixed',
                 analysis_flavor=None):
    """Load the Method-3 completeness model + the 135 K tau histogram + records.

    Returns one bundle dict consumed by the completeness/upper-limit plots:
      tau_grid, E_grid, p4_map, p3_map, default_curve, default_curve3,
      observed_E, tau_hist, tau_edges, tau_samples, hist_source,
      tau_135_records, E_records, measured_135, summary, statement, paths.
    """
    paths = resolve_method3(version=version, variant=variant, repo=repo,
                            pipeline=pipeline, detection=detection,
                            analysis_flavor=analysis_flavor)
    # Route figure saves to this flavor's directory (figures / figures_legacy).
    use_flavor(paths['analysis_flavor'])

    summary = json.loads(paths['summary'].read_text())
    statement = paths['statement'].read_text() if paths['statement'].exists() else ''

    with h5py.File(paths['stage09_h5'], "r") as h5:
        tau_grid = h5["grid/tau_135_seconds"][:]
        E_grid = h5["grid/E_eV"][:]
        p4_map = h5["results/p_characterized_n_good_4"][:]
        p3_map = h5["results/p_characterized_n_good_3"][:]
        p4_intensity_map = (
            h5["results/p_intensity_n_good_4"][:]
            if "p_intensity_n_good_4" in h5["results"] else p4_map
        )
        p3_intensity_map = (
            h5["results/p_intensity_n_good_3"][:]
            if "p_intensity_n_good_3" in h5["results"] else p3_map
        )
        all_oob = h5["diagnostics/all_temperatures_tau_oob"][:].astype(bool)
        known_p4 = h5["validation_known_traps/n_good_4_csv/p_characterized_n_good_4"][:]
        known_tau = h5["validation_known_traps/n_good_4_csv/tau_135_seconds"][:]
        known_E = h5["validation_known_traps/n_good_4_csv/E_eV"][:]

    # Apply the clean-trap energy-fit survival once, here (see the
    # ENERGY_FIT_SURVIVAL comment). Note p4_intensity_map/p3_intensity_map
    # were bound above from the raw arrays, so they remain pure reach even
    # when they fall back to aliasing the pre-multiplication p_characterized
    # datasets.
    p4_map = p4_map * ENERGY_FIT_SURVIVAL
    p3_map = p3_map * ENERGY_FIT_SURVIVAL
    known_p4 = known_p4 * ENERGY_FIT_SURVIVAL
    print(f"Applied energy-fit survival {ENERGY_FIT_SURVIVAL:g} to the "
          f"'characterized' completeness maps/curves (intensity maps stay raw).")

    print("Stage 10 produced at:", summary.get("produced_at"))
    print("Thermal model source:", summary.get("model_notes", {}).get("thermal_model_source"))

    observed_E = _load_observed_E(paths['records4'])
    default_curve = np.array([np.interp(observed_E, E_grid, row).mean() for row in p4_map])
    default_curve3 = np.array([np.interp(observed_E, E_grid, row).mean() for row in p3_map])

    tau_hist, tau_edges, tau_samples, hist_source = _load_tau135_hist(
        paths['tau_hist_npz'], paths['records4'])
    print(f"Histogram source: {hist_source}")
    print(f"Trap count in histogram bins: {int(tau_hist.sum())}")

    tau_135_records, E_records, measured_135 = _load_records(paths['records4'])

    return {
        'paths': paths,
        'summary': summary,
        'statement': statement,
        'tau_grid': tau_grid,
        'E_grid': E_grid,
        'p4_map': p4_map,
        'p3_map': p3_map,
        'p4_intensity_map': p4_intensity_map,
        'p3_intensity_map': p3_intensity_map,
        'all_oob': all_oob,
        'known_p4': known_p4,
        'known_tau': known_tau,
        'known_E': known_E,
        'observed_E': observed_E,
        'default_curve': default_curve,
        'default_curve3': default_curve3,
        'tau_hist': tau_hist,
        'tau_edges': tau_edges,
        'tau_samples': tau_samples,
        'hist_source': hist_source,
        'tau_135_records': tau_135_records,
        'E_records': E_records,
        'measured_135': measured_135,
    }


def _load_observed_E(path):
    values = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            values.append(float(row["E_eV"]))
    return np.asarray(values)


def _load_records(records4_path):
    tau_135_records, E_records, measured_135 = [], [], []
    with Path(records4_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            tau_135_records.append(float(row["tau_135_seconds"]))
            E_records.append(float(row["E_eV"]))
            measured_135.append(parse_bool(row["tau_135_good_intensity_fit"]))
    return (np.asarray(tau_135_records, dtype=float),
            np.asarray(E_records, dtype=float),
            np.asarray(measured_135, dtype=bool))


def _load_tau135_hist(tau_hist_path, records4_path):
    """Load the 135 K tau histogram, rebuilding from the records CSV if the NPZ
    only stores bin edges (cell 16 logic)."""
    tau_hist_path = Path(tau_hist_path)
    with np.load(tau_hist_path) as data:
        tau_edges = data["bin_edges"]
        if "total_taus" in data:
            tau_samples = data["total_taus"]
            tau_hist, _ = np.histogram(tau_samples, bins=tau_edges)
        elif "tau_at_135s" in data:
            tau_samples = data["tau_at_135s"]
            tau_hist, _ = np.histogram(tau_samples, bins=tau_edges)
        else:
            tau_samples = np.array([])
            tau_hist = data["hist"]

    hist_source = tau_hist_path.name
    if tau_samples.size:
        tau_hist, tau_edges = np.histogram(tau_samples, bins=tau_edges)
    elif int(np.sum(tau_hist)) == 0:
        samples = []
        with Path(records4_path).open(newline="") as handle:
            for row in csv.DictReader(handle):
                samples.append(float(row["tau_135_seconds"]))
        tau_samples = np.asarray(samples, dtype=float)
        tau_hist, tau_edges = np.histogram(tau_samples, bins=tau_edges)
        hist_source = f"rebuilt from {Path(records4_path).name} using {tau_hist_path.name} bin edges"
    return tau_hist, tau_edges, tau_samples, hist_source


def plot_completeness_overlay(m3, save=True, log_hist=False):
    """Detection-probability curve overlaid on the measured+extrapolated tau histogram
    (cell 16 figure -> figures/efficiency_completeness.pdf).

    log_hist: if True, the trap-count histogram (right axis) is drawn on a log
    scale and the right y-axis is scaled accordingly."""
    tau_grid = m3['tau_grid']
    default_curve = m3['default_curve']
    tau_hist = m3['tau_hist']
    tau_edges = m3['tau_edges']
    tau_135_records = m3['tau_135_records']
    measured_135 = m3['measured_135']
    extrapolated_135 = ~measured_135

    fig, ax = plt.subplots(figsize=(16 * 0.8, 9 * 0.8))
    ax.plot(tau_grid, default_curve, color="tab:red", linewidth=2.2,
            label="Detection Probability")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("$\\tau_e(135\\,\\mathrm{K})$ [s]")
    ax.set_ylabel("$\\bar{P}(\\mathrm{characterized},E)$", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")

    ax_hist = ax.twinx()
    ax_hist.stairs(tau_hist, tau_edges, color=C_BLUE, linewidth=1.5)
    ax_hist.hist(
        [tau_135_records[extrapolated_135], tau_135_records[measured_135]],
        bins=tau_edges, stacked=True, color=["tab:blue", "tab:green"],
        alpha=0.65, label=["Extrapolated", "Measured"])
    ax_hist.set_ylabel("\\# of Traps", color=C_BLUE)
    ax_hist.tick_params(axis="y", labelcolor=C_BLUE)
    if log_hist:
        ax_hist.set_yscale("log")
        if np.max(tau_hist) > 0:
            ax_hist.set_ylim(0.8, np.max(tau_hist) * 1.5)
    else:
        ax_hist.yaxis.get_major_ticks()[0].set_visible(False)
        if np.max(tau_hist) > 0:
            ax_hist.set_ylim(0, np.max(tau_hist) * 1.15)

    x_min = min(tau_grid[0], tau_edges[0])
    x_max = max(tau_grid[-1], tau_edges[-1])
    ax.set_xlim(x_min, x_max)

    lines, labels = ax.get_legend_handles_labels()
    hist_lines, hist_labels = ax_hist.get_legend_handles_labels()
    ax.legend(lines + hist_lines, labels + hist_labels, loc="upper right", fontsize=small)
    if save:
        ensure_figures_dir()
        plt.savefig(f'{figures_dir()}/efficiency_completeness.pdf', dpi=300)
    plt.show()
    plt.close()


def plot_completeness_map(m3, save=True):
    """2D map of Method-3 completeness over (tau_135, E) with characterized
    traps overlaid (cell 18 -> figures/prob_of_measuring.pdf)."""
    tau_grid = m3['tau_grid']
    E_grid = m3['E_grid']
    p4_map = m3['p4_map']
    known_tau = m3['known_tau']
    known_E = m3['known_E']
    summary = m3['summary']

    fig, ax = plt.subplots(figsize=(16 * 0.8, 9 * 0.8))
    mesh = ax.pcolormesh(tau_grid, E_grid, p4_map.T, shading="auto", cmap="Blues",
                         vmin=0, vmax=1)
    ax.scatter(known_tau, known_E, s=12, c="red", alpha=0.8, label="Characterized Traps")
    ax.set_xscale("log")
    ax.set_xlabel("$\\tau_e(135\\,\\mathrm{K})$ [s]")
    ax.set_ylabel("$E$ [eV]")
    label = (
        "Probability of Characterized Trap"
        if m3.get('paths', {}).get('analysis_flavor') in ('minimal', 'minimal_caldet')
        else "Probability of $\\ge4$ Good Intensity Fits"
    )
    fig.colorbar(mesh, ax=ax, label=label)
    ax.legend(loc="lower right")
    if save:
        ensure_figures_dir()
        plt.savefig(f'{figures_dir()}/prob_of_measuring.pdf', dpi=300)
    plt.show()

    unbounded = summary["unbounded_regime"]["all_temperatures_out_of_stage08_tau_band"]
    print("Unbounded grid fraction:", unbounded["grid_fraction"])
    print("E range:", unbounded["E_eV_min"], "to", unbounded["E_eV_max"], "eV")
    print("tau_135 range:", unbounded["tau_135_seconds_min"], "to",
          unbounded["tau_135_seconds_max"], "s")
    print("max P4 in unbounded regime:", unbounded["p4_max_in_regime"])
    plt.close()


def _caught_temperature_stat(good_temps, scalar):
    """Reduce a trap's good-intensity-fit temperatures to one scalar."""
    if scalar == 'mean':
        return float(np.mean(good_temps))
    if scalar == 'median':
        return float(np.median(good_temps))
    if scalar == 'max':
        return float(np.max(good_temps))
    raise ValueError(f"color_scalar must be 'mean'|'median'|'max', got {scalar!r}")


def _load_characterized_caught_temperature(catalog_h5, analysis_flavor, color_scalar='mean'):
    """Recompute the characterized-trap set straight from the flavor catalog and
    return (tau_135, E, caught_T), matching the stage-09 red-dot selection
    (_passes_final_catalog) and the same SRH tau->135 K extrapolation. caught_T
    is the mean/median/max of each trap's good-intensity-fit temperatures."""
    import dipole as _dip_legacy
    import dipole_new as _dip_minimal
    srh = _dip_minimal if analysis_flavor in ('minimal', 'minimal_caldet') else _dip_legacy
    tau135, E_list, caught = [], [], []
    with h5py.File(catalog_h5, "r") as f:
        for qname, qg in f.items():
            if not isinstance(qg, h5py.Group):
                continue
            for dpname, dg in qg.items():
                if not isinstance(dg, h5py.Group):
                    continue
                a = dg.attrs
                passes = (
                    bool(a.get("WellBehavedTrap", False))
                    and not bool(a.get("EnergyFitFailed", False))
                    and bool(a.get("GoodEnergyFit", False))
                    and bool(a.get("OrientationConsistent", True))
                )
                if not passes or "energy_BestFitEnergy" not in a:
                    continue
                E = float(a["energy_BestFitEnergy"])
                log_sigma = float(np.log(float(a["energy_BestFitCrossSection"])))
                tau_135 = float(np.exp(srh.log_energy_cross_section(135.0, E, log_sigma)))
                good_temps = [int(n.split("_")[1]) for n in dg
                              if n.startswith("temp_") and isinstance(dg[n], h5py.Group)
                              and bool(dg[n].attrs.get("GoodIntensityFit", False))]
                if len(good_temps) < 4:
                    continue
                tau135.append(tau_135)
                E_list.append(E)
                caught.append(_caught_temperature_stat(good_temps, color_scalar))
    return np.asarray(tau135), np.asarray(E_list), np.asarray(caught)


def plot_completeness_map_caught_at_T(m3, color_scalar='mean', cmap=None,
                                      vmin=100, vmax=250, save=True):
    """Variant of plot_completeness_map: characterized traps colored by the
    temperature at which they were actually caught (mean/median/max of their
    good-intensity-fit temperatures).

    The x-axis tau_e(135 K) is an *extrapolation* for long-tau / high-E traps,
    which are far out of window at 135 K and were only detected at high T, where
    their emission time drops into the measurable delay window. Coloring by
    capture temperature shows the apparent low-probability (right) arm is hot --
    i.e. those traps were measured in high-probability regions, not at 135 K.

    Kept separate from plot_completeness_map (the default overlay is unchanged).
    Colormap/normalization default to the example-trap palette (my_cmap, 100-250 K)
    so capture temperature reads consistently across figures."""
    paths = m3['paths']
    tau_grid = m3['tau_grid']
    E_grid = m3['E_grid']
    p4_map = m3['p4_map']
    flavor = paths.get('analysis_flavor', 'legacy')
    catalog_h5 = paths.get('catalog_h5')
    if catalog_h5 is None or not Path(catalog_h5).exists():
        raise FileNotFoundError(
            f"catalog HDF5 not found for flavor {flavor!r}: {catalog_h5}")

    tau135, E, caught = _load_characterized_caught_temperature(
        catalog_h5, flavor, color_scalar)
    known_n = np.asarray(m3.get('known_tau', [])).size
    if known_n and known_n != tau135.size:
        print(f"NOTE: recomputed characterized count ({tau135.size}) != stage-09 "
              f"red-dot count ({known_n}); overlay selection may differ slightly.")
    r = np.corrcoef(np.log10(tau135), caught)[0, 1]
    print(f"characterized traps: {tau135.size}; "
          f"corr(log10 tau_135, {color_scalar} good-T) = {r:+.2f}")

    if cmap is None:
        cmap = my_cmap

    fig, ax = plt.subplots(figsize=(16 * 0.8, 9 * 0.8))
    mesh = ax.pcolormesh(tau_grid, E_grid, p4_map.T, shading="auto", cmap="Blues",
                         vmin=0, vmax=1)
    label = (
        "Probability of Characterized Trap"
        if flavor in ('minimal', 'minimal_caldet')
        else "Probability of $\\geq 4$ Good Intensity Fits"
    )
    fig.colorbar(mesh, ax=ax, label=label, pad=0.02)

    sc = ax.scatter(tau135, E, c=caught, s=22, cmap=cmap,
                    norm=colors.Normalize(vmin=vmin, vmax=vmax),
                    edgecolors="k", linewidths=0.3, alpha=0.9)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label(f"{color_scalar.capitalize()} temperature of "
                 "good intensity fits [K]")

    ax.set_xscale("log")
    ax.set_xlabel("$\\tau_e(135\\,\\mathrm{K})$ [s]")
    ax.set_ylabel("$E$ [eV]")
    proxy = Line2D([0], [0], marker='o', linestyle='None', markersize=8,
                   markerfacecolor='gray', markeredgecolor='k',
                   label="Characterized Traps")
    ax.legend(handles=[proxy], loc="lower right")
    if save:
        ensure_figures_dir()
        plt.savefig(f'{figures_dir()}/prob_of_measuring_caught_at_T_{color_scalar}.pdf',
                    dpi=300)
    plt.show()
    plt.close()


def plot_tau135_energy_scatter(m3):
    """Direct-vs-extrapolated tau_135 in energy space + stacked histogram (cell 19)."""
    tau_135_records = m3['tau_135_records']
    E_records = m3['E_records']
    measured_135 = m3['measured_135']
    extrapolated_135 = ~measured_135
    tau_edges = m3['tau_edges']

    print(f"Direct good 135 K tau fits: {int(measured_135.sum())}")
    print(f"Extrapolated-only tau_135 values: {int(extrapolated_135.sum())}")
    print("Median tau_135 direct 135 K:", np.median(tau_135_records[measured_135]))
    print("Median tau_135 extrapolated-only:", np.median(tau_135_records[extrapolated_135]))

    fig, (ax_scatter, ax_hist) = plt.subplots(
        2, 1, figsize=(16 * 0.8, 2 * 9 * 0.8), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]})
    ax_scatter.scatter(tau_135_records[extrapolated_135], E_records[extrapolated_135],
                       s=12, alpha=0.35, color="tab:blue", label="Extrapolated")
    ax_scatter.scatter(tau_135_records[measured_135], E_records[measured_135],
                       s=16, alpha=0.55, color="tab:green", label="Measured")
    ax_scatter.set_xscale("log")
    ax_scatter.set_ylabel("$E$ [eV]")
    ax_scatter.legend(fontsize=small, loc="best")
    ax_scatter.grid(True, which="both", alpha=0.15)

    ax_hist.hist([tau_135_records[extrapolated_135], tau_135_records[measured_135]],
                 bins=tau_edges, stacked=True, color=["tab:blue", "tab:green"],
                 alpha=0.65, label=["Extrapolated", "Measured"])
    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("$\\tau_e(135\\,\\mathrm{K})$ [s]")
    ax_hist.set_ylabel("\\# of Traps")
    ax_hist.grid(True, which="both", alpha=0.15)
    ax_hist.axvline(3600, label='1 hour', ls='--', color=C_REF)
    ax_hist.axvline(3600 * 24, label='1 day', ls='--', color=C_REF)
    ax_hist.axvline(3600 * 25 * 365.25, label='1 year', ls='--', color=C_REF)
    ax_hist.legend(fontsize=small, loc="best")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_upper_limit_hist(m3, confidence_level=0.90, min_efficiency=0.0,
                          correction_tau_min=1e-4, correction_tau_max=1e10,
                          write=False, write_path=None, save=True):
    """Efficiency-corrected 90% CL upper-limit trap histogram (cell 17).

    Seed-file gating: the NPZ is written only when ``write`` is True (or an
    explicit ``write_path`` is given). When writing without an explicit path,
    the target is taken from the Method-3 bundle's flavor-derived
    ``paths['tau_hist_upper_npz']`` (e.g. ``tau_at_135k_hist_upper_limit.npz``
    for the legacy completeness model, ``tau_at_135k_hist_minimal_caldet_upper_limit.npz``
    for the dipole_new/minimal model), so the saved simulation seed always
    matches the completeness curve that produced it. The figure
    (figures/tau_at_135k_hist_upper_limit.pdf) is always rendered.
    """
    tau_hist = m3['tau_hist']
    tau_edges = m3['tau_edges']
    tau_grid = m3['tau_grid']
    default_curve = m3['default_curve']

    if write_path is None and write:
        write_path = m3['paths']['tau_hist_upper_npz']
        print(f"Upper-limit seed target (flavor='{m3['paths']['analysis_flavor']}'): {write_path}")

    tau_bin_centers = np.sqrt(tau_edges[:-1] * tau_edges[1:])
    # default_curve already includes the energy-fit survival factor
    # (applied once in load_method3; see ENERGY_FIT_SURVIVAL).
    completeness_at_bins = np.interp(np.log10(tau_bin_centers), np.log10(tau_grid),
                                     default_curve, left=np.nan, right=np.nan)
    valid_efficiency = (
        np.isfinite(completeness_at_bins)
        & (completeness_at_bins >= min_efficiency)
        & (tau_bin_centers >= correction_tau_min)
        & (tau_bin_centers <= correction_tau_max)
    )
    print(f"Excluded {np.sum(~valid_efficiency)} bins with completeness below "
          f"{min_efficiency:.0%} or outside the model grid.")

    raw_upper_limits = gamma.ppf(confidence_level, tau_hist + 1)
    corrected_tau_hist = np.full(tau_hist.shape, np.nan, dtype=float)
    corrected_tau_hist[valid_efficiency] = (
        raw_upper_limits[valid_efficiency] / completeness_at_bins[valid_efficiency])

    coord_list = m3['paths']['dipole_coord_list_npz']
    with np.load(coord_list, allow_pickle=True) as _d:
        n_detected = int(sum(len(_d[k]) for k in _d.files))
    print(f'number of total traps after corrections: {np.nansum(corrected_tau_hist)}')
    print(f'scale of total traps after corrections: '
          f'{np.nansum(corrected_tau_hist) / n_detected} '
          f'(n_detected={n_detected} from {coord_list.name})')
    corrected_tau_hist[~valid_efficiency] = 0.0
    assert np.all(np.isfinite(corrected_tau_hist)), "histogram contains NaN/inf"

    if write_path is not None:
        valid_indices = np.flatnonzero(valid_efficiency)
        first, last = valid_indices[0], valid_indices[-1]
        np.savez(write_path,
                 hist=corrected_tau_hist[first:last + 1],
                 bin_edges=tau_edges[first:last + 2])
        print(f"Saved: {write_path}")

    fig, ax = plt.subplots(figsize=(16 * 0.8, 9 * 0.8))
    ax.stairs(raw_upper_limits, tau_edges, color='black', linewidth=1.8, alpha=0.3,
              fill=False, hatch='//', label="90\\% CL UL")
    ax.stairs(tau_hist, tau_edges, color=C_BLUE, linewidth=1.8, fill=True, alpha=0.25,
              label="Characterized Traps")
    ax.stairs(corrected_tau_hist, tau_edges, color="black", linewidth=1.8,
              label="Efficiency-corrected 90\\% CL UL")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(tau_edges[0], tau_edges[-1])
    ax.set_xlabel("$\\tau_e(135\\,\\mathrm{K})$ [s]")
    ax.set_ylabel("Estimated number of traps")
    ax.legend(frameon=False, fontsize=small)
    ax.grid(True, which="both", alpha=0.3)
    if save:
        ensure_figures_dir()
        out = Path(figures_dir()) / "tau_at_135k_hist_upper_limit.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()
    plt.close()
    return corrected_tau_hist


def plot_efficiency_corrected_hist(m3, min_efficiency=0.0,
                                   correction_tau_min=1e-4, correction_tau_max=1e10,
                                   write=False, write_path=None, save=True):
    """Efficiency-corrected (completeness point-estimate) trap histogram.

    Identical to ``plot_upper_limit_hist`` but without the Poisson upper-limit
    inflation: each bin is the characterized count divided by the completeness,
    ``corrected = tau_hist / completeness`` (the Poisson MLE for the true count
    under ``N_obs ~ Poisson(eps * N_true)``), rather than
    ``gamma.ppf(0.90, tau_hist+1) / completeness``. This is the *point estimate*
    of the real trap population, used to seed the zero-dark-current trap-only
    campaign -- NOT a conservative upper limit.

    Seed-file gating mirrors ``plot_upper_limit_hist``: the NPZ is written only
    when ``write`` is True (or an explicit ``write_path`` is given). The default
    target is the bundle's ``paths['tau_hist_upper_npz']`` with ``_upper_limit``
    rewritten to ``_efficiency_corrected`` (e.g.
    ``tau_at_135k_hist_minimal_caldet_efficiency_corrected.npz``), so the saved
    seed always matches the completeness curve that produced it.
    """
    tau_hist = m3['tau_hist']
    tau_edges = m3['tau_edges']
    tau_grid = m3['tau_grid']
    default_curve = m3['default_curve']

    if write_path is None and write:
        write_path = Path(str(m3['paths']['tau_hist_upper_npz']).replace(
            '_upper_limit', '_efficiency_corrected'))
        print(f"Efficiency-corrected seed target (flavor='{m3['paths']['analysis_flavor']}'): {write_path}")

    tau_bin_centers = np.sqrt(tau_edges[:-1] * tau_edges[1:])
    # default_curve already includes the energy-fit survival factor
    # (applied once in load_method3; see ENERGY_FIT_SURVIVAL).
    completeness_at_bins = np.interp(np.log10(tau_bin_centers), np.log10(tau_grid),
                                     default_curve, left=np.nan, right=np.nan)
    valid_efficiency = (
        np.isfinite(completeness_at_bins)
        & (completeness_at_bins >= min_efficiency)
        & (tau_bin_centers >= correction_tau_min)
        & (tau_bin_centers <= correction_tau_max)
    )
    print(f"Excluded {np.sum(~valid_efficiency)} bins with completeness below "
          f"{min_efficiency:.0%} or outside the model grid.")

    point_estimate = tau_hist
    corrected_tau_hist = np.full(tau_hist.shape, np.nan, dtype=float)
    corrected_tau_hist[valid_efficiency] = (
        point_estimate[valid_efficiency] / completeness_at_bins[valid_efficiency])

    coord_list = m3['paths']['dipole_coord_list_npz']
    with np.load(coord_list, allow_pickle=True) as _d:
        n_detected = int(sum(len(_d[k]) for k in _d.files))
    print(f'number of total traps after corrections: {np.nansum(corrected_tau_hist)}')
    print(f'scale of total traps after corrections: '
          f'{np.nansum(corrected_tau_hist) / n_detected} '
          f'(n_detected={n_detected} from {coord_list.name})')
    corrected_tau_hist[~valid_efficiency] = 0.0
    assert np.all(np.isfinite(corrected_tau_hist)), "histogram contains NaN/inf"

    if write_path is not None:
        valid_indices = np.flatnonzero(valid_efficiency)
        first, last = valid_indices[0], valid_indices[-1]
        np.savez(write_path,
                 hist=corrected_tau_hist[first:last + 1],
                 bin_edges=tau_edges[first:last + 2])
        print(f"Saved: {write_path}")

    fig, ax = plt.subplots(figsize=(16 * 0.8, 9 * 0.8))
    ax.stairs(tau_hist, tau_edges, color=C_BLUE, linewidth=1.8, fill=True, alpha=0.25,
              label="Characterized Traps")
    ax.stairs(corrected_tau_hist, tau_edges, color="black", linewidth=1.8,
              label="Efficiency-corrected (point estimate)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(tau_edges[0], tau_edges[-1])
    ax.set_xlabel("$\\tau_e(135\\,\\mathrm{K})$ [s]")
    ax.set_ylabel("Estimated number of traps")
    ax.legend(frameon=False, fontsize=small)
    ax.grid(True, which="both", alpha=0.3)
    if save:
        ensure_figures_dir()
        out = Path(figures_dir()) / "tau_at_135k_hist_efficiency_corrected.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()
    plt.close()
    return corrected_tau_hist


# ---------------------------------------------------------------------------
# 4. Tau histograms by temperature
# ---------------------------------------------------------------------------
def plot_tau_hist_single(tau_temperatures, t=135, save=True,log=True):
    """Stacked measured/extrapolated tau histogram at a single temperature (cell 23)."""
    measured_taus = tau_temperatures[t]['measured']
    extrapolated_taus = tau_temperatures[t]['extrapolated']
    print(len(measured_taus))
    print(len(extrapolated_taus))

    bins = np.geomspace(1e-6, 1e10, 35)
    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.hist(extrapolated_taus, bins, color=C_BLUE, edgecolor='black', alpha=0.8,
            label='Extrapolated')
    ax.hist(measured_taus, bins, color='seagreen', edgecolor='black', alpha=0.8,
            label='Measured')
    ax.set_xlabel('$\\tau_e$ [s]', fontsize=small)
    ax.set_ylabel("\\# of Traps", fontsize=small)
    ax.set_xscale('log')
    if log:
        ax.set_yscale('log')
    plt.axvline(3600, label='1h', ls='--', color=C_REF)
    plt.axvline(3600 * 24, label='1 day', ls='-.', color=C_REF)
    plt.axvline(3600 * 24 * 365, label='1 year', ls=':', color=C_REF)
    ax.legend(frameon=False, loc='upper left')
    ax.minorticks_on()
    if save:
        ensure_figures_dir()
        plt.savefig(f'{figures_dir()}/tau_at_{t}k_hist.pdf', dpi=300)
    plt.show()
    plt.close()


def plot_tau_histograms_by_temperature(tau_temperatures, write_seed=False, save=True):
    """Per-temperature tau histogram + completeness panel for every temperature (cell 24).

    Figures are always written to figures/tau_at_<T>k_hist.pdf. The
    tau_at_<T>k_hist.npz seed files are written only when ``write_seed=True``
    (these are simulation inputs; the 135 K one is also produced by
    run_charge_traps.py, so generation here is off by default).
    """
    if save:
        ensure_figures_dir()
    bins = np.geomspace(1e-7, 1e8, 100)

    for t in tau_temperatures.keys():
        measured_taus = tau_temperatures[t]['measured']
        extrapolated_taus = tau_temperatures[t]['extrapolated']
        total_taus = measured_taus + extrapolated_taus
        if len(total_taus) == 0:
            continue
        print(f"Temperature: {t}K | Measured: {len(measured_taus)} | "
              f"Extrapolated: {len(extrapolated_taus)}")

        fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                                 gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax_hist, ax_eff = axes
        ax_hist.hist([measured_taus, extrapolated_taus], bins=bins, stacked=True,
                     color=['seagreen', C_BLUE], edgecolor='black', alpha=0.8,
                     label=['Measured', 'Extrapolated'])
        ax_hist.set_title(f"Emission $\\tau_e$ of CCD Traps at {t}K", fontsize=20)
        ax_hist.set_ylabel("\\# of Traps", fontsize=16)
        ax_hist.axvline(3600, label='1h', ls='--', color=C_REF)
        ax_hist.axvline(3600 * 24, label='1 day', ls='-.', color=C_REF)
        ax_hist.axvline(3600 * 24 * 365, label='1 year', ls=':', color=C_REF)
        ax_hist.legend(frameon=False, fontsize=16)

        meas_counts, _ = np.histogram(measured_taus, bins=bins)
        extrap_counts, _ = np.histogram(extrapolated_taus, bins=bins)
        tot_counts, _ = np.histogram(total_taus, bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            efficiency = meas_counts / tot_counts
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
        eff_err = np.zeros_like(efficiency)
        valid = tot_counts > 0
        eff_err[valid] = np.sqrt(efficiency[valid] * (1.0 - efficiency[valid]) / tot_counts[valid])

        ax_eff.errorbar(bin_centers[valid], efficiency[valid], yerr=eff_err[valid],
                        fmt='o', color='black', markersize=5, capsize=3)
        ax_eff.set_xlabel('$\\tau_e$ [s]', fontsize=small)
        ax_eff.set_ylabel("Efficiency $\\epsilon$", fontsize=small)
        ax_eff.set_xscale('log')
        ax_eff.set_ylim(-0.1, 1.1)
        ax_eff.axhline(1.0, color='grey', ls=':', alpha=0.5)
        ax_eff.axhline(0.0, color='grey', ls=':', alpha=0.5)
        ax_eff.grid(True, which='both', axis='x', alpha=0.2)
        ax_eff.set_xticks(np.logspace(-8, 8, 17))
        ax_eff.minorticks_on()
        ax_eff.axvline(1e0, ls='--', color=C_REF)
        plt.tight_layout()
        if save:
            plt.savefig(f'{figures_dir()}/tau_at_{t}k_hist.pdf', dpi=300)
        plt.show()
        plt.close()

        if write_seed:
            filename = f'tau_at_{t}k_hist.npz'
            np.savez(filename, hist=tot_counts, bin_edges=bins,
                     measured_taus=measured_taus, extrapolated_taus=extrapolated_taus,
                     total_taus=total_taus, meas_counts=meas_counts,
                     extrap_counts=extrap_counts, efficiency=efficiency, eff_err=eff_err)
            print(f"Saved {filename} for simulation sampling.")


# ---------------------------------------------------------------------------
# 5. Simulation campaign registry + plotting
# ---------------------------------------------------------------------------
# Pre-campaign flat output dirs, registered as named aliases.
LEGACY_SCENARIOS = {
    'minos': './minos_conditions/',
    'minos_upper': './minos_conditions_upperlimit/',
    'snolab': './snolab_conditions/',
    'snolab_upper': './snolab_conditions_upperlimit/',
}


def scenario_dir(condition, population='baseline', vp=3.0, clear='sequencer',
                 order='shuffled', exp_indep='pre_readout', base='campaign',
                 flavor='legacy', binning=1.0, zero_exp_dep=False):
    """Resolve a campaign scenario to its output directory.

    Reuses run_campaign.label_for so the label can never drift from the
    campaign's own naming. ``population`` is 'baseline', 'upper', or 'effcorr'
    (efficiency-corrected point estimate). ``flavor`` ('legacy' / 'minimal_caldet')
    selects the trap catalog: minimal runs append a ``_minimal_caldet`` tag to the
    label, so pass the notebook's PIPELINE here to point at the matching campaign dir.

    ``zero_exp_dep`` mirrors run_campaign's ``--zero-exp-dep`` (zero single-e dark
    current): when True, label_for appends the ``_zedr`` tag, resolving the
    trap-only effcorr campaign dirs.

    ``binning`` is the readout-binning factor (run_campaign --binning-factors).
    The default 1.0 is the unbinned run (no label suffix); a non-default value
    (e.g. 32) appends the ``_bin32`` tag, resolving the binned variants the
    campaign only emits for the standard sequencer clear at the central V_p in
    shuffled order (see run_campaign.binnings_for).
    """
    # The histfile string is only a token carrier for label_for's population
    # detection (it keys on the 'efficiency_corrected'/'upper' substrings); the
    # file itself is not loaded here.
    histfile = {
        'upper': 'tau_at_135k_hist_upper_limit.npz',
        'effcorr': 'tau_at_135k_hist_efficiency_corrected.npz',
    }.get(population, 'tau_at_135k_hist.npz')
    label = label_for(condition, histfile, vp,
                      exp_indep_charge_mode=exp_indep, clear_mode=clear,
                      exposure_order=order, flavor=flavor, binning=binning,
                      zero_exp_dep=zero_exp_dep)
    return os.path.join(base, label) + os.sep


def aggregate_scenario(spec, mask='Halo+Bleed+HotColumn+HotPixel',
                       event_type='1e', vp_values=VP_ORDER):
    """Resolve a scenario spec to its central plot_simulation_results tuple plus
    an optional V_p systematic band.

    ``spec`` is a dict of scenario_dir kwargs (must include ``condition``; may
    include ``population``, ``clear``, ``order``, ``exp_indep``, ``flavor``,
    ``binning``, ``zero_exp_dep``, ``base``) plus an optional ``systematics``
    tuple. If ``'vp'`` is in ``systematics`` the V_p axis is marginalized: the
    central point is the baseline V_p (``VP_BASELINE``) and the band is the
    min/max of the with-traps measured rates (return cols 0=exp_indep,
    1=exp_dep) across whichever V_p dirs exist. V_p is defined as the campaign's
    systematic knob, so this collapses the {1,3,10} scan into one banded point
    instead of three near-identical rows. Without ``'vp'`` in ``systematics`` the
    spec is a single point at its own ``vp`` (default 3) and no band is produced.

    The no-trap branch (cols 2,3) carries no band: it is V_p-independent by
    construction, so a band there would be spurious.

    Missing V_p variants are skipped with a warning; if only the central dir
    exists the band collapses to zero width. Returns ``(None, {})`` if even the
    central dir is absent.

    Returns ``(central_tuple, syst)`` where ``central_tuple`` is the full 10-tuple
    from plot_simulation_results at the central point and ``syst`` maps
    ``{0, 1} -> (lo, hi)`` bounding the ret-col value (empty dict when no band).
    """
    spec = dict(spec)
    systs = spec.pop('systematics', ())
    condition = spec.pop('condition')

    def _run(rundir):
        return plot_simulation_results(rundir, mask=mask, event_type=event_type,
                                       showFit=False, showDensity=False, saveFig=False)

    if 'vp' not in systs:
        rundir = scenario_dir(condition, **spec)
        if not os.path.isdir(rundir):
            print(f"aggregate_scenario: missing results dir:\n  {rundir}")
            return None, {}
        return _run(rundir), {}

    base = {k: v for k, v in spec.items() if k != 'vp'}
    central = None
    band = {0: [], 1: []}
    for vp in vp_values:
        rundir = scenario_dir(condition, vp=vp, **base)
        if not os.path.isdir(rundir):
            if vp == VP_BASELINE:
                print(f"aggregate_scenario: central V_p dir missing, skipping:\n  {rundir}")
                return None, {}
            print(f"aggregate_scenario: skipping missing V_p={vp:g} dir:\n  {rundir}")
            continue
        res = _run(rundir)
        band[0].append(res[0])
        band[1].append(res[1])
        if vp == VP_BASELINE:
            central = res
    syst = {c: (min(band[c]), max(band[c])) for c in (0, 1)}
    return central, syst


def plot_simulation_results(rundir, mask='Halo+Bleed+HotColumn+HotPixel',
                            event_type='1e', showFit=True,showDensity=False, saveFig=True):
    """Aggregate one simulation run dir and fit exposure-(in)dependent rates (cell 38).

    Returns (exp_indep, exp_dep, exp_indep_notraps, exp_dep_notraps, UR_expindep,
    UR_expdep, err_indep, err_dep, err_indep_notraps, err_dep_notraps).
    """
    if event_type not in ['1e', '2e']:
        raise ValueError("event_type must be either '1e' or '2e'")
    count_key = 'counts' if event_type == '1e' else '2e_counts'

    UR_expdep = 4.36e-5 / 24   # e / pix / hour
    UR_expindep = 9.94e-5      # e / pix / image
    colorlist = C_SEQ

    lower, upper = 7e-5, 2e-4
    nbins = 25

    # ``lower``/``upper`` above are only the fallback axis bounds (used for
    # legacy caches that predate adaptive bounds); the compute branch picks the
    # axis from the actual density range so high-occupancy / upper-limit runs,
    # whose densities run well past the old fixed 2e-4 cap, are not clipped.
    def init_histograms(lo, hi):
        return {
            h: Hist(hist.axis.Regular(nbins, lo, hi,
                    name=f"{event_type} Densities Exp {h}h"))
            for h in (0, 4, 6, 10, 20)
        }

    # Empirical per-image-density SEMs (keyed by exposure hour). Populated from
    # the cache (new format) or computed below; None => fall back to Poisson
    # sqrt(N) errors, which under-cover when traps add run-to-run dispersion.
    density_errs_traps = density_errs_notraps = None

    cache_file = os.path.join(rundir, f'aggregated_results_{mask}_{event_type}.json')
    print(cache_file)
    if os.path.exists(cache_file):
        print(f"Loading JSON cached aggregated {event_type} data from {cache_file}...")
        with open(cache_file, 'r') as f:
            cache = json.load(f)

        def restore_keys(d):
            return {int(k): v for k, v in d.items()}

        total_counts_traps = restore_keys(cache['total_counts_traps'])
        total_pix_traps = restore_keys(cache['total_pix_traps'])
        total_counts_notraps = restore_keys(cache['total_counts_notraps'])
        total_pix_notraps = restore_keys(cache['total_pix_notraps'])
        unique_exposures = np.array(cache['unique_exposures'])
        lo = cache.get('hist_lower', lower)
        hi = cache.get('hist_upper', upper)
        histograms = init_histograms(lo, hi)
        histograms_no_traps = init_histograms(lo, hi)
        for k, v in cache['histograms'].items():
            histograms[int(k)][...] = np.array(v)
        for k, v in cache['histograms_no_traps'].items():
            histograms_no_traps[int(k)][...] = np.array(v)
        if 'density_errs_traps' in cache:
            density_errs_traps = restore_keys(cache['density_errs_traps'])
            density_errs_notraps = restore_keys(cache['density_errs_notraps'])
    else:
        print(f"Cache not found. Processing raw HDF5 simulation files for {event_type} events...")
        total_counts_traps = {0: 0, 4: 0, 6: 0, 10: 0, 20: 0}
        total_pix_traps = {0: 0, 4: 0, 6: 0, 10: 0, 20: 0}
        total_counts_notraps = {0: 0, 4: 0, 6: 0, 10: 0, 20: 0}
        total_pix_notraps = {0: 0, 4: 0, 6: 0, 10: 0, 20: 0}
        # Collect the per-image densities so the histogram axis and the error
        # bars can be derived from the actual density distribution rather than
        # assumed (the histograms are built once the full range is known).
        per_img_traps = {0: [], 4: [], 6: [], 10: [], 20: []}
        per_img_notraps = {0: [], 4: [], 6: [], 10: [], 20: []}

        target_files = [f for f in os.listdir(rundir)
                        if f.startswith('ccd_traps_run') and f.endswith('.h5')]
        num_images = len(target_files)

        for i in range(num_images):
            filepath = os.path.join(rundir, f'ccd_traps_run{i}.h5')
            with h5py.File(filepath, 'r') as f:
                exposures_array = f['exposures'][:]
                unique_exposures = np.unique(np.sort(exposures_array))
                for e in range(len(unique_exposures)):
                    exp = unique_exposures[e]
                    expkey = int(exp / 3600)
                    exp_indices = exposures_array == exp
                    counts = f['stats_trap'][mask][count_key][:][exp_indices]
                    counts_notraps = f['stats_notrap'][mask][count_key][:][exp_indices]
                    if mask == 'None':
                        unmasked_pix = unmasked_pix_notraps = int(1024 / 2) * int(6144 / 2)
                        counts_masked = counts
                        counts_masked_notraps = counts_notraps
                    else:
                        counts_masked = f['stats_trap'][mask][count_key][:][exp_indices]
                        unmasked_pix = f['stats_trap'][mask]['unmasked_pix'][:][exp_indices]
                        counts_masked_notraps = f['stats_notrap'][mask][count_key][:][exp_indices]
                        unmasked_pix_notraps = f['stats_notrap'][mask]['unmasked_pix'][:][exp_indices]
                    densities_exp = counts_masked / unmasked_pix
                    densities_exp_notraps = counts_masked_notraps / unmasked_pix_notraps
                    total_counts_traps[expkey] += np.sum(counts_masked)
                    total_pix_traps[expkey] += np.sum(unmasked_pix)
                    total_counts_notraps[expkey] += np.sum(counts_masked_notraps)
                    total_pix_notraps[expkey] += np.sum(unmasked_pix_notraps)
                    per_img_traps[expkey].append(densities_exp)
                    per_img_notraps[expkey].append(densities_exp_notraps)

        # Concatenate the per-image densities collected per exposure.
        per_img_traps = {k: (np.concatenate(v) if v else np.array([]))
                         for k, v in per_img_traps.items()}
        per_img_notraps = {k: (np.concatenate(v) if v else np.array([]))
                           for k, v in per_img_notraps.items()}

        # Adaptive shared axis spanning both branches and all exposures (padded),
        # so upper-limit runs are not clipped at the legacy fixed bounds.
        alld = [d for d in list(per_img_traps.values())
                + list(per_img_notraps.values()) if d.size]
        alld = np.concatenate(alld) if alld else np.array([])
        if alld.size:
            dlo, dhi = float(alld.min()), float(alld.max())
            pad = 0.05 * (dhi - dlo) if dhi > dlo else max(abs(dhi), 1e-9)
            lo, hi = dlo - pad, dhi + pad
        else:
            lo, hi = lower, upper
        histograms = init_histograms(lo, hi)
        histograms_no_traps = init_histograms(lo, hi)
        for k in histograms:
            if per_img_traps[k].size:
                histograms[k].fill(per_img_traps[k])
            if per_img_notraps[k].size:
                histograms_no_traps[k].fill(per_img_notraps[k])

        # Error on the mean density = empirical SEM of the per-image densities,
        # which captures both shot noise and trap-realization dispersion.
        def _sem(d):
            return float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
        density_errs_traps = {k: _sem(per_img_traps[k]) for k in per_img_traps}
        density_errs_notraps = {k: _sem(per_img_notraps[k]) for k in per_img_notraps}

        print("Saving aggregated data to JSON cache...")
        cache_data = {
            'histograms': {str(k): h.values() for k, h in histograms.items()},
            'histograms_no_traps': {str(k): h.values() for k, h in histograms_no_traps.items()},
            'total_counts_traps': total_counts_traps,
            'total_pix_traps': total_pix_traps,
            'total_counts_notraps': total_counts_notraps,
            'total_pix_notraps': total_pix_notraps,
            'density_errs_traps': density_errs_traps,
            'density_errs_notraps': density_errs_notraps,
            'hist_lower': lo,
            'hist_upper': hi,
            'unique_exposures': unique_exposures,
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, cls=NumpyEncoder, indent=4)

    if showDensity:
        fig, ax = plt.subplots(2, figsize=(12, 12))
        for i in range(2):
            hists = histograms if i == 0 else histograms_no_traps
            t = 'with' if i == 0 else 'without'
            for j, h in enumerate([0, 4, 6, 10, 20]):
                hists[h].plot1d(ax=ax[i], color=colorlist[j], label=f'{h}h')
            ax[i].set_title(f"{t} traps")
            ax[i].set_ylabel("Normalized Counts")
            ax[i].set_xlabel(f"{event_type} Densities")
            ax[i].legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    unique_exposures_h = (unique_exposures / 3600).astype(int)

    average_densities, average_density_errs = [], []
    average_densities_notraps, average_density_errs_notraps = [], []
    for h in unique_exposures_h:
        average_densities.append(total_counts_traps[h] / total_pix_traps[h])
        average_densities_notraps.append(total_counts_notraps[h] / total_pix_notraps[h])
        if density_errs_traps is not None:
            # Empirical SEM: captures shot noise + trap-realization dispersion.
            # Poisson sqrt(N) under-covers the high-occupancy/upper-limit runs by
            # ~20-30% (it equals the empirical SEM only when traps add no extra
            # variance, e.g. the low-occupancy baseline runs).
            average_density_errs.append(density_errs_traps[h])
            average_density_errs_notraps.append(density_errs_notraps[h])
        else:
            # Legacy cache without stored SEMs: fall back to Poisson errors.
            average_density_errs.append(np.sqrt(total_counts_traps[h]) / total_pix_traps[h])
            average_density_errs_notraps.append(np.sqrt(total_counts_notraps[h]) / total_pix_notraps[h])

    average_densities = np.array(average_densities)
    average_density_errs = np.array(average_density_errs)
    average_densities_notraps = np.array(average_densities_notraps)
    average_density_errs_notraps = np.array(average_density_errs_notraps)

    normalized_rundir = rundir.replace('\\', '/').lower()
    fit_mask = np.ones_like(unique_exposures_h, dtype=bool)
    if 'bin0h' in normalized_rundir or 'binned_0h' in normalized_rundir:
        fit_mask = unique_exposures_h != 0

    # 1e density is linear in exposure: r(t) = m*t + b. 2e events are accidental
    # pileup of that 1e charge, so the 2e density follows the *same* (m, b) but
    # quadratically: matching the sim's 8-connectivity 2e clustering, each pixel
    # contributes same-pixel doubles (Poisson r^2/2) plus ~4 adjacent pairs
    # (1 horizontal + 1 vertical + 2 diagonal), each ~r^2 -> factor 4.5. Fitting
    # this form recovers the underlying 1e (m, b) through the pileup, directly
    # comparable to the injected 1e truth (UR_expdep, UR_expindep).
    if event_type == '2e':
        PILEUP_FACTOR = 4.5
        def fit_model(t, m, b):
            return PILEUP_FACTOR * (m * t + b) ** 2
        p0 = (UR_expdep, UR_expindep)
    else:
        fit_model = linear_func
        p0 = None

    popt, pcov = curve_fit(fit_model, unique_exposures_h[fit_mask],
                           average_densities[fit_mask],
                           sigma=average_density_errs[fit_mask],
                           absolute_sigma=True, p0=p0)
    exp_dep, exp_indep = popt
    errors = np.sqrt(np.diag(pcov))
    err_dep, err_indep = errors[0], errors[1]

    popt, pcov = curve_fit(fit_model, unique_exposures_h[fit_mask],
                           average_densities_notraps[fit_mask],
                           sigma=average_density_errs_notraps[fit_mask],
                           absolute_sigma=True, p0=p0)
    errors_notraps = np.sqrt(np.diag(pcov))
    err_dep_notraps, err_indep_notraps = errors_notraps[0], errors_notraps[1]
    exp_dep_notraps, exp_indep_notraps = popt

    hours = np.linspace(0, 20, 20)
    fit_no_traps = fit_model(hours, exp_dep_notraps, exp_indep_notraps)
    fit_traps = fit_model(hours, exp_dep, exp_indep)

    fig, axes = plt.subplots(2, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})
    ax = axes[0]
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.plot(hours, fit_traps, color=C_RED, alpha=0.4)
    ax.plot(hours, fit_no_traps, color=C_BLUE, alpha=0.4)
    ax.errorbar(unique_exposures_h, average_densities, yerr=average_density_errs,
                label='with Traps', color=C_RED, ls='None', marker='o', markersize=5, capsize=1)
    ax.errorbar(unique_exposures_h, average_densities_notraps, yerr=average_density_errs_notraps,
                label='without Traps', color=C_BLUE, ls='None', marker='o', markersize=5, capsize=1)
    ax.set_ylabel(f"{event_type} Density [counts/unmasked pix]")
    ax.set_xlabel("Time [hours]")

    # For 2e, fit_model is the quadratic pileup curve, so the truth line is the
    # pileup prediction of the injected 1e truth (same form as the fit); for 1e
    # it is the usual linear truth. Truth scalars are the injected 1e rates.
    true_rate = fit_model(hours, UR_expdep, UR_expindep)
    truth_label = 'Pileup truth' if event_type == '2e' else 'Truth'
    ax.plot(hours, true_rate, label=truth_label, color='black', ls='--')

    if event_type == '2e':
        # Report the pileup coefficients of the fitted 4.5*(m*t + b)^2 curve so
        # the text matches the plotted 2e quantities: the exposure-independent
        # floor 4.5*b^2 (value at t=0, counts/pix) and the pure exposure-dependent
        # term 4.5*m^2 (coeff of t^2, counts/pix/hr^2).
        text_items = [
            (0.30, f'Input 2e Exp-Indep (4.5b²):  {format_sci(PILEUP_FACTOR * UR_expindep ** 2)}', 'black'),
            (0.26, f'Input 2e Exp-Dep² (4.5m²):  {format_sci(PILEUP_FACTOR * UR_expdep ** 2)}', 'black'),
            (0.20, f'2e Exp-Indep with Traps:  {format_sci(PILEUP_FACTOR * exp_indep ** 2)}', C_RED),
            (0.16, f'2e Exp-Dep² with Traps:  {format_sci(PILEUP_FACTOR * exp_dep ** 2)}', C_RED),
            (0.10, f'2e Exp-Indep without Traps:  {format_sci(PILEUP_FACTOR * exp_indep_notraps ** 2)}', C_BLUE),
            (0.06, f'2e Exp-Dep² without Traps:  {format_sci(PILEUP_FACTOR * exp_dep_notraps ** 2)}', C_BLUE),
        ]
    else:
        text_items = [
            (0.30, f'Input Exposure Independent:  {format_sci(UR_expindep)}', 'black'),
            (0.26, f'Input Exposure Dependent:  {format_sci(UR_expdep)}', 'black'),
            (0.20, f'Exposure Independent with Traps:  {format_sci(exp_indep)}', C_RED),
            (0.16, f'Exposure Dependent with Traps:  {format_sci(exp_dep)}', C_RED),
            (0.10, f'Exposure Independent without Traps:  {format_sci(exp_indep_notraps)}', C_BLUE),
            (0.06, f'Exposure Dependent without Traps:  {format_sci(exp_dep_notraps)}', C_BLUE),
        ]
    for y, s, c in text_items:
        ax.text(x=0.99, y=y, s=s, transform=ax.transAxes, horizontalalignment='right',
                verticalalignment='center', fontsize='small', color=c)
    ax.legend(frameon=False)

    diff = average_densities - average_densities_notraps
    differr = np.sqrt(average_density_errs ** 2 + average_density_errs_notraps ** 2)
    axes[1].errorbar(unique_exposures_h, diff, yerr=differr, color='black', ls="None",
                     marker='o', capsize=3)
    axes[1].set_ylabel("Difference")
    axes[1].axhline(0, ls='dashed', color=C_REF)
    axes[1].minorticks_on()
    axes[0].minorticks_on()
    axes[0].set_xticks(np.arange(0, 22, 2))
    plt.tight_layout()

    runname = rundir.replace('/', '').replace('\\', '').replace(".", '')
    # Self-route the save to the flavor that seeded this run (inferred from the
    # rundir path), so figures land correctly even across mixed comparisons and
    # regardless of the global use_flavor() toggle.
    out_dir = figure_dir_for_flavor(flavor_from_rundir(rundir))
    os.makedirs(out_dir, exist_ok=True)
    if saveFig:
        plt.savefig(f'{out_dir}/{runname}_{event_type}_sim_results.pdf', dpi=300)
    if showFit:
        plt.show()
    plt.close()

    return (exp_indep, exp_dep, exp_indep_notraps, exp_dep_notraps, UR_expindep, UR_expdep,
            err_indep, err_dep, err_indep_notraps, err_dep_notraps)


def compare_scenarios(scenarios, mask='Halo+Bleed+HotColumn+HotPixel',
                      event_type='1e', save_as=None):
    """Forest plot comparing exposure-(in)dependent rates across labelled scenarios.

    ``scenarios`` is a list of (display_label, entry). Each ``entry`` is either a
    rundir string (single point, as before) or a scenario spec dict passed to
    ``aggregate_scenario`` (scenario_dir kwargs + optional ``systematics=('vp',)``
    to fold the V_p scan into an asymmetric band). Generalizes the old
    MINOS-vs-SNOLAB forest plot (cell 44) to an arbitrary set of campaign
    scenarios, so any campaign axis can be overlaid.

    When a scenario carries a V_p band it is drawn as a lighter outer error bar
    on the with-traps (red) point, distinct from the inner statistical SEM cap;
    the no-trap (blue) point is never banded (it is V_p-independent).
    Returns the per-scenario central results array (rows align with ``scenarios``).
    """
    labels = [s[0] for s in scenarios]
    ul_present = False
    for s in scenarios:
        if 'ul' in s[0].lower(): ul_present=True
    results = []
    systs = []      # per-scenario V_p band: {0,1}->(lo,hi), empty if none
    rundirs = []    # representative (central) rundir per scenario, for flavor routing
    for _, entry in scenarios:
        if isinstance(entry, dict):
            ctuple, syst = aggregate_scenario(entry, mask=mask, event_type=event_type)
            if ctuple is None:
                raise FileNotFoundError(
                    f"compare_scenarios: no central results for spec {entry}")
            spec = {k: v for k, v in entry.items() if k != 'systematics'}
            cond = spec.pop('condition')
            rundirs.append(scenario_dir(cond, **spec))
            results.append(ctuple)
            systs.append(syst)
        else:
            results.append(plot_simulation_results(entry, mask=mask, event_type=event_type,
                                                    showFit=False, showDensity=False, saveFig=False))
            systs.append({})
            rundirs.append(entry)
    res = np.array(results)
    y_pos = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(12, max(8, 4 * len(labels))), sharey=True)
    fig.subplots_adjust(hspace=0.3)

    def panel(ax, col_indep, col_notraps, col_truth, col_err_traps, col_err_notraps,
              title, xlabel):
        scale = 24 if 'Exposure Dependent' in title else 1
        # V_p systematic: lighter, wider outer bar under the stat cap on the red
        # with-traps point. col_indep is 0 (exp-indep) / 1 (exp-dep), matching the
        # syst dict keys from aggregate_scenario. Drawn first so the stat cap sits
        # on top. Asymmetric about the central value to preserve the ~monotonic
        # V_p response. Also collected into syst_reach for the axis window.
        syst_reach = 0.0
        band_labelled = False
        for i, syst in enumerate(systs):
            if col_indep not in syst:
                continue
            central = res[i, col_indep] * scale
            lo, hi = syst[col_indep][0] * scale, syst[col_indep][1] * scale
            xerr = [[max(central - lo, 0.0)], [max(hi - central, 0.0)]]
            ax.errorbar(central, y_pos[i] - 0.12, xerr=xerr, fmt='none', ecolor=C_RED,
                        alpha=0.45, elinewidth=5, capsize=0,
                        label=None if band_labelled else r'$V_p$ systematic')
            band_labelled = True
            truth_i = res[0, col_truth] * scale
            syst_reach = max(syst_reach, abs(lo - truth_i), abs(hi - truth_i))
        ax.errorbar(res[:, col_indep] * scale, y_pos - 0.12, xerr=res[:, col_err_traps] * scale,
                    fmt='o', color=C_RED, linestyle='none', label='With Traps',
                    markersize=8, capsize=4)
        ax.errorbar(res[:, col_notraps] * scale, y_pos + 0.12, xerr=res[:, col_err_notraps] * scale,
                    fmt='s', color=C_BLUE, linestyle='none', label='Without Traps',
                    markersize=8, capsize=4)
        truth_val = res[0, col_truth] * scale
        ax.axvline(x=truth_val, color='black', linestyle='--', linewidth=1.5,
                   label='Injected Truth')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=small)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=small, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=small)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, axis='y', linestyle='-', alpha=0.2)
        ax.grid(True, axis='x', linestyle=':', alpha=0.5)
        if ul_present:
            ax.set_xlim(truth_val-7e-5,truth_val + 7e-5)
        else:
            # Data-driven window centered on the truth line so deviations are
            # directly comparable: half-width is the farthest point (including
            # its error bar) from truth, plus padding. Adapts per panel instead
            # of a fixed ad-hoc width, which over-zoomed the exp-indep axis.
            pts = np.concatenate([res[:, col_indep], res[:, col_notraps]]) * scale
            errs = np.concatenate([res[:, col_err_traps], res[:, col_err_notraps]]) * scale
            reach = np.max(np.abs(pts - truth_val) + errs)
            reach = max(reach, syst_reach)  # keep the V_p band inside the window
            half = 1.15 * max(reach, 1e-12)
            ax.set_xlim(truth_val - half, truth_val + half)

    panel(axes[0], 0, 2, 4, 6, 8, "Exposure Independent Rate",
          r"Density [$e^-$/superpix/image]")
    panel(axes[1], 1, 3, 5, 7, 9, "Exposure Dependent Rate", r"Rate [$e^-$/pix/day]")

    handles, plot_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, plot_labels, loc='center left', ncol=1, fontsize=small, frameon=False)
    axes[1].legend(handles, plot_labels, loc='center left', ncol=1, fontsize=small, frameon=False)

    if save_as is None:
        # Route to the flavor shared by all scenarios (inferred from their
        # rundirs); if they disagree, fall back to the global use_flavor() dir.
        flavors = {flavor_from_rundir(rundir) for rundir in rundirs}
        out_dir = figure_dir_for_flavor(flavors.pop()) if len(flavors) == 1 else figures_dir()
        os.makedirs(out_dir, exist_ok=True)
        save_as = f'{out_dir}/results_comparison.pdf'
    if save_as:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        plt.savefig(save_as, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return res
