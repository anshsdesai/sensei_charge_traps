"""Regenerate the static figures embedded in the notebook/*.md files.

Each function reads a real analysis cache (named in its docstring), draws one
figure, and writes an SVG into this directory. Run from the repo root with the
project conda env:

    conda run -n sensei_charge_traps_new python notebook/figures/make_figures.py

The functions are deliberately small and self-contained: the cache path and the
numbers each figure illustrates are stated inline, so this script doubles as the
provenance record for the figures. If we later promote the notebook to Quarto,
these same functions become the executable code cells.
"""
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# Minimal + calibrated-detection per-temperature fit cache (the production catalog).
MINIMAL_FIT = os.path.join(ROOT, "fit_dipole_spectra_minimal_caldet_err_4.h5")

# The per-temperature delta-chi2 threshold table the production catalog actually
# used (matches the delta_chi2_threshold attr stored in every temp group).
DETECTION_CALIB = os.path.join(ROOT, "detection_calibration_minimal.npz")

# The minimal error model's baseline (temporal) noise floor, sigma_base(T, quadrant).
PAIR_NOISE = os.path.join(ROOT, "pair_noise_table_minimal.npz")

# Method-3 completeness caches (the production minimal_caldet flavor).
M3_CACHE = os.path.join(ROOT, "trap_completeness_method3", "cache")
# Stage 03 -- trap-free spatial patch-sigma noise map (legacy convention).
STAGE03_NOISE = os.path.join(M3_CACHE, "03_noise_map_v1.h5")
# Stage 05 -- amplitude prior: trap depths + P_c(T), rebuilt from the dipole_new catalog.
STAGE05_PRIOR = os.path.join(M3_CACHE, "05_amplitude_prior_minimal_caldet_v1.npz")

# All temperatures present in the catalog, in Kelvin.
TEMPS = [125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180,
         183, 185, 187, 190, 193, 195, 197, 200, 203, 207, 210]


def _style():
    plt.rcParams.update({
        "text.usetex": False,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def _pumped(t, coeff, tau, offset):
    """The minimal 3-knob curve: signed pumped amplitude + constant pedestal."""
    return 3000.0 * coeff * (np.exp(-t / tau) - np.exp(-8.0 * t / tau)) + offset


def _pumped_no_offset(t, coeff, tau):
    """The 2-knob curve with no pedestal term (offset forced to 0)."""
    return 3000.0 * coeff * (np.exp(-t / tau) - np.exp(-8.0 * t / tau))



def _add_track(image, points, charge=180.0, radius=1):
    """Paint a compact high-energy cluster/track into an integer charge image."""
    rows, cols = image.shape
    offsets = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr = dr * dr + dc * dc
            if rr <= radius * radius:
                offsets.append((dr, dc, np.exp(-0.5 * rr / max(radius, 1))))

    for r, c in points:
        ir = int(round(r))
        ic = int(round(c))
        for dr, dc, weight in offsets:
            rr, cc = ir + dr, ic + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                image[rr, cc] += charge * weight


def _synthetic_fake_image(seed=14, shape=(300, 330)):
    """Small deterministic fake exposure with 1e dots and high-energy clusters."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    image = np.zeros(shape, dtype=np.float64)

    # Sparse isolated 1e events. These are intentionally numerous enough to be
    # visible in a document figure after the nonlinear stretch below.
    n_single = int(0.010 * rows * cols)
    rr = rng.integers(0, rows, n_single)
    cc = rng.integers(0, cols, n_single)
    np.add.at(image, (rr, cc), 1.0)

    # A few deterministic high-energy events: compact blobs plus curved tracks.
    t = np.linspace(0, 1, 115)
    _add_track(
        image,
        zip(rows * (0.72 + 0.10 * t),
            cols * (0.20 + 0.32 * t) + 11 * np.sin(2.8 * np.pi * t)),
        charge=120,
        radius=2,
    )
    t = np.linspace(0, 1, 55)
    _add_track(
        image,
        zip(rows * (0.33 + 0.07 * t + 0.020 * np.sin(4 * np.pi * t)),
            cols * (0.36 + 0.06 * t)),
        charge=95,
        radius=2,
    )
    for r, c, amp, rad in (
            (0.16 * rows, 0.61 * cols, 230, 2),
            (0.51 * rows, 0.54 * cols, 175, 2),
            (0.76 * rows, 0.73 * cols, 210, 2),
            (0.27 * rows, 0.84 * cols, 125, 1),
            (0.82 * rows, 0.09 * cols, 155, 2)):
        theta = np.linspace(0, 2 * np.pi, 28, endpoint=False)
        blob = [(r + rad * np.sin(a), c + rad * np.cos(a)) for a in theta]
        blob.append((r, c))
        _add_track(image, blob, charge=amp, radius=1)

    # Faint horizontal concentrations mimic rows where trap-deferred and isolated
    # events are visually easiest to spot in a full-frame image.
    for frac in (0.15, 0.38, 0.70):
        r0 = int(frac * rows)
        n = max(18, cols // 6)
        cc = rng.choice(cols, size=n, replace=False)
        image[r0 + rng.integers(-1, 2, n), cc] += 1.0

    return image


def _asinh_stretch(image, vmax=45.0, soft=0.28, one_e_floor=0.48):
    """Display stretch that keeps true 1e pixels visible next to HEE cores."""
    stretched = np.clip(np.arcsinh(np.clip(image, 0, None) / soft)
                       / np.arcsinh(vmax / soft), 0, 1)
    single_e = (image >= 0.75) & (image < 2.5)
    return np.maximum(stretched, one_e_floor * single_e)


def simulation_fake_image(out="simulation_fake_image.png", save=True):
    """A notebook-ready fake exposure with single-e and high-energy events.

    Source: deterministic synthetic event field, not a production campaign HDF5.
    The asinh display stretch resolves isolated 1e pixels while allowing the
    high-energy tracks to saturate, matching the visual job of the lab-notebook
    schematic rather than an analysis threshold.
    """
    _style()
    img = _synthetic_fake_image()
    fig, ax = plt.subplots(figsize=(6.2, 6.0), facecolor="white")
    ax.imshow(_asinh_stretch(img), cmap="gray", origin="lower",
              interpolation="nearest")
    ax.set_title("Simulated image", fontsize=22, fontfamily="serif", pad=18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.4)
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path, dpi=220)
        print(f"wrote {path}")
    return fig


def simulation_source_image(out="simulation_source_image.png", save=True):
    """The real MINOS source image used for high-energy cluster transplants.

    Source: minos_image/proc_corr_proc_skp_72000secs_exp_run10_NSAMP_300_36.fits,
    quadrant 0, electronized the same way `CCD.take_fake_image` does before
    calling `transplant_clusters`. This is a reference/background image, not a
    simulated output panel.
    """
    _style()
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from utils import get_qdata, approximate_electronize

    source = os.path.join(
        ROOT, "minos_image",
        "proc_corr_proc_skp_72000secs_exp_run10_NSAMP_300_36.fits",
    )
    image = approximate_electronize(get_qdata(source, 0), 400).T.astype(float)

    fig, ax = plt.subplots(figsize=(11.5, 3.2), facecolor="white")
    ax.imshow(_asinh_stretch(image, vmax=90, soft=0.35, one_e_floor=0.0),
              cmap="gray", origin="upper", interpolation="nearest")
    ax.set_title("MINOS source image for high-energy-event transplants", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.3)
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path, dpi=220)
        print(f"wrote {path}")
    return fig


def _synthetic_condition_image(condition, hours, seed, shape=(260, 320)):
    """Visual fake image for a run condition/exposure panel.

    This is a deterministic display example, not a campaign output. It keeps the
    qualitative simulation rules: exposure-dependent single-e density grows with
    time, and SNOLAB has 10x fewer high-energy events than MINOS.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape
    image = np.zeros(shape, dtype=np.float64)

    # True 1e pixels at the UR per-pixel rates used in the simulation docs:
    # exposure-independent 9.94e-5 e/pix/image plus 4.36e-5 e/pix/day.
    one_e_rate = 9.94e-5 + 4.36e-5 * hours / 24.0
    n_single = rng.poisson(one_e_rate * rows * cols)
    rr = rng.integers(0, rows, n_single)
    cc = rng.integers(0, cols, n_single)
    np.add.at(image, (rr, cc), 1.0)

    # High-energy events: scale with exposure, and SNOLAB is 10x lower. We keep a
    # small deterministic floor so the SNOLAB 20 h panel still has an example HEE.
    minos_expected = 1.0 + 0.38 * hours
    hee_scale = 1.0 if condition == "minos" else 0.10
    n_hee = int(round(minos_expected * hee_scale))
    if condition == "snolab" and hours >= 20:
        n_hee = max(n_hee, 1)

    for _ in range(n_hee):
        length = rng.integers(18, 72)
        theta = rng.uniform(0, 2 * np.pi)
        curve = rng.normal(0, 0.10)
        r0 = rng.uniform(25, rows - 35)
        c0 = rng.uniform(25, cols - 35)
        t = np.linspace(-0.5, 0.5, length)
        rr = r0 + length * 0.42 * t * np.sin(theta) + curve * length * t * t
        cc = c0 + length * 0.42 * t * np.cos(theta) - curve * length * t * t
        charge = rng.uniform(85, 180)
        radius = int(rng.integers(1, 3))
        _add_track(image, zip(rr, cc), charge=charge, radius=radius)

    return image


def simulation_condition_grid(out="simulation_condition_grid.png", save=True):
    """Four-panel fake-image example: MINOS/SNOLAB at 4 h and 20 h.

    Source: deterministic synthetic display images. The panels encode the same
    qualitative condition switch used by `take_fake_image`: SNOLAB gets 10x fewer
    high-energy events than MINOS, while the low-energy single-e population grows
    with exposure time. Intended for visual orientation, not a rate measurement.
    """
    _style()
    specs = [
        ("minos", 4, 104), ("minos", 20, 120),
        ("snolab", 4, 204), ("snolab", 20, 220),
    ]
    images = [_synthetic_condition_image(c, h, s) for c, h, s in specs]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), sharex=True, sharey=True)
    for ax, image, (condition, hours, _) in zip(axes.ravel(), images, specs):
        ax.imshow(_asinh_stretch(image, vmax=55, soft=0.24, one_e_floor=0.56),
                  cmap="gray", origin="upper", interpolation="nearest")
        ax.set_title(f"{condition.upper()}, {hours} h", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Fake images by exposure and background condition", y=0.98, fontsize=14)
    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path, dpi=220)
        print(f"wrote {path}")
    return fig


def simulation_trap_effect(out="simulation_trap_effect.png", save=True):
    """Zoomed proof image where a high-energy event fills visible traps.

    Source: deterministic synthetic high-energy event passed through
    ccd_simulation.fast_readout_numba. This is not the production trap
    distribution: selected traps with visible capture probability are placed in
    the high-energy event's readout path so they fill from that event and emit a
    delayed single-electron trail. The frame is zoomed and displayed with a
    boosted 1e stretch so the trail is visible by eye.
    """
    _style()
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from ccd_simulation import fast_readout_numba, seed_numba

    rows, cols = 140, 180
    no_trap = np.zeros((rows, cols), dtype=np.float64)
    tpix_vertical = 49.10
    rng = np.random.default_rng(52)

    # Sparse ambient 1e pixels for scale. These remain true 1e events; the
    # display stretch, not the data, makes them easy to see.
    ambient = 120
    rr = rng.integers(0, rows, ambient)
    cc = rng.integers(0, cols, ambient)
    np.add.at(no_trap, (rr, cc), 1.0)

    # One compact high-energy event. It is the source packet that fills the
    # selected traps; no isolated 1e probe row is injected for the trap effect.
    he_rows = 82 + np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    he_cols = 72 + np.array([0, 1, 2, 3, 5, 7, 9, 11, 12, 13])
    core_points = []
    for r, c in zip(he_rows, he_cols):
        core_points.append((r, c))
    _add_track(no_trap, core_points, charge=240, radius=3)
    _add_track(no_trap, [(91, 88), (92, 90), (93, 93), (94, 96)],
               charge=160, radius=2)

    # Selected traps downstream of the high-energy packet. Their columns overlap
    # the event footprint; their row positions and tau values make the emitted
    # charge land above the event in the displayed image.
    trap_rows = []
    trap_cols = []
    for c in range(72, 88, 2):
        for r in range(96, 134, 3):
            trap_rows.append(r)
            trap_cols.append(c)
    trap_rows = np.array(trap_rows, dtype=np.int64)
    trap_cols = np.array(trap_cols, dtype=np.int64)
    n_traps = trap_rows.size

    # tau_e is chosen to be comparable to a few row dwells: long enough to defer
    # charge away from the HEE core, short enough that many traps emit during the
    # same readout. V1-like phase choice avoids same-step recapture in this
    # didactic example, so the HEE-sourced trail is cleanly visible.
    tau = np.exp(rng.uniform(np.log(1.5 * tpix_vertical),
                             np.log(7.0 * tpix_vertical), n_traps))
    emit_probs = (1.0 - np.exp(-tpix_vertical / tau)).astype(np.float64)
    capture_alpha = np.full(n_traps, 4.0, dtype=np.float64)
    trap_is_v3 = np.zeros(n_traps, dtype=np.uint8)
    trapped = np.zeros(n_traps, dtype=np.float64)

    seed_numba(5207)
    exp_acc = np.zeros_like(no_trap)
    out_flat = fast_readout_numba(
        no_trap.copy(), exp_acc, tpix_vertical, trap_rows, trap_cols,
        emit_probs, capture_alpha, trap_is_v3, trapped,
    )
    with_traps = np.flipud(np.fliplr(out_flat.reshape(rows, cols)))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True, sharey=True)
    for ax, data, title in (
            (axes[0], no_trap, "No traps"),
            (axes[1], with_traps, "With selected traps")):
        ax.imshow(_asinh_stretch(data, vmax=55, soft=0.22, one_e_floor=0.58),
                  cmap="gray", origin="upper", interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    from matplotlib.patches import Rectangle
    for ax in axes:
        ax.add_patch(Rectangle((66, 76), 38, 26, fill=False, ec="#e6c229",
                               lw=1.2, alpha=0.9))
    axes[1].annotate(
        "HEE-filled traps\nemit delayed 1e trail",
        xy=(82, 55), xytext=(108, 38), color="#f0f0f0", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#f0f0f0", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="0.6", alpha=0.65),
    )
    axes[0].text(0.03, 0.95, "high-energy event only", transform=axes[0].transAxes,
                 va="top", fontsize=9, color="black",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.75", alpha=0.85))
    axes[1].text(0.03, 0.95, "same event + trap trail", transform=axes[1].transAxes,
                 va="top", fontsize=9, color="black",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.75", alpha=0.85))

    fig.suptitle("High-energy event filling traps: visible deferred-charge trail",
                 y=0.98, fontsize=13)
    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path, dpi=220)
        print(f"wrote {path}")
    return fig


def pedestal(out="pedestal.svg",
             quad=2, trap="dp_475_269", temp=170, save=True):
    """The readout pedestal: what it is (panel A) and how it grows (panel B).

    Source: fit_dipole_spectra_minimal_caldet_err_4.h5 (well-behaved traps).

    Panel A -- one representative hot trap's signed intensity-vs-delay curve,
    fit with the pedestal term (lands, reduced chi2 ~ 1) and without it (a
    single decaying shape cannot represent the flat offset, so it misses).
    Panel B -- across every well-behaved trap, the median |pedestal| and its
    median significance (|offset| / error) vs temperature: ~90 e- and 8 sigma
    in the cold, climbing past 1000 e- and ~75-95 sigma by 170-185 K, the
    signature of a fixed-charge-per-cycle dark-current deferral.
    """
    _style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    with h5py.File(MINIMAL_FIT, "r") as f:
        # ---- Panel A: mechanism on one trap ----
        tg = f[f"quad_{quad}/{trap}/temp_{temp}"]
        t = tg["seconds"][:]
        y = tg["intensities"][:]
        yerr = tg["intensity_err"][:]
        coeff = tg.attrs["fit_coeff"]
        tau = tg.attrs["fit_tau"]
        off = tg.attrs["fit_offset"]
        rchi2_full = tg.attrs["fit_reduced_chi_squared"]

        order = np.argsort(t)
        t, y, yerr = t[order], y[order], yerr[order]
        tt = np.linspace(t.min(), t.max(), 400)

        # Refit the no-pedestal model to the same data, to show what it costs.
        try:
            p0 = [coeff, tau]
            popt, _ = curve_fit(_pumped_no_offset, t, y, p0=p0, sigma=yerr,
                                absolute_sigma=True, maxfev=20000)
            resid = (y - _pumped_no_offset(t, *popt)) / yerr
            rchi2_noff = np.sum(resid**2) / (len(t) - 2)
            y_noff = _pumped_no_offset(tt, *popt)
        except Exception:
            popt, rchi2_noff, y_noff = None, np.nan, None

        tms = t * 1e3
        ttms = tt * 1e3
        axA.errorbar(tms, y, yerr=yerr, fmt="o", ms=4, color="0.2",
                     capsize=2, label="data", zorder=5)
        axA.plot(ttms, _pumped(tt, coeff, tau, off), "-", color="C0", lw=2,
                 label=f"3-knob w/ pedestal ($\\chi^2_\\nu$={rchi2_full:.2f})")
        if y_noff is not None:
            axA.plot(ttms, y_noff, "--", color="C3", lw=2,
                     label=f"2-knob no pedestal ($\\chi^2_\\nu$={rchi2_noff:.1f})")
        axA.axhline(off, ls=":", color="C0", lw=1.3, alpha=0.8)
        axA.annotate(f"fitted pedestal $I_0$ = {off:.0f} e$^-$",
                     xy=(ttms[-1], off), xytext=(0.42, 0.10),
                     textcoords="axes fraction", color="C0", fontsize=9.5)
        axA.set_xlabel("pump delay (ms)")
        axA.set_ylabel("signed dipole intensity (e$^-$)")
        axA.set_title(f"A. The pedestal on one trap ({temp} K)")
        axA.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        # ---- Panel B: rise across all well-behaved traps ----
        med_off, med_sig = [], []
        for T in TEMPS:
            offs, sig = [], []
            for q in range(4):
                gq = f.get(f"quad_{q}")
                if gq is None:
                    continue
                for name in gq:
                    g = gq[name]
                    if not g.attrs.get("WellBehavedTrap", False):
                        continue
                    ttg = g.get(f"temp_{T}")
                    if ttg is None:
                        continue
                    o = ttg.attrs.get("fit_offset", np.nan)
                    oe = ttg.attrs.get("fit_offset_err", np.nan)
                    if np.isfinite(o):
                        offs.append(abs(o))
                        if oe and np.isfinite(oe):
                            sig.append(abs(o) / oe)
            med_off.append(np.median(offs) if offs else np.nan)
            med_sig.append(np.median(sig) if sig else np.nan)

    T = np.array(TEMPS)
    med_off = np.array(med_off)
    med_sig = np.array(med_sig)

    l1, = axB.plot(T, med_off, "o-", color="C0", label="median |pedestal|")
    axB.set_xlabel("temperature (K)")
    axB.set_ylabel("median |pedestal| (e$^-$)", color="C0")
    axB.tick_params(axis="y", labelcolor="C0")
    axB.set_title("B. Pedestal grows with dark current")

    axB2 = axB.twinx()
    axB2.grid(False)
    l2, = axB2.plot(T, med_sig, "s--", color="C3", ms=4,
                    label="median |offset| / error")
    axB2.set_ylabel("median significance ($\\sigma$)", color="C3")
    axB2.tick_params(axis="y", labelcolor="C3")
    axB2.axhline(3, ls=":", color="0.5", lw=1)
    axB.axvline(temp, ls=":", color="0.6", lw=1)
    axB.legend([l1, l2], [l1.get_label(), l2.get_label()],
               loc="upper left", fontsize=8.5)

    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


def detection_gates(out="detection_gates.svg", quad=1, trap="dp_493_1447", temp=160,
                    save=True):
    """The two per-temperature detection gates, entirely from stored catalog fields.

    Source: fit_dipole_spectra_minimal_caldet_err_4.h5 +
    detection_calibration_minimal.npz. Nothing is refit -- every plotted number
    (amplitude_significance, delta_chi2_vs_constant, delta_chi2_threshold) is read
    straight out of the per-temperature groups that dipole_new.fitTrapIntensity wrote.

    Panel A -- both gates on one representative trap: the signed curve, its pumped
    fit (same model as dipole_new.intensity_function, via _pumped), and the best
    flat line the delta-chi2 is measured against.
    Panel B -- gate 1: |A|/sigma_A across every good-temperature fit vs the flat 3 cut.
    Panel C -- gate 2: delta-chi2-over-flat for every detection vs the calibrated
    per-temperature threshold table the catalog actually used.
    """
    _style()
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.3))
    cal = np.load(DETECTION_CALIB)
    thrT, thrV = cal["temperature_K"], cal["threshold"]

    asig_all, T_all, d_all = [], [], []
    with h5py.File(MINIMAL_FIT, "r") as f:
        tg = f[f"quad_{quad}/{trap}/temp_{temp}"]
        t = tg["seconds"][:]; y = tg["intensities"][:]; yerr = tg["intensity_err"][:]
        coeff = tg.attrs["fit_coeff"]; tau = tg.attrs["fit_tau"]; off = tg.attrs["fit_offset"]
        asig_ex = float(tg.attrs["amplitude_significance"])
        dchi_ex = float(tg.attrs["delta_chi2_vs_constant"])
        thr_ex = float(tg.attrs["delta_chi2_threshold"])
        o = np.argsort(t); t, y, yerr = t[o], y[o], yerr[o]
        w = 1.0 / yerr**2
        flat = np.sum(y * w) / np.sum(w)          # the same weighted-mean constant delta_chi2 uses
        tt = np.linspace(t.min(), t.max(), 400)

        # populations: read the two stored statistics off every good-temperature fit
        for q in range(4):
            gq = f.get(f"quad_{q}")
            if gq is None:
                continue
            for name in gq:
                g = gq[name]
                if not g.attrs.get("WellBehavedTrap", False):
                    continue
                for k in g:
                    if not k.startswith("temp_"):
                        continue
                    ttg = g[k]
                    if not ttg.attrs.get("GoodIntensityFit", False):
                        continue
                    asig_all.append(ttg.attrs.get("amplitude_significance", np.nan))
                    T_all.append(int(k.split("_")[1]))
                    d_all.append(ttg.attrs.get("delta_chi2_vs_constant", np.nan))

    # ---- Panel A: both gates on one trap ----
    axA.errorbar(t * 1e3, y, yerr=yerr, fmt="o", ms=4, color="0.2", capsize=2,
                 label="data", zorder=5)
    axA.plot(tt * 1e3, _pumped(tt, coeff, tau, off), "-", color="C0", lw=2, label="pumped fit")
    axA.axhline(flat, ls="--", color="C3", lw=1.6, label="best flat line")
    axA.axhline(off, ls=":", color="C0", lw=1.2, alpha=0.8, label="pedestal $I_0$")
    axA.set_xlabel("pump delay (ms)")
    axA.set_ylabel("signed dipole intensity (e$^-$)")
    axA.set_title(f"A. Both gates on one trap ({temp} K)")
    axA.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    axA.text(0.03, 0.04,
             f"gate 1:  $|A|/\\sigma_A$ = {asig_ex:.0f}   ($\\geq$ 3, pass)\n"
             f"gate 2:  $\\Delta\\chi^2$ vs flat = {dchi_ex:.0f}   ($>$ {thr_ex:.0f}, pass)",
             transform=axA.transAxes, fontsize=9, va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    asig_all = np.asarray(asig_all); T_all = np.asarray(T_all); d_all = np.asarray(d_all)

    # ---- Panel B: gate 1 across all detections ----
    a = asig_all[np.isfinite(asig_all) & (asig_all > 0)]
    axB.hist(a, bins=np.logspace(np.log10(0.5), np.log10(max(a.max(), 100.0)), 50),
             color="C0", alpha=0.85)
    axB.set_xscale("log")
    axB.axvline(3, ls="--", color="C3", lw=1.8)
    axB.axvline(asig_ex, ls=":", color="0.3", lw=1.2)
    axB.text(3.2, axB.get_ylim()[1] * 0.88, "cut = 3", color="C3", fontsize=9.5)
    axB.set_xlabel("amplitude significance  $|A|/\\sigma_A$")
    axB.set_ylabel("good-temperature fits")
    axB.set_title("B. Gate 1 — pump strength $> 3\\sigma$")

    # ---- Panel C: gate 2 vs the calibrated per-temperature table ----
    m = np.isfinite(d_all) & (d_all > 0)
    axC.scatter(T_all[m], d_all[m], s=6, color="C0", alpha=0.12, rasterized=True,
                label="detections")
    order = np.argsort(thrT)
    axC.plot(thrT[order], thrV[order], "s-", color="C3", lw=1.8, ms=4,
             label="calibrated threshold (0.1% FPR)")
    axC.scatter([temp], [dchi_ex], s=90, color="k", marker="*", zorder=6,
                label="trap in panel A")
    axC.set_yscale("log")
    axC.set_xlabel("temperature (K)")
    axC.set_ylabel("$\\Delta\\chi^2$ over flat line")
    axC.set_title("C. Gate 2 — per-temperature threshold")
    axC.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


def noise_model(out="noise_model.svg", save=True):
    """Why the minimal error bar (~37 e-) is ~5x smaller than the legacy patch spread (~185 e-).

    Source: fit_dipole_spectra_minimal_caldet_err_4.h5 (patch_sigma vs intensity_err
    datasets) + pair_noise_table_minimal.npz (the sigma_base(T, quadrant) table).

    Panel A -- the legacy 'patch sigma' measures the spread of pixel VALUES across a
    35x35 patch, so it lumps the honest shot-to-shot noise together with the fixed
    pixel-to-pixel pattern; that fixed pattern is ~5x the real noise and is exactly
    what same-pixel subtraction cancels. Panel B -- the resulting temporal noise
    floor, tabulated per temperature and quadrant (rises with dark current at high T).
    """
    _style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    patch, ierr = [], []
    with h5py.File(MINIMAL_FIT, "r") as f:
        for q in range(4):
            gq = f.get(f"quad_{q}")
            if gq is None:
                continue
            for name in gq:
                g = gq[name]
                for k in g:
                    if not k.startswith("temp_"):
                        continue
                    tg = g[k]
                    if "patch_sigma" in tg and "intensity_err" in tg:
                        ps = tg["patch_sigma"][:]; ie = tg["intensity_err"][:]
                        if ps.size and ie.size:
                            patch.append(np.median(ps)); ierr.append(np.median(ie))
    patch = np.asarray(patch); ierr = np.asarray(ierr)
    mp, mi = np.median(patch), np.median(ierr)
    fixed = np.sqrt(max(mp**2 - mi**2, 0.0))

    bins = np.logspace(np.log10(5), np.log10(500), 60)
    axA.hist(patch, bins=bins, color="C3", alpha=0.55,
             label=f"legacy patch $\\sigma$  (median {mp:.0f} e$^-$)")
    axA.hist(ierr, bins=bins, color="C0", alpha=0.75,
             label=f"minimal error bar  (median {mi:.0f} e$^-$)")
    axA.set_xscale("log")
    axA.set_xlabel("per-point noise (e$^-$)")
    axA.set_ylabel("temperature-groups")
    axA.set_title("A. Patch spread is $\\sim5\\times$ the real noise")
    axA.legend(loc="upper left", fontsize=8.3)
    axA.text(0.97, 0.55,
             "patch$^2$ = fixed-pattern$^2$ + noise$^2$\n"
             f"fixed pattern $\\approx$ {fixed:.0f} e$^-$\n"
             "  (cancels when you subtract\n"
             "   the same pixel in two images)\n"
             f"noise floor $\\approx$ {mi:.0f} e$^-$",
             transform=axA.transAxes, ha="right", va="top", fontsize=8.3,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    tb = np.load(PAIR_NOISE)
    T, Q, sb = tb["temperature_K"], tb["quadrant"], tb["sigma_base_e"]
    for q in range(4):
        m = Q == q
        order = np.argsort(T[m])
        axB.plot(T[m][order], sb[m][order], "o-", ms=3, label=f"quad {q}")
    axB.set_xlabel("temperature (K)")
    axB.set_ylabel("baseline noise floor $\\sigma_\\mathrm{base}$ (e$^-$)")
    axB.set_title("B. The noise table (temperature $\\times$ quadrant)")
    axB.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


def catalog_funnel(out="catalog_funnel.svg", save=True):
    """Where the 9,333 found dipoles go: the real selection funnel to 3,798 characterized.

    Source: fit_dipole_spectra_minimal_caldet_err_4.h5 (per-dipole flags + per-temperature
    GoodIntensityFit). Panel A -- the funnel; the dominant loss is the >=4-good-temperatures
    requirement. Panel B -- the good-temperature count per found dipole: most never reach 4.
    """
    _style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    ng = []
    n_found = n_wb = n_wb_or = n_char = 0
    with h5py.File(MINIMAL_FIT, "r") as f:
        for q in range(4):
            gq = f.get(f"quad_{q}")
            if gq is None:
                continue
            for name in gq:
                g = gq[name]; n_found += 1
                c = sum(1 for k in g if k.startswith("temp_")
                        and g[k].attrs.get("GoodIntensityFit", False))
                ng.append(c)
                wb = bool(g.attrs.get("WellBehavedTrap", False))
                oc = bool(g.attrs.get("OrientationConsistent", False))
                ge = bool(g.attrs.get("GoodEnergyFit", False))
                ef = bool(g.attrs.get("EnergyFitFailed", False))
                if wb:
                    n_wb += 1
                if wb and oc:
                    n_wb_or += 1
                if wb and oc and ge and not ef:
                    n_char += 1
    ng = np.asarray(ng)

    stages = ["found\n(dipoles)", "$\\geq$4 good\ntemperatures",
              "orientation\nconsistent", "characterized\n(catalog)"]
    vals = [n_found, n_wb, n_wb_or, n_char]
    losses = [f"$-${vals[i] - vals[i + 1]}" for i in range(len(vals) - 1)]
    loss_why = ["< 4 good temps", "sign-inconsistent", "energy fit fails"]
    y = np.arange(len(stages))[::-1]
    axA.barh(y, vals, color=["0.6", "C0", "C0", "C2"], height=0.6)
    for yi, v in zip(y, vals):
        axA.text(v + 120, yi, f"{v}", va="center", fontsize=10)
    for i in range(len(vals) - 1):
        axA.text(vals[i + 1] + 120, y[i + 1] + 0.5, f"{losses[i]}  ({loss_why[i]})",
                 va="center", fontsize=8, color="C3")
    axA.set_yticks(y); axA.set_yticklabels(stages, fontsize=9)
    axA.set_xlim(0, n_found * 1.18)
    axA.set_xlabel("dipoles")
    axA.set_title("A. Selection funnel: 9,333 $\\to$ 3,798")
    axA.grid(axis="y", visible=False)

    axB.hist(ng, bins=np.arange(0, ng.max() + 2) - 0.5, color="C0", alpha=0.8)
    axB.axvline(3.5, ls="--", color="C3", lw=1.8)
    axB.text(3.8, axB.get_ylim()[1] * 0.82, "keep $\\geq$4", color="C3", fontsize=9.5)
    axB.set_xlabel("good temperatures per found dipole")
    axB.set_ylabel("dipoles")
    axB.set_title("B. Most found dipoles never reach 4")

    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


def _srh_logtau(temperatures, E, logsigma):
    """Replicates dipole_new.log_energy_cross_section: returns ln(tau) for the SRH line."""
    kb = 8.617333262e-5; h = 4.135667696e-15; me = 0.510998950e6; ccms = 2.99792458e10
    M_DENS, M_COND = 0.94, 0.41                    # hole effective masses (m_e units)
    denom = 2 * np.sqrt(3) * (2 * np.pi)**1.5 * (M_DENS * me)**1.5 / np.sqrt(M_COND * me)
    kbT = kb * np.asarray(temperatures, float)
    scaling = (h**3) * (ccms**2) / denom
    return np.log(scaling) - logsigma - 2 * np.log(kbT) + E / kbT


def failure_gallery(out="failure_gallery.svg", save=True):
    """One representative dropped trap per current-analysis rejection stage.

    Source: fit_dipole_spectra_minimal_caldet_err_4.h5. Every example is a real
    catalog record identified by its stored flags. A and B are per-temperature
    (detection) failures shown as intensity-vs-delay curves; C and D are per-trap
    failures (orientation and the SRH energy fit).
    """
    _style()
    fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(11, 8.6))

    def curve(g, T):
        tg = g[f"temp_{T}"]
        t = tg["seconds"][:]; y = tg["intensities"][:]; e = tg["intensity_err"][:]
        o = np.argsort(t)
        return t[o] * 1e3, y[o], e[o], tg

    def goods(g):
        return [k for k in g if k.startswith("temp_") and g[k].attrs.get("GoodIntensityFit", False)]

    box = dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9)
    with h5py.File(MINIMAL_FIT, "r") as f:
        # A -- noise-like, 0 good temperatures
        gA = f["quad_0/dp_100_2729"]
        Ta = 160 if "temp_160" in gA else int(
            [k for k in gA if k.startswith("temp_")][0].split("_")[1])
        t, y, e, _ = curve(gA, Ta)
        axA.errorbar(t, y, yerr=e, fmt="o", ms=3, color="0.3", capsize=2)
        axA.axhline(0, ls=":", color="0.6")
        axA.set_title(f"A. Noise-like: 0 good temperatures ({Ta} K)")
        axA.text(0.04, 0.96, "no significant pumped bump at any T\n→ 0 good temps → cut at the ≥4 bar\n(3,295 found dipoles look like this)",
                 transform=axA.transAxes, va="top", fontsize=8.5, bbox=box)
        axA.set_xlabel("pump delay (ms)"); axA.set_ylabel("signed intensity (e$^-$)")

        # B -- sub-threshold, exactly 3 good temperatures
        gB = f["quad_0/dp_102_1932"]
        gb = goods(gB)
        Tb = max((int(k.split("_")[1]) for k in gb),
                 key=lambda T: gB[f"temp_{T}"].attrs["amplitude_significance"])
        tg = gB[f"temp_{Tb}"]
        t, y, e, _ = curve(gB, Tb)
        tt = np.linspace(t.min() / 1e3, t.max() / 1e3, 300)
        axB.errorbar(t, y, yerr=e, fmt="o", ms=3, color="0.3", capsize=2)
        axB.plot(tt * 1e3, _pumped(tt, tg.attrs["fit_coeff"], tg.attrs["fit_tau"],
                                   tg.attrs["fit_offset"]), "-", color="C0", lw=2)
        axB.set_title(f"B. Sub-threshold: only 3 good temps ({Tb} K shown)")
        axB.text(0.04, 0.96, "a genuine bump here, but the trap passes\nthe per-temp gates at only 3 temps\n→ below the ≥4 bar (recoverable at ≥3)",
                 transform=axB.transAxes, va="top", fontsize=8.5, bbox=box)
        axB.set_xlabel("pump delay (ms)"); axB.set_ylabel("signed intensity (e$^-$)")

        # C -- orientation-inconsistent (dual_response): the sign flips across temperature
        gC = f["quad_0/dp_104_2640"]
        gg = [(int(k.split("_")[1]), float(gC[k].attrs["fit_coeff"]),
               float(gC[k].attrs["amplitude_significance"])) for k in goods(gC)]
        Tp = max((z for z in gg if z[1] > 0), key=lambda z: z[2])[0]
        Tn = max((z for z in gg if z[1] < 0), key=lambda z: z[2])[0]
        for T, col, lab in [(Tp, "C0", f"+ bump ({Tp} K)"), (Tn, "C3", f"− dip ({Tn} K)")]:
            t, y, e, tg = curve(gC, T)
            tt = np.linspace(t.min() / 1e3, t.max() / 1e3, 300)
            axC.errorbar(t, y, yerr=e, fmt="o", ms=3, color=col, capsize=2, alpha=0.7)
            axC.plot(tt * 1e3, _pumped(tt, tg.attrs["fit_coeff"], tg.attrs["fit_tau"],
                                       tg.attrs["fit_offset"]), "-", color=col, lw=2, label=lab)
        axC.axhline(0, ls=":", color="0.6")
        axC.set_title("C. Orientation-inconsistent (dual_response)")
        axC.text(0.04, 0.05, "same pixel pumps + at one T and − at another\n→ two traps blended in one pixel → cut",
                 transform=axC.transAxes, va="bottom", fontsize=8.5, bbox=box)
        axC.set_xlabel("pump delay (ms)"); axC.set_ylabel("signed intensity (e$^-$)")
        axC.legend(fontsize=8.5, loc="upper right")

        # D -- energy-fit failure (reduced chi2 >= 10): tau(T) does not follow one SRH line
        gD = f["quad_0/dp_111_2769"]
        pts = sorted((int(k.split("_")[1]), float(gD[k].attrs["fit_tau"]),
                      float(gD[k].attrs["fit_tau_err"])) for k in goods(gD))
        Td = np.array([p[0] for p in pts]); tau = np.array([p[1] for p in pts])
        tauerr = np.array([p[2] for p in pts])
        E = float(gD.attrs["energy_BestFitEnergy"]); sig = float(gD.attrs["energy_BestFitCrossSection"])
        rc = float(gD.attrs["energy_reduced_chi2"])
        ln10 = np.log(10.0)
        axD.errorbar(1000.0 / Td, np.log10(tau), yerr=(tauerr / tau) / ln10,
                     fmt="o", ms=4, color="0.2", capsize=2, label="good-temperature τ")
        Tg = np.linspace(Td.min(), Td.max(), 200)
        axD.plot(1000.0 / Tg, _srh_logtau(Tg, E, np.log(sig)) / ln10, "-", color="C3",
                 lw=2, label=f"best SRH fit ($\\chi^2_\\nu$={rc:.0f})")
        axD.set_title("D. Energy-fit failure: τ(T) not one SRH line")
        axD.text(0.04, 0.05, "≥4 good temps & one consistent sign, but the\nτ(T) points scatter off a single SRH law\n→ non-SRH blend → cut",
                 transform=axD.transAxes, va="bottom", fontsize=8.5, bbox=box)
        axD.set_xlabel("1000 / T  (K$^{-1}$)"); axD.set_ylabel("log$_{10}$ τ  (s)")
        axD.legend(fontsize=8.5, loc="upper right")

    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


# ---------------------------------------------------------------------------
# Method-3 completeness stage figures (completeness_efficiency.qmd sec 3-4)
# ---------------------------------------------------------------------------
def _load_m3():
    """Load the minimal_caldet Method-3 bundle (adds repo root to sys.path)."""
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import figure_utils as fu
    return fu.load_method3(pipeline="minimal", detection="calibrated")


def completeness_noise_map(out="completeness_noise_map.svg", save=True):
    """Stage 03 -- the injection noise, and why the flavor matters (WS1).

    Sources: 03_noise_map_v1.h5 (577,200 trap-free 34x34 patch sigmas) and
    pair_noise_table_minimal.npz (temporal pair-noise floor sigma_base(T,q)).

    Panel A -- the spatial patch-sigma distribution the *legacy* injection
    marginalizes over, as median +/- (p16,p84) across quadrants vs temperature
    (~180-200 e-).
    Panel B -- patch sigma (legacy) against the ~5x smaller temporal pair
    noise (~35 e-) that the *minimal* pipeline actually detects with. Injecting
    patch sigma for the minimal flavor over-noised it ~5x -- the WS1 bug (sec 5).
    """
    _style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    with h5py.File(STAGE03_NOISE, "r") as f:
        s = f["summary"]
        temp = np.asarray(s["temperature_K"])
        med = np.asarray(s["trap_free_median"])
        p16 = np.asarray(s["trap_free_p16"])
        p84 = np.asarray(s["trap_free_p84"])

    # Aggregate the 4 quadrants into one curve per temperature.
    utemp = np.array(sorted(set(temp.tolist())))
    a_med = np.array([np.median(med[temp == T]) for T in utemp])
    a_lo = np.array([np.median(p16[temp == T]) for T in utemp])
    a_hi = np.array([np.median(p84[temp == T]) for T in utemp])

    axA.plot(utemp, a_med, "o-", color="C0", label="median patch $\\sigma$")
    axA.fill_between(utemp, a_lo, a_hi, color="C0", alpha=0.20,
                     label="p16-p84 across patches")
    axA.set_xlabel("temperature (K)")
    axA.set_ylabel("spatial patch $\\sigma$ (e$^-$)")
    axA.set_title("A. Stage-03 trap-free noise map (legacy convention)")
    axA.legend(fontsize=9)

    pn = np.load(PAIR_NOISE)
    pt = np.asarray(pn["temperature_K"])
    pbase = np.asarray(pn["sigma_base_e"])
    a_pair = np.array([np.median(pbase[pt == T]) for T in utemp])

    axB.plot(utemp, a_med, "o-", color="C0", label="patch $\\sigma$ (legacy inject)")
    axB.plot(utemp, a_pair, "s--", color="C3", ms=4,
             label="pair noise $\\sigma_{\\rm base}$ (minimal inject)")
    axB.set_yscale("log")
    axB.set_xlabel("temperature (K)")
    axB.set_ylabel("per-point $\\sigma$ (e$^-$)")
    axB.set_title("B. WS1: minimal detects at ~5x lower noise")
    axB.annotate("~5x gap\n(the WS1 fix)", xy=(131, np.sqrt(a_med.mean() * a_pair.mean())),
                 fontsize=9, color="0.3", ha="left", va="center")
    axB.legend(fontsize=8.5, loc="upper right")

    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


def completeness_amplitude_pc(out="completeness_amplitude_pc.svg", save=True):
    """Stage 05 -- the amplitude prior: trap depths and capture probability.

    Source: 05_amplitude_prior_minimal_caldet_v1.npz (rebuilt from the
    dipole_new catalog). Amplitude A = N_pumps * D_t * P_c(T).

    Panel A -- the distribution of trap depths D_t (at the 135 K anchor) over
    the 1920 high-confidence traps that seed the injection amplitude grid.
    Panel B -- the temperature scaling P_c(T), anchored to 1 at 135 K. It rises
    to ~1.25 near 160-170 K then falls to 0.681 at 210 K -- a *real* warm-T
    incompleteness signal (the dominant open systematic, sec 5).
    """
    _style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    d = np.load(STAGE05_PRIOR)
    depth = np.asarray(d["default_depth_electrons_at_pc135"])
    temps = np.asarray(d["temperatures_K"])
    pc = np.asarray(d["pc_temperature_factor"])

    axA.hist(depth, bins=40, range=(0, np.percentile(depth, 99)),
             color="C0", alpha=0.85)
    axA.axvline(np.median(depth), ls="--", color="0.3", lw=1.2,
                label=f"median {np.median(depth):.0f} e$^-$")
    axA.set_xlabel("trap depth $D_t$ at $P_c$(135 K)  (e$^-$)")
    axA.set_ylabel("number of traps")
    axA.set_title(f"A. Depth distribution ({depth.size} high-conf. traps)")
    axA.legend(fontsize=9)

    axB.plot(temps, pc, "o-", color="C0")
    axB.axhline(1.0, ls=":", color="0.5", lw=1)
    axB.axvline(135, ls=":", color="0.6", lw=1)
    i210 = int(np.argmin(np.abs(temps - 210)))
    axB.annotate(f"$P_c$(210 K) = {pc[i210]:.3f}",
                 xy=(temps[i210], pc[i210]), xytext=(178, 0.74),
                 fontsize=9.5, color="C3",
                 arrowprops=dict(arrowstyle="->", color="C3", lw=1))
    axB.set_xlabel("temperature (K)")
    axB.set_ylabel("$P_c(T)$  (anchored to 135 K)")
    axB.set_title("B. Capture probability falls at high T")

    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


def completeness_example_curves(out="completeness_example_curves.svg", save=True):
    """Stages 06/08 -- the injection-recovery failure modes it must reproduce.

    Illustrative (synthesized with the minimal 3-knob curve + a ~36 e- pair-noise
    draw), one panel per named case in single_curve_recovery._tau_cases. A trap is
    "recovered" only if the noisy fit clears the live cuts (amplitude significance
    >= 3, tau rel-err <= 0.5, peak reachable in the dwell window). The losses --
    tau peak before the first delay, tau peak beyond the last delay, or a rising
    edge below threshold -- are exactly the incompleteness the grid quantifies.
    """
    _style()
    rng = np.random.default_rng(7)
    t = np.geomspace(5e-5, 0.2, 20)          # representative pump-delay grid (s)
    x_peak = np.log(8.0) / 7.0                # analytic peak of the 3-knob shape
    sigma = 36.0                              # minimal pair-noise floor (e-)
    coeff = 1.0                               # ~2000 e- peak amplitude

    # tau values mirror single_curve_recovery._tau_cases (relative to this grid).
    cases = [
        ("short_outside_band", 2.0e-5, coeff,
         "peak before first delay -> lost"),
        ("near_peak_reachable", t[10] / x_peak, coeff,
         "peak well sampled -> recovered"),
        ("long_reachable_peak", t[-2] / x_peak, coeff,
         "peak near long-delay end -> recovered"),
        ("long_rising_edge", 2.0, coeff,
         "only the rising edge is seen -> marginal"),
        ("effectively_undetectable_long", 20.0, coeff,
         "rising edge below threshold -> lost"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    axes = axes.ravel()
    for ax, (name, tau, cf, verdict) in zip(axes, cases):
        true = _pumped(t, cf, tau, 0.0)
        noisy = true + rng.normal(0.0, sigma, t.size)
        ax.errorbar(t * 1e3, noisy, yerr=sigma, fmt="o", ms=3.2, color="0.25",
                    capsize=2, label="noisy", zorder=5)
        ax.plot(t * 1e3, true, "-", color="C0", lw=1.6, label="true")
        recovered = None
        try:
            popt, pcov = curve_fit(_pumped, t, noisy, p0=[cf, tau, 0.0],
                                   sigma=np.full(t.size, sigma),
                                   absolute_sigma=True, maxfev=20000)
            perr = np.sqrt(np.diag(pcov))
            tt = np.geomspace(t.min(), t.max(), 300)
            ax.plot(tt * 1e3, _pumped(tt, *popt), "--", color="C3", lw=1.5,
                    label="fit")
            sig = abs(popt[0]) / perr[0] if perr[0] > 0 else 0.0
            tau_relerr = perr[1] / abs(popt[1]) if popt[1] != 0 else np.inf
            peak_reach = abs(_pumped(t, *popt)).max() >= 3 * sigma
            recovered = (sig >= 3) and (tau_relerr <= 0.5) and peak_reach
        except Exception:
            recovered = False
        ok = "PASS" if recovered else "FAIL"
        col = "C2" if recovered else "C3"
        ax.set_xscale("log")
        ax.set_title(f"{name}\n[{ok}] {verdict}", fontsize=8.6, color=col)
        ax.set_xlabel("pump delay (ms)")
        ax.legend(fontsize=7.5, loc="upper right")
    axes[0].set_ylabel("signed intensity (e$^-$)")
    axes[3].set_ylabel("signed intensity (e$^-$)")
    axes[5].axis("off")
    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


def completeness_pdet_grid(out="completeness_pdet_grid.svg", save=True):
    """Stage 09 -- the P(characterized | tau135, E) map with the real catalog.

    Source: 09_characterization_probability_minimal_caldet_v1.h5 via
    figure_utils.load_method3 (energy-fit survival 0.972 applied). Each grid
    point walks its tau(T) trajectory through all 23 temperatures and combines
    the per-T detection probabilities with a Poisson-binomial n_good>=4 tail.
    The 3798 characterized traps (red) sit almost entirely in the high-P region;
    the lower-right (long tau135, low E) is where the catalog goes blind.
    """
    _style()
    m3 = _load_m3()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    mesh = ax.pcolormesh(m3["tau_grid"], m3["E_grid"], m3["p4_map"].T,
                         cmap="Blues", shading="auto", vmin=0, vmax=1)
    ax.scatter(m3["known_tau"], m3["known_E"], s=3, color="C3", alpha=0.7,
               label=f"characterized traps (n={m3['known_tau'].size})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau_e$(135 K)  [s]")
    ax.set_ylabel("E  [eV]")
    ax.set_title("Stage-09 completeness map (minimal, n_good$\\geq$4)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("P(characterized)")
    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


def completeness_noise_patches(out="completeness_noise_patches.svg", save=True):
    """Stage 03 -- the noise the injection reproduces, on real data (slide 3).

    Source: the cached example FITS (160 K, dtph 650000, quad 2). Left column is
    a real characterized dipole patch (dp_368_689); right column is a trap-free
    patch from the same image. Top row: the 34x34 pixel patches on a shared
    symmetric electron scale; bottom row: their pixel-value histograms with the
    patch sigma. The dipole pair (the two bright/dark pixels) rides on the same
    per-pixel scatter the trap-free patch shows -- which is why injecting into
    trap-free patches reproduces the real detection noise.
    """
    _style()
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from utils import get_qdata, crop_qdata, approximate_electronize

    fits = os.path.join(
        ROOT, "proc",
        "proc_skp_dp_scan1_160k_binned_NROW580_NBINROW1_NCOL3600_NBINCOL1"
        "_SC300000_vl-2.75_vh7.5_dtph650000_NPUMPS3000_2_17.fits")
    quad, half = 2, 17
    img = get_qdata(fits, quad)
    img = crop_qdata(img)
    img = approximate_electronize(img, 400)
    img = (img.T - np.median(img, axis=1)).T

    def patch(r, c):
        return img[r - half:r + half, c - half:c + half]

    dip = patch(368, 689)          # characterized dipole dp_368_689
    free = patch(91, 3033)         # trap-free reference
    vlim = np.percentile(np.abs(np.concatenate([dip.ravel(), free.ravel()])), 99.5)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8),
                             gridspec_kw={"height_ratios": [3, 1.4]})
    for ax, data, ttl in ((axes[0, 0], dip, "Characterized dipole patch"),
                          (axes[0, 1], free, "Trap-free noise patch")):
        im = ax.imshow(data, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("patch column [pix]")
        ax.grid(False)
    from matplotlib.patches import Rectangle
    axes[0, 0].add_patch(Rectangle((half - 2.5, half - 2.5), 5, 5, ec="yellow",
                                   fc="none", lw=1.6))
    axes[0, 0].set_ylabel("patch row [pix]")
    fig.colorbar(im, ax=axes[0, :], shrink=0.8,
                 label="row-median-subtracted charge (e$^-$)")

    for ax, data in ((axes[1, 0], dip), (axes[1, 1], free)):
        ax.hist(data.ravel(), bins=40, range=(-vlim, vlim), color="0.5")
        ax.set_xlabel("pixel charge (e$^-$)")
        ax.set_title(f"patch $\\sigma$ = {np.std(data.ravel()):.0f} e$^-$",
                     fontsize=10)
    axes[1, 0].set_ylabel("pixels")
    fig.suptitle("Stage 03: real dipole vs trap-free noise (160 K, dtph 650000)",
                 fontsize=11)
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    return fig


def completeness_characterization_maps(out="completeness_characterization_maps.svg",
                                       save=True):
    """Stage 09 -- P(characterized) for n_good>=4 and >=3, plus tau-out-of-band.

    Source: 09_characterization_probability_minimal_caldet_v1.h5 via load_method3
    (energy-fit survival 0.972 applied to the two probability maps). The n_good>=3
    map is more forgiving than >=4 (its blind wedge starts later). The right panel
    is the fraction of the 23 temperatures whose tau(T) falls outside the Stage-08
    grid band -- the geometric origin of the two blind corners.
    """
    _style()
    m3 = _load_m3()
    import h5py as _h5
    with _h5.File(m3["paths"]["stage09_h5"], "r") as h:
        oob_frac = h["diagnostics/tau_oob_fraction"][:]

    tau, E = m3["tau_grid"], m3["E_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, data, ttl, cmap in (
            (axes[0], m3["p4_map"].T, "P(characterized), $n_{\\rm good}\\geq4$", "Blues"),
            (axes[1], m3["p3_map"].T, "P(characterized), $n_{\\rm good}\\geq3$", "Blues"),
            (axes[2], oob_frac.T, "$\\tau$ out-of-band fraction", "magma")):
        mesh = ax.pcolormesh(tau, E, data, cmap=cmap, shading="auto",
                             vmin=0, vmax=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau_e$(135 K)  [s]")
        ax.set_title(ttl, fontsize=11)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("E  [eV]")
    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    return fig


def completeness_known_trap_check(out="completeness_known_trap_check.svg", save=True):
    """Stage 10 -- validation: do the known traps land in high-P regions?

    Source: load_method3 known_p4 (P4 evaluated at each of the 3798 characterized
    n_good=4 traps, survival applied). Almost all sit at P4~1; only a handful fall
    below the 0.8 line -- the self-consistency check that the completeness map
    does not misclassify the traps it was built to cover.
    """
    _style()
    m3 = _load_m3()
    kp = np.asarray(m3["known_p4"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(kp, bins=np.linspace(0, 1.0, 51), color="C0")
    ax.axvline(0.8, ls="--", color="k", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("P4 at known characterized trap")
    ax.set_ylabel("trap count")
    frac = float((kp >= 0.8).mean())
    ax.set_title(f"Known traps mostly lie in high-probability regions "
                 f"({frac:.2%} at P4$\\geq$0.8)")
    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


def _fu_setup(use_tex=True):
    """Load the minimal bundle and apply figure_utils' paper style."""
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import figure_utils as fu
    fu.setup_style(use_tex=use_tex)
    return fu, fu.load_method3(pipeline="minimal", detection="calibrated")


def _capture_fu(plot_call, out, save):
    """Run a figure_utils plot that internally does plt.show()/plt.close(),
    neutralise those so the figure survives, and return it for Quarto to embed."""
    _show, _close = plt.show, plt.close
    plt.show = lambda *a, **k: None
    plt.close = lambda *a, **k: None
    try:
        plot_call()
        fig = plt.gcf()
    finally:
        plt.show, plt.close = _show, _close
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    return fig


def completeness_overlay_fig(out="completeness_overlay.svg", save=True):
    """Stage 09/10 -- the completeness curve over the tau135 catalog histogram.

    Thin wrapper around figure_utils.plot_completeness_overlay (the same cell used
    in the analysis notebook), so this doc shows the exact production figure. No
    seed files are written.
    """
    fu, m3 = _fu_setup()
    return _capture_fu(lambda: fu.plot_completeness_overlay(m3, save=False), out, save)


def completeness_efficiency_hist(out="completeness_efficiency_hist.svg", save=True):
    """Stage 10 -- efficiency-corrected (point-estimate) tau135 population.

    Wrapper around figure_utils.plot_efficiency_corrected_hist with the production
    correction band (tau in [6e-5, 1e9] s). write=False so the simulation seed NPZ
    is never regenerated from a doc render.
    """
    fu, m3 = _fu_setup()
    return _capture_fu(
        lambda: fu.plot_efficiency_corrected_hist(
            m3, correction_tau_min=6e-5, correction_tau_max=1e9,
            write=False, save=False),
        out, save)


def completeness_upper_limit_hist(out="completeness_upper_limit_hist.svg", save=True):
    """Stage 10 -- efficiency-corrected 90% CL upper-limit tau135 population.

    Wrapper around figure_utils.plot_upper_limit_hist with the production
    correction band (tau in [6e-5, 5e7] s). write=False so the seed NPZ is never
    regenerated from a doc render.
    """
    fu, m3 = _fu_setup()
    return _capture_fu(
        lambda: fu.plot_upper_limit_hist(
            m3, correction_tau_min=6e-5, correction_tau_max=5e7,
            write=False, save=False),
        out, save)


# ---------------------------------------------------------------------------
# Results page -- the measured catalog's trap properties (energies, sigmas, tau)
# ---------------------------------------------------------------------------
def _load_agg():
    """Load the minimal_caldet trap catalog and reduce to per-trap (E, sigma, tau)."""
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import figure_utils as fu
    return fu.aggregate_trap_fits(fu.load_trap_fits(MINIMAL_FIT))


def results_energy_sigma(out="results_energy_sigma.svg", save=True):
    """The measured catalog: fitted trap depth and capture cross-section.

    Source: fit_dipole_spectra_minimal_caldet_err_4.h5 (good-energy-fit traps).
    A -- trap depth E [eV]. B -- capture cross-section sigma [cm^2] (log). C --
    the joint (sigma, E) density showing the SRH degeneracy ridge (small sigma
    <-> small E; see the sigma-degeneracy note).
    """
    _style()
    agg = _load_agg()
    E = np.asarray(agg["energies"], float)
    sig = np.asarray(agg["cross_sections"], float)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13, 4.2))

    axA.hist(E, bins=np.linspace(0, 0.8, 50))
    axA.set_yscale("log")
    axA.set_xlabel(r"Fitted trap energy $E$ [eV]")
    axA.set_ylabel("Traps")
    axA.set_title(f"Trap depth (n = {len(E)})")

    axB.hist(sig, bins=np.geomspace(1e-25, 1e-10, 50))
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel(r"Capture cross-section $\sigma$ [cm$^2$]")
    axB.set_ylabel("Traps")
    axB.set_title("Cross-section")

    h = axC.hist2d(sig, E,
                   bins=[np.geomspace(1e-25, 1e-10, 40), np.linspace(0, 0.7, 40)],
                   cmap="viridis", norm=matplotlib.colors.LogNorm())
    axC.set_xscale("log")
    axC.set_xlabel(r"$\sigma$ [cm$^2$]")
    axC.set_ylabel(r"$E$ [eV]")
    axC.set_title("Joint density")
    fig.colorbar(h[3], ax=axC, label="Traps")

    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    return fig


def results_tau135_census(out="results_tau135_census.svg", save=True):
    """The characterized trap population vs emission time tau at 135 K.

    Source: the Method-3 bundle's tau135 histogram -- the ~3,798 characterized
    traps extrapolated to the 135 K anchor. This is the census the simulation
    seeds and that the completeness / upper-limit corrections re-weight.
    """
    _style()
    m3 = _load_m3()
    edges = np.asarray(m3["tau_edges"], float)
    hist = np.asarray(m3["tau_hist"], float)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.stairs(hist, edges, fill=True, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$ at 135 K [s]")
    ax.set_ylabel("Characterized traps")
    ax.set_title(rf"$\tau_{{135}}$ census ($\Sigma$ = {hist.sum():.0f} traps)")
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    return fig


def results_tau135_energy(out="results_tau135_energy.svg", save=True):
    """tau(135 K) vs fitted energy for the characterized traps (production figure).

    Thin wrapper around figure_utils.plot_tau135_energy_scatter(m3).
    """
    fu, m3 = _fu_setup()
    return _capture_fu(lambda: fu.plot_tau135_energy_scatter(m3), out, save)


# ---------------------------------------------------------------------------
# Results page -- the simulation campaign (current numbers; will be superseded
# by the v3_phase_fraction=0.5 rerun). All reuse figure_utils production plots.
# ---------------------------------------------------------------------------
# Common campaign axes for the representative scenarios (minimal_caldet flavor,
# spurious charge injected post-readout, shuffled exposure order).
_CAMPAIGN_KW = dict(
    exp_indep="post_readout",
    order="shuffled",
    flavor="minimal_caldet",
    base=os.path.join(ROOT, "campaign"),
)


def _scenario_spec(condition, population, clear, binning=1.0, vp_syst=True):
    """A compare_scenarios spec dict for one campaign point (central V_p=3)."""
    s = dict(condition=condition, population=population, clear=clear,
             binning=binning, **_CAMPAIGN_KW)
    if vp_syst:
        s["systematics"] = ("vp",)
    return s


def results_scenario_null(out="results_scenario_null.svg", save=True):
    """Single-scenario detail for the BASELINE (characterized) population -- the null.

    MINOS, baseline, V_p=3, three_hour clear. The with-traps single-electron
    density tracks the no-trap truth across exposure; the difference panel is
    consistent with ~zero. Production figure_utils.plot_simulation_results.
    Current campaign numbers -- superseded by the phase-split rerun.
    """
    fu, _ = _fu_setup()
    rundir = fu.scenario_dir("minos", population="baseline", vp=3.0,
                             clear="three_hour", **_CAMPAIGN_KW)
    return _capture_fu(
        lambda: fu.plot_simulation_results(rundir, showFit=False, showDensity=False,
                                           saveFig=False),
        out, save)


def results_scenario_upper(out="results_scenario_upper.svg", save=True):
    """Single-scenario detail for the UPPER-LIMIT population -- the allowance.

    MINOS, upper, V_p=3, three_hour clear. The 1/eps-inflated long-tau population
    lifts the with-traps density above truth, producing a clear exposure-dependent
    slope excess (the 90% CL allowance, not a measured effect). Production
    figure_utils.plot_simulation_results. Superseded by the phase-split rerun.
    """
    fu, _ = _fu_setup()
    rundir = fu.scenario_dir("minos", population="upper", vp=3.0,
                             clear="three_hour", **_CAMPAIGN_KW)
    return _capture_fu(
        lambda: fu.plot_simulation_results(rundir, showFit=False, showDensity=False,
                                           saveFig=False),
        out, save)


def results_forest_headline(out="results_forest_headline.svg", save=True):
    """Forest: baseline (null) vs upper-limit (allowance), MINOS & SNOLAB, three_hour.

    The four-scenario headline comparison -- with-traps (red) vs no-trap (blue)
    against the injected truth, with the V_p systematic band. Baseline points sit
    on truth; the UL points show the large allowance. Production
    figure_utils.compare_scenarios. Superseded by the phase-split rerun.
    """
    fu, _ = _fu_setup()
    scenarios = [
        ("MINOS 3h clear",     _scenario_spec("minos",  "baseline", "three_hour")),
        ("MINOS UL 3h clear",  _scenario_spec("minos",  "upper",    "three_hour")),
        ("SNOLAB 3h clear",    _scenario_spec("snolab", "baseline", "three_hour")),
        ("SNOLAB UL 3h clear", _scenario_spec("snolab", "upper",    "three_hour")),
    ]
    return _capture_fu(lambda: fu.compare_scenarios(scenarios), out, save)


def results_forest_clearmodes(out="results_forest_clearmodes.svg", save=True):
    """Forest: the BASELINE null across all clear modes (zoomed), MINOS & SNOLAB.

    Eight baseline scenarios across sequencer / three_hour / binned-0h / binned
    (bin32) clears. The data-driven zoom shows the with-traps points sit on the
    injected truth to within the V_p systematic in every clear mode -- the null is
    robust. Production figure_utils.compare_scenarios. Superseded by the rerun.
    """
    fu, _ = _fu_setup()
    scenarios = [
        ("MINOS clear seq",        _scenario_spec("minos",  "baseline", "sequencer")),
        ("MINOS 3h clear seq",     _scenario_spec("minos",  "baseline", "three_hour")),
        ("MINOS 0h bin clear",     _scenario_spec("minos",  "baseline", "binned_0h")),
        ("MINOS binned clear seq", _scenario_spec("minos",  "baseline", "sequencer", binning=32.0)),
        ("SNOLAB clear seq",       _scenario_spec("snolab", "baseline", "sequencer")),
        ("SNOLAB 3h clear seq",    _scenario_spec("snolab", "baseline", "three_hour")),
        ("SNOLAB 0h bin clear",    _scenario_spec("snolab", "baseline", "binned_0h")),
        ("SNOLAB binned clear seq",_scenario_spec("snolab", "baseline", "sequencer", binning=32.0)),
    ]
    return _capture_fu(lambda: fu.compare_scenarios(scenarios), out, save)


# ---------------------------------------------------------------------------
# High-T Arrhenius lean: profile measurement + tau135 bracket (signed_refit.qmd §6b)
# ---------------------------------------------------------------------------
_LEAN_KNEE_K = 165.0          # lean onset; T <= 165 K = "cold" (SRH line holds)
_LEAN_MIN_COLD_PTS = 3        # cold-anchored reference fit needs 2 params + 1 dof
_LEAN_MIN_PTS_PER_T = 20      # min pooled residuals to quote a lean value at a T
_LEAN_HOUR_S = 3600.0
_LEAN_DRIVER_BAND = (3e5, 5e7)  # tau135 band driving the exposure-dependent UL slope
_LEAN_CACHE = {}


def _lean_srh_fit(T, lntau, lnerr):
    """The live energy-fit call (dipole_new.py:570-576, absolute errors)."""
    popt, _ = curve_fit(_srh_logtau, T, lntau, sigma=lnerr,
                        bounds=([0, -100], [2, -1]), absolute_sigma=True)
    return popt


def _lean_load_and_measure():
    """Measure the high-T Arrhenius lean and its tau135 bracket from the catalog.

    Source: fit_dipole_spectra_minimal_caldet_err_4.h5. Selection replicates the
    stage-09 characterized set (WellBehavedTrap, GoodEnergyFit, orientation,
    >= 4 good temps — figure_utils._load_characterized_caught_temperature).

    Lean profile: for traps with >= 3 good cold temps and >= 1 hot temp, fit the
    SRH line through the COLD points only; delta(T) = median dex offset of the
    hot points from that line (bootstrap SEM over traps' points).

    Bracket: refit every characterized trap with the live energy-fit call,
    (A) with points as stored — reproduces the production tau135 exactly —
    (B) with hot points raised by -delta(T) ("un-leaned", i.e. the cold-end SRH
    behaviour transported to the hot temps), and
    (C) with the Henry-Lang activated model taken as true: sigma(T) =
    sigma_LT (1 + R e^{-Eb/kT}) with the ensemble (R, Eb) fit to the profile;
    every point gets +ln(1 + R e^{-Eb/kT_i}), the standard SRH line is fit to
    the corrected points (recovering E and sigma_LT), and the anchor is
    back-corrected by -ln(1 + R e^{-Eb/k 135}). Computed once and cached.
    """
    if _LEAN_CACHE:
        return _LEAN_CACHE
    ln10 = np.log(10.0)
    traps = []
    with h5py.File(MINIMAL_FIT, "r") as f:
        for qname in sorted(f.keys()):
            qg = f[qname]
            if not isinstance(qg, h5py.Group):
                continue
            for dpname in sorted(qg.keys()):
                dg = qg[dpname]
                if not isinstance(dg, h5py.Group):
                    continue
                a = dg.attrs
                passes = (bool(a.get("WellBehavedTrap", False))
                          and not bool(a.get("EnergyFitFailed", False))
                          and bool(a.get("GoodEnergyFit", False))
                          and bool(a.get("OrientationConsistent", True)))
                if not passes or "energy_BestFitEnergy" not in a:
                    continue
                T, tau, terr = [], [], []
                for n in dg:
                    if not (n.startswith("temp_") and isinstance(dg[n], h5py.Group)):
                        continue
                    tg = dg[n]
                    if not bool(tg.attrs.get("GoodIntensityFit", False)):
                        continue
                    tv = float(tg.attrs.get("fit_tau", np.nan))
                    te = float(tg.attrs.get("fit_tau_err", np.nan))
                    if np.isfinite(tv) and tv > 0 and np.isfinite(te) and te > 0:
                        T.append(int(n.split("_")[1])); tau.append(tv); terr.append(te)
                if len(T) < 4:
                    continue
                o = np.argsort(T)
                traps.append((np.array(T, float)[o], np.log(np.array(tau)[o]),
                              np.array(terr)[o] / np.array(tau)[o]))

    # -- lean profile from cold-anchored reference traps
    per_T = {}
    n_ref = 0
    for T, lntau, lnerr in traps:
        cold = T <= _LEAN_KNEE_K
        if cold.sum() < _LEAN_MIN_COLD_PTS or cold.all():
            continue
        try:
            popt = _lean_srh_fit(T[cold], lntau[cold], lnerr[cold])
        except RuntimeError:
            continue
        n_ref += 1
        hot = ~cold
        for tv, r in zip(T[hot], (lntau[hot] - _srh_logtau(T[hot], *popt)) / ln10):
            per_T.setdefault(int(tv), []).append(float(r))
    rng = np.random.default_rng(20260703)
    prof_T, prof_med, prof_sem = [], [], []
    for tv in sorted(per_T):
        arr = np.array(per_T[tv])
        if len(arr) < _LEAN_MIN_PTS_PER_T:
            continue
        idx = rng.integers(0, len(arr), size=(500, len(arr)))
        prof_T.append(tv); prof_med.append(float(np.median(arr)))
        prof_sem.append(float(np.std(np.median(arr[idx], axis=1))))
    prof_T = np.array(prof_T, float); prof_med = np.array(prof_med); prof_sem = np.array(prof_sem)

    # one-parameter curvature readout: delta(T) = -n_x * log10(T / knee)
    x = np.log10(prof_T / _LEAN_KNEE_K)
    w = 1.0 / prof_sem**2
    n_x = -np.sum(w * x * prof_med) / np.sum(w * x**2)
    n_x_err = 1.0 / np.sqrt(np.sum(w * x**2))

    # -- activated-mechanism (Henry-Lang) ensemble fit to the profile, for C.
    # Same processed-template construction as arrhenius_lean_mechanisms: the
    # mechanism's straight-Arrhenius component is absorbed by a cold a + b/kT
    # fit before comparing hot residuals.
    from scipy.optimize import least_squares
    KB = 8.617333262e-5
    cold_grid = np.array([140., 145., 150., 155., 160., 165.])

    def _act_ln(T, p):
        return -np.log1p(10.0 ** p[0] * np.exp(-p[1] / (KB * np.asarray(T, float))))

    def _act_processed(p, T_hot):
        xc = 1.0 / (KB * cold_grid)
        A = np.vstack([np.ones_like(xc), xc]).T
        coef, *_ = np.linalg.lstsq(A, _act_ln(cold_grid, p), rcond=None)
        xh = 1.0 / (KB * np.asarray(T_hot, float))
        return (_act_ln(T_hot, p) - (coef[0] + coef[1] * xh)) / ln10

    act = least_squares(lambda p: (prof_med - _act_processed(p, prof_T)) / prof_sem,
                        [3.0, 0.15], bounds=([-2.0, 0.005], [10.0, 1.0]))
    act_p0, act_eb = float(act.x[0]), float(act.x[1])

    def _f_act(T):
        return np.log1p(10.0 ** act_p0 * np.exp(-act_eb / (KB * np.asarray(T, float))))

    # -- tau135 bracket
    tau135_A, tau135_B, tau135_C = [], [], []
    for T, lntau, lnerr in traps:
        corr = np.zeros_like(lntau)
        hot = T > _LEAN_KNEE_K
        if hot.any():
            corr[hot] = -np.interp(T[hot], prof_T, prof_med) * ln10
        try:
            pA = _lean_srh_fit(T, lntau, lnerr)
            pB = _lean_srh_fit(T, lntau + corr, lnerr)
            pC = _lean_srh_fit(T, lntau + _f_act(T), lnerr)
        except RuntimeError:
            continue
        tau135_A.append(float(np.exp(_srh_logtau(135.0, *pA))))
        tau135_B.append(float(np.exp(_srh_logtau(135.0, *pB))))
        tau135_C.append(float(np.exp(_srh_logtau(135.0, *pC) - _f_act(135.0))))

    _LEAN_CACHE.update(dict(
        n_ref=n_ref, prof_T=prof_T, prof_med=prof_med, prof_sem=prof_sem,
        n_x=float(n_x), n_x_err=float(n_x_err),
        act_p0=act_p0, act_eb=act_eb,
        tau135_A=np.array(tau135_A), tau135_B=np.array(tau135_B),
        tau135_C=np.array(tau135_C)))
    return _LEAN_CACHE


def arrhenius_lean_profile(out="arrhenius_lean_profile.svg", save=True):
    """The measured high-T lean profile delta(T) with the one-power overlay.

    Source: fit_dipole_spectra_minimal_caldet_err_4.h5 (re-derived at render
    time by _lean_load_and_measure)."""
    _style()
    d = _lean_load_and_measure()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(d["prof_T"], d["prof_med"], yerr=d["prof_sem"], fmt="o",
                color="0.15", ms=4.5, capsize=2, label="median hot-point offset (data)")
    tt = np.linspace(_LEAN_KNEE_K, d["prof_T"].max(), 200)
    ax.plot(tt, -d["n_x"] * np.log10(tt / _LEAN_KNEE_K), "--", color="C3", lw=2,
            label=(rf"empirical descriptor $-n\,\log_{{10}}(T/{_LEAN_KNEE_K:.0f}\,\mathrm{{K}})$,"
                   rf" $n = {d['n_x']:.2f} \pm {d['n_x_err']:.2f}$"
                   "\n(summary only — NOT a mechanism; see the shape comparison)"))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("offset from cold-anchored SRH line (dex)")
    ax.set_title(f"High-T Arrhenius lean, minimal catalog ({d['n_ref']} reference traps)")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


def arrhenius_lean_mechanisms(out="arrhenius_lean_mechanisms.svg", save=True):
    """Literature mechanism shapes vs the measured lean profile.

    Each candidate mechanism's addition Delta(T) to ln(tau) is processed
    EXACTLY like the data: first fit with the SRH family a + b/(kT) over the
    cold temperatures (140-165 K, the dominant cold coverage), so the
    cold-anchored line absorbs any straight-Arrhenius component; only the
    hot-side residual is compared to the measured profile. Free amplitudes are
    chi2-fit to the profile. Mechanisms and anchors:
      - Green mass: hole DOS mass m_dv(T) via the N_v(T) quadratic after
        Green 1990 [JAP 67, 2944] (parameter-free; 300-500 K data,
        extrapolated down);
      - gap pinning: Varshni shrinkage (alpha=4.73e-4 eV/K, beta=636 K)
        entering the hole-emission depth for a conduction-pinned level
        (Thurmond 1975; Van Vechten & Thurmond 1976), pinning fraction free;
      - activated sigma: multiphonon capture sigma_LT + sigma_inf e^{-Eb/kT}
        (Henry & Lang 1977, PRB 15, 989), (R, Eb) free;
      - power law sigma ~ T^n, n free (the naive reading of the empirical
        descriptor — included to show its processed shape does NOT fit).
    Source: fit_dipole_spectra_minimal_caldet_err_4.h5 via
    _lean_load_and_measure."""
    from scipy.optimize import least_squares
    _style()
    d = _lean_load_and_measure()
    Ts, med, sem = d["prof_T"], d["prof_med"], d["prof_sem"]
    KB = 8.617333262e-5
    VA, VB = 4.73e-4, 636.0
    cold = np.array([140., 145., 150., 155., 160., 165.])

    def process(fn, T_hot):
        xc = 1.0 / (KB * cold)
        A = np.vstack([np.ones_like(xc), xc]).T
        coef, *_ = np.linalg.lstsq(A, fn(cold), rcond=None)
        xh = 1.0 / (KB * np.asarray(T_hot, float))
        return (fn(np.asarray(T_hot, float)) - (coef[0] + coef[1] * xh)) / np.log(10.0)

    def nv_green_ln(T):
        t = np.asarray(T, float) / 300.0
        return np.log(-0.17 + 0.93 * t + 2.34 * t**2) - 1.5 * np.log(np.asarray(T, float))

    templates = {
        "Green mass (0 par)": (lambda p: (lambda T: -nv_green_ln(T)), 0, [], None),
        "gap pinning (1 par)": (lambda p: (lambda T: -p[0] * (VA * T**2 / (T + VB)) / (KB * T)),
                                1, [0.5], ([0.0], [1.5])),
        "activated $\\sigma$, Henry–Lang (2 par)":
            (lambda p: (lambda T: -np.log1p(10.0**p[0] * np.exp(-p[1] / (KB * T)))),
             2, [3.0, 0.15], ([-2.0, 0.005], [10.0, 1.0])),
        "power $\\sigma\\sim T^n$ (1 par)": (lambda p: (lambda T: -p[0] * np.log(T)),
                                             1, [1.5], ([0.0], [6.0])),
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.errorbar(Ts, med, yerr=sem, fmt="o", color="k", ms=4.5, capsize=2, zorder=5,
                label="measured lean")
    fitted = {}
    for (name, (make, npar, p0, bounds)), col in zip(templates.items(),
                                                     ["C0", "C1", "C2", "C3"]):
        if npar == 0:
            pred = process(make(None), Ts)
            pars = []
        else:
            res = least_squares(lambda p: (med - process(make(p), Ts)) / sem,
                                p0, bounds=bounds)
            pred, pars = process(make(res.x), Ts), list(res.x)
        chi2red = float(np.sum(((med - pred) / sem) ** 2)) / max(len(Ts) - npar, 1)
        fitted[name] = (pars, chi2red)
        extra = ""
        if "Henry" in name:
            extra = rf", $E_b = {pars[1]*1e3:.0f}$ meV"
        ax.plot(Ts, pred, "-", color=col, lw=1.8,
                label=rf"{name}: $\chi^2/\mathrm{{dof}} = {chi2red:.0f}${extra}")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("lean (dex)")
    ax.set_title("Mechanism shapes, processed like the data")
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


def arrhenius_lean_bracket(out="arrhenius_lean_bracket.svg", save=True):
    """tau135 census under the lean bracket: as-measured (A), hot points
    un-leaned (B), and mechanism-implied activated sigma(T) (C).

    Source: fit_dipole_spectra_minimal_caldet_err_4.h5 (re-derived at render
    time by _lean_load_and_measure)."""
    _style()
    d = _lean_load_and_measure()
    tau_A, tau_B, tau_C = d["tau135_A"], d["tau135_B"], d["tau135_C"]
    fA = float(np.mean(tau_A > _LEAN_HOUR_S)); fB = float(np.mean(tau_B > _LEAN_HOUR_S))
    fC = float(np.mean(tau_C > _LEAN_HOUR_S))
    nA = int(np.sum((tau_A > _LEAN_DRIVER_BAND[0]) & (tau_A < _LEAN_DRIVER_BAND[1])))
    nB = int(np.sum((tau_B > _LEAN_DRIVER_BAND[0]) & (tau_B < _LEAN_DRIVER_BAND[1])))
    nC = int(np.sum((tau_C > _LEAN_DRIVER_BAND[0]) & (tau_C < _LEAN_DRIVER_BAND[1])))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bins = np.linspace(-6, 10, 81)
    ax.hist(np.log10(tau_A), bins=bins, histtype="step", lw=1.6, color="0.15",
            label=f"A — hot points as measured (production): {fA*100:.2f}% > 1 hr")
    ax.hist(np.log10(tau_B), bins=bins, histtype="step", lw=1.6, color="C3",
            label=f"B — hot points un-leaned: {fB*100:.2f}% > 1 hr")
    ax.hist(np.log10(tau_C), bins=bins, histtype="step", lw=1.6, color="C0",
            label=(rf"C — activated $\sigma(T)$ taken as true "
                   rf"($E_b={d['act_eb']*1e3:.0f}$ meV): {fC*100:.2f}% > 1 hr"))
    ax.axvline(np.log10(_LEAN_HOUR_S), color="0.5", ls=":", lw=1.2)
    ax.text(np.log10(_LEAN_HOUR_S), ax.get_ylim()[1], " 1 hr", va="top", fontsize=8.5, color="0.4")
    ax.axvspan(np.log10(_LEAN_DRIVER_BAND[0]), np.log10(_LEAN_DRIVER_BAND[1]),
               alpha=0.12, color="C1",
               label=f"UL-slope driver band: {nA} (A) / {nB} (B) / {nC} (C) traps")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}\,\tau_{135}$  (s)")
    ax.set_ylabel("characterized traps")
    ax.set_title("Lean bracket on the extrapolated τ135 census")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    if save:
        path = os.path.join(HERE, out)
        fig.savefig(path)
        print(f"wrote {path}")
    return fig


if __name__ == "__main__":
    simulation_fake_image()
    simulation_source_image()
    simulation_condition_grid()
    simulation_trap_effect()
    pedestal()
    detection_gates()
    noise_model()
    catalog_funnel()
    failure_gallery()
    completeness_noise_map()
    completeness_noise_patches()
    completeness_amplitude_pc()
    completeness_example_curves()
    completeness_pdet_grid()
    completeness_characterization_maps()
    completeness_known_trap_check()
    completeness_overlay_fig()
    completeness_efficiency_hist()
    completeness_upper_limit_hist()
    results_energy_sigma()
    results_tau135_census()
    results_tau135_energy()
    results_scenario_null()
    results_scenario_upper()
    results_forest_headline()
    results_forest_clearmodes()
    arrhenius_lean_profile()
    arrhenius_lean_mechanisms()
    arrhenius_lean_bracket()
    plt.close("all")
