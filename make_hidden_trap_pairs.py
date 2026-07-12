"""Generate the (tau135, sigma) MIXTURE pairs file used by the CCD simulation's
efficiency-corrected / upper-limit trap populations (`upper`, `effcorr`).

------------------------------------------------------------------------------
Why this file exists (the bug it fixes)
------------------------------------------------------------------------------
The upper-limit / efficiency-corrected seed histograms inflate the per-tau135
trap count by 1/eps(tau135) to represent traps the pocket-pumping measurement
would have MISSED (completeness_efficiency.qmd sec.8). But the *standard* pairs
file (make_trap_pairs.py) assigns every simulated trap a sigma resampled from the
DETECTED/characterized catalog, which by selection has detectable (large) sigma.
So the inflated population was injecting the missed traps with the sigma of the
caught traps -- "injecting traps we would have caught" -- overstating their SER
impact (SRH capture rate kc ~ sigma; ccd_simulation.py trap_kc = sigma*v_th/V).

------------------------------------------------------------------------------
The fix: a per-tau MIXTURE of measured and conditional-missed sigma
------------------------------------------------------------------------------
Each inflated tau bin (count N_det/eps) is really two groups: a fraction eps(tau)
are measured-equivalent traps and a fraction 1-eps(tau) are the missed ones. So
the correct sigma law per tau is a mixture:

    sigma | tau ~  eps(tau)      * [ measured sigma | tau ]
                 + (1 - eps(tau)) * [ conditional-missed sigma | tau ]

- measured sigma | tau : resampled from the measured pairs
  (trap_tau135_sigma_pairs<flavor>.npz), exactly as the baseline sim does, so the
  measured-equivalent fraction KEEPS the detected cross sections.
- conditional-missed sigma | tau : the missed traps are conditioned on detection
  FAILURE, so their energies follow
        q(E | tau, missed)  proportional to  g_obs(E) * (1 - Pbar_char(tau, E))
  (Bayes on the observed-E prior; the standard count-completeness derivation).
  Each sampled E maps to sigma via the SRH relation dipole.log_energy_cross_section
  at 135 K (logsigma = L0(E) - log(tau135)). We SAMPLE the distribution (not a
  scalar) because capture/recapture is nonlinear in sigma, so the sim must average
  the response, not a representative sigma.

eps(tau) and Pbar_char are taken with the SAME convention figure_utils.load_method3
uses to build the seed histograms: Pbar_char = raw p_characterized * 0.972
(ENERGY_FIT_SURVIVAL), eps = mean of Pbar_char over the observed-E points. That
keeps the mixture ratio consistent with the histogram counts (no double use of
Pbar_char: it sets the count normalization N_det/eps once, and the missed
composition q(E|missed) separately; their product is the expected missed component
of a parent population N_det/eps).

When the sim resamples this file with its K=20 nearest-in-log(tau) window
(ccd_simulation.py ~1210), it reproduces the mixture: short/mid tau (eps~1) ->
mostly measured sigma; long tau (eps small) -> mostly conditional-missed sigma.
Run `python claude_scripts/verify_hidden_pairs_mixture.py` (closure) to confirm the
resampled distribution matches the intended mixture before trusting the file.

------------------------------------------------------------------------------
Scope / conditionality (this is a SCENARIO, not an upper limit)
------------------------------------------------------------------------------
Paired with the effcorr (point-estimate) or upper-limit histogram, this answers
"IF a hidden population consistent with the completeness point-estimate existed,
how much of the measured SER residual would it explain?" Conditional on: (a)
hidden traps sharing the observed-E prior g_obs(E); (b) the amplitude prior /
A-independent-of-(E,tau) assumption. Empirical support for (b): trap depth D_t
depends only weakly on sigma across the detected catalog (Pearson(log sigma, log
depth)=+0.38; depth rises ~3x over 5 decades of sigma, i.e. depth ~ sigma^0.1, far
from the depth ~ sigma of unsaturated capture), and the injected sigma range is
already populated by real characterized traps -- so this is empirical support for
weak dependence over the measured range, NOT a proof of pumping saturation.

Usage:
    python make_hidden_trap_pairs.py \
        --stage09 trap_completeness_method3/cache/09_characterization_probability_minimal_caldet_v1.h5 \
        --measured-pairs trap_tau135_sigma_pairs_minimal_caldet.npz \
        --out trap_tau135_sigma_pairs_minimal_caldet_hidden.npz
"""
import argparse

import h5py
import numpy as np

from dipole import log_energy_cross_section
from figure_utils import ENERGY_FIT_SURVIVAL

T_ANCHOR = 135.0  # K -- tau135 anchor for the SRH E<->sigma mapping


def sigma_of_E(tau135, E):
    """sigma [cm^2] at fixed tau135 and energy E via SRH:
    logsigma = L0(E) - log(tau135), L0(E) = log_energy_cross_section(135, E, 0)."""
    return np.exp(log_energy_cross_section(T_ANCHOR, E, 0.0) - np.log(tau135))


def make_mixture_pairs(stage09, measured_pairs, out,
                       n_nodes=2000, per_node=20, k_measured=20, seed=0):
    rng = np.random.default_rng(seed)

    with h5py.File(stage09, "r") as f:
        tau_grid = f["grid/tau_135_seconds"][:]                    # (Ntau,)
        E_grid = f["grid/E_eV"][:]                                 # (NE,)
        p4_raw = f["results/p_characterized_n_good_4"][:]          # (Ntau, NE)
        observed_E = f["validation_known_traps/n_good_4_csv/E_eV"][:]  # (Nobs,)

    p4 = p4_raw * ENERGY_FIT_SURVIVAL                              # match load_method3
    logtau_grid = np.log(tau_grid)

    # Pmat[t, i] = Pbar_char at map-tau node t, evaluated at observed-E point i.
    # (161 E-interps total; then interpolate in log-tau to arbitrary nodes.)
    Pmat = np.array([np.interp(observed_E, E_grid, p4[t]) for t in range(len(tau_grid))])

    # Measured pairs, sorted in log-tau for K-nearest resampling (as the sim does).
    md = np.load(measured_pairs)
    m_order = np.argsort(md['tau135'])
    m_logtau = np.log(md['tau135'][m_order])
    m_sigma = md['sigma'][m_order]
    # Physical ceiling: no hidden trap should exceed the largest OBSERVED sigma.
    # The SRH degeneracy maps grid-edge E to unphysically large sigma at short tau
    # (tau<1e-3 s, fast-release traps); clip to keep capture rates physical.
    sigma_ceiling = float(m_sigma.max())

    def measured_sigma_at(logtau, n):
        ins = np.searchsorted(m_logtau, logtau)
        lo = np.clip(ins - k_measured // 2, 0, len(m_sigma) - k_measured)
        return m_sigma[lo + rng.integers(0, k_measured, size=n)]

    # Dense output nodes across the map tau range; each node emits `per_node`
    # points, tau-jittered within the node so no exact duplicates confuse the
    # sim's searchsorted window.
    node_logtau = np.linspace(logtau_grid[0], logtau_grid[-1], n_nodes)
    dlog = node_logtau[1] - node_logtau[0]

    taus_out, sig_out, E_out, is_hidden = [], [], [], []
    eps_check = {}
    for lt in node_logtau:
        # Pbar_char over observed-E at this node (interp between map-tau rows).
        j = np.clip(np.searchsorted(logtau_grid, lt) - 1, 0, len(tau_grid) - 2)
        w = (lt - logtau_grid[j]) / (logtau_grid[j + 1] - logtau_grid[j])
        Prow = (1 - w) * Pmat[j] + w * Pmat[j + 1]            # (Nobs,)
        eps = float(np.mean(Prow))                            # count-consistent eps
        miss_w = np.clip(1.0 - Prow, 0.0, None)               # q(E|missed) weights
        eps_check[float(np.exp(lt))] = eps

        n_hidden = int(rng.binomial(per_node, min(max(1 - eps, 0.0), 1.0)))
        n_meas = per_node - n_hidden
        pt_logtau = lt + rng.uniform(-dlog / 2, dlog / 2, size=per_node)

        # measured-equivalent fraction: keep detected sigma
        if n_meas:
            ms = measured_sigma_at(lt, n_meas)
            taus_out.append(np.exp(pt_logtau[:n_meas]))
            sig_out.append(ms); E_out.append(np.full(n_meas, np.nan))
            is_hidden.append(np.zeros(n_meas, bool))
        # missed fraction: conditional-missed sigma
        if n_hidden:
            wsum = miss_w.sum()
            if wsum <= 0:      # eps~1 degenerate: fall back to measured
                ms = measured_sigma_at(lt, n_hidden)
                taus_out.append(np.exp(pt_logtau[n_meas:]))
                sig_out.append(ms); E_out.append(np.full(n_hidden, np.nan))
                is_hidden.append(np.zeros(n_hidden, bool))
            else:
                Es = rng.choice(observed_E, size=n_hidden, p=miss_w / wsum)
                taus_h = np.exp(pt_logtau[n_meas:])
                sig_out.append(np.minimum(sigma_of_E(taus_h, Es), sigma_ceiling))
                taus_out.append(taus_h); E_out.append(Es)
                is_hidden.append(np.ones(n_hidden, bool))

    tau135 = np.concatenate(taus_out)
    sigma = np.concatenate(sig_out)
    energy = np.concatenate(E_out)
    hidden = np.concatenate(is_hidden)
    order = np.argsort(tau135)
    tau135, sigma, energy, hidden = tau135[order], sigma[order], energy[order], hidden[order]

    # eps cross-check vs the notebook (completeness_efficiency.qmd sec.4).
    print("eps cross-check (expect ~0.31, 0.011, 0.0043 for minimal_caldet):")
    for tau, want in [(1e5, 0.31), (1e6, 0.011), (1e7, 0.0043)]:
        i = min(eps_check, key=lambda t: abs(np.log(t) - np.log(tau)))
        print(f"  tau135~{tau:.0e}: eps={eps_check[i]:.4f} (notebook {want})")

    np.savez(out, tau135=tau135, sigma=sigma, energy=energy, is_hidden=hidden)
    frac_h = hidden.mean()
    print(f"\nMixture pairs: {len(tau135)} points "
          f"({frac_h:.1%} conditional-missed, {1-frac_h:.1%} measured).")
    print(f"tau135 {tau135.min():.2e}..{tau135.max():.2e} s; "
          f"sigma {sigma.min():.2e}..{sigma.max():.2e} cm^2")
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage09", type=str,
                   default="trap_completeness_method3/cache/"
                           "09_characterization_probability_minimal_caldet_v1.h5")
    p.add_argument("--measured-pairs", type=str,
                   default="trap_tau135_sigma_pairs_minimal_caldet.npz")
    p.add_argument("--out", type=str,
                   default="trap_tau135_sigma_pairs_minimal_caldet_hidden.npz")
    p.add_argument("--n-nodes", type=int, default=2000)
    p.add_argument("--per-node", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    make_mixture_pairs(args.stage09, args.measured_pairs, args.out,
                       n_nodes=args.n_nodes, per_node=args.per_node, seed=args.seed)


if __name__ == "__main__":
    main()
