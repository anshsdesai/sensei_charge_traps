import sys, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trap_completeness_method3" / "src"))
import numpy as np
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from validation_sensitivity import read_known_traps, tau_at_temperature

CACHE = ROOT / "trap_completeness_method3" / "cache"
LEG = CACHE / "01_records_ngood4.csv"
MIN = CACHE / "01_records_minimal_caldet_ngood4.csv"

with h5py.File(CACHE / "08_pdet_grid_tau1000_v1.h5", "r") as h5:
    temps = np.sort(h5["grid/temperature_K"][:].astype(float))
itemps = [int(round(T)) for T in temps]
bins = np.geomspace(1e-7, 1e8, 75)
centers = np.sqrt(bins[:-1] * bins[1:])

def parse_set(s):
    s = (s or "").strip()
    return set(int(round(float(x))) for x in s.split(",") if x.strip()) if s else set()

def naive_curve(csv_path):
    rec = read_known_traps(csv_path)
    tauT = tau_at_temperature(rec["tau_135_seconds"], rec["E_eV"], temps)  # (N,T)
    good = [set(int(round(x)) for x in gs) for gs in rec["good_temperatures"]]
    measured = np.zeros(tauT.shape, bool)
    for i, gs in enumerate(good):
        for t, T in enumerate(itemps):
            measured[i, t] = T in gs
    total, _ = np.histogram(tauT.reshape(-1), bins=bins)
    meas, _ = np.histogram(tauT[measured], bins=bins)
    eff = np.divide(meas, total, out=np.zeros_like(total, float), where=total > 0)
    return eff, total

def per_T_rate(csv_path):
    good_n = {T: 0 for T in itemps}; meas_n = {T: 0 for T in itemps}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            gs = parse_set(r.get("good_temperatures_K", ""))
            ms = parse_set(r.get("measured_temperatures_K", ""))
            for T in itemps:
                if T in ms:
                    meas_n[T] += 1
                    if T in gs:
                        good_n[T] += 1
    return np.array([good_n[T] / meas_n[T] if meas_n[T] else np.nan for T in itemps])

def window(eff, total, lo, hi, agg):
    sel = (centers >= lo) & (centers <= hi) & (total > 0)
    return float(agg(eff[sel])) if sel.any() else float("nan")

print("grid temps:", itemps)
eff_L, tot_L = naive_curve(LEG)
eff_M, tot_M = naive_curve(MIN)
rate_L, rate_M = per_T_rate(LEG), per_T_rate(MIN)

for name, eff, tot in [("legacy ", eff_L, tot_L), ("minimal", eff_M, tot_M)]:
    plat = window(eff, tot, 1e-4, 1e-2, np.mean)
    dip = window(eff, tot, 3e-3, 3e-2, np.min)
    peak = window(eff, tot, 1e-1, 2.0, np.mean)
    print(f"{name}: plateau(1e-4..1e-2)={plat:.3f}  dip_min(3e-3..3e-2)={dip:.3f}  "
          f"peak(1e-1..2)={peak:.3f}  dip/peak={dip/peak:.2f}")

print("\nper-T good-fit rate (good/measured):")
print("  T      :", " ".join(f"{T:4d}" for T in itemps))
print("  legacy :", " ".join(f"{x:4.2f}" for x in rate_L))
print("  minimal:", " ".join(f"{x:4.2f}" for x in rate_M))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
a1.plot(centers[tot_L > 0], eff_L[tot_L > 0], "o-", ms=3, color="#b3331f", label="legacy")
a1.plot(centers[tot_M > 0], eff_M[tot_M > 0], "o-", ms=3, color="#1f5fa8", label="minimal")
a1.axvspan(3e-3, 3e-2, color="gray", alpha=0.15, label="dip window")
a1.set_xscale("log"); a1.set_xlabel(r"$\tau_e(T)$ [s]")
a1.set_ylabel("Observed naive efficiency"); a1.legend(); a1.set_ylim(-0.02, 1.05)
a1.set_title("Naive efficiency vs tau(T)")
a2.plot(temps, rate_L, "o-", color="#b3331f", label="legacy")
a2.plot(temps, rate_M, "o-", color="#1f5fa8", label="minimal")
a2.set_xlabel("Temperature [K]"); a2.set_ylabel("per-T good-fit rate (good/measured)")
a2.set_title("Per-temperature fit survival"); a2.legend(); a2.set_ylim(0, 1.05)
fig.tight_layout()
out = ROOT / "figures" / "naive_dip_legacy_vs_minimal.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print("\nsaved", out)
