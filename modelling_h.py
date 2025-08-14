import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from pathlib import Path

# ---------------------- USER INPUTS ----------------------
FILE_PATH     = "M2_2.18mM_pressure_o2_release.csv"  # your CSV
# Stoichiometry-based total O2 for THIS run:
N_TOTAL_O2_MOL = 13.08e-6        # mol  (2.18 mM × 6 mL → 13.08 µmol)
V_HEADSPACE    = 0.002           # L
TEMP_K         = 297.15          # K (~24 °C)
R_GAS          = 0.08206         # L·atm/(mol·K)

# Volumes for partitioning:
V_SOLUTION     = 0.006           # L

# Henry constants to sweep:
H_LIST = [0.5e-3, 1.0e-3, 1.3e-3, 2.0e-3, 3.0e-3]  # mol/(L·atm)

OUT_DIR = Path("fits_by_H_fixed_Pmax")  # images will be saved here
# --------------------------------------------------------

def load_pressure_data(file_path):
    df = pd.read_csv(file_path).dropna()
    if "DWT denoised pressure (kPa)" in df.columns:
        df["Pressure (atm)"] = df["DWT denoised pressure (kPa)"] / 101.325
    return df

def logistic_henry(time, P_total_max, k, t_lag, P_baseline, H,
                   V_solution=V_SOLUTION, V_headspace=V_HEADSPACE,
                   R=R_GAS, T=TEMP_K):
    # Logistic "would-be" total pressure
    P_total = P_baseline + (P_total_max - P_baseline) / (1.0 + np.exp(-k * (time - t_lag)))
    # Henry partitioning (constant fraction for given H)
    denom = (V_headspace / (R * T)) + (H * V_solution)
    part_frac = (V_headspace / (R * T)) / denom
    return P_baseline + (P_total - P_baseline) * part_frac

def fit_for_H_fixed_Pmax(time_s, P_meas_atm, H_value, P_total_max_fixed):
    # Baseline/plateau guesses
    n = len(P_meas_atm)
    P_baseline_guess = float(np.mean(P_meas_atm[:max(3, n//10)]))
    P_plateau_guess  = float(np.mean(P_meas_atm[-max(5, n//5):]))

    # Slope-based guesses for t_lag and k
    y_smooth = np.convolve(P_meas_atm - P_baseline_guess, np.ones(5)/5, mode="same")
    dy = np.gradient(y_smooth, time_s)
    t0_guess = float(time_s[np.argmax(dy)])
    denom = max(P_plateau_guess - P_baseline_guess, 1e-6)
    k_guess = float(abs(4.0 * np.max(dy) / denom)) if np.max(dy) > 0 else 1e-3

    # Model with P_total_max fixed; fit k, t_lag, P_baseline
    def f_fixedH(t, k, t_lag, P_baseline):
        return logistic_henry(t, P_total_max_fixed, k, t_lag, P_baseline, H_value)

    lower = [k_guess/10, t0_guess*0.5, 0.0]
    upper = [k_guess*10, t0_guess*2.0, P_plateau_guess]  # keep baseline sensible
    p0    = [k_guess,    t0_guess,      max(0.0, P_baseline_guess)]

    popt, _ = curve_fit(f_fixedH, time_s, P_meas_atm,
                        p0=p0, bounds=(lower, upper), maxfev=40000)
    pred = f_fixedH(time_s, *popt)
    r2 = r2_score(P_meas_atm, pred)
    return popt, pred, r2

def main():
    # Fix P_total_max from stoichiometry (all O2 in gas headspace)
    P_total_max_fixed = (N_TOTAL_O2_MOL * R_GAS * TEMP_K) / V_HEADSPACE  # atm
    print(f"Fixed P_total_max from stoichiometry: {P_total_max_fixed:.5f} atm")

    # Load data
    df = load_pressure_data(FILE_PATH)
    t = df["Time (s)"].values
    P = df["Pressure (atm)"].values

    OUT_DIR.mkdir(exist_ok=True)

    # Fit & plot one image per H
    for H in H_LIST:
        (k_fit, tlag_fit, Pbase_fit), pred, r2 = fit_for_H_fixed_Pmax(
            t, P, H, P_total_max_fixed
        )

        # Plot this H only
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(t/3600, P, s=18, c="black", label="Experimental Data", zorder=3)
        ax.plot(t/3600, pred, linewidth=2.8, label=f"Fit (R²={r2:.3f})")

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Pressure (atm)")
        ax.set_title(f"Logistic + Henry (P_total_max fixed)\nH = {H:.2e} mol/(L·atm)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Annotate fitted params and constants
        txt = (f"P_total_max (fixed) = {P_total_max_fixed:.4f} atm\n"
               f"k = {k_fit:.3e} s⁻¹\n"
               f"t_lag = {tlag_fit:.1f} s\n"
               f"P_baseline = {Pbase_fit:.4f} atm\n"
               f"V_sol = {V_SOLUTION:.3f} L, V_head = {V_HEADSPACE:.3f} L, T = {TEMP_K:.2f} K")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))

        fname = OUT_DIR / f"fit_fixedPmax_H_{H:.1e}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close(fig)

        print(f"H={H:.2e}  R²={r2:.3f}  k={k_fit:.3e}  t_lag={tlag_fit:.1f} s  "
              f"P_baseline={Pbase_fit:.4f} atm  ->  {fname}")

if __name__ == "__main__":
    main()
