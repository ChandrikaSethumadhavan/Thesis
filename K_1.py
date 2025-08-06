import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Load your CSV file
df = pd.read_csv("M2_2.18mM_pressure_o2_release.csv")
clean_df = df.dropna()

# Extract time and pressure
time = df["Time (s)"]
if 'DWT denoised pressure (kPa)' in df.columns:
    df['Pressure (atm)'] = df['DWT denoised pressure (kPa)'] / 101.325
pressure_atm = df["Pressure (atm)"].values

# 1. First-order exponential: y = a * (1 - exp(-b * t))
def first_order(t, a, b):
    return a * (1 - np.exp(-b * t))

# 2. Logistic fit: y = L / (1 + exp(-k*(t - t0)))
def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

# 3. Exponential with offset: y = a * (1 - exp(-b * t)) + c
def exponential(t, a, b, c):
    return a * (1 - np.exp(-b * t)) + c

# Curve fitting with bounds
print("="*60)
print("FITTING RESULTS - RATE CONSTANT EXTRACTION")
print("="*60)

# Fit first-order model
popt_first, _ = curve_fit(first_order, time, pressure_atm, 
                         p0=[max(pressure_atm), 0.01], 
                         bounds=([0, 0], [2, 1]))
pred_first = first_order(time, *popt_first)
r2_first = r2_score(pressure_atm, pred_first)

print(f"\n1. FIRST-ORDER MODEL: y = a * (1 - exp(-k*t))")
print(f"   Parameters: a = {popt_first[0]:.6f} atm, k = {popt_first[1]:.6e} s⁻¹")
print(f"   Rate constant k₁ = {popt_first[1]:.6e} s⁻¹")


# Fit logistic model
popt_logistic, _ = curve_fit(logistic, time, pressure_atm, 
                            p0=[max(pressure_atm), 0.1, np.median(time)],
                            bounds=([0, 0, 0], [2, 10, max(time)]))
pred_logistic = logistic(time, *popt_logistic)
r2_logistic = r2_score(pressure_atm, pred_logistic)

print(f"\n2. LOGISTIC MODEL: y = L / (1 + exp(-k*(t - t₀)))")
print(f"   Parameters: L = {popt_logistic[0]:.6f} atm, k = {popt_logistic[1]:.6e} s⁻¹, t₀ = {popt_logistic[2]:.2f} s")
print(f"   Rate constant k₂ = {popt_logistic[1]:.6e} s⁻¹")


# Fit exponential with offset
popt_exp, _ = curve_fit(exponential, time, pressure_atm,
                       p0=[0.15, 0.00001, 0.001])
pred_exp = exponential(time, *popt_exp)
r2_exp = r2_score(pressure_atm, pred_exp)

print(f"\n3. EXPONENTIAL MODEL: y = a * (1 - exp(-k*t)) + c")
print(f"   Parameters: a = {popt_exp[0]:.6f} atm, k = {popt_exp[1]:.6e} s⁻¹, c = {popt_exp[2]:.6f} atm")
print(f"   Rate constant k₃ = {popt_exp[1]:.6e} s⁻¹")


# Generate smooth time values
t_fit = np.linspace(min(time), max(time), 500)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(time, pressure_atm, 'ko', label="Raw Data", markersize=3)
plt.plot(t_fit, first_order(t_fit, *popt_first), 'r--', linewidth=2, 
         label=f"First-order Fit (k₁ = {popt_first[1]:.2e} s⁻¹)")
plt.plot(t_fit, logistic(t_fit, *popt_logistic), 'g-', linewidth=2, 
         label=f"Logistic Fit (k₂ = {popt_logistic[1]:.2e} s⁻¹)")
plt.plot(t_fit, exponential(t_fit, *popt_exp), 'b:', linewidth=2, 
         label=f"Exponential Fit (k₃ = {popt_exp[1]:.2e} s⁻¹)")

plt.xlabel("Time (s)", fontsize=12, fontweight='bold')
plt.ylabel("Pressure (atm)", fontsize=12, fontweight='bold')
plt.title("Pressure (atm) vs Time with Curve Fits", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Add text box with rate constants
rate_constants_text = f"""Rate Constants Extracted:
• First-order: k₁ = {popt_first[1]:.2e} s⁻¹
• Logistic: k₂ = {popt_logistic[1]:.2e} s⁻¹  
• Exponential: k₃ = {popt_exp[1]:.2e} s⁻¹
"""

plt.text(0.02, 0.98, rate_constants_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.show()

# Summary comparison
print(f"\n" + "="*60)
print("RATE CONSTANT COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<15} {'Rate Constant (s⁻¹)':<20} {'R²':<8} {'Best Fit?':<10}")
print("-" * 55)
print(f"{'First-order':<15} {popt_first[1]:.2e}            {r2_first:.3f}    {r2_first == max(r2_first, r2_logistic, r2_exp) and '✓' or ''}")
print(f"{'Logistic':<15} {popt_logistic[1]:.2e}            {r2_logistic:.3f}    {r2_logistic == max(r2_first, r2_logistic, r2_exp) and '✓' or ''}")
print(f"{'Exponential':<15} {popt_exp[1]:.2e}            {r2_exp:.3f}    {r2_exp == max(r2_first, r2_logistic, r2_exp) and '✓' or ''}")

# Rate constant differences
k_values = [popt_first[1], popt_logistic[1], popt_exp[1]]
k_mean = np.mean(k_values)
k_std = np.std(k_values)

print(f"\nRate Constant Statistics:")
print(f"  Mean: {k_mean:.2e} s⁻¹")
print(f"  Std Dev: {k_std:.2e} s⁻¹")
print(f"  Coefficient of Variation: {(k_std/k_mean)*100:.1f}%")

if (k_std/k_mean)*100 < 20:
    print(f"  ✅ Rate constants are consistent across models (<20% variation)")
else:
    print(f"  ⚠️  Rate constants vary significantly across models (>20% variation)")