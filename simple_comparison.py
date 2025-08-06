import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load your CSV file
df = pd.read_csv( "M2_2.18mM_pressure_o2_release.csv")
clean_df = df.dropna()
# Extract time and pressure
time = df["Time (s)"]
if 'DWT denoised pressure (kPa)' in df.columns:
    df['Pressure (atm)'] = df['DWT denoised pressure (kPa)'] / 101.325
pressure_atm = df["Pressure (atm)"].values

# pressure = df['DWT denoised pressure (kPa)']

# 1. First-order exponential: y = a * (1 - exp(-b * t))
def first_order(t, a, b):
    return a * (1 - np.exp(-b * t))

# 2. Logistic fit: y = L / (1 + exp(-k*(t - t0)))
def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

def exponential(t, a, b, c):
    return a * (1 - np.exp(-b * t)) + c

# # Curve fitting with bounds
popt_first, _ = curve_fit(first_order, time, pressure_atm, p0=[max(pressure_atm), 0.01], bounds=([0, 0], [2, 1]))
popt_logistic, _ = curve_fit(logistic, time, pressure_atm, p0=[max(pressure_atm), 0.1, np.median(time)],
                             bounds=([0, 0, 0], [2, 10, max(time)]))
# popt_exp, _ = curve_fit(exponential, time, pressure_atm, p0=[pressure_atm[0], -0.01], bounds=([0, -10], [2, 0]))
popt_exp, _ = curve_fit(
    exponential,
    time,
    pressure_atm,
    p0=[0.15, 0.00, 5000]   # prevent explosive growth
)

# Generate smooth time values
t_fit = np.linspace(min(time), max(time), 500)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time, pressure_atm, 'ko', label="Raw Data", markersize=3)
plt.plot(t_fit, first_order(t_fit, *popt_first), 'r--', label="First-order Fit")
plt.plot(t_fit, logistic(t_fit, *popt_logistic), 'g-', label="Logistic Fit")
plt.plot(t_fit, exponential(t_fit, *popt_exp), 'b:', label="Exponential Fit")

plt.xlabel("Time (s)")
plt.ylabel("Pressure (atm)")
plt.title("Pressure (atm) vs Time with Curve Fits")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


