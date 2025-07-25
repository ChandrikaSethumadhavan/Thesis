import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load data ===
csv_path = "M2_2.18mM_pressure_o2_release.csv"
df = pd.read_csv(csv_path)

# Extract time column
time = df['Time (s)'].values  # Time in seconds

# === Fixed parameters ===
R_gas = 0.08206       # L·atm/mol·K
T_K = 310.15          # 37 °C in Kelvin
V_solution = 0.006    # L
V_headspace = 0.002   # L
n_total_max = 13.08   # µmol (theoretical)
H_0 = 0.0221          # Henry’s constant in pure water (mol/L/atm)
half_life_37C = 4.7 * 3600  # Convert hours to seconds
k = np.log(2) / half_life_37C

# === Sechenov correction for salt ===
k_s = 0.12  # L/mol
salt_concs = [0.0, 0.15, 0.5, 1.0]  # in mol/L
colors = plt.cm.viridis(np.linspace(0, 1, len(salt_concs)))

def henry_law_model(time, n_total_max, rate_constant, H, V_solution, V_headspace, R, T):
    n_total = n_total_max * (1 - np.exp(-rate_constant * time))
    denominator = (V_headspace / (R * T)) + (H * V_solution)
    partition_fraction = (V_headspace / (R * T)) / denominator
    return n_total * partition_fraction

# === Simulate O2 release at different salt concentrations ===
plt.figure(figsize=(10, 6))

for i, C_salt in enumerate(salt_concs):
    H_salt = H_0 * 10 ** (-k_s * C_salt)
    n_gas = henry_law_model(time, n_total_max, k, H_salt, V_solution, V_headspace, R_gas, T_K)
    plt.plot(time / 3600, n_gas, label=f"{C_salt:.2f} M NaCl", color=colors[i])  # Convert time to hours

# === Plot formatting ===
plt.xlabel("Time (hours)")
plt.ylabel("O₂ Released in Headspace (µmol)")
plt.title("O₂ Release at Varying Salt Concentrations (2.18 mM ANT-EPO, 37 °C)")
plt.legend(title="Salt Concentration")
plt.grid(True)
plt.tight_layout()
plt.show()
