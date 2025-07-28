#effect of concentration on O2 release 
import numpy as np
import matplotlib.pyplot as plt

# Time
time = np.linspace(0, 48 * 3600, 1000)  # simulate for 48 hrs

# Fixed parameters
R = 0.08206
T = 310.15  # 37 °C
V_solution = 0.006
V_headspace = 0.002
H = 0.0221  # Henry's constant
half_life = 4.7 * 3600
k_base = np.log(2) / half_life

# ANT-EPO concentrations in mM
concentrations = np.linspace(1, 10, 10)  # 1 to 10 mM
o2_detected = []
o2_theoretical = []

def henry_law_detected_o2(n_total_max, k, H):
    n_total = n_total_max * (1 - np.exp(-k * time))
    denom = (V_headspace / (R * T)) + H * V_solution
    partition_fraction = (V_headspace / (R * T)) / denom
    return n_total * partition_fraction

for conc in concentrations:
    # Scale theoretical max O₂ with ANT-EPO concentration
    n_total = 13.08 * (conc / 2.18)  # base: 13.08 µmol at 2.18 mM
    o2_theoretical.append(n_total)

    # Optional: introduce non-linearity (e.g., self-quenching or limited conversion)
    n_total_effective = n_total * (1 - np.exp(-0.3 * conc))  # saturation behavior

    # Optional: simulate k suppression
    k_effective = k_base / (1 + 0.3 * conc)  # slower kinetics at high conc

    # Simulate
    n_gas = henry_law_detected_o2(n_total_effective, k_effective, H)
    o2_detected.append(n_gas[-1])  # final O₂ detected after full duration

# Plot
plt.figure(figsize=(8, 5))
plt.plot(concentrations, o2_theoretical, '--', label="Theoretical Max O₂ (µmol)")
plt.plot(concentrations, o2_detected, 'o-', label="Simulated Detected O₂ (µmol)")
plt.xlabel("ANT-EPO Concentration (mM)")
plt.ylabel("O₂ Released in Headspace (µmol)")
plt.title("O₂ Release vs ANT-EPO Concentration (with limiting effects)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
