# # ===============================================================================================================================================
# # This code simulates the effect of salt on the rate of oxygen release in a solution, using a modified Henry's law model that has a dynamic adjustment for the reaction rate based on salt concentration.

# #(see below for the code with no dynamic adjustment for the reaction rate based on salt concentration)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # === Simulated Time ===
# time = np.linspace(0, 50 * 3600, 1000)  # Simulate up to 5 hours

# # === Fixed Parameters ===
# R_gas = 0.08206       # L·atm/mol·K
# T_K = 310.15          # 37 °C in Kelvin
# V_solution = 0.006    # L
# V_headspace = 0.002   # L
# n_total_max = 13.08   # µmol (theoretical max)
# H_0 = 0.0221          # Henry’s constant (mol/L/atm)
# k_s = 0.12            # Sechenov coefficient (L/mol)
# base_k = np.log(2) / (4.7 * 3600)  # First-order k at 37°C, 4.7 hrs half-life

# # === Salt concentrations ===
# salt_concs = [0.0, 0.5, 0.75, 0.9, 1.0, 2.0, 4.0, 5.0]  # mol/L NaCl
# colors = plt.cm.viridis(np.linspace(0, 1, len(salt_concs)))

# # === Function for O2 in headspace ===
# def henry_law_model(time, n_total_max, rate_constant, H, V_solution, V_headspace, R, T):
#     n_total = n_total_max * (1 - np.exp(-rate_constant * time))
#     denominator = (V_headspace / (R * T)) + (H * V_solution)
#     partition_fraction = (V_headspace / (R * T)) / denominator
#     return n_total * partition_fraction

# #a = 1.5  # sensitivity coefficient 
# a = 1.0

# # === Model how k decreases with salt ===
# def salt_dependent_k(base_k, C_salt, model='exp'):
#     """
#     Stress-test model: simulate drop in k due to salt.
#     Options:
#     - 'linear': k = base_k * (1 - a * C_salt)
#     - 'exp':    k = base_k * exp(-a * C_salt)
#     """
   
#     if model == 'linear':
#         return max(base_k * (1 - a * C_salt), 1e-6)  # every mole/L of salt reduces k by a * C_salt proportionally.
#     elif model == 'exp':
#         return base_k * np.exp(-a * C_salt) #exponential decay of k with salt.
#     else:
#         return base_k  # default fallback

# # === Simulate with salt-dependent k ===
# plt.figure(figsize=(10, 6))

# for i, C_salt in enumerate(salt_concs):
#     H_salt = H_0 * 10 ** (-k_s * C_salt)  # Henry's law correction
#     k_adjusted = salt_dependent_k(base_k, C_salt, model='exp')  # Try 'exp' too
#     n_gas = henry_law_model(time, n_total_max, k_adjusted, H_salt, V_solution, V_headspace, R_gas, T_K)
#     plt.plot(time / 3600, n_gas, label=f"{C_salt:.2f} M NaCl", color=colors[i])

# # === Plot settings ===
# plt.xlabel("Time (hours)")
# plt.ylabel("O₂ in Headspace (µmol)")
# plt.title(f"Stress Test: O₂ Detection with Salt-Dependent Reaction Rate in exp model for {a} sensitivity factor ")
# plt.legend(title="NaCl Concentration")
# plt.grid(True)
# plt.tight_layout()
# plt.show()












# # ===============================================================================================================================================

#code with no dynamic adjustment for the reaction rate based on salt concentration.
#CONSTANT paramaters assumed for all salt concentrations.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load data ===
csv_path = "M2_2.18mM_pressure_o2_release.csv"
df = pd.read_csv(csv_path)

# # Extract time column
# time = df['Time (s)'].values  # Time in seconds

time = np.linspace(0, 50 * 3600, 1000)  # 0 to 5 hours in seconds



# === Fixed parameters ===
R_gas = 0.08206       # L·atm/mol·K
T_K = 310.15          # 37 °C in Kelvin
V_solution = 0.006    # L
V_headspace = 0.002   # L
n_total_max = 13.08   # µmol (theoretical)
#n_total_max = 4.83  # µmol from actual data

H_0 = 0.0221          # Henry’s constant in pure water (mol/L/atm)
half_life_37C = 4.7 * 3600  # Convert hours to seconds
k = np.log(2) / half_life_37C

# === Sechenov correction for salt ===
k_s = 0.12  # L/mol
#salt_concs = [0.0, 0.15, 0.5, 1.0]  # in mol/L
salt_concs = [0.0, 0.5, 0.75, 0.9, 1.0, 2.0, 4.0, 5.0]  # mol/L NaCl
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





