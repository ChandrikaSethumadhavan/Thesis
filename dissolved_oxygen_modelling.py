# Schlenk Setup Parameters
V_solution_ml = 6.0                  # mL
V_headspace_ml = 2.0                # mL
T_C = 24.0                          # °C
R = 0.08206                         # L·atm/(mol·K)
H = 1.30e-3                         # mol/(L·atm)
O2_total_umol = 13.08               # µmol from stoichiometry (1 ANT-EPO → 1 O2)
ANT_EPO_conc_mM = 2.5               # mM concentration

# Convert units
V_solution = V_solution_ml / 1000   # L
V_headspace = V_headspace_ml / 1000 # L
T_K = T_C + 273.15                  # K
O2_total_mol = O2_total_umol / 1e6  # mol

# Theoretical max pressure (all O2 in gas phase)
P_total_theoretical = (O2_total_mol * R * T_K) / V_headspace  # atm

# Measured pressure from CSV / sensor
P_measured = 0.059                 # atm (from your data)

# Dissolved pressure (not seen by sensor)
P_dissolved = P_total_theoretical - P_measured  # atm

# Convert dissolved pressure to mols using Ideal Gas Law
# PV = nRT → n = PV / RT
O2_dissolved_mol = (P_dissolved * V_headspace) / (R * T_K)
O2_dissolved_umol = O2_dissolved_mol * 1e6       # µmol

# Sanity check
O2_measured_umol = O2_total_umol - O2_dissolved_umol

# Print results
print("=== Schlenk Setup Validation ===")
print(f"Theoretical Total Pressure (atm): {P_total_theoretical:.6f}")
print(f"Measured Pressure (atm): {P_measured:.6f}")
print(f"Dissolved O2 Pressure (atm): {P_dissolved:.6f}")
print(f"O2 Dissolved in DI Water: {O2_dissolved_umol:.4f} µmol")
print(f"O2 Measured by Sensor: {O2_measured_umol:.4f} µmol")
