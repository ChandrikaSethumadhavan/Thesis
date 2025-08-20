
#==================================================

""" Note : This code considers pressure in Kpa
kH was calculated in the following way:
1. 1.3 x 10^-3 mol / L /atm 
2. 1 atm = 101.325 kPa
3. kH = 1.3 x 10^-3 mol / L / atm * (1 atm / 101.325 kPa) = 1.283 x 10^-5 mol / L / kPa        #note it is 1/atm not just atm."""




# imports, file loading and constants


# import pandas as pd 
# import matplotlib.pyplot as plt

# csv_path  = "M2_2.18mM_pressure_o2_release.csv"
# time_col  = "Time (s)"                    
# press_col = "DWT denoised pressure (kPa)" # PO2 in kPa (no +ATM)
# gas_col   = "O2 Released (µmol)"          # headspace O2 from csv
# df = pd.read_csv(csv_path)
# df.columns = [c.strip() for c in df.columns]
# time_h     = pd.to_numeric(df[time_col])
# p_kpa      = pd.to_numeric(df[press_col])
# n_gas_umol = pd.to_numeric(df[gas_col])


# n_theory  = 13.08                         # µmol (theoretical)
# kH = 1.283e-5           # mol/L/kPa
# Vsol = 0.006           # L
# Vg = 0.002
# R = 8.314462           # L kPa/mol/K
# T = 297.15             # K


# def n_aq_umol_from_kpa(p_kpa: pd.Series) -> pd.Series:
    
#     n_aq_umol = p_kpa * Vsol * kH * 1e6
#     return n_aq_umol

# def plotting_pressure(p_kpa: pd.Series) -> pd.Series:

#     pressure_plot = p_kpa
#     return pressure_plot

# def plotting_pressure_in_atm(p_kpa: pd.Series) -> pd.Series:
#     pressure_atm = p_kpa / 101.325
#     return pressure_atm


# #n_aq_umol = n_aq_umol_from_kpa(p_kpa)
# # pressure_plot = plotting_pressure(p_kpa)
# pressure_atm_plot = plotting_pressure_in_atm(p_kpa)
# df_out = pd.DataFrame({
#     "Elapsed Time (h)": time_h,
#     # "n_aq (µmol)": n_aq_umol,
#     "p_kpa (kPa)": p_kpa,
#     "p_atm (atm)": pressure_atm_plot
# })

# plt.figure(figsize=(7.2, 4.5))



# plt.plot(df_out["Elapsed Time (h)"], df_out["p_atm (atm)"], linewidth=2)
# plt.ylabel("Pressure in atm")
# plt.xlabel("Elapsed Time (h)")
# plt.title("Pressure (atm) profile throughout experiment")





# plt.plot(df_out["Elapsed Time (h)"], df_out["p_kpa (kPa)"], linewidth=2)
# plt.ylabel("Pressure in kPa")
# plt.xlabel("Elapsed Time (h)")
# plt.title("Pressure (kPa) profile throughout experiment")



# plt.plot(df_out["Elapsed Time (h)"], df_out["n_aq (µmol)"], linewidth=2)
# plt.ylabel("Dissolved oxygen n_aq in µmol")
# plt.xlabel("Elapsed Time (h)")
# plt.title("Dissolved Oxygen (n_aq) calculated from Pressure (kPa)")






# # Add detailed info box
# info_text = f"""Used : kH = {kH:.2e} mol/L/kPa
# R= {R} L kPa/mol/K
# T = {T} K"""

# plt.text(0.98, 0.02, info_text, 
#          transform=plt.gca().transAxes,
#          verticalalignment='bottom',
#          horizontalalignment='right',
#          fontsize=9,
#          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))



# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

#========================================================================


# Using the equation directly to plot graphs
""" Units used:
K, atm, mol, L, 
KH = 1.3 x 10^-3 mol / L / atm, 
R = 0.082057 L atm / K / mol, 
T = 297.15 K





Using : Vg / (R * T) + kH * Vsol


"""






# import pandas as pd 
# import matplotlib.pyplot as plt

# csv_path  = "M2_2.18mM_pressure_o2_release.csv"
# time_col  = "Time (s)"                    
# press_col = "DWT denoised pressure (kPa)" # PO2 in kPa (no +ATM)
# gas_col   = "O2 Released (µmol)"          # headspace O2 from csv
# df = pd.read_csv(csv_path)
# df.columns = [c.strip() for c in df.columns]
# time_h     = pd.to_numeric(df[time_col])
# p_kpa      = pd.to_numeric(df[press_col])
# p_atm      = p_kpa / 101.325
# n_gas_umol = pd.to_numeric(df[gas_col])

# #constants
# Vg = 0.002           # L
# Vsol = 0.006         # L
# T = 297.15  
# KH = 1.3e-03
# R = 0.082057




# def calculate_n_total (p_atm: pd.Series, Vg: float, Vsol: float, R: float, T: float, KH: float) -> pd.Series:
#     ntot = p_atm * ( (Vg / (R*T)) + (KH * Vsol)) 
#     ntot_umol = ntot * 1e6  # Convert to µmol
#     return ntot_umol



# ntot_umol = calculate_n_total(p_atm, Vg, Vsol, R, T, KH)

# print(ntot_umol.values.max())
# df_out = pd.DataFrame({
#     "Elapsed Time (h)": time_h,
#     "n_total (µmol)": ntot_umol
# })






# plt.figure(figsize=(7.2, 4.5))
# plt.plot(df_out["Elapsed Time (h)"], df_out["n_total (µmol)"], linewidth=2)
# plt.ylabel("n_total (µmol)")
# plt.xlabel("Elapsed Time (h)")
# plt.title("Total oxygen (n_total) profile throughout experiment")

# # Add detailed info box
# info_text = f"""Used : kH = {KH:.2e} mol/L/atm
# R= {R} L atm/mol/K
# T = {T} K
# Max o2 value in umol = {ntot_umol.values.max():.3f}"""

# plt.text(0.98, 0.02, info_text, 
#          transform=plt.gca().transAxes,
#          verticalalignment='bottom',
#          horizontalalignment='right',
#          fontsize=9,
#          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))




# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()













#===========================================================================

# Same but instead of convertiung units to atm from kpa, keeping pressure units in Kpa

""" Units used:
K, kpa, mol, L, 
KH = 1.3 x 10^-3 mol / L / atm, 
R = 0.082057 L atm / K / mol, 
T = 297.15 K


Sticking with K instead of C. The ideal gas law and henry's law both scale with kelvin so using celcius scaling will become inappropriate.


Using : Vg / (R * T) + kH * Vsol


"""

# import pandas as pd 
# import matplotlib.pyplot as plt

# csv_path  = "M2_2.18mM_pressure_o2_release.csv"
# time_col  = "Time (s)"                    
# press_col = "DWT denoised pressure (kPa)" # PO2 in kPa (no +ATM)
# gas_col   = "O2 Released (µmol)"          # headspace O2 from csv
# df = pd.read_csv(csv_path)
# df.columns = [c.strip() for c in df.columns]
# time_h     = pd.to_numeric(df[time_col])
# p_kpa      = pd.to_numeric(df[press_col])
# n_gas_umol = pd.to_numeric(df[gas_col])

# #constants
# KH = 1.283e-5           # mol/L/kPa
# Vsol = 0.006           # L
# Vg = 0.002  
# R = 8.314462           # L kPa/mol/K
# T = 297.15             # K



# def calculate_n_total (p_kpa: pd.Series, Vg: float, Vsol: float, R: float, T: float, KH: float) -> pd.Series:
#     ntot = p_kpa * ( (Vg / (R*T)) + (KH * Vsol)) 
#     ntot_umol = ntot * 1e6  # Convert to µmol
#     return ntot_umol



# ntot_umol = calculate_n_total(p_kpa, Vg, Vsol, R, T, KH)

# print(ntot_umol.values.max())
# df_out = pd.DataFrame({
#     "Elapsed Time (h)": time_h,
#     "n_total (µmol)": ntot_umol
# })






# plt.figure(figsize=(7.2, 4.5))
# plt.plot(df_out["Elapsed Time (h)"], df_out["n_total (µmol)"], linewidth=2)
# plt.ylabel("n_total (µmol)")
# plt.xlabel("Elapsed Time (h)")
# plt.title("Total oxygen (n_total) profile throughout experiment")

# # Add detailed info box
# info_text = f"""Used : kH = {KH:.2e} mol/L/kPa
# R= {R} L kPa/mol/K
# T = {T} K
# Max o2 value in umol = {ntot_umol.values.max():.3f}"""

# plt.text(0.98, 0.02, info_text, 
#          transform=plt.gca().transAxes,
#          verticalalignment='bottom',
#          horizontalalignment='right',
#          fontsize=9,
#          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))




# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()















#==========================================================================


#===========================================================================

# converting units from kpa to atm only at the end so that no miscalculations occur

""" Units used:
K, kpa, mol, L, 
KH = 1.283e-5          
R = 8.314462  
T = 297.15 K


Sticking with K instead of C. The ideal gas law and henry's law both scale with kelvin so using celcius scaling will become inappropriate.


Using : Vg / (R * T) + kH * Vsol


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- inputs ----------------
csv_path  = "M2_2.18mM_pressure_o2_release.csv"
time_col  = "Time (s)"                     # your column is actually hours
press_col = "DWT denoised pressure (kPa)"  # pressure in kPa
ASSUME_GAUGE = False   # True -> add 101.325 to make absolute; False -> already absolute
# ----------------------------------------

# Load & clean
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]
t_h   = pd.to_numeric(df[time_col], errors="coerce")
P_kPa = pd.to_numeric(df[press_col], errors="coerce")
mask = t_h.notna() & P_kPa.notna()
t_h, P_kPa = t_h[mask].reset_index(drop=True), P_kPa[mask].reset_index(drop=True)
t_h = t_h - t_h.iloc[0]  # start time at 0 h

# Absolute pressure
P_abs_kPa = P_kPa + 101.325 if ASSUME_GAUGE else P_kPa
P_abs_atm = P_abs_kPa / 101.325

# Geometry & constants
Vg   = 0.002     # L
Vsol = 0.006     # L
T_K  = 297.15    # K

# --- (A) kPa-guided calculation ---
R_kPa  = 8.31446261815324              # L·kPa·mol^-1·K^-1
kH_kPa = 1.3e-3 / 101.325              # mol·L^-1·kPa^-1 (~1.283e-5)
A_kPa  = Vg/(R_kPa*T_K) + kH_kPa*Vsol  # mol/kPa
n_tot_kPa_umol = P_abs_kPa * A_kPa * 1e6   # µmol

# --- (B) atm-guided calculation ---
R_atm  = 0.082057366080960             # L·atm·mol^-1·K^-1
kH_atm = 1.3e-3                        # mol·L^-1·atm^-1
A_atm  = Vg/(R_atm*T_K) + kH_atm*Vsol  # mol/atm
n_tot_atm_umol = P_abs_atm * A_atm * 1e6   # µmol

# Residuals
final_kpa = max(n_tot_kPa_umol)
final_atm = max(n_tot_atm_umol)
diff_umol = final_kpa - final_atm
print(f"max values of o2 in umol produced by keeping pressure in kpa : {final_kpa}")
print(f"max values of o2 in umol produced by keeping pressure in atm : {final_atm}")
print(f"Max |difference|: {np.abs(diff_umol):.6e} µmol")



#next task :  how to know where exactly (the values which are 
# differing) this difference of print(f"Max |difference|: {np.max(np.abs(diff_umol)):.6e} µmol") is happening between both of them?

#literature vant hoff and KK'
# model KH









#====================================================================
# Comparison of n_gas detected by the sensor and n_total calculated by us