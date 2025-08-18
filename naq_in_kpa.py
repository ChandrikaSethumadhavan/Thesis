
#==================================================

""" Note : This code considers pressure in Kpa, """



# imports, file loading and constants


import pandas as pd 
import matplotlib.pyplot as plt

csv_path  = "M2_2.18mM_pressure_o2_release.csv"
time_col  = "Time (s)"                    
press_col = "DWT denoised pressure (kPa)" # PO2 in kPa (no +ATM)
gas_col   = "O2 Released (µmol)"          # headspace O2 from csv
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]
time_h     = pd.to_numeric(df[time_col])
p_kpa      = pd.to_numeric(df[press_col])
n_gas_umol = pd.to_numeric(df[gas_col])


n_theory  = 13.08                         # µmol (theoretical)
kH = 1.283e-5           # mol/L/kPa
Vsol = 0.006           # L
Vg = 0.002
R = 8.314462           # L kPa/mol/K
T = 297.15             # K


def n_aq_umol_from_kpa(p_kpa: pd.Series) -> pd.Series:
    
    n_aq_umol = p_kpa * Vsol * kH * 1e6
    return n_aq_umol


n_aq_umol = n_aq_umol_from_kpa(p_kpa)
df_out = pd.DataFrame({
    "Elapsed Time (h)": time_h,
    "n_aq (µmol)": n_aq_umol,
    "p_kpa (kPa)": p_kpa
})

plt.figure(figsize=(7.2, 4.5))

plt.plot(df_out["Elapsed Time (h)"], df_out["n_aq (µmol)"], linewidth=2)
# plt.plot(df_out["Elapsed Time (h)"], df_out["p_kpa (kPa)"], linewidth=2)
# plt.ylabel("Pressure in kPa")
plt.xlabel("Elapsed Time (h)")
plt.ylabel("Dissolved oxygen n_aq in µmol")
plt.title("Dissolved Oxygen (n_aq) calculated from Pressure (kPa)")




plt.plot(df_out["Elapsed Time (h)"], df_out["n_aq (µmol)"], linewidth=2)

# Add detailed info box
info_text = f"""Used : kH = {kH:.2e} mol/L/kPa
R= {R} L kPa/mol/K
T = {T} K"""

plt.text(0.98, 0.02, info_text, 
         transform=plt.gca().transAxes,
         verticalalignment='bottom',
         horizontalalignment='right',
         fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))



plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#========================================================================


# Straight forward plotting of just dissolved oxygen naq (in kPa)



















#===========================================================================

# #comparison of headspace and dissolved 















#==========================================================================
# Combined values of n_aq and n_gas to see how the n_total will look like










#====================================================================
# Comparison of n_gas detected by the sensor and n_total calculated by us