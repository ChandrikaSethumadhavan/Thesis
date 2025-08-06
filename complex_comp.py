import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

# Style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_pressure_data(file_path):
    df = pd.read_csv(file_path)
    clean_df = df.dropna()

    # Unit conversions
    if 'O2 Released (µmol)' in clean_df.columns:
        clean_df['O2 Released (mol)'] = clean_df['O2 Released (µmol)'] / 1e6

    if 'Max O2 Possible (µmol)' in clean_df.columns:
        clean_df['Max O2 Possible (mol)'] = clean_df['Max O2 Possible (µmol)'] / 1e6

    if 'DWT denoised pressure (kPa)' in clean_df.columns:
        clean_df['Pressure (atm)'] = clean_df['DWT denoised pressure (kPa)'] / 101.325

    return clean_df

# Logistic + Henry’s law model
def logistic_henry_model_pressure(time, P_total_max, k, t_lag, H, 
                                  V_solution=0.006, V_headspace=0.002, R=0.08206, T=297.15):
    P_total = P_total_max / (1 + np.exp(-k * (time - t_lag)))
    denominator = (V_headspace / (R * T)) + (H * V_solution)
    partition_fraction = (V_headspace / (R * T)) / denominator
    P_gas = P_total * partition_fraction
    return P_gas

def fit_logistic_henry_model(df):
    time_data = df['Time (s)'].values
    pressure_measured = df['Pressure (atm)'].values

    results = {}

    try:
        P_max_measured = pressure_measured.max()
        t0_measured = time_data[np.argmax(np.gradient(pressure_measured))]

        # Initial guesses
        P_total_guess = P_max_measured * 3
        k_guess = 0.5
        t_lag_guess = t0_measured
        H_guess = 0.02

        popt, pcov = curve_fit(
            logistic_henry_model_pressure, time_data, pressure_measured,
            p0=[P_total_guess, k_guess, t_lag_guess, H_guess],
            bounds=([P_max_measured, 0, 0.8 * t0_measured, 1e-6],
                    [10 * P_max_measured, 1, 1.2 * t0_measured, 1.0]),
            maxfev=15000
        )

        P_pred = logistic_henry_model_pressure(time_data, *popt)

        results['logistic_henry'] = {
            'params': popt,
            'prediction': P_pred,
            'r2': r2_score(pressure_measured, P_pred),
            'rmse': np.sqrt(mean_squared_error(pressure_measured, P_pred)),
            'k_value': popt[1],
            'model_name': 'Logistic + Henry\'s Law'
        }

    except Exception as e:
        print(f"Model fitting failed: {e}")
        results['logistic_henry'] = None

    return results, time_data, pressure_measured

def plot_logistic_henry(results, time_data, pressure_measured):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot experimental data
    ax.scatter(time_data / 3600, pressure_measured * 1000, alpha=0.7, s=50,
               color='black', label='Experimental Data', zorder=5)

    # Plot model prediction
    result = results['logistic_henry']
    if result is not None:
        ax.plot(time_data / 3600, result['prediction'] * 1000,
                color='#4ECDC4', linestyle='--', linewidth=3,
                label=f"{result['model_name']} (k={result['k_value']:.2e} s⁻¹, R²={result['r2']:.3f})")

    ax.set_xlabel('Time (hours)', fontsize=14)
    ax.set_ylabel('Pressure (mPa)', fontsize=14)
    ax.set_title('Logistic + Henry’s Law Fit to Pressure Data', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    file_path = "M2_2.18mM_pressure_o2_release.csv"  # <- Change to your CSV file
    df = load_pressure_data(file_path)

    print(f"Data loaded: {len(df)} points")

    results, time_data, pressure_measured = fit_logistic_henry_model(df)

    result = results['logistic_henry']
    if result is not None:
        print(f"\nModel: {result['model_name']}")
        print(f"  Rate constant k: {result['k_value']:.2e} s⁻¹")
        print(f"  R²: {result['r2']:.4f}")
        print(f"  RMSE: {result['rmse']*1e6:.3f} µatm")

    plot_logistic_henry(results, time_data, pressure_measured)

if __name__ == "__main__":
    main()
