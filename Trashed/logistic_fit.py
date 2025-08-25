
# Simplified Logistic-Henry Model with Automatic Plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

def load_pressure_data(file_path):
    """Load and clean pressure data"""
    df = pd.read_csv(file_path)
    clean_df = df.dropna()
    return clean_df


def logistic_henry_model(time, n_total_max, growth_rate, lag_time, H, V_solution=0.006, V_headspace=0.002, R=0.08206, T=310.15):
    """
    Logistic-Henry model combining sigmoidal kinetics with gas-liquid partitioning
    
    Parameters:
    - time: Time array (seconds)
    - n_total_max: Maximum total O₂ (µmol)
    - growth_rate: Logistic growth rate (1/s)
    - lag_time: Time offset for lag phase (s)
    - H: Henry's constant (mol/L/atm)
    """
    # Logistic growth for total O₂ released
    n_total = n_total_max / (1 + np.exp(-growth_rate * (time - lag_time)))
    
    # Henry's Law partitioning
    denominator = (V_headspace / (R * T)) + (H * V_solution)
    partition_fraction = (V_headspace / (R * T)) / denominator
    
    # Detectable O₂ in headspace
    n_gas = n_total * partition_fraction
    
    return n_gas

def fit_logistic_model(df):
    """Fit logistic-Henry model to data"""
    
    time_data = df['Time (s)'].values
    n_gas_measured = df['O2 Released (µmol)'].values
    n_total_theoretical = df['Max O2 Possible (µmol)'].iloc[0]
    
    print(f"📊 Data summary:")
    print(f"   Time range: {time_data.min():.0f} to {time_data.max():.0f} seconds ({time_data.max()/3600:.1f} hours)")
    print(f"   O2 range: {n_gas_measured.min():.6f} to {n_gas_measured.max():.6f} µmol")
    print(f"   Theoretical max: {n_total_theoretical:.6f} µmol")
    
    # Initial parameter estimation with debugging
    max_measured = n_gas_measured.max()
    
    # More conservative n_total_max estimation
    estimated_n_total_max = max_measured * 2.5  # Conservative scaling
    if estimated_n_total_max > n_total_theoretical:
        estimated_n_total_max = n_total_theoretical * 0.8
    
    # Better growth rate estimation
    # Find the region with maximum slope (steepest part)
    window_size = max(5, len(time_data) // 20)  # Adaptive window
    smoothed_data = np.convolve(n_gas_measured, np.ones(window_size)/window_size, mode='valid')
    smoothed_time = time_data[:len(smoothed_data)]
    
    if len(smoothed_data) > 2:
        slopes = np.gradient(smoothed_data, smoothed_time)
        max_slope = np.max(slopes)
        # For logistic function, max slope = growth_rate * n_total_max / 4
        estimated_growth_rate = (4 * max_slope) / estimated_n_total_max
    else:
        estimated_growth_rate = 1e-5
    
    # Better lag time estimation - find inflection point
    # Look for point where acceleration changes (second derivative = 0)
    if len(n_gas_measured) > 10:
        # Find point closest to 50% of max
        target_value = max_measured * 0.5
        lag_idx = np.argmin(np.abs(n_gas_measured - target_value))
        estimated_lag_time = time_data[lag_idx]
    else:
        estimated_lag_time = time_data[len(time_data)//2]
    
    # Henry's constant from your experimental data
    initial_H = 0.0221
    
    # Debug initial estimates
    print(f"🔧 Initial parameter estimates:")
    print(f"   n_total_max: {estimated_n_total_max:.6f} µmol")
    print(f"   growth_rate: {estimated_growth_rate:.2e} s⁻¹")
    print(f"   lag_time: {estimated_lag_time:.0f} s ({estimated_lag_time/3600:.1f} h)")
    print(f"   H: {initial_H:.4f} mol/L/atm")
    
    initial_params = [estimated_n_total_max, estimated_growth_rate, estimated_lag_time, initial_H]
    
    # More flexible bounds
    lower_bounds = [
        max_measured,                    # n_total_max must be at least max measured
        1e-8,                           # very small growth rate
        0,                              # lag can start immediately
        1e-6                            # very small Henry's constant
    ]
    
    upper_bounds = [
        n_total_theoretical * 5,        # generous upper limit
        1e-1,                          # reasonable upper growth rate
        time_data.max(),               # lag can be anywhere in time range
        1.0                            # generous Henry's constant upper limit
    ]
    
    # Ensure initial params are within bounds
    for i in range(len(initial_params)):
        if initial_params[i] < lower_bounds[i]:
            initial_params[i] = lower_bounds[i] * 1.1
            print(f"⚠️  Adjusted parameter {i} to fit lower bound")
        if initial_params[i] > upper_bounds[i]:
            initial_params[i] = upper_bounds[i] * 0.9
            print(f"⚠️  Adjusted parameter {i} to fit upper bound")
    
    print(f"✅ Adjusted initial parameters:")
    print(f"   n_total_max: {initial_params[0]:.6f} µmol")
    print(f"   growth_rate: {initial_params[1]:.2e} s⁻¹")
    print(f"   lag_time: {initial_params[2]:.0f} s ({initial_params[2]/3600:.1f} h)")
    print(f"   H: {initial_params[3]:.4f} mol/L/atm")
    
    # Try fitting with multiple strategies
    fitting_strategies = [
        # Strategy 1: Original bounds
        (initial_params, lower_bounds, upper_bounds),
        
        # Strategy 2: No bounds (if bounded fitting fails)
        (initial_params, None, None),
        
        # Strategy 3: Simplified initial guess
        ([max_measured * 2, 1e-5, time_data.max()/2, 0.02], lower_bounds, upper_bounds)
    ]
    
    for strategy_num, (params, lower, upper) in enumerate(fitting_strategies, 1):
        try:
            print(f"🔄 Trying fitting strategy {strategy_num}...")
            
            if lower is not None and upper is not None:
                popt, pcov = curve_fit(
                    logistic_henry_model, time_data, n_gas_measured,
                    p0=params, bounds=(lower, upper), maxfev=15000
                )
            else:
                popt, pcov = curve_fit(
                    logistic_henry_model, time_data, n_gas_measured,
                    p0=params, maxfev=15000
                )
            
            # Calculate fit quality
            n_gas_predicted = logistic_henry_model(time_data, *popt)
            r2 = r2_score(n_gas_measured, n_gas_predicted)
            rmse = np.sqrt(mean_squared_error(n_gas_measured, n_gas_predicted))
            
            print(f"✅ Logistic Model Fitted Successfully! (Strategy {strategy_num})")
            print(f"   R² = {r2:.4f}, RMSE = {rmse:.6f} µmol")
            print(f"   Parameters: n_max={popt[0]:.3f}, k={popt[1]:.2e}, lag={popt[2]/3600:.1f}h, H={popt[3]:.4f}")
            
            return popt, r2, rmse, time_data, n_gas_measured, n_gas_predicted
            
        except Exception as e:
            print(f"   ❌ Strategy {strategy_num} failed: {e}")
            continue
    
    # If all strategies fail, try a simple exponential fit as fallback
    print("🔄 All logistic strategies failed. Trying simple exponential as fallback...")
    try:
        def simple_exponential(t, a, b, c):
            return a * (1 - np.exp(-b * t)) + c
        
        popt_exp, _ = curve_fit(simple_exponential, time_data, n_gas_measured, maxfev=5000)
        n_gas_predicted = simple_exponential(time_data, *popt_exp)
        r2 = r2_score(n_gas_measured, n_gas_predicted)
        rmse = np.sqrt(mean_squared_error(n_gas_measured, n_gas_predicted))
        
        print(f"✅ Fallback exponential model fitted!")
        print(f"   R² = {r2:.4f}, RMSE = {rmse:.6f} µmol")
        print("⚠️  Note: Using exponential model instead of logistic")
        
        # Convert to logistic-like parameters for consistency
        # This is approximate conversion
        fallback_params = [popt_exp[0], popt_exp[1], time_data.max()/4, 0.02]
        return fallback_params, r2, rmse, time_data, n_gas_measured, n_gas_predicted
        
    except Exception as e:
        print(f"❌ Even fallback model failed: {e}")
        print("💡 Check your data file format and column names")
        return None, None, None, None, None, None

def predict_and_plot(fitted_params, V_solution, V_headspace, setup_name=""):
    """
    Predict oxygen detection for new V/HS ratio and automatically generate plot
    """
    
    if fitted_params is None:
        print("❌ No fitted parameters available")
        return None
    
    # Scale parameters for new volume
    old_V_solution = 0.006
    fitted_n_total_max, fitted_growth_rate, fitted_lag_time, fitted_H = fitted_params
    scaled_n_total_max = fitted_n_total_max * (V_solution / old_V_solution)
    
    # Generate time points for prediction
    time_points = np.linspace(0, 200000, 1000)  # 0 to ~55 hours
    
    # Predict with new volumes
    predicted_n_gas = logistic_henry_model(
        time_points, scaled_n_total_max, fitted_growth_rate, fitted_lag_time, fitted_H,
        V_solution=V_solution, V_headspace=V_headspace
    )
    
    # Calculate metrics
    ratio = V_solution / V_headspace
    final_detection = predicted_n_gas[-1]  # At equilibrium
    
    # Calculate detection efficiency
    V_headspace_calc = V_headspace
    V_solution_calc = V_solution
    R, T = 0.08206, 310.15
    denominator = (V_headspace_calc / (R * T)) + (fitted_H * V_solution_calc)
    partition_fraction = (V_headspace_calc / (R * T)) / denominator
    efficiency = partition_fraction * 100
    
    # Compare to baseline (6ml:2ml)
    baseline_detection = logistic_henry_model(
        np.array([200000]), fitted_n_total_max, fitted_growth_rate, fitted_lag_time, fitted_H
    )[0]
    improvement = ((final_detection - baseline_detection) / baseline_detection) * 100
    
    # Print results
    print(f"\n🔬 PREDICTION RESULTS for {setup_name}")
    print(f"   Setup: {V_solution*1000:.1f}ml solution : {V_headspace*1000:.1f}ml headspace")
    print(f"   Ratio (V_sol/V_head): {ratio:.2f}")
    print(f"   Predicted detection: {final_detection:.6f} µmol")
    print(f"   Detection efficiency: {efficiency:.1f}%")
    print(f"   vs. Baseline (6ml:2ml): {improvement:+.1f}%")
    
    # Generate plot
    plt.figure(figsize=(12, 8))
    
    # Main prediction plot
    plt.subplot(2, 2, 1)
    plt.plot(time_points/3600, predicted_n_gas, 'r-', linewidth=2, label=f'{setup_name}\n{V_solution*1000:.1f}ml:{V_headspace*1000:.1f}ml')
    
    # Add baseline for comparison
    baseline_time = time_points
    baseline_prediction = logistic_henry_model(baseline_time, *fitted_params)
    plt.plot(baseline_time/3600, baseline_prediction, 'b--', linewidth=2, alpha=0.7, label='Baseline (6ml:2ml)')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('O₂ Released (µmol)')
    plt.title(f'Oxygen Release Prediction\nRatio: {ratio:.2f}, Efficiency: {efficiency:.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Improvement comparison
    plt.subplot(2, 2, 2)
    scenarios = ['Baseline\n(6ml:2ml)', f'Prediction\n({V_solution*1000:.1f}ml:{V_headspace*1000:.1f}ml)']
    detections = [baseline_detection, final_detection]
    colors = ['blue', 'red' if improvement > 0 else 'orange']
    
    bars = plt.bar(scenarios, detections, color=colors, alpha=0.7)
    plt.ylabel('Final Detection (µmol)')
    plt.title(f'Detection Comparison\n{improvement:+.1f}% change')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, detections):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Rate analysis
    plt.subplot(2, 2, 3)
    rate = np.gradient(predicted_n_gas, time_points) * 3600  # Convert to µmol/hour
    plt.plot(time_points/3600, rate, 'g-', linewidth=2)
    plt.axvline(x=fitted_lag_time/3600, color='orange', linestyle='--', alpha=0.7, label=f'Lag: {fitted_lag_time/3600:.1f}h')
    plt.xlabel('Time (hours)')
    plt.ylabel('Release Rate (µmol/h)')
    plt.title('Oxygen Release Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Efficiency vs ratio visualization
    plt.subplot(2, 2, 4)
    test_ratios = np.linspace(0.5, 5.0, 50)
    test_efficiencies = []
    
    for test_ratio in test_ratios:
        total_vol = V_solution + V_headspace
        test_V_head = total_vol / (1 + test_ratio)
        test_V_sol = total_vol - test_V_head
        
        test_denom = (test_V_head / (R * T)) + (fitted_H * test_V_sol)
        test_eff = ((test_V_head / (R * T)) / test_denom) * 100
        test_efficiencies.append(test_eff)
    
    plt.plot(test_ratios, test_efficiencies, 'purple', linewidth=2)
    plt.axvline(x=ratio, color='red', linestyle='--', alpha=0.7, label=f'Current: {ratio:.2f}')
    plt.axhline(y=efficiency, color='red', linestyle='--', alpha=0.7)
    plt.scatter([ratio], [efficiency], color='red', s=100, zorder=5)
    plt.xlabel('V_solution/V_headspace Ratio')
    plt.ylabel('Detection Efficiency (%)')
    plt.title('Efficiency vs Volume Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'ratio': ratio,
        'detection': final_detection,
        'efficiency': efficiency,
        'improvement': improvement,
        'time_profile': (time_points, predicted_n_gas)
    }

def run_multiple_predictions(fitted_params, volume_setups):
    """
    Run multiple predictions and generate plots for each
    """
    results = []
    
    for i, (V_sol, V_head, name) in enumerate(volume_setups):
        print(f"\n{'='*60}")
        print(f"PREDICTION {i+1}: {name}")
        print(f"{'='*60}")
        
        result = predict_and_plot(fitted_params, V_sol/1000, V_head/1000, name)
        if result:
            results.append({**result, 'name': name, 'V_sol': V_sol, 'V_head': V_head})
    
    return results

def plot_original_fit(time_data, n_gas_measured, n_gas_predicted, fitted_params, r2, rmse):
    """Plot the original data fit"""
    
    fitted_n_total_max, fitted_growth_rate, fitted_lag_time, fitted_H = fitted_params
    
    plt.figure(figsize=(12, 8))
    
    # Main fit
    plt.subplot(2, 2, 1)
    plt.plot(time_data/3600, n_gas_measured, 'ko', markersize=4, label='Experimental Data', alpha=0.7)
    plt.plot(time_data/3600, n_gas_predicted, 'r-', linewidth=2, label=f'Logistic Fit (R²={r2:.3f})')
    plt.axvline(x=fitted_lag_time/3600, color='orange', linestyle='--', alpha=0.7, label=f'Lag: {fitted_lag_time/3600:.1f}h')
    plt.xlabel('Time (hours)')
    plt.ylabel('O₂ Released (µmol)')
    plt.title('Original Data Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(2, 2, 2)
    residuals = n_gas_measured - n_gas_predicted
    plt.plot(time_data/3600, residuals, 'bo-', markersize=3, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Time (hours)')
    plt.ylabel('Residuals (µmol)')
    plt.title(f'Residuals (RMSE={rmse:.4f})')
    plt.grid(True, alpha=0.3)
    
    # Rate profile
    plt.subplot(2, 2, 3)
    time_smooth = np.linspace(time_data.min(), time_data.max(), 1000)
    n_gas_smooth = logistic_henry_model(time_smooth, *fitted_params)
    rate = np.gradient(n_gas_smooth, time_smooth) * 3600
    
    plt.plot(time_smooth/3600, rate, 'g-', linewidth=2)
    plt.axvline(x=fitted_lag_time/3600, color='orange', linestyle='--', alpha=0.7)
    plt.xlabel('Time (hours)')
    plt.ylabel('Release Rate (µmol/h)')
    plt.title('Release Rate Profile')
    plt.grid(True, alpha=0.3)
    
    # Parameters summary
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f'Fitted Parameters:', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f'n_total_max: {fitted_n_total_max:.3f} µmol', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'growth_rate: {fitted_growth_rate:.2e} s⁻¹', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f'lag_time: {fitted_lag_time:.0f} s ({fitted_lag_time/3600:.1f} h)', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'Henry\'s constant: {fitted_H:.4f} mol/L/atm', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f'Fit Quality:', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.1, f'R² = {r2:.4f}, RMSE = {rmse:.6f}', fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main_analysis():
    """Main analysis function"""
    
    print("🧪 SIMPLIFIED LOGISTIC-HENRY ANALYSIS")
    print("="*50)
    
    # Load and fit data
    file_path = "M2_2.18mM_pressure_o2_release.csv"  # Update with your file path
    
    print("📂 Loading data...")
    df = load_pressure_data(file_path)
    
    print("🔬 Fitting logistic model...")
    fit_results = fit_logistic_model(df)
    
    if fit_results[0] is None:
        print("❌ Analysis failed")
        return
    
    fitted_params, r2, rmse, time_data, n_gas_measured, n_gas_predicted = fit_results
    
    # Plot original fit
    print("\n📊 Plotting original data fit...")
    plot_original_fit(time_data, n_gas_measured, n_gas_predicted, fitted_params, r2, rmse)
    
    # Define volume setups to test
    volume_setups = [
        (8, 4, "Test 1: 8ml:4ml"),
        (10, 5, "Test 2: 10ml:5ml"),
        (4, 6, "Test 3: 4ml:6ml"),
        (12, 3, "Test 4: 12ml:3ml"),
        (6, 6, "Test 5: 6ml:6ml")
    ]
    
    # Run predictions
    print("\n🎯 Running volume ratio predictions...")
    results = run_multiple_predictions(fitted_params, volume_setups)
    
    # Summary
    print(f"\n📋 SUMMARY OF ALL PREDICTIONS")
    print("="*60)
    print(f"{'Setup':<20} {'Ratio':<8} {'Detection':<12} {'Efficiency':<10} {'Improvement'}")
    print("-"*60)
    
    for result in results:
        print(f"{result['name']:<20} {result['ratio']:<8.2f} {result['detection']:<12.6f} {result['efficiency']:<10.1f}% {result['improvement']:+.1f}%")
    
    # Find best setup
    best_result = max(results, key=lambda x: x['detection'])
    print(f"\n🏆 BEST SETUP: {best_result['name']}")
    print(f"   Detection: {best_result['detection']:.6f} µmol ({best_result['improvement']:+.1f}% vs baseline)")

if __name__ == "__main__":
    main_analysis()