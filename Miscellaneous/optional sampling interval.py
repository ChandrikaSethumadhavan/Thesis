import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error
import seaborn as sns

def load_pressure_data(file_path):
    """Load and clean the pressure data CSV file"""
    df = pd.read_csv(file_path)
    clean_df = df.dropna()
    return clean_df

def calculate_information_content(time_series, sampling_interval):
    """
    Calculate information content using signal variance and autocorrelation
    Higher information content = better sampling interval
    """
    # Subsample the data at given interval
    subsampled_indices = np.arange(0, len(time_series), sampling_interval)
    subsampled_data = time_series.iloc[subsampled_indices]
    
    if len(subsampled_data) < 3:
        return 0
    
    # Calculate signal variance (information content)
    signal_variance = np.var(subsampled_data)
    
    # Calculate autocorrelation to detect redundancy
    if len(subsampled_data) > 10:
        autocorr = np.corrcoef(subsampled_data[:-1], subsampled_data[1:])[0,1]
        autocorr = np.nan_to_num(autocorr)  # Handle NaN cases
    else:
        autocorr = 0
    
    # Information score: high variance, low redundancy
    information_score = signal_variance * (1 - abs(autocorr))
    
    return information_score

def calculate_reconstruction_error(original_data, sampling_interval):
    """
    Calculate how well we can reconstruct the original signal 
    from subsampled data using interpolation
    """
    if sampling_interval >= len(original_data):
        return float('inf')
    
    # Subsample
    subsampled_indices = np.arange(0, len(original_data), sampling_interval)
    subsampled_values = original_data.iloc[subsampled_indices]
    
    if len(subsampled_values) < 2:
        return float('inf')
    
    # Interpolate back to original resolution
    try:
        interp_func = interp1d(subsampled_indices, subsampled_values, 
                              kind='linear', fill_value='extrapolate')
        reconstructed = interp_func(np.arange(len(original_data)))
        
        # Calculate reconstruction error
        mse = mean_squared_error(original_data, reconstructed)
        return mse
    except:
        return float('inf')

def analyze_signal_characteristics(df):
    """
    Analyze the characteristics of different signals to understand
    their optimal sampling requirements
    """
    signals_to_analyze = {
        'O2_Released': 'O2 Released (µmol)',
        'Temperature': 'calibrated temperature (C)',
        'Conversion': '% conversion',
        'Pressure': 'DWT denoised pressure (kPa)'
    }
    
    signal_characteristics = {}
    
    for signal_name, column_name in signals_to_analyze.items():
        if column_name in df.columns:
            signal_data = df[column_name]
            
            # Calculate signal properties
            signal_range = signal_data.max() - signal_data.min()
            signal_std = signal_data.std()
            signal_mean = signal_data.mean()
            
            # Calculate rate of change
            rate_of_change = np.gradient(signal_data)
            max_rate = np.max(np.abs(rate_of_change))
            mean_rate = np.mean(np.abs(rate_of_change))
            
            # Nyquist-like analysis: find dominant frequency  We want to find: How fast do the small wiggles happen?

           
           
            try:
                # Remove trend for frequency analysis
                detrended = signal.detrend(signal_data)  #0.0 → 1.2 → 2.8 → 4.1 → 4.7 → 4.8 → 4.8 → 4.8... is converted to  0.0 → -0.1 → 0.05 → -0.02 → 0.01 → 0.0 → 0.0...
                                                                                                            #     (just the fluctuations around the trend)
                fft = np.fft.fft(detrended)  # gives different frequency's strengths
                freqs = np.fft.fftfreq(len(detrended))
                
                # Find dominant frequency (excluding DC component)
                dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
                dominant_freq = abs(freqs[dominant_freq_idx])
                
                # Nyquist criterion: sample at least 2x dominant frequency
                """
                WHAT DOES NQUIST SAY ? 
                To perfectly reconstruct a signal, sample at least 2× the highest frequency
                """
                nyquist_interval = 1 / (2 * dominant_freq) if dominant_freq > 0 else len(signal_data)
                nyquist_interval = max(1, int(nyquist_interval))
                
            except:
                nyquist_interval = 10  # Default fallback
            
            signal_characteristics[signal_name] = {
                'range': signal_range,
                'std': signal_std,
                'mean': signal_mean,
                'max_rate': max_rate,
                'mean_rate': mean_rate,
                'nyquist_interval': nyquist_interval,
                'column': column_name
            }
    
    return signal_characteristics

# def find_optimal_sampling_intervals(df):
#     """
#     Find optimal sampling intervals for different signals using multiple criteria
#     """
#     signal_characteristics = analyze_signal_characteristics(df)
    
#     # Test different sampling intervals
#     max_interval = min(len(df) // 10, 100)  # Don't test intervals too large
#     test_intervals = range(1, max_interval, max(1, max_interval // 50))  # Result: [1, 3, 5, 7, 9, 11, ..., 95, 97, 99] these are the intervals we will test
    
#     results = {}
#     """this will look like : 
#     results = {
#     'O2_Released': {
#         'information_scores': [score1, score2, ...],
#         'reconstruction_errors': [error1, error2, ...],
#         # ... will be filled in the loop
#     },
#     'Temperature': {
#         # ... same structure
#     }
# }"""
    
#     for signal_name, char in signal_characteristics.items():
#         column_name = char['column']  # 'O2 Released (µmol)'
#         signal_data = df[column_name]  ## Extract the actual data column
        
#         information_scores = []
#         reconstruction_errors = []
        
#         for interval in test_intervals:
            
#             """
#             this is just for O2_Released signal, but it will be done for all signals
#             test_intervals =        [1,   3,   5,   7,  ...,  97,  99]
#             information_scores =    [0.1, 0.3, 0.5, 0.6, ..., 0.8, 0.9]  # Higher = better
#             reconstruction_errors = [0.0, 0.1, 0.2, 0.3, ..., 2.1, 2.5]  # Lower = better
#             """

#             # Calculate information content
#             #interval = one interval from the test_intervals, e.g. 1, 3, 5, ...
#             info_score = calculate_information_content(signal_data, interval) #information_scores = [score_for_interval_1, score_for_interval_3, score_for_interval_5, ...]
#             information_scores.append(info_score) 
            
#             # Calculate reconstruction error
#             recon_error = calculate_reconstruction_error(signal_data, interval)
#             reconstruction_errors.append(recon_error) #reconstruction_errors = [error_for_interval_1, error_for_interval_3, error_for_interval_5, ...]
        
#         # Normalize scores for comparison
#         if max(information_scores) > 0:
#             norm_info_scores = np.array(information_scores) / max(information_scores)
#         else:
#             norm_info_scores = np.zeros_like(information_scores)
        
#         if max(reconstruction_errors) > 0 and max(reconstruction_errors) != float('inf'):
#             norm_recon_errors = 1 - (np.array(reconstruction_errors) / max(reconstruction_errors))
#             # Handle infinite errors
#             norm_recon_errors[reconstruction_errors == float('inf')] = 0
#         else:
#             norm_recon_errors = np.ones_like(reconstruction_errors)
        
#         # Combined score: balance information content and reconstruction quality
#         combined_scores = 0.6 * norm_info_scores + 0.4 * norm_recon_errors
        
#         # Find optimal interval
#         optimal_idx = np.argmax(combined_scores)
#         optimal_interval = list(test_intervals)[optimal_idx]
        
#         results[signal_name] = {
#             'optimal_interval': optimal_interval,
#             'intervals': list(test_intervals),
#             'information_scores': information_scores,
#             'reconstruction_errors': reconstruction_errors,
#             'combined_scores': combined_scores,
#             'nyquist_interval': char['nyquist_interval'],
#             'signal_characteristics': char
#         }
    
#     return results


def find_optimal_sampling_intervals(df):
    """
    Find optimal sampling intervals with extended search range
    """
    signal_characteristics = analyze_signal_characteristics(df)
    
    results = {}
    
    for signal_name, char in signal_characteristics.items():
        column_name = char['column']
        signal_data = df[column_name]
        
        # Call the extended search function
        global_optimal, test_intervals, scores = find_true_global_optimum(
            signal_data, max_test_interval=2000
        )
        
        # Store results in same format as before
        results[signal_name] = {
            'optimal_interval': global_optimal,
            'intervals': list(test_intervals),
            'combined_scores': scores,
            'nyquist_interval': char['nyquist_interval'],
            'signal_characteristics': char
        }
    
    return results

def find_true_global_optimum(signal_data, max_test_interval=2000):
    """
    Test intervals up to much higher values
    """
    # Test every 10th interval up to max_test_interval
    test_intervals = range(10, max_test_interval, 10)
    
    information_scores = []
    reconstruction_errors = []
    combined_scores = []
    
    for interval in test_intervals:
        # Calculate information content
        info_score = calculate_information_content(signal_data, interval)
        information_scores.append(info_score)
        
        # Calculate reconstruction error
        recon_error = calculate_reconstruction_error(signal_data, interval)
        reconstruction_errors.append(recon_error)
        
        # Handle edge cases for combined score
        if len(signal_data) // interval < 3:  # Too few points
            combined_score = 0
        elif recon_error == 0 or recon_error == float('inf'):
            combined_score = info_score
        else:
            # Normalize scores (simplified version)
            norm_info = info_score
            norm_recon = 1 / max(recon_error, 1e-10)
            combined_score = 0.6 * norm_info + 0.4 * norm_recon
        
        combined_scores.append(combined_score)
    
    # Find global maximum
    if combined_scores:
        global_opt_idx = np.argmax(combined_scores)
        global_optimal = test_intervals[global_opt_idx]
    else:
        global_optimal = 10  # Fallback
    
    return global_optimal, test_intervals, combined_scores


def compare_search_strategies(df):
    """
    Compare the original limited search vs extended search
    """
    print("🔍 COMPARING SEARCH STRATEGIES")
    print("="*50)
    
    for signal_name in ['O2_Released', 'Temperature', 'Pressure']:
        if signal_name == 'O2_Released':
            column_name = 'O2 Released (µmol)'
        elif signal_name == 'Temperature':
            column_name = 'calibrated temperature (C)'
        elif signal_name == 'Pressure':
            column_name = 'DWT denoised pressure (kPa)'
        else:
            continue
            
        if column_name not in df.columns:
            continue
            
        signal_data = df[column_name]
        
        # Original limited search (max=100)
        limited_optimal = find_limited_optimal(signal_data, max_interval=100)
        
        # Extended search (max=2000)  
        extended_optimal, _, _ = find_true_global_optimum(signal_data, max_test_interval=2000)
        
        print(f"\n{signal_name}:")
        print(f"  Limited search (≤100):  {limited_optimal}")
        print(f"  Extended search (≤2000): {extended_optimal}")
        print(f"  Improvement factor: {extended_optimal/limited_optimal:.1f}x")

def find_limited_optimal(signal_data, max_interval=100):
    """
    Original search strategy for comparison
    """
    test_intervals = range(1, max_interval, max(1, max_interval // 50))
    
    best_score = 0
    best_interval = 1
    
    for interval in test_intervals:
        info_score = calculate_information_content(signal_data, interval)
        recon_error = calculate_reconstruction_error(signal_data, interval)
        
        if recon_error == 0 or recon_error == float('inf'):
            combined_score = info_score
        else:
            combined_score = 0.6 * info_score + 0.4 * (1/recon_error)
        
        if combined_score > best_score:
            best_score = combined_score
            best_interval = interval
    
    return best_interval

def calculate_measurement_efficiency(df, sampling_results):
    """
    Calculate how much measurement efficiency we gain with optimal sampling
    """
    total_measurements = len(df)
    
    efficiency_analysis = {}
    
    for signal_name, result in sampling_results.items():
        optimal_interval = result['optimal_interval']
        
        # Calculate measurement reduction
        optimal_measurements = len(df) // optimal_interval
        measurement_reduction = (total_measurements - optimal_measurements) / total_measurements * 100
        
        # Calculate time savings (assuming each measurement takes time)
        time_savings = measurement_reduction  # Same percentage
        
        # Calculate information retention
        max_info_score = max(result['information_scores'])
        optimal_info_score = result['information_scores'][result['intervals'].index(optimal_interval)]
        information_retention = (optimal_info_score / max_info_score) * 100 if max_info_score > 0 else 0
        
        efficiency_analysis[signal_name] = {
            'measurement_reduction_%': measurement_reduction,
            'time_savings_%': time_savings,
            'information_retention_%': information_retention,
            'measurements_needed': optimal_measurements,
            'original_measurements': total_measurements
        }
    
    return efficiency_analysis

def plot_optimal_sampling_analysis(df, sampling_results, efficiency_analysis):
    """
    Create comprehensive plots showing optimal sampling analysis
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Signal characteristics comparison
    plt.subplot(2, 4, 1)
    signals = list(sampling_results.keys())
    optimal_intervals = [sampling_results[s]['optimal_interval'] for s in signals]
    nyquist_intervals = [sampling_results[s]['nyquist_interval'] for s in signals]
    
    x = np.arange(len(signals))
    width = 0.35
    
    plt.bar(x - width/2, optimal_intervals, width, label='Optimal Interval', alpha=0.7)
    plt.bar(x + width/2, nyquist_intervals, width, label='Nyquist Interval', alpha=0.7)
    plt.xlabel('Signal Type')
    plt.ylabel('Sampling Interval (data points)')
    plt.title('Optimal vs Nyquist Sampling Intervals')
    plt.xticks(x, signals, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2-5: Individual signal analysis
    for i, (signal_name, result) in enumerate(list(sampling_results.items())[:4]):
        plt.subplot(2, 4, i + 2)
        
        intervals = result['intervals']
        combined_scores = result['combined_scores']
        optimal_interval = result['optimal_interval']
        
        plt.plot(intervals, combined_scores, 'b-', linewidth=2, alpha=0.7)
        plt.axvline(x=optimal_interval, color='red', linestyle='--', linewidth=2, 
                   label=f'Optimal: {optimal_interval}')
        plt.axvline(x=result['nyquist_interval'], color='green', linestyle='--', linewidth=2,
                   label=f'Nyquist: {result["nyquist_interval"]}')
        
        plt.xlabel('Sampling Interval')
        plt.ylabel('Combined Score')
        plt.title(f'{signal_name} Optimal Sampling')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    # Plot 6: Efficiency comparison
    plt.subplot(2, 4, 6)
    signals = list(efficiency_analysis.keys())
    measurement_reductions = [efficiency_analysis[s]['measurement_reduction_%'] for s in signals]
    info_retentions = [efficiency_analysis[s]['information_retention_%'] for s in signals]
    
    x = np.arange(len(signals))
    width = 0.35
    
    plt.bar(x - width/2, measurement_reductions, width, label='Measurement Reduction %', alpha=0.7)
    plt.bar(x + width/2, info_retentions, width, label='Information Retention %', alpha=0.7)
    plt.xlabel('Signal Type')
    plt.ylabel('Percentage')
    plt.title('Sampling Efficiency Analysis')
    plt.xticks(x, signals, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Time series comparison example
    plt.subplot(2, 4, 7)
    # Use O2 signal as example
    if 'O2_Released' in sampling_results:
        signal_data = df[sampling_results['O2_Released']['signal_characteristics']['column']]
        optimal_interval = sampling_results['O2_Released']['optimal_interval']
        
        time_full = np.arange(len(signal_data))
        time_sampled = np.arange(0, len(signal_data), optimal_interval)
        
        plt.plot(time_full, signal_data, 'b-', alpha=0.3, label='Full Resolution', linewidth=1)
        plt.plot(time_sampled, signal_data.iloc[time_sampled], 'ro-', 
                label=f'Optimal Sampling (1:{optimal_interval})', markersize=3)
        
        plt.xlabel('Time Index')
        plt.ylabel('O2 Released (µmol)')
        plt.title('Sampling Comparison Example')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 8: Summary statistics table
    plt.subplot(2, 4, 8)
    plt.axis('off')
    
    # Create summary table
    summary_text = "OPTIMAL SAMPLING SUMMARY\n\n"
    for signal_name, eff in efficiency_analysis.items():
        summary_text += f"{signal_name}:\n"
        summary_text += f"  Reduction: {eff['measurement_reduction_%']:.1f}%\n"
        summary_text += f"  Info Retained: {eff['information_retention_%']:.1f}%\n"
        summary_text += f"  Interval: 1:{sampling_results[signal_name]['optimal_interval']}\n\n"
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main_sampling_analysis(file_path):
    """
    Main function with extended search capability
    """
    print("🔍 Loading and analyzing dataset for optimal sampling intervals...")
    
    # Load data
    df = load_pressure_data(file_path)
    print(f"✅ Loaded {len(df)} data points")
    
    # Add comparison analysis
    print("\n" + "="*60)
    print("🆚 COMPARING SEARCH STRATEGIES")
    print("="*60)
    compare_search_strategies(df)
    
    # Run extended analysis  
    print("\n" + "="*60)
    print("🎯 EXTENDED OPTIMAL SAMPLING ANALYSIS")
    print("="*60)
    
    # Find optimal sampling intervals (now with extended search)
    sampling_results = find_optimal_sampling_intervals(df)
    
    # Calculate efficiency gains
    efficiency_analysis = calculate_measurement_efficiency(df, sampling_results)
    
    # Display results
    for signal_name, result in sampling_results.items():
        eff = efficiency_analysis[signal_name]
        print(f"\n{signal_name.upper()}:")
        print(f"  🎯 Optimal Interval: 1:{result['optimal_interval']} data points")
        print(f"  💾 Measurement Reduction: {eff['measurement_reduction_%']:.1f}%")
        print(f"  📊 Information Retained: {eff['information_retention_%']:.1f}%")
    
    # Create visualization
    plot_optimal_sampling_analysis(df, sampling_results, efficiency_analysis)
    
    return sampling_results, efficiency_analysis

# Example usage
if __name__ == "__main__":
    # Replace with your actual CSV file path
    file_path = "M2_2.18mM_pressure_o2_release.csv"
    
    try:
        sampling_results, efficiency_analysis = main_sampling_analysis(file_path)
        
        
        print(" Analysis complete! Check the plots above for detailed results.")
        
    
        
    except FileNotFoundError:
        print(f" Could not find file: {file_path}")
        print("Please make sure the CSV file exists and the path is correct.")
    except Exception as e:
        print(f" Error during analysis: {str(e)}")
        