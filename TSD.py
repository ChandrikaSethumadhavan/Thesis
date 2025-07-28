import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    time_data = df['Time (s)'].values
    temp_data = df['calibrated temperature (C)'].values
    o2_data = df['O2 Released (µmol)'].values
    
    return time_data, temp_data, o2_data

def decompose_signals(time_data, temp_data, o2_data):
    # Create time series with regular intervals
    time_hours = time_data / 3600
    
    # Interpolate to regular grid for decomposition
    regular_time = np.linspace(time_hours.min(), time_hours.max(), len(time_data))
    temp_interp = np.interp(regular_time, time_hours, temp_data)
    o2_interp = np.interp(regular_time, time_hours, o2_data)
    
    # Convert to pandas Series for statsmodels
    temp_series = pd.Series(temp_interp, index=pd.date_range('2024-01-01', periods=len(temp_interp), freq='h'))
    o2_series = pd.Series(o2_interp, index=pd.date_range('2024-01-01', periods=len(o2_interp), freq='h'))
    
    # Decompose signals (additive model)
    temp_decomp = seasonal_decompose(temp_series, model='additive', period=min(24, len(temp_series)//4))
    o2_decomp = seasonal_decompose(o2_series, model='additive', period=min(24, len(o2_series)//4))
    
    return regular_time, temp_decomp, o2_decomp

def frequency_analysis(time_data, temp_data, o2_data):
    # Sampling frequency
    dt = np.median(np.diff(time_data))
    fs = 1 / dt  # Hz
    
    # Remove DC component and detrend
    temp_detrend = signal.detrend(temp_data)
    o2_detrend = signal.detrend(o2_data)
    
    # Apply window function
    window = np.hanning(len(temp_data))
    temp_windowed = temp_detrend * window
    o2_windowed = o2_detrend * window
    
    # FFT
    temp_fft = fft(temp_windowed)
    o2_fft = fft(o2_windowed)
    freqs = fftfreq(len(temp_data), dt)
    
    # Power spectral density
    temp_psd = np.abs(temp_fft)**2
    o2_psd = np.abs(o2_fft)**2
    
    # Cross-correlation
    correlation = np.correlate(temp_detrend, o2_detrend, mode='full')
    correlation_lags = np.arange(-len(o2_detrend)+1, len(temp_detrend)) * dt
    
    return freqs, temp_psd, o2_psd, correlation, correlation_lags

def wavelet_analysis(time_data, temp_data, o2_data):
    # Simple continuous wavelet transform approximation using Morlet wavelets
    scales = np.geomspace(1, len(time_data)//10, 50)
    dt = np.median(np.diff(time_data))
    
    def morlet_wavelet(t, scale):
        return np.exp(-t**2/(2*scale**2)) * np.cos(5*t/scale) / np.sqrt(scale)
    
    temp_cwt = np.zeros((len(scales), len(temp_data)))
    o2_cwt = np.zeros((len(scales), len(o2_data)))
    
    for i, scale in enumerate(scales):
        for j in range(len(time_data)):
            # Local wavelet analysis
            start_idx = max(0, j - int(3*scale))
            end_idx = min(len(time_data), j + int(3*scale))
            
            if end_idx - start_idx > 5:
                local_time = time_data[start_idx:end_idx] - time_data[j]
                wavelet = morlet_wavelet(local_time, scale * dt)
                
                if len(wavelet) == len(temp_data[start_idx:end_idx]):
                    temp_cwt[i, j] = np.sum(temp_data[start_idx:end_idx] * wavelet)
                    o2_cwt[i, j] = np.sum(o2_data[start_idx:end_idx] * wavelet)
    
    return scales * dt, temp_cwt, o2_cwt

def phase_space_analysis(temp_data, o2_data):
    # Phase space reconstruction using time delay embedding
    delay = 10  # Time delay
    
    if len(temp_data) > delay:
        temp_delayed = temp_data[:-delay]
        temp_current = temp_data[delay:]
        o2_delayed = o2_data[:-delay]
        o2_current = o2_data[delay:]
        
        return temp_delayed, temp_current, o2_delayed, o2_current
    else:
        return temp_data[:-1], temp_data[1:], o2_data[:-1], o2_data[1:]

def plot_time_series_decomposition(time_data, temp_data, o2_data, regular_time, temp_decomp, o2_decomp):
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Complete Time Series Decomposition Analysis', fontsize=16, fontweight='bold')
    
    time_hours = time_data / 3600
    
    # Temperature decomposition
    plt.subplot(4, 4, 1)
    plt.plot(regular_time, temp_decomp.observed, 'b-', linewidth=1)
    plt.title('Temperature: Original Signal')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 2)
    plt.plot(regular_time, temp_decomp.trend, 'r-', linewidth=2)
    plt.title('Temperature: Trend Component')
    plt.ylabel('Trend (°C)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 3)
    plt.plot(regular_time, temp_decomp.seasonal, 'g-', linewidth=1)
    plt.title('Temperature: Periodic Component')
    plt.ylabel('Periodic (°C)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 4)
    plt.plot(regular_time, temp_decomp.resid, 'm-', linewidth=1, alpha=0.7)
    plt.title('Temperature: Residual/Noise')
    plt.ylabel('Residual (°C)')
    plt.grid(True, alpha=0.3)
    
    # O2 decomposition
    plt.subplot(4, 4, 5)
    plt.plot(regular_time, o2_decomp.observed, 'b-', linewidth=1)
    plt.title('O₂: Original Signal')
    plt.ylabel('O₂ (µmol)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 6)
    plt.plot(regular_time, o2_decomp.trend, 'r-', linewidth=2)
    plt.title('O₂: Trend Component')
    plt.ylabel('Trend (µmol)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 7)
    plt.plot(regular_time, o2_decomp.seasonal, 'g-', linewidth=1)
    plt.title('O₂: Periodic Component')
    plt.ylabel('Periodic (µmol)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 8)
    plt.plot(regular_time, o2_decomp.resid, 'm-', linewidth=1, alpha=0.7)
    plt.title('O₂: Residual/Noise')
    plt.ylabel('Residual (µmol)')
    plt.grid(True, alpha=0.3)
    
    # Frequency analysis
    freqs, temp_psd, o2_psd, correlation, correlation_lags = frequency_analysis(time_data, temp_data, o2_data)
    
    plt.subplot(4, 4, 9)
    positive_freqs = freqs[:len(freqs)//2]
    positive_temp_psd = temp_psd[:len(temp_psd)//2]
    plt.loglog(positive_freqs[1:], positive_temp_psd[1:], 'b-', linewidth=1)
    plt.title('Temperature: Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 10)
    positive_o2_psd = o2_psd[:len(o2_psd)//2]
    plt.loglog(positive_freqs[1:], positive_o2_psd[1:], 'r-', linewidth=1)
    plt.title('O₂: Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True, alpha=0.3)
    
    # Cross-correlation
    plt.subplot(4, 4, 11)
    max_lag_plot = min(len(correlation)//4, 100)
    center_idx = len(correlation) // 2
    lag_slice = slice(center_idx - max_lag_plot, center_idx + max_lag_plot)
    plt.plot(correlation_lags[lag_slice]/3600, correlation[lag_slice], 'purple', linewidth=2)
    plt.title('Temperature-O₂ Cross-Correlation')
    plt.xlabel('Time Lag (hours)')
    plt.ylabel('Correlation')
    plt.grid(True, alpha=0.3)
    
    # Phase space analysis
    temp_delayed, temp_current, o2_delayed, o2_current = phase_space_analysis(temp_data, o2_data)
    
    plt.subplot(4, 4, 12)
    plt.scatter(temp_delayed, temp_current, c=time_hours[:-10], cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(label='Time (hours)')
    plt.xlabel('Temperature(t)')
    plt.ylabel('Temperature(t+Δt)')
    plt.title('Temperature: Phase Space')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 13)
    plt.scatter(o2_delayed, o2_current, c=time_hours[:-10], cmap='plasma', s=10, alpha=0.6)
    plt.colorbar(label='Time (hours)')
    plt.xlabel('O₂(t)')
    plt.ylabel('O₂(t+Δt)')
    plt.title('O₂: Phase Space')
    plt.grid(True, alpha=0.3)
    
    # Combined phase space
    plt.subplot(4, 4, 14)
    plt.scatter(temp_data, o2_data, c=time_hours, cmap='coolwarm', s=15, alpha=0.7)
    plt.colorbar(label='Time (hours)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('O₂ Released (µmol)')
    plt.title('Combined Phase Space')
    plt.grid(True, alpha=0.3)
    
    # Trend comparison
    plt.subplot(4, 4, 15)
    plt.plot(regular_time, temp_decomp.trend / np.max(temp_decomp.trend), 'b-', linewidth=2, label='Temperature Trend')
    plt.plot(regular_time, o2_decomp.trend / np.max(o2_decomp.trend), 'r-', linewidth=2, label='O₂ Trend')
    plt.title('Normalized Trends Comparison')
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Signal statistics
    plt.subplot(4, 4, 16)
    plt.axis('off')
    
    # Calculate statistics
    temp_trend_var = np.var(temp_decomp.trend)
    temp_periodic_var = np.var(temp_decomp.seasonal)
    temp_noise_var = np.var(temp_decomp.resid.dropna())
    
    o2_trend_var = np.var(o2_decomp.trend)
    o2_periodic_var = np.var(o2_decomp.seasonal)
    o2_noise_var = np.var(o2_decomp.resid.dropna())
    
    temp_total_var = temp_trend_var + temp_periodic_var + temp_noise_var
    o2_total_var = o2_trend_var + o2_periodic_var + o2_noise_var
    
    # Correlation between trends
    trend_correlation = np.corrcoef(temp_decomp.trend, o2_decomp.trend)[0,1]
    
    stats_text = f"""
SIGNAL DECOMPOSITION STATISTICS

Temperature Signal:
• Trend: {temp_trend_var/temp_total_var*100:.1f}% of variance
• Periodic: {temp_periodic_var/temp_total_var*100:.1f}% of variance  
• Noise: {temp_noise_var/temp_total_var*100:.1f}% of variance

O₂ Signal:
• Trend: {o2_trend_var/o2_total_var*100:.1f}% of variance
• Periodic: {o2_periodic_var/o2_total_var*100:.1f}% of variance
• Noise: {o2_noise_var/o2_total_var*100:.1f}% of variance

Cross-Correlation:
• Trend correlation: {trend_correlation:.3f}
• Max correlation: {np.max(correlation):.3f}
• Best lag: {correlation_lags[np.argmax(correlation)]/3600:.2f} hours

Signal Quality:
• Temp SNR: {(temp_trend_var + temp_periodic_var)/temp_noise_var:.1f}
• O₂ SNR: {(o2_trend_var + o2_periodic_var)/o2_noise_var:.1f}
"""
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def wavelet_time_frequency_plot(time_data, temp_data, o2_data):
    scales, temp_cwt, o2_cwt = wavelet_analysis(time_data, temp_data, o2_data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Wavelet Time-Frequency Analysis', fontsize=14, fontweight='bold')
    
    time_hours = time_data / 3600
    
    # Temperature wavelet
    im1 = ax1.contourf(time_hours, scales, np.abs(temp_cwt), levels=50, cmap='viridis')
    ax1.set_ylabel('Time Scale (seconds)')
    ax1.set_xlabel('Time (hours)')
    ax1.set_title('Temperature: Wavelet Transform')
    ax1.set_yscale('log')
    plt.colorbar(im1, ax=ax1, label='Magnitude')
    
    # O2 wavelet
    im2 = ax2.contourf(time_hours, scales, np.abs(o2_cwt), levels=50, cmap='plasma')
    ax2.set_ylabel('Time Scale (seconds)')
    ax2.set_xlabel('Time (hours)')
    ax2.set_title('O₂: Wavelet Transform')
    ax2.set_yscale('log')
    plt.colorbar(im2, ax=ax2, label='Magnitude')
    
    # Wavelet coherence (simplified)
    coherence = np.abs(temp_cwt * np.conj(o2_cwt))
    im3 = ax3.contourf(time_hours, scales, coherence, levels=50, cmap='RdYlBu_r')
    ax3.set_ylabel('Time Scale (seconds)')
    ax3.set_xlabel('Time (hours)')
    ax3.set_title('Temperature-O₂ Coherence')
    ax3.set_yscale('log')
    plt.colorbar(im3, ax=ax3, label='Coherence')
    
    # Average power vs scale
    temp_avg_power = np.mean(np.abs(temp_cwt)**2, axis=1)
    o2_avg_power = np.mean(np.abs(o2_cwt)**2, axis=1)
    
    ax4.loglog(scales, temp_avg_power, 'b-', linewidth=2, label='Temperature')
    ax4.loglog(scales, o2_avg_power, 'r-', linewidth=2, label='O₂')
    ax4.set_xlabel('Time Scale (seconds)')
    ax4.set_ylabel('Average Power')
    ax4.set_title('Scale-Averaged Power')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main_time_series_analysis():
    file_path = "M2_2.18mM_pressure_o2_release.csv"
    
    print("📊 TIME SERIES DECOMPOSITION ANALYSIS")
    print("="*50)
    
    time_data, temp_data, o2_data = load_data(file_path)
    
    print(f"📈 Analyzing {len(time_data)} data points over {time_data[-1]/3600:.1f} hours")
    
    # Decompose signals
    regular_time, temp_decomp, o2_decomp = decompose_signals(time_data, temp_data, o2_data)
    
    # Main decomposition plot
    plot_time_series_decomposition(time_data, temp_data, o2_data, regular_time, temp_decomp, o2_decomp)
    
    # Wavelet analysis
    wavelet_time_frequency_plot(time_data, temp_data, o2_data)
    
    print("✅ Time series decomposition complete!")
    print("📊 Check plots for trend, periodic, and noise components")
    
    return temp_decomp, o2_decomp

if __name__ == "__main__":
    temp_decomp, o2_decomp = main_time_series_analysis()