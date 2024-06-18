import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

# Generate white noise data
num_samples = 1000
white_noise = np.random.randn(num_samples)

# Plot white noise data
plt.figure(figsize=(10, 6))
plt.plot(white_noise)
plt.title('White Noise')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Generate time series data
time = np.arange(num_samples)
time_series = 0.05 * time + 10 * np.sin(2 * np.pi * time / 50)

# Plot time series and white noise data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, time_series, label='Time Series Data', color='blue')
plt.title('Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, white_noise, label='White Noise', color='green')
plt.title('White Noise')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Load and plot sunspots data
sunspots_data = sm.datasets.sunspots.load_pandas().data['SUNACTIVITY'].values
plt.figure(figsize=(12, 6))
plt.plot(sunspots_data, color='blue')
plt.title('Sunspot Activity Over Time')
plt.xlabel('Time')
plt.ylabel('Sunspot Number')
plt.grid(True)
plt.show()

# Perform ADF and KPSS tests on sunspots data
adf_result = adfuller(sunspots_data)
kpss_result = kpss(sunspots_data)
print("\nADF Test (Sunspots Data):")
print("ADF Statistic:", adf_result[0], "p-value:", adf_result[1], "Critical Values:", adf_result[4])
print("\nKPSS Test (Sunspots Data):")
print("KPSS Statistic:", kpss_result[0], "p-value:", kpss_result[1], "Critical Values:", kpss_result[3])
