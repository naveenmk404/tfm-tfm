import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

# Generate sample data
np.random.seed(0)
n = 100
data = np.random.normal(loc=0, scale=1, size=n)

# Create a DataFrame
df = pd.DataFrame({'Time': np.arange(n), 'Data': data})

# Plot data
plt.figure(figsize=(10, 4))
plt.plot(df['Time'], df['Data'])
plt.title('Sample Time Series Data')
plt.xlabel('Time')
plt.ylabel('Data')
plt.grid(True)
plt.show()

# Plot ACF and PACF
plot_acf(df['Data'], lags=20, title='ACF')
plt.show()
plot_pacf(df['Data'], lags=20, title='PACF')
plt.show()

# Fit AR models and print summaries
for lag in [5, 10, 15, 20]:
    result = AutoReg(df['Data'], lags=lag).fit()
    print(f"\nAutoregression Model with {lag} lags:")
    print(result.summary())
