import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Function to generate ARIMA data, fit model, and evaluate metrics
def process_arima(ar_params, ma_params, order):
    data = ArmaProcess(ar_params, ma_params).generate_sample(nsample=1000)
    fitted_model = ARIMA(data, order=order).fit()
    residuals = fitted_model.resid
    return (acf(residuals, nlags=20), pacf(residuals, nlags=20),
            np.mean(residuals), np.var(residuals),
            np.sqrt(mean_squared_error(np.zeros_like(residuals), residuals)))

# Parameters
np.random.seed(0)
params_list = [([1, -0.5], [1], (1, 0, 0)), ([1, -0.5, 0.25], [1], (2, 0, 0))]

# Processing ARIMA models and extracting results
results = [process_arima(*params) for params in params_list]

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
titles = ['ACF of ARIMA(1) Residuals', 'PACF of ARIMA(1) Residuals', 
          'ACF of ARIMA(2) Residuals', 'PACF of ARIMA(2) Residuals']

for i, result in enumerate(results):
    acf_vals, pacf_vals = result[:2]
    axes[i, 0].stem(acf_vals, use_line_collection=True)
    axes[i, 0].set_title(titles[2 * i])
    axes[i, 0].set_xlabel('Lag')
    axes[i, 0].set_ylabel('ACF')
    axes[i, 1].stem(pacf_vals, use_line_collection=True)
    axes[i, 1].set_title(titles[2 * i + 1])
    axes[i, 1].set_xlabel('Lag')
    axes[i, 1].set_ylabel('PACF')

plt.tight_layout()
plt.show()

# Print evaluation metrics
for i, result in enumerate(results, 1):
    mean_val, var_val, rmse_val = result[2:]
    print(f'ARIMA({i}) Residuals - Mean: {mean_val:.4f}, Variance: {var_val:.4f}, RMSE: {rmse_val:.4f}')
