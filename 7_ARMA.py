import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Function to generate ARMA data, fit model, compute residuals, and evaluate metrics
def process_arma(ar_params, ma_params, order):
    arma_process = ArmaProcess(ar_params, ma_params)
    data = arma_process.generate_sample(nsample=1000)
    model = ARIMA(data, order=order).fit()
    residuals = model.resid
    mean_val = np.mean(residuals)
    var_val = np.var(residuals)
    rmse_val = np.sqrt(mean_squared_error(np.zeros_like(residuals), residuals))
    return residuals, mean_val, var_val, rmse_val

# Parameters
np.random.seed(0)
arma_params = [([1, -0.5], [1], (1, 0, 0)), ([1, -0.5, 0.25], [1], (2, 0, 0))]

# Processing ARMA models
results = [process_arma(*params) for params in arma_params]
residuals_list, metrics = zip(*[(res, (mean, var, rmse)) for res, mean, var, rmse in results])

# Compute ACF and PACF for residuals
acf_pacf_results = [(acf(res, nlags=20), pacf(res, nlags=20)) for res in residuals_list]

# Plotting
n_models = len(arma_params)
fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models))
titles = ['ACF of ARMA(1) Residuals', 'PACF of ARMA(1) Residuals', 
          'ACF of ARMA(2) Residuals', 'PACF of ARMA(2) Residuals']

for i, ((acf_vals, pacf_vals), title_acf, title_pacf) in enumerate(zip(acf_pacf_results, titles[::2], titles[1::2])):
    axes[i, 0].stem(acf_vals, use_line_collection=True)
    axes[i, 0].set_title(title_acf)
    axes[i, 0].set_xlabel('Lag')
    axes[i, 0].set_ylabel('ACF')
    
    axes[i, 1].stem(pacf_vals, use_line_collection=True)
    axes[i, 1].set_title(title_pacf)
    axes[i, 1].set_xlabel('Lag')
    axes[i, 1].set_ylabel('PACF')

plt.tight_layout()
plt.show()

# Print evaluation metrics
for i, (mean_val, var_val, rmse_val) in enumerate(metrics, 1):
    print(f'ARMA({i}) Residuals - Mean: {mean_val:.4f}, Variance: {var_val:.4f}, RMSE: {rmse_val:.4f}')
