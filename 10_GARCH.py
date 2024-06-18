!pip install arch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and preprocess the data
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
data.columns = ['Passengers']

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(data['Passengers'], lags=30, ax=plt.gca(), title='Autocorrelation Function (ACF)')
plt.subplot(2, 1, 2)
plot_pacf(data['Passengers'], lags=30, ax=plt.gca(), title='Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Function to fit GARCH model and print summary
def fit_garch_model(data, p, q):
    model = arch_model(data, vol='GARCH', p=p, q=q).fit()
    print(f"\nGARCH({p},{q}) Model Summary:")
    print(model.summary())
    plt.figure(figsize=(10, 6))
    plt.plot(model.conditional_volatility, label=f'Volatility (GARCH({p},{q}))')
    plt.title(f'GARCH({p},{q}) Model - Conditional Volatility')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()
    mae = mean_absolute_error(data, model.conditional_volatility)
    rmse = np.sqrt(mean_squared_error(data, model.conditional_volatility))
    print(f"GARCH({p},{q}) Model - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Fit and evaluate GARCH(1,1) and GARCH(2,2) models
for (p, q) in [(1, 1), (2, 2)]:
    fit_garch_model(data['Passengers'], p, q)
