import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
import matplotlib.pyplot as plt

# Load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, header=0, parse_dates=['Month'], index_col='Month')
df.index.freq = 'MS'

# Split data into train and test sets
train, test = df.iloc[:-12], df.iloc[-12:]

# Forecasting functions
def simple_exp_smoothing(train, test):
    model = SimpleExpSmoothing(train).fit()
    forecast = model.forecast(len(test))
    return forecast, mean_absolute_error(test, forecast), mean_squared_error(test, forecast)

def simple_moving_average(train, test, window_size=12):
    forecast = np.repeat(train.rolling(window=window_size).mean().iloc[-1], len(test))
    return forecast, mean_absolute_error(test, forecast), mean_squared_error(test, forecast)

def holt_winters(train, test, seasonal_periods=12):
    model = ExponentialSmoothing(train, seasonal_periods=seasonal_periods, trend='add', seasonal='add').fit()
    forecast = model.forecast(len(test))
    return forecast, mean_absolute_error(test, forecast), mean_squared_error(test, forecast)

# Calculate forecasts and errors
ses_forecast, ses_mae, ses_mse = simple_exp_smoothing(train, test)
sma_forecast, sma_mae, sma_mse = simple_moving_average(train, test)
hw_forecast, hw_mae, hw_mse = holt_winters(train, test)

# Calculate RMSE
ses_rmse = np.sqrt(ses_mse)
sma_rmse = np.sqrt(sma_mse)
hw_rmse = np.sqrt(hw_mse)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(test.index, ses_forecast, label=f'SES (MAE={ses_mae:.2f}, RMSE={ses_rmse:.2f})')
plt.plot(test.index, sma_forecast, label=f'SMA (MAE={sma_mae:.2f}, RMSE={sma_rmse:.2f})')
plt.plot(test.index, hw_forecast, label=f'Holt-Winters (MAE={hw_mae:.2f}, RMSE={hw_rmse:.2f})')
plt.title('Airline Passengers Forecasting')
plt.xlabel('Year')
plt.ylabel('Passengers')
plt.legend()
plt.grid(True)
plt.show()
