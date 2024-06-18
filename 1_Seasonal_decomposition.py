# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url)

# Check for missing values
print(df.isnull().sum())

# Fill missing values with linear interpolation
df['Passengers'] = df['Passengers'].interpolate(method='linear')

# Convert the 'Month' column to datetime
df['Month'] = pd.to_datetime(df['Month'])

# Set the 'Month' column as the index
df.set_index('Month', inplace=True)

# Plot the time series data
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Passengers'], label='Passengers')
plt.xlabel('Time')
plt.ylabel('Number of Passengers')
plt.title('Monthly Number of Air Passengers')
plt.legend()
plt.show()

# Decompose the time series
result = seasonal_decompose(df['Passengers'], model='multiplicative')

# Plot the decomposition
result.plot()
plt.show()
