import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load the dataset
df = pd.read_csv("AirPassengers.csv")

# Preview the dataset
print(df.head())

# Convert 'Month' to datetime and set as index
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Extract the time series
data = df['#Passengers']

# Step 2: Plot the original data
data.plot(title='Monthly Air Passengers', figsize=(10, 6), grid=True)
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.show()

# Step 2 (continued): Rolling statistics
rolling_mean = data.rolling(window=12).mean()
rolling_std = data.rolling(window=12).std()

plt.figure(figsize=(10, 6))
plt.plot(data, label='Original')
plt.plot(rolling_mean, color='red', label='12-Month Rolling Mean')
plt.plot(rolling_std, color='green', label='12-Month Rolling Std Dev')
plt.title('Rolling Statistics')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid(True)
plt.show()

# Step 3: ADF Test
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

print("ADF Test on Original Data:")
adf_test(data)

# Step 4: Differencing
data_diff = data.diff().dropna()

data_diff.plot(title='Differenced Data', figsize=(10, 6), grid=True)
plt.xlabel('Date')
plt.ylabel('Differenced Passengers')
plt.show()

print("\nADF Test on Differenced Data:")
adf_test(data_diff)

# Step 5: ACF and PACF
plot_acf(data_diff, lags=20)
plt.title('Autocorrelation (ACF)')
plt.show()

plot_pacf(data_diff, lags=20)
plt.title('Partial Autocorrelation (PACF)')
plt.show()

# Step 6: Fit ARIMA model
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()

# Step 7: Residual Analysis
print("\nARIMA Model Summary:")
print(model_fit.summary())

residuals = model_fit.resid

plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals')
plt.grid(True)
plt.show()

plt.hist(residuals, bins=15, color='lightblue', edgecolor='black')
plt.title('Histogram of Residuals')
plt.grid(True)
plt.show()

# Step 8: Forecast next 12 months
forecast = model_fit.forecast(steps=12)
forecast_index = pd.date_range(data.index[-1] + pd.DateOffset(1), periods=12, freq='MS')

# Step 9: Plot Forecast
plt.figure(figsize=(10, 6))
plt.plot(data, label='Historical Data')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid(True)
plt.show()
