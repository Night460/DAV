import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

# Step 1: Generate synthetic monthly sales data
dates = pd.date_range('2020-01', '2023-12', freq='MS')  # Monthly Start
sales = [200 + i*2 + np.random.randn()*10 for i in range(len(dates))]
data = pd.Series(sales, index=dates)

# Step 2: Plot the original data
data.plot(title='Monthly Sales', figsize=(10, 6), grid=True)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Step 2 (continued): Add Rolling Mean and Standard Deviation
rolling_mean = data.rolling(window=12).mean()
rolling_std = data.rolling(window=12).std()

plt.figure(figsize=(10, 6))
plt.plot(data, label='Original Sales')
plt.plot(rolling_mean, color='red', label='12-Month Rolling Mean')
plt.plot(rolling_std, color='green', label='12-Month Rolling Std Dev')
plt.title('Monthly Sales with Rolling Mean and Std Deviation')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Step 3: Check for Stationarity with the Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

# Perform ADF test on original data
print("ADF Test on Original Data:")
adf_test(data)

# Step 4: Apply differencing if not stationary
data_diff = data.diff().dropna()

# Plot differenced data
data_diff.plot(title='Differenced Sales Data', figsize=(10, 6), grid=True)
plt.xlabel('Date')
plt.ylabel('Differenced Sales')
plt.show()

# ADF test after differencing
print("\nADF Test on Differenced Data:")
adf_test(data_diff)

# Step 5: Plot ACF and PACF
plot_acf(data_diff, lags=20)
plt.title('Autocorrelation (ACF)')
plt.show()

plot_pacf(data_diff, lags=20)
plt.title('Partial Autocorrelation (PACF)')
plt.show()

# Step 6: Fit ARIMA model (example: p=1, d=1, q=1)
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()

# Step 7: Residual analysis
print("\nARIMA Model Summary:")
print(model_fit.summary())

# Plot residuals
residuals = model_fit.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals from ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Histogram of residuals
plt.hist(residuals, bins=15, color='skyblue', edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Step 8: Forecast future values (next 12 months)
forecast = model_fit.forecast(steps=12)
forecast_index = pd.date_range(data.index[-1] + pd.DateOffset(1), periods=12, freq='MS')

# Step 9: Plot historical + forecasted values
plt.figure(figsize=(10, 6))
plt.plot(data, label='Historical Sales')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.title('ARIMA Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()
