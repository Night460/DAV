import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# Generate sample monthly data
dates = pd.date_range('2020-01', '2023-12', freq='MS')
sales = [200 + i*2 + np.random.randn()*10 for i in range(len(dates))]
data = pd.Series(sales, index=dates)

# Line Plot: Original Time Series
data.plot(title='Monthly Sales', figsize=(8, 4), grid=True)
plt.ylabel("Sales")
plt.xlabel("Date")
plt.show()

# Histogram: Distribution of values
plt.hist(data, bins=12, color='skyblue', edgecolor='black')
plt.title('Histogram of Sales Data')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Box Plot: To detect outliers
plt.boxplot(data, vert=False)
plt.title('Box Plot of Sales Data')
plt.xlabel('Sales')
plt.grid(True)
plt.show()

# Decomposition
seasonal_decompose(data, model='additive', period=12).plot()
plt.tight_layout()
plt.show()

# ACF and PACF
plot_acf(data, lags=20)
plt.title('Autocorrelation (ACF)')
plt.show()

plot_pacf(data, lags=20)
plt.title('Partial Autocorrelation (PACF)')
plt.show()

# ADF Test for Stationarity
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f"   {key}: {value}")
    if result[1] <= 0.05:
        print("=> Data is stationary.")
    else:
        print("=> Data is NOT stationary. Differencing is required.")

print("\n--- Augmented Dickey-Fuller Test ---")
adf_test(data)

# ARIMA Modeling
model = ARIMA(data, order=(1, 1, 1)).fit()
forecast = model.forecast(12)

# Forecast Plot
data.plot(label='History')
forecast.index = pd.date_range(data.index[-1] + pd.DateOffset(1), periods=12, freq='MS')
forecast.plot(label='Forecast', color='red')
plt.title('Sales Forecast')
plt.legend()
plt.grid(True)
plt.show()
