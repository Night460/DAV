import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Load data and set datetime index
df = pd.read_csv(
    "AirPassengers.csv",
    parse_dates=['Month'],
    infer_datetime_format=True
)
df.rename(columns={'#Passengers': 'Passengers'}, inplace=True)
df.set_index('Month', inplace=True)

# 2. Time series decomposition
decomp = seasonal_decompose(df['Passengers'], model='additive', period=12)
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
decomp.observed.plot(ax=axes[0], title='Original')
decomp.trend.plot(ax=axes[1], title='Trend')
decomp.seasonal.plot(ax=axes[2], title='Seasonal')
decomp.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()

# 3. ACF and PACF
plot_acf(df['Passengers'], lags=50, title='Autocorrelation')  # lags up to 50
plot_pacf(df['Passengers'], lags=50, title='Partial Autocorrelation')
plt.show()

# 4. Rolling mean smoothing
rolling_mean = df['Passengers'].rolling(window=12).mean()
plt.plot(df['Passengers'], label='Original')
plt.plot(rolling_mean, label='12‑Month Rolling Mean', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.title('Original vs. Rolling Mean')
plt.show()

# 5. Feature engineering
df_feat = df.copy()
df_feat['month'] = df_feat.index.month
df_feat['year'] = df_feat.index.year

X = df_feat[['month', 'year']]
y = df_feat['Passengers']

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=False
)

# 7. Model training and forecasting
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. Evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Training R²:  {model.score(X_train, y_train):.3f}")
print(f"Testing R²:   {model.score(X_test, y_test):.3f}")
print(f"Test RMSE:    {rmse:.2f}")

# 9. Plot forecasts vs. actuals
plt.plot(y_train.index, y_train, label='Train', alpha=0.7)
plt.plot(y_test.index, y_test, label='Actual', color='orange')
plt.plot(y_test.index, y_pred, label='Forecast', color='green')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.title('AirPassengers Forecast')
plt.legend()
plt.show()
