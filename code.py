import sys
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")
sys.setrecursionlimit(2000)

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------
symbol = "AAPL"
today = date.today().strftime("%Y-%m-%d")
api_key = "YOUR_API_KEY"

url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-03-16/{today}?adjusted=true&sort=asc&limit=400&apiKey={api_key}"

# ----------------------------------------------------
# FETCH DATA
# ----------------------------------------------------
response = requests.get(url)
data = response.json()

if "results" not in data:
    print("No data found.")
    exit()

df = pd.DataFrame(data["results"])
df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
df.set_index("timestamp", inplace=True)

df = df.asfreq("D").ffill()
close_prices = df["c"]

# ----------------------------------------------------
# VISUALIZATION: Closing Prices
# ----------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(close_prices)
plt.title(f"{symbol} Closing Prices")
plt.grid()
plt.show()

# ----------------------------------------------------
# SEASONAL DECOMPOSITION
# ----------------------------------------------------
if len(close_prices) > 60:
    decomposition = seasonal_decompose(close_prices, model="additive", period=30)
    decomposition.plot()
    plt.suptitle("Trend / Seasonality / Residual")
    plt.show()

# ----------------------------------------------------
# ACF
# ----------------------------------------------------
plot_acf(close_prices, lags=40)
plt.title("ACF")
plt.show()

# ----------------------------------------------------
# ADF TEST
# ----------------------------------------------------
adf = adfuller(close_prices)
print("\nADF Test Results:")
print(f"ADF Statistic: {adf[0]}")
print(f"p-value: {adf[1]}")
print("Stationary" if adf[1] < 0.05 else "Non-stationary")

# ----------------------------------------------------
# ROLLING AVERAGE
# ----------------------------------------------------
rolling_mean = close_prices.rolling(20).mean()
plt.figure(figsize=(12, 6))
plt.plot(close_prices, label="Original")
plt.plot(rolling_mean, label="20-Day MA")
plt.legend()
plt.grid()
plt.show()

# ----------------------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------------------
train_size = int(len(close_prices) * 0.8)
train, test = close_prices[:train_size], close_prices[train_size:]

# ----------------------------------------------------
# ARIMA GRID SEARCH
# ----------------------------------------------------
def find_best_arima(train_series):
    best_aic = float("inf")
    best_order = None
    best_model = None

    for p in range(3):
        for d in range(3):
            for q in range(3):
                try:
                    model = ARIMA(train_series, order=(p, d, q))
                    fit = model.fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                        best_model = fit
                except:
                    continue

    return best_order, best_model


best_order, arima_model = find_best_arima(train)

# ----------------------------------------------------
# ARIMA FORECAST
# ----------------------------------------------------
arima_forecast = arima_model.forecast(steps=len(test))
arima_mae = mean_absolute_error(test, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))

print("\nBest ARIMA Order:", best_order)
print("ARIMA MAE:", arima_mae)
print("ARIMA RMSE:", arima_rmse)

# ----------------------------------------------------
# SARIMA MODEL
# ----------------------------------------------------
sarima_model = SARIMAX(train,
                       order=(1,1,1),
                       seasonal_order=(1,1,1,12))
sarima_fit = sarima_model.fit()

sarima_forecast = sarima_fit.forecast(steps=len(test))
sarima_mae = mean_absolute_error(test, sarima_forecast)
sarima_rmse = np.sqrt(mean_squared_error(test, sarima_forecast))

print("\nSARIMA MAE:", sarima_mae)
print("SARIMA RMSE:", sarima_rmse)

# ----------------------------------------------------
# MODEL COMPARISON PLOT
# ----------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label="Actual", color="black")
plt.plot(test.index, arima_forecast, label="ARIMA", color="red")
plt.plot(test.index, sarima_forecast, label="SARIMA", color="green")
plt.legend()
plt.title("ARIMA vs SARIMA Forecast")
plt.grid()
plt.show()

# ----------------------------------------------------
# FUTURE FORECAST (7 DAYS)
# ----------------------------------------------------
N = 7
future_forecast = sarima_fit.forecast(steps=N)
future_dates = [close_prices.index[-1] + timedelta(days=i) for i in range(1, N+1)]

plt.figure(figsize=(12, 6))
plt.plot(close_prices.index, close_prices, label="Historical")
plt.plot(future_dates, future_forecast, marker="o", label="7-Day Forecast")
plt.legend()
plt.grid()
plt.show()

