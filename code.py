import sys
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")
sys.setrecursionlimit(2000)

today = date.today().strftime("%Y-%m-%d")

symbol = "AAPL"

url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-03-16/{today}?adjusted=true&sort=asc&limit=400&apiKey=DjUOG_OOpvoSrMPpBbecBcJ_YAtNasP8"

response = requests.get(url)
data = response.json()

if "results" in data:
    df = pd.DataFrame(data["results"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    plt.figure(figsize=(12, 8))
    plt.plot(df["timestamp"],df['c'])
    plt.grid(True)
    plt.show()
    # Remove duplicates and handle missing values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.set_index("timestamp", inplace=True)

    df = df.asfreq("D").ffill()
    close_prices = df["c"]

    if len(close_prices) > 30:
        decomposition = seasonal_decompose(close_prices, model="additive", period=30)
        plt.figure(figsize=(12, 8))
        plt.subplot(411)
        plt.plot(close_prices, label="Original Data")
        plt.legend()
        plt.subplot(412)
        plt.plot(decomposition.trend, label="Trend", color="red")
        plt.legend()
        plt.subplot(413)
        plt.plot(decomposition.seasonal, label="Seasonality", color="green")
        plt.legend()
        plt.subplot(414)
        plt.plot(decomposition.resid, label="Residual (Cyclic)", color="purple")
        plt.legend()
        plt.suptitle(f"Trend, Seasonality & Cyclic Components - {symbol}")
        plt.tight_layout()
        plt.show()
    plt.figure(figsize=(10, 5))
    plot_acf(close_prices, lags=50)
    plt.title(f"Autocorrelation Function (ACF) for {symbol}")
    plt.show()

    adf_test = adfuller(close_prices)
    print("ADF Test Results:")
    print(f"ADF Statistic: {adf_test[0]}")
    print(f"p-value: {adf_test[1]}")
    print("Critical Values:", adf_test[4])
    print("Stationary" if adf_test[1] < 0.05 else "Non-stationary")

    df_daily = close_prices.resample("D").mean()
    df_weekly = close_prices.resample("W").mean()
    df_monthly = close_prices.resample("M").mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df_daily, label="Daily Avg", marker="o")
    plt.plot(df_weekly, label="Weekly Avg", marker="s")
    plt.plot(df_monthly, label="Monthly Avg", marker="^")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title(f"Stock Prices Averages for {symbol}")
    plt.legend()
    plt.grid(True)
    plt.show()

    df["hour"] = df.index.hour
    df["day"] = df.index.day
    heatmap_data = df.pivot_table(values="c", index="day", columns="hour", aggfunc=np.mean)
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Day of the Month")
    plt.title(f"Stock Price Heatmap for {symbol}")
    plt.show()

    train_size = int(len(close_prices) * 0.8)
    train, test = close_prices[:train_size], close_prices[train_size:]

    def find_best_arima(train_series, p_values, d_values, q_values):
        best_aic = float("inf")
        best_order = None
        best_model = None

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(train_series, order=(p, d, q))
                        model_fit = model.fit()
                        aic = model_fit.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            best_model = model_fit
                    except:
                        continue

        return best_order, best_model

    # Search for best (p, d, q)
    p_values = range(0, 3)
    d_values = range(0, 3)
    q_values = range(0, 3)

from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import timedelta

best_order, best_model = find_best_arima(train, p_values, d_values, q_values)

if best_model:
    print(f"Best ARIMA Order: {best_order}")
    print(best_model.summary())
    N = 7  
    forecast_obj = best_model.get_forecast(steps=N)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    forecast_dates = [df.index[-1] + timedelta(days=i) for i in range(1, N+1)]

    test_forecast = best_model.get_forecast(steps=len(test)).predicted_mean
    mae = mean_absolute_error(test, test_forecast)
    rmse_val = rmse(test, test_forecast)

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse_val}")
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test, label="Actual", color="blue")
    plt.plot(test.index, test_forecast, label="Predicted", color="red")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title(f"ARIMA Model Forecast for {symbol}")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.plot(close_prices.index, close_prices, label="Historical Data", color="blue")
    plt.plot(forecast_dates, forecast, label="Forecast", color="red", marker="o")
    plt.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="pink", alpha=0.3)
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title(f"Forecasted Prices with Confidence Interval for {symbol}")
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("No valid ARIMA model found.")