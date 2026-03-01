from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
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

app = Flask(__name__)


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------

def fetch_stock_data(symbol):
    today = date.today().strftime("%Y-%m-%d")
    api_key = "YOUR_API_KEY"
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-03-16/{today}?adjusted=true&sort=asc&limit=400&apiKey={api_key}"

    response = requests.get(url)
    data = response.json()

    if "results" not in data:
        return None

    df = pd.DataFrame(data["results"])
    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.asfreq("D").ffill()

    return df["c"]


def adf_analysis(series):
    result = adfuller(series)
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Stationary": result[1] < 0.05
    }


def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    return mae, rmse


def find_best_arima(train):
    best_aic = float("inf")
    best_order = None
    best_model = None

    for p in range(3):
        for d in range(3):
            for q in range(3):
                try:
                    model = ARIMA(train, order=(p, d, q))
                    fit = model.fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                        best_model = fit
                except:
                    continue

    return best_order, best_model


# ---------------------------------------------------
# Route
# ---------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        symbol = request.form['symbol'].upper()
        close_prices = fetch_stock_data(symbol)

        if close_prices is None:
            return render_template('index.html', error="Invalid symbol or no data found.")

        # -------------------------
        # ADF TEST
        # -------------------------
        adf_results = adf_analysis(close_prices)

        # -------------------------
        # VISUALIZATION: Closing Price
        # -------------------------
        plt.figure(figsize=(12, 6))
        plt.plot(close_prices)
        plt.title(f"{symbol} Closing Prices")
        plt.grid()
        plt.savefig("static/closing_prices.png")
        plt.close()

        # -------------------------
        # SEASONAL DECOMPOSITION
        # -------------------------
        if len(close_prices) > 60:
            decomposition = seasonal_decompose(close_prices, model="additive", period=30)
            fig = decomposition.plot()
            fig.savefig("static/decomposition.png")
            plt.close()

        # -------------------------
        # ACF
        # -------------------------
        plt.figure(figsize=(10, 5))
        plot_acf(close_prices, lags=40)
        plt.savefig("static/acf.png")
        plt.close()

        # -------------------------
        # Rolling Averages
        # -------------------------
        rolling_mean = close_prices.rolling(20).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(close_prices, label="Original")
        plt.plot(rolling_mean, label="20-Day MA")
        plt.legend()
        plt.savefig("static/rolling_avg.png")
        plt.close()

        # -------------------------
        # Train/Test Split
        # -------------------------
        train_size = int(len(close_prices) * 0.8)
        train, test = close_prices[:train_size], close_prices[train_size:]

        # -------------------------
        # ARIMA
        # -------------------------
        best_order, arima_model = find_best_arima(train)

        arima_forecast = arima_model.forecast(steps=len(test))
        arima_mae, arima_rmse = evaluate_model(test, arima_forecast)

        # -------------------------
        # SARIMA
        # -------------------------
        sarima_model = SARIMAX(train,
                               order=(1, 1, 1),
                               seasonal_order=(1, 1, 1, 12))
        sarima_fit = sarima_model.fit()

        sarima_forecast = sarima_fit.forecast(steps=len(test))
        sarima_mae, sarima_rmse = evaluate_model(test, sarima_forecast)

        # -------------------------
        # Forecast Plot Comparison
        # -------------------------
        plt.figure(figsize=(12, 6))
        plt.plot(test.index, test, label="Actual", color="black")
        plt.plot(test.index, arima_forecast, label="ARIMA", color="red")
        plt.plot(test.index, sarima_forecast, label="SARIMA", color="green")
        plt.legend()
        plt.title("ARIMA vs SARIMA Forecast")
        plt.grid()
        plt.savefig("static/forecast_comparison.png")
        plt.close()

        return render_template(
            'index.html',
            symbol=symbol,
            adf_results=adf_results,
            arima_mae=arima_mae,
            arima_rmse=arima_rmse,
            sarima_mae=sarima_mae,
            sarima_rmse=sarima_rmse,
            best_order=best_order
        )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
