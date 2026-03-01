# StockPredPy

StockPredPy is a Python-based project that performs time series forecasting on stock prices. It uses the [Polygon.io](https://polygon.io/) API to fetch real-time historical stock data and applies statistical models (ARIMA and SARIMA) to predict future prices. The project includes both a standalone Python script for terminal execution and a Flask-based web application for an interactive user interface.

## Features

- **Real-Time Data Fetching:** Automatically retrieves the latest daily adjusted closing prices for any given stock ticker using the Polygon.io API.
- **Time Series Analysis:**
  - **ADF (Augmented Dickey-Fuller) Test:** Checks for stationarity in the stock price data.
  - **Seasonal Decomposition:** Breaks down the series into trend, seasonality, and residual components.
  - **ACF (Autocorrelation Function) Plot:** Helps in identifying the moving average component (MA) for the ARIMA model.
  - **Rolling Averages:** Plots a 20-day moving average alongside the actual prices to visualize short-term trends.
- **Predictive Modeling:**
  - **ARIMA (AutoRegressive Integrated Moving Average):** Automatically finds the best hyperparameters (p, d, q) using Grid Search to fit the dataset.
  - **SARIMA (Seasonal ARIMA):** Accounts for seasonal trends (e.g., monthly/yearly patterns) for more accurate forecasting.
- **Evaluation Metrics:** Evaluates both models using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Web Interface (Flask):** Allows users to dynamically input a stock ticker and view visualizations and predictions directly in the browser.

## Project Structure

```
stockpredpy/
│
├── main.py               # Flask web application handling routes, data processing, and rendering.
├── stock.py              # Standalone Python script for terminal-based execution/testing.
├── templates/
│   └── index.html        # HTML template for the Flask frontend.
├── static/               # Directory where Flask saves generated Matplotlib plots.
│   ├── closing_prices.png
│   ├── decomposition.png
│   ├── acf.png
│   ├── rolling_avg.png
│   └── forecast_comparison.png
└── README.md             # Project documentation.
```

## Prerequisites

Ensure you have Python 3.7+ installed. Then, install the required Python packages:

```bash
pip install flask requests pandas numpy seaborn matplotlib statsmodels scikit-learn
```

## How to Run the Project

### Method 1: Flask Web Application (Recommended)

1. Open a terminal and navigate to the project directory:
   ```bash
   cd /path/to/stockpredpy
   ```
2. Run the `main.py` file:
   ```bash
   python main.py
   ```
3. Open your web browser and go to `http://127.0.0.1:5000/`.
4. Enter any valid US Stock Ticker symbol (e.g., `AAPL`, `TSLA`, `MSFT`) into the search bar to view the analysis and forecast.

### Method 2: Standalone Python Script

If you want to run the analysis directly through your terminal and view the plots in separate windows, use the standalone script:

1. Open `stock.py` in your editor and update the `symbol` variable at the top if you want to analyze a stock other than Apple (`AAPL`).
2. Run the script:
   ```bash
   python stock.py
   ```
3. Close each plotted graph window as it appears to let the script progress to the next step and finish model training/evaluation.

## Configuration (API Key)

Both `main.py` and `stock.py` use a Polygon.io API key defined in the code:
```python
api_key = "DjUOG_OOpvoSrMPpBbecBcJ_YAtNasP8"
```
*Note: Make sure your Polygon API tier covers the requests being made or substitute it with your own API key if you hit rate limits.*

## Disclaimer

This project is for educational and demonstrative purposes only. It is not intended to be used as financial advice or used directly for real trading and investments. Financial markets are highly volatile, and statistical models do not guarantee future performance.




