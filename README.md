# Stock Price Analysis Web App

This is a Flask web application that retrieves stock price data using the Polygon API, performs time series analysis, and visualizes trends using ARIMA modeling.

# Features
- Fetches stock price data for a given symbol
- Performs time series decomposition (trend, seasonality, residuals)
- Plots autocorrelation function (ACF)
- Conducts Augmented Dickey-Fuller (ADF) test for stationarity
- Generates stock price heatmaps and moving averages
- Implements ARIMA forecasting with optimal parameters
- Displays various visualizations for stock trend analysis

# Project Structure
stock_analysis_app/
│
|
├── app.py                     
│
|
├── templates/                
│   └── index.html            
│
├── static/                  

# Files
- app.py: Contains the Flask web application and stock analysis code.
- code.py: Contains the core logic without Flask for standalone processing.
- index.html: UI interface for user input and displaying results.

# Usage
- Enter a stock symbol in the web UI.
- View data analysis, trends, and forecasts.
- Forecast future stock prices with ARIMA.

# Notes
- Replace api_key in app.py with your own Polygon API key.
- Ensure static/ folder exists for saving images.




