#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Define the stocks to analyze
stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-A', 'V', 'JPM', 'JNJ']

# Data loading function
@st.cache_data  # Updated caching method
def load_data(stocks, start_date, end_date):
    # Download stock data from Yahoo Finance
    stock_data = yf.download(stocks, start=start_date, end=end_date)['Close']
    
    # Calculate daily returns
    daily_returns = stock_data.pct_change().dropna()
    
    return stock_data, daily_returns

# Ask user to input start and end date
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-03-01"))

# Get today's date
today = datetime.today().date()  # Using datetime.today().date() to get today's date

# If the end date is in the future, set it to today's date
if end_date > today:
    st.warning(f"End date cannot be in the future. The end date has been set to {today}.")
    end_date = today

# Load data based on the selected date range
if start_date < end_date:
    stock_data, daily_returns = load_data(stocks, start_date, end_date)

    # Compute mean returns and covariance matrix
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    
    # Display results
    st.write(f"Data range: {start_date} - {end_date}")
    
else:
    st.error("Start date must be before the end date.")


def optimal_portfolio(selected_stocks, method='SLSQP'):
    num_assets = len(selected_stocks)
    selected_returns = mean_returns[selected_stocks]
    selected_cov_matrix = cov_matrix.loc[selected_stocks, selected_stocks]

    if method == 'SLSQP':
        def negative_sharpe(weights):
            portfolio_return = np.sum(weights * selected_returns)
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(selected_cov_matrix, weights)))
            return - (portfolio_return / portfolio_stddev)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.ones(num_assets) / num_assets

        optimized = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = optimized.x
        optimal_return = np.sum(optimal_weights * selected_returns)
        optimal_stddev = np.sqrt(np.dot(optimal_weights.T, np.dot(selected_cov_matrix, optimal_weights)))
        optimal_sharpe = optimal_return / optimal_stddev

    elif method == 'Monte Carlo':
        num_simulations = 10000
        all_weights = np.zeros((num_simulations, num_assets))
        ret_arr = np.zeros(num_simulations)
        vol_arr = np.zeros(num_simulations)
        sharpe_arr = np.zeros(num_simulations)

        for i in range(num_simulations):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            all_weights[i, :] = weights
            ret_arr[i] = np.sum(weights * selected_returns)
            vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(selected_cov_matrix, weights)))
            sharpe_arr[i] = ret_arr[i] / vol_arr[i]

        max_sharpe_idx = sharpe_arr.argmax()
        optimal_weights = all_weights[max_sharpe_idx]
        optimal_return = ret_arr[max_sharpe_idx]
        optimal_stddev = vol_arr[max_sharpe_idx]
        optimal_sharpe = sharpe_arr[max_sharpe_idx]
    
    return optimal_weights, optimal_return, optimal_stddev, optimal_sharpe

def predict_stock_price(stock_data, stock, future_days=30):
    # Prepare the data for regression
    stock_prices = stock_data[stock].values
    dates = np.array(range(len(stock_prices))).reshape(-1, 1)
    
    # Linear regression model
    model = LinearRegression()
    model.fit(dates, stock_prices)
    
    # Predict future stock prices
    future_dates = np.array(range(len(stock_prices), len(stock_prices) + future_days)).reshape(-1, 1)
    predicted_prices = model.predict(future_dates)
    
    # Return model coefficients (intercept and slope) and predictions
    intercept = model.intercept_
    slope = model.coef_[0]
    
    return predicted_prices, future_dates, intercept, slope

st.sidebar.title("Investment Application")
page = st.sidebar.radio("Select Page:", ["Portfolio Analysis", "Stock Data"])

if page == "Portfolio Analysis":
    st.title("Create Portfolio Scenarios and Risk Analysis")
    
    selected_stocks = st.sidebar.multiselect("Select stocks to add to your portfolio:", stocks, default=stocks[:5])
    investment_amount = st.sidebar.number_input("Enter Investment Amount ($):", min_value=100, value=1000, step=100)
    time_period = st.sidebar.radio("Select Time Period:", ["Daily", "Monthly", "Yearly"])
    optimization_method = st.sidebar.radio("Optimization Method:", ["SLSQP", "Monte Carlo"])
    
    period_factors = {"Daily": 1, "Monthly": 21, "Yearly": 252}
    period_factor = period_factors.get(time_period, 1)

    if len(selected_stocks) < 2:
        st.error("Please select at least two stocks.")
    else:
        optimal_weights, optimal_return, optimal_stddev, optimal_sharpe = optimal_portfolio(selected_stocks, method=optimization_method)
        allocation = {selected_stocks[i]: optimal_weights[i] * investment_amount for i in range(len(selected_stocks))}
        
        adjusted_return = optimal_return * period_factor * 100
        adjusted_stddev = optimal_stddev * np.sqrt(period_factor) * 100
        adjusted_sharpe = adjusted_return / adjusted_stddev if adjusted_stddev != 0 else 0
        
        st.header("Optimal Portfolio Analysis")
        st.write(f"**Portfolio {time_period} Return:** {adjusted_return:.2f}%")
        st.write(f"**Portfolio {time_period} Volatility (Risk):** {adjusted_stddev:.2f}%")
        st.write(f"**Portfolio {time_period} Sharpe Ratio:** {adjusted_sharpe:.4f}")

        st.subheader("Stock Allocation")
        allocation_df = pd.DataFrame({
            "Stock": selected_stocks,
            "Weight (%)": (optimal_weights * 100).round(2),
            "Investment Amount ($)": [allocation[s] for s in selected_stocks]
        })
        st.table(allocation_df)

elif page == "Stock Data":
    st.title("Daily Stock Data and Price Prediction")
    
    selected_stock = st.selectbox("Select the stock you want to view:", stocks)
    
    # Display stock data for selected stock
    st.subheader("Stock Performance")
    avg_return = mean_returns[selected_stock] * 100
    std_dev = np.sqrt(cov_matrix.loc[selected_stock, selected_stock]) * 100
    st.write(f"**Average Daily Return:** {avg_return:.2f}%")
    st.write(f"**Daily Volatility (Risk):** {std_dev:.2f}%")
    
    # Plot the stock data using line chart
    st.line_chart(stock_data[selected_stock])
    
    # Add Stock Price Prediction functionality
    prediction_days = st.sidebar.slider("Number of days to predict:", 1, 365, 30)
    
    # Get the predicted stock prices and regression coefficients
    predicted_prices, future_dates, intercept, slope = predict_stock_price(stock_data, selected_stock, future_days=prediction_days)
    
    # Create future dates for prediction
    future_dates = pd.to_datetime([stock_data.index[-1] + pd.Timedelta(days=i) for i in range(1, prediction_days + 1)])
    
    # Create prediction DataFrame
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices})
    
    st.write(f"Predicted stock prices for {selected_stock} for the next {prediction_days} days:")
    st.line_chart(prediction_df.set_index('Date')['Predicted Price'])
    
    st.table(prediction_df)
    
    # Display regression equation
    st.subheader("Regression Equation")
    regression_equation = f"y = {slope:.4f}x + {intercept:.2f}"
    st.write(f"The regression equation is: {regression_equation}")


# In[2]:


def run_streamlit():
    script_content = """

import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Define the stocks to analyze
stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-A', 'V', 'JPM', 'JNJ']

# Data loading function
@st.cache_data  # Updated caching method
def load_data(stocks, start_date, end_date):
    # Download stock data from Yahoo Finance
    stock_data = yf.download(stocks, start=start_date, end=end_date)['Close']
    
    # Calculate daily returns
    daily_returns = stock_data.pct_change().dropna()
    
    return stock_data, daily_returns

# Ask user to input start and end date
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-03-01"))

# Get today's date
today = datetime.today().date()  # Using datetime.today().date() to get today's date

# If the end date is in the future, set it to today's date
if end_date > today:
    st.warning(f"End date cannot be in the future. The end date has been set to {today}.")
    end_date = today

# Load data based on the selected date range
if start_date < end_date:
    stock_data, daily_returns = load_data(stocks, start_date, end_date)

    # Compute mean returns and covariance matrix
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    
    # Display results
    st.write(f"Data range: {start_date} - {end_date}")
    
else:
    st.error("Start date must be before the end date.")


def optimal_portfolio(selected_stocks, method='SLSQP'):
    num_assets = len(selected_stocks)
    selected_returns = mean_returns[selected_stocks]
    selected_cov_matrix = cov_matrix.loc[selected_stocks, selected_stocks]

    if method == 'SLSQP':
        def negative_sharpe(weights):
            portfolio_return = np.sum(weights * selected_returns)
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(selected_cov_matrix, weights)))
            return - (portfolio_return / portfolio_stddev)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.ones(num_assets) / num_assets

        optimized = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = optimized.x
        optimal_return = np.sum(optimal_weights * selected_returns)
        optimal_stddev = np.sqrt(np.dot(optimal_weights.T, np.dot(selected_cov_matrix, optimal_weights)))
        optimal_sharpe = optimal_return / optimal_stddev

    elif method == 'Monte Carlo':
        num_simulations = 10000
        all_weights = np.zeros((num_simulations, num_assets))
        ret_arr = np.zeros(num_simulations)
        vol_arr = np.zeros(num_simulations)
        sharpe_arr = np.zeros(num_simulations)

        for i in range(num_simulations):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            all_weights[i, :] = weights
            ret_arr[i] = np.sum(weights * selected_returns)
            vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(selected_cov_matrix, weights)))
            sharpe_arr[i] = ret_arr[i] / vol_arr[i]

        max_sharpe_idx = sharpe_arr.argmax()
        optimal_weights = all_weights[max_sharpe_idx]
        optimal_return = ret_arr[max_sharpe_idx]
        optimal_stddev = vol_arr[max_sharpe_idx]
        optimal_sharpe = sharpe_arr[max_sharpe_idx]
    
    return optimal_weights, optimal_return, optimal_stddev, optimal_sharpe

def predict_stock_price(stock_data, stock, future_days=30):
    # Prepare the data for regression
    stock_prices = stock_data[stock].values
    dates = np.array(range(len(stock_prices))).reshape(-1, 1)
    
    # Linear regression model
    model = LinearRegression()
    model.fit(dates, stock_prices)
    
    # Predict future stock prices
    future_dates = np.array(range(len(stock_prices), len(stock_prices) + future_days)).reshape(-1, 1)
    predicted_prices = model.predict(future_dates)
    
    # Return model coefficients (intercept and slope) and predictions
    intercept = model.intercept_
    slope = model.coef_[0]
    
    return predicted_prices, future_dates, intercept, slope

st.sidebar.title("Investment Application")
page = st.sidebar.radio("Select Page:", ["Portfolio Analysis", "Stock Data"])

if page == "Portfolio Analysis":
    st.title("Create Portfolio Scenarios and Risk Analysis")
    
    selected_stocks = st.sidebar.multiselect("Select stocks to add to your portfolio:", stocks, default=stocks[:5])
    investment_amount = st.sidebar.number_input("Enter Investment Amount ($):", min_value=100, value=1000, step=100)
    time_period = st.sidebar.radio("Select Time Period:", ["Daily", "Monthly", "Yearly"])
    optimization_method = st.sidebar.radio("Optimization Method:", ["SLSQP", "Monte Carlo"])
    
    period_factors = {"Daily": 1, "Monthly": 21, "Yearly": 252}
    period_factor = period_factors.get(time_period, 1)

    if len(selected_stocks) < 2:
        st.error("Please select at least two stocks.")
    else:
        optimal_weights, optimal_return, optimal_stddev, optimal_sharpe = optimal_portfolio(selected_stocks, method=optimization_method)
        allocation = {selected_stocks[i]: optimal_weights[i] * investment_amount for i in range(len(selected_stocks))}
        
        adjusted_return = optimal_return * period_factor * 100
        adjusted_stddev = optimal_stddev * np.sqrt(period_factor) * 100
        adjusted_sharpe = adjusted_return / adjusted_stddev if adjusted_stddev != 0 else 0
        
        st.header("Optimal Portfolio Analysis")
        st.write(f"**Portfolio {time_period} Return:** {adjusted_return:.2f}%")
        st.write(f"**Portfolio {time_period} Volatility (Risk):** {adjusted_stddev:.2f}%")
        st.write(f"**Portfolio {time_period} Sharpe Ratio:** {adjusted_sharpe:.4f}")

        st.subheader("Stock Allocation")
        allocation_df = pd.DataFrame({
            "Stock": selected_stocks,
            "Weight (%)": (optimal_weights * 100).round(2),
            "Investment Amount ($)": [allocation[s] for s in selected_stocks]
        })
        st.table(allocation_df)

elif page == "Stock Data":
    st.title("Daily Stock Data and Price Prediction")
    
    selected_stock = st.selectbox("Select the stock you want to view:", stocks)
    
    # Display stock data for selected stock
    st.subheader("Stock Performance")
    avg_return = mean_returns[selected_stock] * 100
    std_dev = np.sqrt(cov_matrix.loc[selected_stock, selected_stock]) * 100
    st.write(f"**Average Daily Return:** {avg_return:.2f}%")
    st.write(f"**Daily Volatility (Risk):** {std_dev:.2f}%")
    
    # Plot the stock data using line chart
    st.line_chart(stock_data[selected_stock])
    
    # Add Stock Price Prediction functionality
    prediction_days = st.sidebar.slider("Number of days to predict:", 1, 365, 30)
    
    # Get the predicted stock prices and regression coefficients
    predicted_prices, future_dates, intercept, slope = predict_stock_price(stock_data, selected_stock, future_days=prediction_days)
    
    # Create future dates for prediction
    future_dates = pd.to_datetime([stock_data.index[-1] + pd.Timedelta(days=i) for i in range(1, prediction_days + 1)])
    
    # Create prediction DataFrame
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices})
    
    st.write(f"Predicted stock prices for {selected_stock} for the next {prediction_days} days:")
    st.line_chart(prediction_df.set_index('Date')['Predicted Price'])
    
    
    
    # Display regression equation
    st.subheader("Regression Equation")
    regression_equation = f"y = {slope:.4f}x + {intercept:.2f}"
    st.write(f"The regression equation is: {regression_equation}")
    st.table(prediction_df)

"""
    with open("streamlit_app.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    subprocess.Popen(["streamlit", "run", "streamlit_app.py"], shell=True)

run_streamlit()


# In[ ]:




