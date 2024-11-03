# import all necessary datasets
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the tickers and allocations for each portfolio
tickers_a = ["SPY", "EWC"] 
weights_a = [0.4, 0.6]

tickers_b = ["SPY", "EWC", "VGK", "MCHI", "EWJ", "EWY", "INDA"]
weights_b = [0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]

# Define the time period
start_date = "2018-01-01"
end_date = "2024-10-20"

# Download historical price data
data_a = yf.download(tickers_a, start=start_date, end=end_date)["Adj Close"] # local equities portfolio
data_b = yf.download(tickers_b, start=start_date, end=end_date)["Adj Close"] # international equities portfolio
# Show the returns of each ETF
returns_a = data_a.iloc[:, 0:].pct_change() * 100
returns_a.fillna(0)
returns_b = data_b.iloc[:, 0:].pct_change() * 100
returns_b.fillna(0)
# Function to scale stock prices based on their initial starting price
# The objective of this function is to set all prices to start at a value of 1 
def price_scaling(raw_prices_df):
    scaled_prices_df = raw_prices_df.copy()
    for i in raw_prices_df.columns[0:]:
          scaled_prices_df[i] = raw_prices_df[i]/raw_prices_df[i][0]
    return scaled_prices_df
# Scale stock prices using the "price_scaling" function that we defined earlier (make all stock values start at 1)
portfolio_a = data_a.copy()
scaled_a = price_scaling(portfolio_a)

portfolio_b = data_b.copy()
scaled_b = price_scaling(portfolio_b)
# Use enumerate() method to obtain the stock names along with a counter "i" (0, 1, 2, 3,..etc.)
# This counter "i" will be used as an index to access elements in the "weights" list
initial_investment = 100000
for i, stock in enumerate(scaled_a.columns[0:]):
    portfolio_a[stock] = weights_a[i] * scaled_a[stock]  * initial_investment
portfolio_a.round(1)
initial_investment = 100000
for i, stock in enumerate(scaled_a.columns[0:]):
    portfolio_b[stock] = weights_b[i] * scaled_b[stock]  * initial_investment
portfolio_b.round(1)
def asset_allocation(df, weights, initial_investment):
    portfolio_a = df.copy()

    # Scale stock prices using the "price_scaling" function that we defined earlier (Make them all start at 1)
    scaled_a = price_scaling(df)
  
    for i, stock in enumerate(scaled_a.columns[0:]):
        portfolio_a[stock] = scaled_a[stock] * weights[i] * initial_investment

    # Sum up all values and place the result in a new column titled "portfolio value [$]" 
    # Note that we excluded the date column from this calculation
    portfolio_a['Portfolio Value [$]'] = portfolio_a[portfolio_a != 'Date'].sum(axis = 1, numeric_only = True)
    
            
    # Calculate the portfolio percentage daily return and replace NaNs with zeros
    portfolio_a['Portfolio Daily Return [%]'] = portfolio_a['Portfolio Value [$]'].pct_change(1) * 100 
    portfolio_a.replace(np.nan, 0, inplace = True)
    
    
    return portfolio_a
# Lets test out the asset allocation function

portfolio_b = asset_allocation(data_b, weights_b, 100000)
portfolio_b.round(2)
portfolio_a = asset_allocation(data_a, weights_a, 100000)
portfolio_a.round(2)

from matplotlib.ticker import FuncFormatter

# Function to format y-axis labels as 'k' for thousands
def thousand_formatter(x, pos):
    return f'{int(x)}k' if x >= 1 else str(int(x))

# Assuming `portfolio_a` and `portfolio_b` have daily returns data
# Calculate daily returns for each portfolio
daily_returns_a = portfolio_a['Portfolio Value [$]'].pct_change().dropna()
daily_returns_b = portfolio_b['Portfolio Value [$]'].pct_change().dropna()

# Annualized Return
annual_return_a = (1 + daily_returns_a.mean())**252 - 1
annual_return_b = (1 + daily_returns_b.mean())**252 - 1

# Annualized Volatility (Standard Deviation of Daily Returns)
annual_volatility_a = daily_returns_a.std() * np.sqrt(252)
annual_volatility_b = daily_returns_b.std() * np.sqrt(252)

# Assuming a daily risk-free rate derived from an annual rate of 3%
risk_free_rate_daily = 0.03 / 252

# Sharpe Ratio for each portfolio
excess_return_a = daily_returns_a - risk_free_rate_daily
excess_return_b = daily_returns_b - risk_free_rate_daily
sharpe_ratio_a = (excess_return_a.mean() / daily_returns_a.std()) * np.sqrt(252)
sharpe_ratio_b = (excess_return_b.mean() / daily_returns_b.std()) * np.sqrt(252)

# Print results
print(f"Portfolio A - Annual Return: {annual_return_a:.2%}, Volatility: {annual_volatility_a:.2%}, Sharpe Ratio: {sharpe_ratio_a:.2f}")
print(f"Portfolio B - Annual Return: {annual_return_b:.2%}, Volatility: {annual_volatility_b:.2%}, Sharpe Ratio: {sharpe_ratio_b:.2f}")

# Plot cumulative returns for both portfolios (actual values in thousands)
plt.figure(figsize=(12, 6))
# Divide by 1000 to plot values in thousands
plt.plot(portfolio_a['Portfolio Value [$]'] / 1000, label="Portfolio A (Local Equities)", color="blue")  
plt.plot(portfolio_b['Portfolio Value [$]'] / 1000, label="Portfolio B (Diversified)", color="green")  
plt.title("Cumulative Returns: Portfolio A vs. Portfolio B")
plt.xlabel("Date")
plt.ylabel("Portfolio Value [$] ")  
plt.xticks(rotation=45)

# Set y-axis formatter
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_formatter))

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Run a Monte Carlo Simulation to get the optimal portfolio
# Define the number of simulations and the risk-free rate
num_portfolios = 250000
risk_free_rate = 0.03  # Assuming 3% risk-free rate
results = np.zeros((4, num_portfolios))
weights_record = []

# Monte Carlo Simulation
np.random.seed(75)  # For reproducibility
for i in range(num_portfolios):
    # Generate random weights
    weights = np.random.random(len(tickers_b))
    weights /= np.sum(weights)
    
    # Calculate portfolio return and volatility
    portfolio_return = np.sum(returns_b.mean() * weights * 252)  # Annualized return
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(returns_b.cov() * 252, weights)))  # Annualized volatility
    
    # Calculate Sharpe Ratio with the given risk-free rate
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
    
    # Store results
    results[0, i] = portfolio_return
    results[1, i] = portfolio_stddev
    results[2, i] = sharpe_ratio
    results[3, i] = i  # Just storing the index
    weights_record.append(weights)  # Record the weights

# Convert results to a DataFrame
results_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Sharpe Ratio", "Index"])

# Find the portfolio with the maximum Sharpe Ratio
max_sharpe_index = results_df["Sharpe Ratio"].idxmax()
max_sharpe_portfolio = results_df.iloc[max_sharpe_index]
optimal_weights = weights_record[max_sharpe_index]

# Find the portfolio with the minimum volatility
min_vol_index = results_df["Volatility"].idxmin()
min_vol_portfolio = results_df.iloc[min_vol_index]

# Plot the efficient frontier
plt.figure(figsize=(12, 8))
plt.scatter(results_df["Volatility"], results_df["Return"], c=results_df["Sharpe Ratio"], cmap="viridis", marker="o", s=10, alpha=0.5)
plt.colorbar(label="Sharpe Ratio")
plt.scatter(max_sharpe_portfolio["Volatility"], max_sharpe_portfolio["Return"], color="r", marker="*", s=200, label="Max Sharpe Ratio")
plt.scatter(min_vol_portfolio["Volatility"], min_vol_portfolio["Return"], color="b", marker="*", s=200, label="Min Volatility")
plt.title("Efficient Frontier: Monte Carlo Simulation")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Return")
plt.legend()
plt.show()

# Display the optimal portfolios with weights in percentage
print("Portfolio with Maximum Sharpe Ratio")
print(f"Return: {max_sharpe_portfolio['Return']:.2f}, Volatility: {max_sharpe_portfolio['Volatility']:.2f}, Sharpe Ratio: {max_sharpe_portfolio['Sharpe Ratio']:.2f}")
print("Optimal Weights for Maximum Sharpe Ratio Portfolio:")
for ticker, weight in zip(tickers_b, optimal_weights):
    print(f"{ticker}: {weight * 100:.2f}%")

print("\nPortfolio with Minimum Volatility")
print(f"Return: {min_vol_portfolio['Return']:.2f}, Volatility: {min_vol_portfolio['Volatility']:.2f}")
tickers_c = tickers_b.copy() # Use the tickers for optimal portfolio
weights_c = optimal_weights.copy() # Use the optimal weights for the portfolio
portfolio_c = data_b.copy()
scaled_c = price_scaling(portfolio_c)
portfolio_c = asset_allocation(data_b, weights_c, 100000)

# Function to format y-axis labels as 'k' for thousands
def thousand_formatter(x, pos):
    return f'{int(x)}k' if x >= 1 else str(int(x))

# Plot cumulative returns for both portfolios (actual values in thousands)
plt.figure(figsize=(12, 6))
# Divide by 1000 to plot values in thousands
plt.plot(portfolio_a['Portfolio Value [$]'] / 1000, label="Portfolio A (Local Equities)", color="blue")  
plt.plot(portfolio_b['Portfolio Value [$]'] / 1000, label="Portfolio B (Diversified)", color="green")  
plt.plot(portfolio_c['Portfolio Value [$]'] / 1000, label="Portfolio C (Optimal Portfolio)", color="cyan")  
plt.title("Cumulative Returns: Portfolio A vs. Portfolio B vs. Optimal Portfolio")
plt.xlabel("Date")
plt.ylabel("Portfolio Value [$] ")  
plt.xticks(rotation=45)

# Set y-axis formatter
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_formatter))

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()