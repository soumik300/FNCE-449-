import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Define ETF tickers
etf_tickers = ["SPY", "EWC", "VGK", "MCHI", "EWJ", "EWY", "INDA"]

# Define time period
start_date = "2018-01-01"
end_date = "2024-10-20"

# Download adjusted close price data for each ETF
etf_data = yf.download(etf_tickers, start=start_date, end=end_date)["Adj Close"]

# Calculate daily returns for each ETF
etf_returns = etf_data.pct_change().dropna()

# Calculate cumulative returns for each ETF and convert to percentages
cumulative_returns = (1 + etf_returns).cumprod() - 1  # Subtract 1 for percentage change

# Plot cumulative returns 
plt.figure(figsize=(14, 8))

for ticker in etf_tickers:
    plt.plot(cumulative_returns[ticker] * 100, label=ticker)  # Multiply by 100 for %

# Plot the chart 
plt.title("Cumulative Returns of Selected ETFs (2018-2024)", fontsize=18, fontweight="bold", color="navy")
plt.xlabel("Date", fontsize=14, fontweight="bold", color="gray")
plt.ylabel("Cumulative Return (%)", fontsize=14, fontweight="bold", color="gray")
plt.legend(title="ETFs", title_fontsize=12, fontsize=10)
plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_color("lightgray")
plt.gca().spines["bottom"].set_color("lightgray")
plt.gca().xaxis.label.set_color("gray")
plt.gca().yaxis.label.set_color("gray")
plt.tick_params(axis="x", colors="gray")
plt.tick_params(axis="y", colors="gray")

plt.tight_layout()
plt.show()

