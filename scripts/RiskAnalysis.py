import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

# Generate simulated historical returns for 5 assets
np.random.seed(42)  # For reproducibility

assets = ['Stock A', 'Stock B', 'Bond A', 'Bond B', 'Real Estate']
returns = np.random.normal(loc=0.001, scale=0.02, size=(252, len(assets)))  # Simulated daily returns for 1 year

# Create DataFrame
portfolio_returns = pd.DataFrame(returns, columns=assets)

# Number of simulation runs
num_simulations = 10000
num_assets = len(assets)

# Simulated portfolio weights
weights = np.random.dirichlet(np.ones(num_assets), size=num_simulations)

# Expected returns and covariance matrix
mean_daily_returns = portfolio_returns.mean()
cov_matrix = portfolio_returns.cov()

# Portfolio simulations
port_returns = []
port_volatility = []
sharpe_ratios = []

for w in weights:
    annualized_return = np.sum(mean_daily_returns * w) * 252  # Assuming 252 trading days
    annualized_volatility = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility

    port_returns.append(annualized_return)
    port_volatility.append(annualized_volatility)
    sharpe_ratios.append(sharpe_ratio)

# Create a DataFrame for the results
simulation_results = pd.DataFrame({
    'Return': port_returns,
    'Volatility': port_volatility,
    'Sharpe Ratio': sharpe_ratios
})

# Identify the portfolio with the maximum Sharpe Ratio
max_sharpe_idx = simulation_results['Sharpe Ratio'].idxmax()
optimal_portfolio = weights[max_sharpe_idx]

print(f"Optimal Portfolio Weights: {dict(zip(assets, optimal_portfolio))}")

# Efficient Frontier Plot
plt.figure(figsize=(12, 8))
plt.scatter(simulation_results['Volatility'], simulation_results['Return'], c=simulation_results['Sharpe Ratio'], cmap='viridis', alpha=0.8)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(simulation_results.loc[max_sharpe_idx, 'Volatility'], simulation_results.loc[max_sharpe_idx, 'Return'], color='red', marker='*', s=200, label='Maximum Sharpe Ratio')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.legend()
plt.show()

# Save simulation results to a CSV
simulation_results.to_csv('portfolio_simulation_results.csv', index=False)

# Save optimal portfolio weights
optimal_weights_df = pd.DataFrame({
    'Asset': assets,
    'Weight': optimal_portfolio
})
optimal_weights_df.to_csv('optimal_portfolio_weights.csv', index=False)
