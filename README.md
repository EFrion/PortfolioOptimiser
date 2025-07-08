![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)


# Portfolio optimiser

A portfolio optimisation framework implementing modern portfolio theory with practical constraints and multiple optimisation strategies. Built for research and educational purposes in quantitative finance.

## Table of Contents
- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Output](#output)
- [Technical Implementation](#technical-implementation)
- [In Development](#in-development)
- [Academic References](#academic-references)
- [Contact](#contact)
- [Disclaimer](#disclaimer)
- [License](#license)

## Overview

This Python application optimises investment portfolios using historical stock data, supporting both static and dynamic covariance models with practical constraints such as position limits and sector diversification requirements.

**Key Features:**
- Multiple optimisation strategies (Minimum Variance, Maximum Sharpe, Sortino, MVSK)
- Dynamic covariance estimation with rolling windows
- Monte Carlo simulation for portfolio space exploration
- Multi-currency support with automatic FX conversion
- Sector and individual position constraints
- Comprehensive visualisation including efficient frontiers and correlation heatmap

## Mathematical Foundation

The optimiser implements several portfolio optimisation models:

- **Mean-Variance optimisation**: Classic Markowitz framework
- **Maximum Sharpe Ratio**: Tangency portfolio on the efficient frontier
- **Sortino Ratio**: Downside risk-adjusted optimisation
- **MVSK**: Mean-Variance-Skewness-Kurtosis utility maximisation

All optimisation models incorporate two diversification constraints: individual positions are capped at 5% of total portfolio value, and sector allocations are limited to 25% maximum. The efficient frontier is computed only when these constraints can be satisfied simultaneously. Portfolios with insufficient assets to meet minimum diversification requirements will not generate frontier plots. This typically requires a minimum of 20 stocks if individual position limits are set to 5%.

## Quick Start

### Prerequisites
Tested with Python **3.12+**

```bash
pip install pandas numpy scipy matplotlib seaborn adjustText
```

### Data Structure
```
./
├── portfolio_optimiser_v1.0.py
├── config.json
└── Stocks_Data/
    ├── USD/
    │   ├── AMAT.csv
    │   └── AMT.csv
    ├── EUR/
    │   ├── BNP.csv
    │   └── BMW.csv
    └── usdeur.csv
```

### CSV Format Requirements

Data should have a formatting similar to CSV files from:
- **US Stocks**: nasdaq.com historical data format - [Nasdaq Historical Data](https://www.nasdaq.com/market-activity/stocks)
- **EUR Stocks**: fr.investing.com historical data format - [Investing Historical Data](https://fr.investing.com/equities/)
- **FX rates**: EUR/USD exchange rates from fr.investing.com - [Investing.com EUR/USD](https://fr.investing.com/currencies/usd-eur-historical-data)

Expected columns:

- **Date column**: Consistent date format (MM/DD/YYYY for USD, DD/MM/YYYY for EUR). More specifically, the code uses format='%d/%m/%Y' for EUR if decimal_separator is a comma (,) and relies on errors='coerce' otherwise.
- **Price column**: 'Close/Last' (USD), 'Dernier' (EUR)
- **FX data**: EUR/USD exchange rates with 'Date' and 'Dernier' columns

Example:
| Field | USD Stocks | EUR Stocks | FX Data |
|-------|------------|------------|---------|
| Date  | `Date` (MM/DD/YYYY) | `Date` (DD/MM/YYYY) | `Date` |
| Price | `Close/Last` | `Dernier` | `Dernier` |
| Notes | Use U.S. locale (.) | Use French locale (,) | Use French locale (,) |

## Running the Application

There are two primary ways to run the Portfolio Optimiser:

### 1. Running the Executable
For a quick start without needing a Python environment setup, you can download the pre-compiled executable.

- Go to [Releases page](https://https://github.com/EFrion/PortfolioOptimiser/releases/tag/v1.0.0) and download the executable
- Ensure your Stocks_Data folder and config.json file are placed in the same directory as the extracted executable
- Run the executable

### 2. Running from Source (Python Script)

If you prefer to run the Python script directly, or wish to modify the code, follow these steps:

- Clone the repository or download the source code
- Ensure you have met the Prerequisites (Python and required libraries)
- Prepare your data as described in Data Structure and CSV Format Requirements
- Go to the project's root directory in your terminal and execute the script:
```bash
python portfolio_optimiser_v1.0.py
```


## Configuration

The `config.json` file controls all optimisation parameters:

```json
{
  "feature_toggles": {
    "RUN_STATIC_PORTFOLIO": true,
    "RUN_DYNAMIC_PORTFOLIO": true,
    "RUN_MONTE_CARLO_SIMULATION": true,
    "RUN_MVO_OPTIMISATION": true,
    "RUN_SHARPE_OPTIMISATION": true,
    "RUN_SORTINO_OPTIMISATION": true,
    "RUN_MVSK_OPTIMISATION": true
  },
  "data_paths": {
    "STOCK_ROOT_FOLDER": "Stocks_Data",
    "EXCHANGE_RATE_FILE_NAME": "usdeur.csv"
  },
  "portfolio_parameters": {
    "RISK_FREE_RATE": 0.02,
    "LAMBDA_S": 0.1,
    "LAMBDA_K": 0.1,
    "NUM_FRONTIER_POINTS": 100,
    "CONFIGURED_MAX_STOCK_WEIGHT": 0.05,
    "CONFIGURED_MAX_SECTOR_WEIGHT": 0.25,
    "ROLLING_WINDOW_DAYS": 252
  },
  "output_settings": {
    "OUTPUT_DIR": "portfolio_results",
    "OUTPUT_FILENAME": "portfolio_optimisation_results.csv"
  },
  "stock_sectors": {
    "AMAT": "Technology",
    "BNP": "Financials"
  }
}
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `RISK_FREE_RATE` | Annualised risk-free rate for Sharpe/Sortino calculations | 0.02 |
| `LAMBDA_S` | Skewness preference coefficient (positively rewards positive skew) | 0.1 |
| `LAMBDA_K` | Kurtosis penalty coefficient (positively penalises fat tails) | 0.1 |
| `ROLLING_WINDOW_DAYS` | Days for dynamic covariance estimation | 252 |
| `CONFIGURED_MAX_STOCK_WEIGHT` | Maximum individual position size | 0.05 |
| `CONFIGURED_MAX_SECTOR_WEIGHT` | Maximum sector concentration | 0.25 |

## Output

The code generates:

1. **portfolio_optimisation_results.csv**: Detailed portfolio metrics and weights
2. **optimised_portfolios.png**: Efficient frontier with optimal portfolios
3. **stocks_heatmap.png**: Asset correlation matrix visualisation

>**Note**: Output files are saved to the directory specified in `config.json` under `"output_settings" → "OUTPUT_DIR"`. Default is `"portfolio_results/"`.

## Technical Implementation

- **Optimisation**: Uses scipy.optimize with the Sequential Least Squares Programming (SLSQP) method for constrained non-linear optimisation
- **Covariance Estimation**: Supports both sample covariance and rolling window estimation using Pandas
- **Constraint Handling**: Implements linear equality and inequality constraints for total weight, individual position limits, and sector diversification
- **Performance Metrics**: Sharpe ratio, Sortino ratio, skewness, kurtosis calculations
- **Data Preprocessing**: Robust handling of missing data, date alignment, and currency conversion using pandas for efficient time-series operations

## In development

- [ ] **Backtesting Module**: Historical performance evaluation
- [ ] **Bayesian Portfolio Optimisation**: Treats expected returns and volatility as distributions for more robust portfolios accounting for estimation risk and uncertainty in input parameters (e.g. Black-Litterman model)

## Academic References

This implementation draws from:
- Markowitz, H. (1952). "Portfolio Selection"
- Sortino, F. & Price, L. (1994). "Performance Measurement in a Downside Risk Framework"
- Jondeau, E. & Rockinger, M. (2006). "Optimal Portfolio Allocation Under Higher Moments"
- Black, F. & Litterman, R. (1992). "Global Portfolio Optimization."

## Contact

For questions, suggestions, or feedback, please open an issue or reach out via GitHub.

> I built this as part of my exploration of portfolio theory in Python while applying for quant research roles.


## Disclaimer

This tool is designed for research and educational purposes in quantitative finance. It does not constitute financial advice. All optimisation results should be validated independently before any investment decisions.

## License

MIT License - See LICENSE file for details.

---

*Built with Python for quantitative finance research and education.*
