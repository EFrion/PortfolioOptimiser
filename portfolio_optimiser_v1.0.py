import pandas as pd
import numpy as np
import os
import csv
import sys
import json
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
from scipy.optimize import minimize

def load_config(config_file_path: str = 'config.json') -> dict:
    """
    Loads configuration settings from a JSON file.

    Args:
        config_file_path (str): The path to the configuration JSON file.

    Returns:
        dict: A dictionary containing the loaded configuration settings.
    """
    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded successfully from {config_file_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_file_path}")
        print("Please ensure 'config.json' is in the same directory as the script.")
        exit(1) # Exit the script if config file is missing
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {config_file_path}. Please check file format.")
        print(f"Details: {e}")
        exit(1) # Exit the script if JSON is malformed
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        exit(1)
        
# Function to load exchange rate data
def load_exchange_rate_data(exchange_rate_file, column_mapping=None, decimal_separator='.'):
    """
    Loads historical EUR/USD exchange rate data.
    Assumes the file has 'Date' and 'Close' (or similar) columns.
    """
    if column_mapping is None:
        column_mapping = {
            'Date': 'Date',
            'Close': 'Close' # Default mapping for exchange rate close price
        }

    try:
        eur_usd_df = pd.read_csv(exchange_rate_file, decimal=decimal_separator)
        
        # Apply column mapping
        eur_usd_df = eur_usd_df.rename(columns={v: k for k, v in column_mapping.items()})

        # Convert 'Date' column to datetime objects, assuming European format if decimal_separator is ','
        if decimal_separator == ',':
            eur_usd_df['Date'] = pd.to_datetime(eur_usd_df['Date'], format='%d/%m/%Y', errors='coerce')
        else:
            eur_usd_df['Date'] = pd.to_datetime(eur_usd_df['Date'], errors='coerce') # Coerce errors will turn invalid dates into NaT
        
        eur_usd_df = eur_usd_df.dropna(subset=['Date']) # Drop rows where date conversion failed

        eur_usd_df = eur_usd_df.sort_values(by='Date')
        
        # Rename 'Close' to 'EURUSD_Rate' for clarity during merge
        if 'Close' not in eur_usd_df.columns:
            raise KeyError(f"Mapped 'Close' column not found in exchange rate file '{exchange_rate_file}'. Check column_mapping.")

        eur_usd_df = eur_usd_df[['Date', 'Close']].rename(columns={'Close': 'EURUSD_Rate'})
        print(f"Successfully loaded exchange rate data from {exchange_rate_file}")
        return eur_usd_df
    except FileNotFoundError:
        print(f"Warning: Exchange rate file '{exchange_rate_file}' not found. Cannot convert EUR stocks to USD.")
        return None
    except KeyError as e:
        print(f"Error: Exchange rate file '{exchange_rate_file}' must contain the expected columns. Details: {e}")
        return None
    except Exception as e:
        print(f"An error occurred loading exchange rate data: {e}")
        return None


def process_single_stock_data(file_path, ticker, is_eur_stock=False, eur_usd_df=None, column_mapping=None, decimal_separator='.'):
    """
    Loads stock price data from a CSV, cleans it, and performs currency conversion if needed.
    Returns the DataFrame with 'Date' and 'Close/Last' (adjusted for currency if applicable).

    Args:
        file_path (str): The full path to the input CSV file.
        ticker (str): The ticker symbol for the stock.
        is_eur_stock (bool): True if the stock is denominated in EUR and needs conversion.
        eur_usd_df (pd.DataFrame, optional): DataFrame containing EUR/USD exchange rates.
        column_mapping (dict, optional): A dictionary to map standard column names
                                        (e.g., 'Date', 'Close/Last') to actual names
                                        in the CSV file.
        decimal_separator (str): The decimal separator used in the CSV ('.' or ',').

    Returns:
        pd.DataFrame: DataFrame with 'Date' and 'Close/Last' (price) columns, or None if error.
        float: Annualised standard deviation.
        float: Compound Annual Growth Rate (CAGR).
        float: Annualised average return.
        str: Ticker symbol.
        float: The last closing price of the stock.
        pd.Timestamp: The earliest valid date for this stock.
    """
    if column_mapping is None:
        column_mapping = {
            'Date': 'Date',
            'Close/Last': 'Close/Last',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low'
        }

    try:
        df = pd.read_csv(file_path, decimal=decimal_separator)
        df = df.rename(columns={v: k for k, v in column_mapping.items()})

        for col in ['Close/Last', 'Open', 'High', 'Low']:
            if col in df.columns:
                df[col] = df[col].astype(str).replace({r'\$': '', '€': ''}, regex=True).astype(float)
            else:
                if col in column_mapping:
                    raise KeyError(f"Missing expected column '{column_mapping[col]}' (mapped to '{col}') in file: {file_path}")

        if is_eur_stock and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        else:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        df = df.dropna(subset=['Date'])
        df = df.sort_values(by='Date')

        if is_eur_stock and eur_usd_df is not None:
            original_rows = len(df)
            df = pd.merge(df, eur_usd_df, on='Date', how='inner')
            if len(df) < original_rows:
                print(f"Warning: Missing exchange rate data for some dates for {ticker}. Filtered to {len(df)} common dates.")
            if df.empty:
                print(f"Error: No overlapping dates between {ticker} and EUR/USD exchange rates. Cannot process.")
                return None, None, None, None, None, None, None # Added None for earliest_date
            df['Close/Last'] = df['Close/Last'] * df['EURUSD_Rate']
            print(f"Converted {ticker} (EUR) prices to USD.")

        # Calculate metrics for individual stock (for printing, not for backtest loop directly)
        daily_returns = df['Close/Last'].pct_change().dropna()
        if len(daily_returns) < 2:
            print(f"Warning: Not enough valid data points for {ticker} after cleaning for volatility/return calculations.")
            return None, None, None, None, None, None, None # Added None for earliest_date

        annualised_std_dev = daily_returns.std() * np.sqrt(252)
        annualised_avg_return = daily_returns.mean() * 252

        beginning_value = df['Close/Last'].iloc[0]
        ending_value = df['Close/Last'].iloc[-1]
        beginning_date = df['Date'].iloc[0]
        ending_date = df['Date'].iloc[-1]
        number_of_years = (ending_date - beginning_date).days / 365.25
        cagr = (ending_value / beginning_value)**(1 / number_of_years) - 1 if number_of_years > 0 else np.nan
        
        last_close_price = df['Close/Last'].iloc[-1] if not df['Close/Last'].empty else np.nan
        
        earliest_valid_date = df['Date'].min() # Get the earliest date for this specific stock

        return df[['Date', 'Close/Last']].rename(columns={'Close/Last': ticker}), annualised_std_dev, cagr, annualised_avg_return, ticker, last_close_price, earliest_valid_date

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None, None, None, None, None, None
    except KeyError as e:
        print(f"Error: Missing expected column in '{file_path}'. Please check column names and mapping. Details: {e}")
        return None, None, None, None, None, None, None
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
        return None, None, None, None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred processing '{file_path}': {e}")
        return None, None, None, None, None, None, None

# --- Portfolio optimisation functions ---

def portfolio_volatility(weights, annualised_covariance_matrix):
    """
    Objective function to minimise: Portfolio Standard Deviation (Volatility).
    """
    return np.sqrt(np.dot(weights.T, np.dot(annualised_covariance_matrix, weights)))

def portfolio_return(weights, annual_returns):
    """
    Calculates the portfolio's expected return.
    """
    return np.sum(weights * annual_returns)

def calculate_portfolio_metrics(weights, annual_returns, annualised_covariance_matrix):
    """
    Calculates portfolio return and standard deviation for a given set of weights.

    Args:
        weights (np.array): Array of weights for each asset.
        annual_returns (np.array): Array of annualised returns for each asset.
        annualised_covariance_matrix (np.array): Annualised covariance matrix of asset returns.

    Returns:
        tuple: (portfolio_return, portfolio_std_dev)
    """
    portfolio_ret = np.sum(weights * annual_returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(annualised_covariance_matrix, weights)))
    return portfolio_ret, portfolio_std

def negative_sharpe_ratio(weights, annual_returns, annualised_covariance_matrix, risk_free_rate):
    """
    Objective function to minimise for finding the Tangency Portfolio:
    Negative Sharpe Ratio.
    """
    p_return = portfolio_return(weights, annual_returns)
    p_volatility = portfolio_volatility(weights, annualised_covariance_matrix)
    if p_volatility == 0:
        return -np.inf # Avoid division by zero, return negative infinity for extremely good Sharpe
    return -(p_return - risk_free_rate) / p_volatility

def downside_deviation(weights, daily_returns_df_slice, risk_free_rate):
    """
    Calculates the annualised downside deviation for a portfolio.
    Only considers returns below the Minimum Acceptable Return (MAR), which is the risk-free rate.
    
    Args:
        weights (np.array): Array of weights for each asset.
        daily_returns_df_slice (pd.DataFrame): Daily returns for the lookback period.
        risk_free_rate (float): Annualised risk-free rate.
    """
    # Calculate portfolio daily returns for the slice
    portfolio_daily_returns = daily_returns_df_slice.dot(weights)
    
    # Calculate daily MAR
    daily_mar = (1 + risk_free_rate)**(1/252) - 1 # Convert annualised risk-free rate to daily

    # Filter for returns below the MAR
    downside_returns = portfolio_daily_returns[portfolio_daily_returns < daily_mar]
    
    if downside_returns.empty:
        return 0.0 # No downside returns, so downside deviation is 0

    # Calculate downside deviation (standard deviation of downside returns)
    downside_std = np.sqrt(np.mean((downside_returns - daily_mar)**2))
    
    # Annualize downside deviation
    annualised_downside_std = downside_std * np.sqrt(252)
    return annualised_downside_std

def negative_sortino_ratio(weights, annual_returns, daily_returns_df_slice, annualised_covariance_matrix, risk_free_rate):
    """
    Objective function to minimise for finding the Sortino Ratio optimised portfolio:
    Negative Sortino Ratio.
    """
    p_return = portfolio_return(weights, annual_returns)
    # Pass the relevant slice of daily returns to downside_deviation
    p_downside_dev = downside_deviation(weights, daily_returns_df_slice, risk_free_rate)
    
    if p_downside_dev == 0:
        return -np.inf # Avoid division by zero, return negative infinity for extremely good Sortino
    return -(p_return - risk_free_rate) / p_downside_dev

def portfolio_skewness(weights, daily_returns_df_slice):
    """
    Calculates the skewness for a portfolio's daily returns.
    """
    portfolio_daily_returns = daily_returns_df_slice.dot(weights)
    return portfolio_daily_returns.skew()

def portfolio_kurtosis(weights, daily_returns_df_slice):
    """
    Calculates the kurtosis for a portfolio's daily returns.
    """
    portfolio_daily_returns = daily_returns_df_slice.dot(weights)
    return portfolio_daily_returns.kurtosis()

def negative_mvsk_utility(weights, annual_returns, daily_returns_df_slice, annualised_covariance_matrix, risk_free_rate, lambda_s, lambda_k):
    """
    Objective function to minimise for Mean-Variance-Skewness-Kurtosis (MVSK) optimisation.
    Maximizes a utility function that considers mean, variance, skewness, and kurtosis.
    
    Args:
        weights (np.array): Portfolio weights.
        annual_returns (np.array): Annualised expected returns of assets.
        daily_returns_df_slice (pd.DataFrame): Daily returns for the lookback period.
        annualised_covariance_matrix (np.array): Annualised covariance matrix.
        risk_free_rate (float): Risk-free rate.
        lambda_s (float): Coefficient for skewness (positive to prefer higher skewness).
        lambda_k (float): Coefficient for kurtosis (positive to prefer lower kurtosis).
    """
    p_return = portfolio_return(weights, annual_returns)
    p_volatility = portfolio_volatility(weights, annualised_covariance_matrix)
    p_skewness = portfolio_skewness(weights, daily_returns_df_slice)
    p_kurtosis = portfolio_kurtosis(weights, daily_returns_df_slice)

    if p_volatility == 0:
        return np.inf 

    # A common MVSK objective for maximization (we minimise the negative):
    utility = (p_return - risk_free_rate) / p_volatility + lambda_s * p_skewness - lambda_k * p_kurtosis
    
    return -utility

def find_latest_nan_date(df):
    """
    Finds the latest date (index) in a DataFrame where any NaN appears across columns,
    and identifies which tickers have NaNs on that specific date.
    
    Args:
        df (pd.DataFrame): The DataFrame to check for NaNs. Assumes datetime index.
        
    Returns:
        tuple: (pd.Timestamp or None, list of str): 
               The latest date with a NaN, and a list of tickers with NaNs on that date.
               Returns (None, []) if no NaNs are found.
    """
    # Find rows that contain any NaN
    nan_rows = df[df.isnull().any(axis=1)]
    
    if nan_rows.empty:
        return None, [] # No NaNs found
    else:
        # Get the maximum date (latest date) among rows with NaNs
        latest_date_with_nan = nan_rows.index.max()
        
        # Filter the original DataFrame for this specific date
        nans_on_latest_date = df.loc[latest_date_with_nan]
        
        # Identify which columns (tickers) have NaN values on this specific date
        tickers_with_nans = nans_on_latest_date[nans_on_latest_date.isnull()].index.tolist()
        
        return latest_date_with_nan, tickers_with_nans

def write_portfolio_weights_to_csv(
    filename: str,
    portfolio_type: str,
    optimisation_type: str,
    metrics: dict,
    weights: list,
    portfolio_tickers: list
):
    """
    Writes portfolio optimisation results (metrics and weights) to a CSV file.

    Args:
        filename (str): The name of the CSV file to write to.
        portfolio_type (str): Describes the portfolio (e.g., "Static Covariance", "Dynamic Covariance").
        optimisation_type (str): The type of optimisation (e.g., "Minimum Variance", "Tangency", "Sortino", "MVSK").
        metrics (dict): A dictionary of key metrics (Return, Volatility).
        weights (list): A list of numerical weights for each ticker.
        portfolio_tickers (list): A list of ticker symbols corresponding to the weights.
    """
    
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Define all possible metric headers that might appear across any portfolio type
    all_possible_metric_keys = [
        "Sharpe Ratio",
        "Sortino Ratio",
        "Skewness",
        "Kurtosis"
    ]
    
    # Prepare header for weights (e.g., 'Weight_AAPL', 'Weight_MSFT')
    weight_headers = [f"{ticker}" for ticker in portfolio_tickers]

    # Define the base headers that will always be present
    base_headers = ["Portfolio Type", "Optimisation Type", "Return", "Volatility"]

    # Combine all headers into a single, fixed list
    all_headers = base_headers + all_possible_metric_keys + weight_headers


    # Initialize with all keys from `all_headers` to ensure consistency
    row_data = {header: '' for header in all_headers}
    
    # Prepare the data row
    row_data = {
        "Portfolio Type": portfolio_type,
        "Optimisation Type": optimisation_type,
        "Return": f"{metrics.get('Return', 0.0):.2%}",
        "Volatility": f"{metrics.get('Volatility', 0.0):.2%}"
    }

    # Add dynamic metrics to the row data with conditional formatting
    for header_key in all_possible_metric_keys:
        if header_key in metrics:
            value = metrics[header_key]
            if header_key in ["Sharpe Ratio", "Sortino Ratio", "Skewness", "Kurtosis"]:
                row_data[header_key] = f"{value:.4f}"
            else:
                # Fallback for any other unexpected metric, just format as general float
                row_data[header_key] = f"{value}"
        else:
            # Ensure the key exists in row_data, even if value is empty
            row_data[header_key] = ''


    # Add weights to the row data
    for i, ticker in enumerate(portfolio_tickers):
        if i < len(weights): # Safety check to prevent index out of bounds
            row_data[f"{ticker}"] = f"{weights[i]:.2%}"
        else:
            # If for some reason weights list is shorter than tickers, fill with empty
            row_data[f"{ticker}"] = ''

    try:
        # Check if file exists to determine if header needs to be written
        file_exists_and_not_empty = os.path.exists(filepath) and os.path.getsize(filepath) > 0

        with open(filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_headers)

            if not file_exists_and_not_empty: # Write header only if file is new or being overwritten
                writer.writeheader()
            writer.writerow(row_data)
        #print(f"Successfully wrote {optimisation_type} ({portfolio_type}) data to {filepath}")
    except IOError as e:
        print(f"Error writing to CSV file {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def clear_portfolio_results_csv(filename: str):
    """
    Clears (overwrites) the specified CSV file, effectively preparing it for new data.
    Called once at the very beginning of script execution.

    Args:
        filename (str): The name of the CSV file to clear.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, 'w', newline='') as csvfile:
            # Just opening in 'w' mode and immediately closing clears the file.
            pass
        print(f"Cleared existing data in {filepath} for a new run.")
    except IOError as e:
        print(f"Error clearing CSV file {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while clearing the CSV: {e}")


# --- Main script execution ---
if __name__ == "__main__":
    # --- Load Configuration ---
    config = load_config('config.json')

    # --- FEATURE TOGGLES (Set to True or False to enable/disable features) ---
    RUN_STATIC_PORTFOLIO = config['feature_toggles']['RUN_STATIC_PORTFOLIO']
    RUN_DYNAMIC_PORTFOLIO = config['feature_toggles']['RUN_DYNAMIC_PORTFOLIO']
    RUN_MONTE_CARLO_SIMULATION = config['feature_toggles']['RUN_MONTE_CARLO_SIMULATION']
    RUN_MVO_OPTIMISATION = config['feature_toggles']['RUN_MVO_OPTIMISATION']
    RUN_SHARPE_OPTIMISATION = config['feature_toggles']['RUN_SHARPE_OPTIMISATION']
    RUN_SORTINO_OPTIMISATION = config['feature_toggles']['RUN_SORTINO_OPTIMISATION']
    RUN_MVSK_OPTIMISATION = config['feature_toggles']['RUN_MVSK_OPTIMISATION']
    # -------------------------------------------------------------------------

    # --- Data Paths ---
    STOCK_ROOT_FOLDER = config['data_paths']['STOCK_ROOT_FOLDER']
    USD_FOLDER = os.path.join(STOCK_ROOT_FOLDER, 'USD') # Dynamically construct
    EUR_FOLDER = os.path.join(STOCK_ROOT_FOLDER, 'EUR') # Dynamically construct
    EXCHANGE_RATE_FILE = os.path.join(STOCK_ROOT_FOLDER, config['data_paths']['EXCHANGE_RATE_FILE_NAME'])
    # -------------------------------------------------------------------------

    # --- Portfolio Parameters --- 
    RISK_FREE_RATE = config['portfolio_parameters']['RISK_FREE_RATE'] # Define a risk-free rate for Sharpe Ratio calculation (e.g., U.S. 3-Month Treasury Bill Rate)
    LAMBDA_S = config['portfolio_parameters']['LAMBDA_S'] # A positive lambda_s rewards higher (more positive) skewness.
    LAMBDA_K = config['portfolio_parameters']['LAMBDA_K'] # A positive lambda_k penalizes higher (more positive) kurtosis (fat tails).
    NUM_FRONTIER_POINTS = config['portfolio_parameters']['NUM_FRONTIER_POINTS'] # For efficient frontier. Fine-grain return percentage with more points
    CONFIGURED_MAX_STOCK_WEIGHT = config['portfolio_parameters']['CONFIGURED_MAX_STOCK_WEIGHT'] # Stock constraint
    CONFIGURED_MAX_SECTOR_WEIGHT = config['portfolio_parameters']['CONFIGURED_MAX_SECTOR_WEIGHT'] # Sector constraint
    ROLLING_WINDOW_DAYS = config['portfolio_parameters']['ROLLING_WINDOW_DAYS'] # For rolling covariance (Default: 252, 1 year of trading days)
    # -------------------------------------------------------------------------
    
    # Output Settings
    OUTPUT_DIR = config['output_settings']['OUTPUT_DIR']
    OUTPUT_FILENAME = config['output_settings']['OUTPUT_FILENAME']
    output_filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME) # Dynamically construct
    # -------------------------------------------------------------------------
    
    # Ensure output directory exists before any file operations
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Ensures output file is cleared every time the script starts.
    clear_portfolio_results_csv(OUTPUT_FILENAME)

    # This dictionary maps each stock ticker (in uppercase) to its sector.
    # You MUST populate this accurately for sector constraints to work.
    STOCK_SECTORS = config['stock_sectors']


    # Create main folder and subfolders if they don't exist
    for folder in [USD_FOLDER, EUR_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}.")
    if not os.path.exists(STOCK_ROOT_FOLDER):
         os.makedirs(STOCK_ROOT_FOLDER)
         print(f"Created folder: {STOCK_ROOT_FOLDER}.")

    print("Please ensure your stock CSV files are in the 'Stocks_Data/USD' or 'Stocks_Data/EUR' subfolders.")
    print(f"For EUR conversion, also place '{os.path.basename(EXCHANGE_RATE_FILE)}' (with 'Date' and 'Dernier' columns, European format) in '{STOCK_ROOT_FOLDER}'.")

    # --- Define Column Mappings for different currencies/formats ---
    usd_column_mapping = {
        'Date': 'Date',
        'Close/Last': 'Close/Last',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low'
    }

    # Corrected column names based on user's feedback
    eur_column_mapping = {
        'Date': 'Date',
        'Close/Last': 'Dernier', # Map 'Dernier' from EUR CSV to 'Close/Last' for internal use
        'Open': 'Ouv.',
        'High': ' Plus Haut', # Retain leading space as requested
        'Low': 'Plus Bas'
    }

    exchange_rate_column_mapping = {
        'Date': 'Date',
        'Close': 'Dernier' # The 'Dernier' column for the rate, assuming it's the closing rate
    }

    # --- Load Exchange Rate Data ---
    eur_usd_rates_df = load_exchange_rate_data(EXCHANGE_RATE_FILE,
                                              column_mapping=exchange_rate_column_mapping,
                                              decimal_separator=',')

    # individual_stock_metrics will now only store tickers and their full history metrics for initial printout
    individual_stock_metrics = [] 
    all_stock_prices = pd.DataFrame() # To store all stock prices for merging
    earliest_dates_per_stock = {} # To store the earliest valid date for each stock

    # --- Process USD Stocks ---
    if os.path.exists(USD_FOLDER):
        print(f"\nProcessing stocks in {USD_FOLDER}...")
        for filename in os.listdir(USD_FOLDER):
            if filename.endswith('.csv'):
                file_path = os.path.join(USD_FOLDER, filename)
                ticker = os.path.splitext(filename)[0].upper()
                stock_price_df, annual_std, cagr, annual_avg_ret, _, last_price, earliest_date_for_stock = \
                    process_single_stock_data(file_path, ticker, is_eur_stock=False,
                                            column_mapping=usd_column_mapping, decimal_separator='.')
                if stock_price_df is not None:
                    individual_stock_metrics.append({
                        'ticker': ticker,
                        'annualised_std': annual_std,
                        'cagr': cagr,
                        'annual_return': annual_avg_ret,
                        'last_price': last_price
                    })
                    if all_stock_prices.empty:
                        all_stock_prices = stock_price_df
                    else:
                        all_stock_prices = pd.merge(all_stock_prices, stock_price_df, on='Date', how='outer')
                    earliest_dates_per_stock[ticker] = earliest_date_for_stock
    else:
        print(f"Folder '{USD_FOLDER}' not found. Skipping USD stock processing.")

    # --- Process EUR Stocks ---
    if os.path.exists(EUR_FOLDER):
        print(f"\nProcessing stocks in {EUR_FOLDER}...")
        if eur_usd_rates_df is None:
            print("Cannot process EUR stocks as EUR/USD exchange rate data is not available.")
        else:
            for filename in os.listdir(EUR_FOLDER):
                if filename.endswith('.csv'):
                    file_path = os.path.join(EUR_FOLDER, filename)
                    ticker = os.path.splitext(filename)[0].upper()
                    stock_price_df, annual_std, cagr, annual_avg_ret, _, last_price, earliest_date_for_stock = \
                        process_single_stock_data(file_path, ticker, is_eur_stock=True, eur_usd_df=eur_usd_rates_df.copy(),
                                                column_mapping=eur_column_mapping, decimal_separator=',')
                    if stock_price_df is not None:
                        individual_stock_metrics.append({
                            'ticker': ticker,
                            'annualised_std': annual_std,
                            'cagr': cagr,
                            'annual_return': annual_avg_ret,
                            'last_price': last_price
                        })
                        if all_stock_prices.empty:
                            all_stock_prices = stock_price_df
                        else:
                            all_stock_prices = pd.merge(all_stock_prices, stock_price_df, on='Date', how='outer')
                        earliest_dates_per_stock[ticker] = earliest_date_for_stock
    else:
        print(f"Folder '{EUR_FOLDER}' not found. Skipping EUR stock processing.")
    
    
    # Sort the list of stock metrics alphabetically by ticker
    individual_stock_metrics.sort(key=lambda x: x['ticker'])

    # --- Finalize Price Data and Calculate Daily Returns for all stocks ---
    if all_stock_prices.empty:
        print(f'Error: no stock price data collected. Cannot proceed with portfolio optimisation.')
        exit()

    # Sort all_stock_prices chronologically (.set_index('Date').sort_index()) and alphabetically (.sort_index(axis=1))
    all_stock_prices = all_stock_prices.set_index('Date').sort_index().sort_index(axis=1)    
    
    # --- Determine global common start date and truncate ---
    if earliest_dates_per_stock:
        global_common_start_date = max(earliest_dates_per_stock.values())
        print(f"\nGlobal common start date for all loaded stocks: {global_common_start_date.strftime('%Y-%m-%d')}")
        
        # Truncate all_stock_prices to start from this common date
        all_stock_prices = all_stock_prices[all_stock_prices.index >= global_common_start_date]
        print(f"New all_stock_prices shape after common start date truncation: {all_stock_prices.shape}")



    # Apply ffill() and bfill() to fill any *remaining* internal missing values
    # (after ensuring all stocks actually existed for the period)
    #print("\nDEBUG: Applying ffill() and bfill() to fill any internal missing price data...")
    all_stock_prices_before_fill = all_stock_prices.copy() # Keep a copy before filling for diagnostic if needed
    all_stock_prices = all_stock_prices.ffill().bfill()

    # Drop columns (tickers) that are still entirely NaN after filling (meaning no data at all even after common start date)
    initial_num_columns = all_stock_prices.shape[1]
    all_stock_prices.dropna(axis=1, how='all', inplace=True)
    if all_stock_prices.shape[1] < initial_num_columns:
        # Identify dropped columns by comparing sets of columns
        current_cols_set = set(all_stock_prices.columns)
        dropped_columns_after_fill = [col for col in columns_before_dropna_all_nan if col not in current_cols_set]
        print(f"DEBUG: Dropped entirely NaN columns (tickers with no data after common start date and filling): {dropped_columns_after_fill}")


    # --- DEBUGGING START ---
    #print(f"\nDEBUG: all_stock_prices after common start date truncation, ffill, bfill, and dropping all-NaN columns:")
    #print(f"DEBUG: Shape: {all_stock_prices.shape}")
    #print(f"DEBUG: Is empty: {all_stock_prices.empty}")
    if not all_stock_prices.empty:
        print("\nHead of all_stock_prices:")
        print(all_stock_prices.head())
        print("\nTail of all_stock_prices:")
        print(all_stock_prices.tail())
    else:
        print("DEBUG: all_stock_prices is EMPTY after cleaning.")
        print("Please check your input CSV files for overlapping dates and complete data.")
        exit() # Exit early if no data after cleaning
    # --- DEBUGGING END ---

    # Now calculate daily returns from the aligned price data
    # .dropna() here will remove the first row (NaN from pct_change) and any other rows
    # where a NaN might have persisted (e.g., if a stock truly has no data at the very start/end
    # that ffill/bfill couldn't cover).
    daily_returns = all_stock_prices.pct_change().dropna()

    # --- DEBUGGING START ---
    #print(f"\nDEBUG: daily_returns after pct_change().dropna():")
    #print(f"DEBUG: Shape: {daily_returns.shape}")
    #print(f"DEBUG: Is empty: {daily_returns.empty}")
    if not daily_returns.empty:
        print("\nHead of daily_returns:")
        print(daily_returns.head())
        print("\nTail of daily_returns:")
        print(daily_returns.tail())
    else:
        print("DEBUG: daily_returns is EMPTY. This will prevent further calculations.")
        print("Please ensure your data has at least two trading days with valid prices for all selected stocks.")
        exit() # Exit early if no daily returns
    # --- DEBUGGING END ---


    # Ensure we have enough data after merging and cleaning
    if len(daily_returns.columns) < 2:
        print("Error: Not enough common valid stock data (min 2) across all files after date alignment. Plotting cannot proceed.")
        print("Please check your CSV files for overlapping dates and valid price data.")
        exit()

    portfolio_tickers = sorted(list(daily_returns.columns))
    #print(portfolio_tickers)
    num_assets = len(portfolio_tickers)

    # --- Calculate the annualised mean annual_returns_array based on the common daily_returns times the average number of trading days ---
    annual_returns_array = daily_returns.mean() * 252
    annual_returns_array = annual_returns_array.values # Convert to numpy array for optimisation functions
    
    print("\n--- Individual Stock Metrics (All in USD) ---")
    # Using the metrics from the full individual history for this printout, as it's descriptive.
    # The optimisation will use the aligned annual_returns_array.
    for stock in individual_stock_metrics:
        print(f"Ticker: {stock['ticker']}")
        print(f"  Annualised Return: {stock['annual_return']:.2%}")
        print(f"  Annualised Std Dev: {stock['annualised_std']:.2%}")
        print(f"  CAGR: {stock['cagr']:.2%}")
        print(f"  Last Close Price (USD): {stock['last_price']:.2f}\n")

    
    # --- Optimisation Setup for Constraints (common for all optimisations) ---
    bounds = tuple((0, 1) for asset in range(num_assets)) # Weights between 0 and 1 (no short-selling)

    # Base constraint: sum of weights equals 1
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]

    # Ideally, individual stock weight constraint should be at most 5%.
    # For a small number of assets though, calculate an effective maximum stock weight to ensure feasibility 
    # It should be at least 1.0 / num_assets to allow sum of weights to be 1.0
    effective_max_stock_weight = max(CONFIGURED_MAX_STOCK_WEIGHT, 1.0 / num_assets)
    print(f"Maximum individual stock weight: {effective_max_stock_weight:.2%}")
    if num_assets <= 20 and RUN_MVO_OPTIMISATION:
        print("WARNING: with 20 assets or less, the efficient frontier is reduced to a single point (MVP) because the code currently constraints each asset being at most 5% of your portfolio.")
    elif num_assets > 20 and RUN_MVO_OPTIMISATION:
        print("More than 20 assets detected. We will attempt to draw the efficient frontier.\n")

    # Add individual stock weight constraint (max 5%)
    for i in range(num_assets):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, idx=i: effective_max_stock_weight - x[idx]
        })

    # Add sector weight constraints (max 25%)
    sectors = {}
    for i, ticker in enumerate(portfolio_tickers):
        sector = STOCK_SECTORS.get(ticker)
        if sector:
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(i) # Store index of stock in portfolio_tickers
        else:
            print(f"Warning: Ticker '{ticker}' not found in STOCK_SECTORS. It will not be subject to sector constraints.")

    # Determine a dynamic maximum sector weight based on the number of assets
    # If num_assets is less than 20, allow sectors to take up to 100% (effectively disabling the hard cap)
    effective_max_sector_weight_for_constraint = CONFIGURED_MAX_SECTOR_WEIGHT
    if num_assets <= 20:
        effective_max_sector_weight_for_constraint = 1.0 # Allow up to 100% for sectors if only 1 or 2 assets
        #print(f"effective_max_sector_weight_for_constraint:{effective_max_sector_weight_for_constraint}") 
    
    
    for sector_name, stock_indices in sectors.items():
        # Maximum possible weight a sector can have given individual stock limits.
        sum_of_effective_max_stock_weights_in_sector = sum(effective_max_stock_weight for _ in stock_indices)
        
        # Determine the effective maximum sector weight for this specific sector.
        current_sector_limit = min(effective_max_sector_weight_for_constraint, sum_of_effective_max_stock_weights_in_sector)
        
        print(f"Maximum weight for sector '{sector_name}': {current_sector_limit:.2%}")

        
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, indices=stock_indices, effective_limit=current_sector_limit: effective_limit - np.sum(x[indices]),
            'args': (stock_indices, current_sector_limit,)
        })

    initial_guess = np.array(num_assets * [1. / num_assets])

    # --- STATIC COVARIANCE MODEL ---
    static_annualised_covariance_matrix = daily_returns.cov() * 252
    static_annualised_correlation_matrix = daily_returns.corr()
    
    plt.figure(figsize=(18, 10))
    sns.heatmap(static_annualised_correlation_matrix, cmap="Blues", annot=True, fmt='.2f')
    plt.savefig(os.path.join(OUTPUT_DIR, "stocks_heatmap.png"), bbox_inches='tight', dpi=100)
    plt.close()
    print(f"\nHeatmap saved in {os.path.join(OUTPUT_DIR, 'stocks_heatmap.png')}")
    #plt.show()

    if RUN_STATIC_PORTFOLIO:
        print("\n--- OPTIMISATION WITH STATIC COVARIANCE MODEL ---")

        # 1. Find Minimum Variance Portfolio (MVP) - Static
        if RUN_MVO_OPTIMISATION:
            # Find optimum weights
            mvp_results_static = minimize(portfolio_volatility, initial_guess, args=(static_annualised_covariance_matrix.values,),
                                   method='SLSQP', bounds=bounds, constraints=constraints)

            mvp_volatility_static = mvp_results_static.fun
            mvp_return_static = portfolio_return(mvp_results_static.x, annual_returns_array)
            mvp_weights_static = mvp_results_static.x

            print("\nMinimum Variance Portfolio (MVP) - Static:")
            print(f"Return: {mvp_return_static:.2%}")
            print(f"Volatility: {mvp_volatility_static:.2%}")
            print("Weights:")
            for i, ticker in enumerate(portfolio_tickers):
                print(f"  {ticker}: {mvp_weights_static[i]:.2%}")
                
            
            # Trace the Efficient Frontier - Static
            if num_assets > 20:
                min_return_for_frontier_static = mvp_return_static
                max_return_for_frontier_static = max(annual_returns_array) + 0.001 # Add a small buffer to ensure the highest return point is included.
                
                if min_return_for_frontier_static >= max_return_for_frontier_static:
                    max_return_for_frontier_static = min_return_for_frontier_static + 0.001 

                #print("min_return_for_frontier_static: ", min_return_for_frontier_static, "max_return_for_frontier_static: ", max_return_for_frontier_static)
                target_returns_static = np.linspace(min_return_for_frontier_static, max_return_for_frontier_static, NUM_FRONTIER_POINTS)

                # mvp_results_static as found
                efficient_frontier_std_devs_static = [mvp_volatility_static]
                efficient_frontier_returns_static = [mvp_return_static]
                current_initial_guess = mvp_weights_static # Start with MVP weights
                #print("weights: ", current_initial_guess)
                
                failures_in_a_row = 0
                last_achieved_target = 0
                for target_ret in target_returns_static:
                    return_constraint = {'type': 'eq',
                                         'fun': lambda x: portfolio_return(x, annual_returns_array) - target_ret}
                    all_constraints = constraints + [return_constraint]

                    result = minimize(portfolio_volatility, current_initial_guess, args=(static_annualised_covariance_matrix.values,),
                                      method='SLSQP', bounds=bounds, constraints=all_constraints)

                    #print(f"DEBUG: Static Frontier optimisation for target return {target_ret:.2%}: Success={result.success}, Message={result.message}")
                    if result.success:
                        efficient_frontier_std_devs_static.append(result.fun)
                        efficient_frontier_returns_static.append(portfolio_return(result.x, annual_returns_array))
                        current_initial_guess = result.x
                        #print("weights: ", current_initial_guess)
                        failures_in_a_row = 0
                        last_achieved_target = target_ret
                    else:
                        #print(f"Optimisation failed at target return {target_ret:.2%}: {result.message}")
                        failures_in_a_row += 1
                        if failures_in_a_row >= 5:
                            print(f"\nMaximum return target: {last_achieved_target:.2%} (Static efficient frontier)")
                            break
               
                # Ensures the frontier is drawn smoothly and monotonically left-to-right.
                optimised_points_static = sorted(list(zip(efficient_frontier_std_devs_static, efficient_frontier_returns_static)))
                efficient_frontier_std_devs_static = [p[0] for p in optimised_points_static]
                efficient_frontier_returns_static = [p[1] for p in optimised_points_static]

        # 2. Find Tangency Portfolio (Maximum Sharpe Ratio Portfolio) - Static
        if RUN_SHARPE_OPTIMISATION:
            tangency_portfolio_results_static = minimize(negative_sharpe_ratio, initial_guess,
                                                  args=(annual_returns_array, static_annualised_covariance_matrix.values, RISK_FREE_RATE),
                                                  method='SLSQP', bounds=bounds, constraints=constraints)

            tangency_volatility_static = portfolio_volatility(tangency_portfolio_results_static.x, static_annualised_covariance_matrix.values)
            tangency_return_static = portfolio_return(tangency_portfolio_results_static.x, annual_returns_array)
            tangency_weights_static = tangency_portfolio_results_static.x
            tangency_sharpe_ratio_static = (tangency_return_static - RISK_FREE_RATE) / tangency_volatility_static

            print(f"\nTangency Portfolio (Max Sharpe Ratio = {tangency_sharpe_ratio_static:.4f}) - Static:")
            print(f"Risk-Free Rate: {RISK_FREE_RATE:.2%}")
            print(f"Return: {tangency_return_static:.2%}")
            print(f"Volatility: {tangency_volatility_static:.2%}")
            print("Weights:")
            for i, ticker in enumerate(portfolio_tickers):
                print(f"  {ticker}: {tangency_weights_static[i]:.2%}")

        # 3. Find Sortino Ratio Optimised Portfolio - Static
        if RUN_SORTINO_OPTIMISATION:
            sortino_portfolio_results_static = minimize(negative_sortino_ratio, initial_guess,
                                                 args=(annual_returns_array, daily_returns, static_annualised_covariance_matrix.values, RISK_FREE_RATE),
                                                 method='SLSQP', bounds=bounds, constraints=constraints)

            sortino_volatility_static = portfolio_volatility(sortino_portfolio_results_static.x, static_annualised_covariance_matrix.values)
            sortino_return_static = portfolio_return(sortino_portfolio_results_static.x, annual_returns_array)
            sortino_weights_static = sortino_portfolio_results_static.x
            sortino_ratio_static = (sortino_return_static - RISK_FREE_RATE) / downside_deviation(sortino_weights_static, daily_returns, RISK_FREE_RATE)

            print(f"\nSortino Portfolio (Max Sortino Ratio = {sortino_ratio_static:.4f}) - Static:")
            print(f"Risk-Free Rate: {RISK_FREE_RATE:.2%}")
            print(f"Return: {sortino_return_static:.2%}")
            print(f"Volatility: {sortino_volatility_static:.2%}")
            print("Weights:")
            for i, ticker in enumerate(portfolio_tickers):
                print(f"  {ticker}: {sortino_weights_static[i]:.2%}")
        
        # 4. Find MVSK Optimised Portfolio - Static
        if RUN_MVSK_OPTIMISATION:
            mvsk_portfolio_results_static = minimize(negative_mvsk_utility, initial_guess,
                                             args=(annual_returns_array, daily_returns, static_annualised_covariance_matrix.values, RISK_FREE_RATE, LAMBDA_S, LAMBDA_K),
                                             method='SLSQP', bounds=bounds, constraints=constraints)

            mvsk_return_static = portfolio_return(mvsk_portfolio_results_static.x, annual_returns_array)
            mvsk_volatility_static = portfolio_volatility(mvsk_portfolio_results_static.x, static_annualised_covariance_matrix.values)
            mvsk_skewness_static = portfolio_skewness(mvsk_portfolio_results_static.x, daily_returns)
            mvsk_kurtosis_static = portfolio_kurtosis(mvsk_portfolio_results_static.x, daily_returns)
            mvsk_weights_static = mvsk_portfolio_results_static.x

            print(f"\nMean-Variance-Skewness-Kurtosis Portfolio - Static:")
            print(f"Return: {mvsk_return_static:.2%}")
            print(f"Volatility: {mvsk_volatility_static:.2%}")
            print(f"Skewness: {mvsk_skewness_static:.4f}")
            print(f"Kurtosis: {mvsk_kurtosis_static:.4f}")
            print("Weights:")
            for i, ticker in enumerate(portfolio_tickers):
                print(f"  {ticker}: {mvsk_weights_static[i]:.2%}")


    # --- DYNAMIC COVARIANCE MODEL (Rolling Window) ---
    dynamic_covariance_available = False
    if RUN_DYNAMIC_PORTFOLIO: 
        if len(daily_returns) < ROLLING_WINDOW_DAYS:
            print(f"\nWarning: Not enough historical data ({len(daily_returns)} days) for a rolling window of {ROLLING_WINDOW_DAYS} days.")
            print("Skipping dynamic covariance optimisation. Please ensure sufficient historical data.")
        else:
            # Get the last valid rolling covariance matrix
            rolling_cov_matrix_full = daily_returns.rolling(window=ROLLING_WINDOW_DAYS).cov().dropna()

            if not rolling_cov_matrix_full.empty:
                # Get the last timestamp for which a complete matrix was calculated
                last_timestamp = rolling_cov_matrix_full.index.get_level_values(0).unique()[-1]
                # Select the block corresponding to this last timestamp
                dynamic_annualised_covariance_matrix = rolling_cov_matrix_full.loc[last_timestamp].values * 252
            else:
                print("\nError: No complete rolling covariance matrices found after dropping NaNs.")
                RUN_DYNAMIC_PORTFOLIO = False

            #print("rolling_cov_matrix_full: ", rolling_cov_matrix_full)
            #print("last_timestamp: ", last_timestamp)
            #print("dynamic_annualised_covariance_matrix: ", dynamic_annualised_covariance_matrix)
            
            # Ensure dynamic_annualised_covariance_matrix is not NaN itself
            if np.isnan(dynamic_annualised_covariance_matrix).any():
                print("\nWarning: The selected dynamic covariance matrix still contains NaN values.")
                print("This might happen if individual stock returns within the last window had NaNs before the .dropna() call.")
                # You might want to fall back to static or fill with zeros, or raise error
                RUN_DYNAMIC_PORTFOLIO = False # Prevent optimization with bad data

            
            dynamic_covariance_available = True
            print(f"\n--- OPTIMISATION WITH DYNAMIC COVARIANCE MODEL (Last {ROLLING_WINDOW_DAYS} Days) ---")

            
            # 1. Find Minimum Variance Portfolio (MVP) - Dynamic
            if RUN_MVO_OPTIMISATION:
                mvp_results_dynamic = minimize(portfolio_volatility, initial_guess, args=(dynamic_annualised_covariance_matrix,),
                                       method='SLSQP', bounds=bounds, constraints=constraints)

                mvp_volatility_dynamic = mvp_results_dynamic.fun
                mvp_return_dynamic = portfolio_return(mvp_results_dynamic.x, annual_returns_array)
                mvp_weights_dynamic = mvp_results_dynamic.x

                print("\nMinimum Variance Portfolio (MVP) - Dynamic:")
                print(f"Return: {mvp_return_dynamic:.2%}")
                print(f"Volatility: {mvp_volatility_dynamic:.2%}")
                print("Weights:")
                for i, ticker in enumerate(portfolio_tickers):
                    print(f"  {ticker}: {mvp_weights_dynamic[i]:.2%}")

                if num_assets>=20:
                    # Trace the Efficient Frontier - Dynamic
                    min_return_for_frontier_dynamic = mvp_return_dynamic
                    max_return_for_frontier_dynamic = max(annual_returns_array) + 0.001

                    if min_return_for_frontier_dynamic >= max_return_for_frontier_dynamic:
                        max_return_for_frontier_dynamic = min_return_for_frontier_dynamic + 0.05

                    target_returns_dynamic = np.linspace(min_return_for_frontier_dynamic, max_return_for_frontier_dynamic, NUM_FRONTIER_POINTS)

                    efficient_frontier_std_devs_dynamic = []
                    efficient_frontier_returns_dynamic = []
                 
                    failures_in_a_row = 0
                    last_achieved_target = 0
                    for target_ret_dyn in target_returns_dynamic:
                        return_constraint_dyn = {'type': 'eq',
                                                 'fun': lambda x: portfolio_return(x, annual_returns_array) - target_ret_dyn}
                        all_constraints_dyn = constraints + [return_constraint_dyn]

                        result_dyn = minimize(portfolio_volatility, initial_guess, args=(dynamic_annualised_covariance_matrix,),
                                              method='SLSQP', bounds=bounds, constraints=all_constraints_dyn)
                        
                        #print(f"DEBUG: Dynamic Frontier optimisation for target return {target_ret_dyn:.2%}: Success={result_dyn.success}, Message={result_dyn.message}")
                        if result_dyn.success:
                            efficient_frontier_std_devs_dynamic.append(result_dyn.fun)
                            efficient_frontier_returns_dynamic.append(portfolio_return(result_dyn.x, annual_returns_array))
                            failures_in_a_row = 0
                            last_achieved_target = target_ret_dyn
                        else:
                            #print(f"Optimisation failed at target return {target_ret:.2%}: {result.message}")
                            failures_in_a_row += 1
                            if failures_in_a_row >= 5:
                                print(f"\nMaximum return target: {last_achieved_target:.2%} (Dynamic efficient frontier)")
                                break
                    
                
                    optimised_points_dynamic = sorted(list(zip(efficient_frontier_std_devs_dynamic, efficient_frontier_returns_dynamic)))
                    efficient_frontier_std_devs_dynamic = [p[0] for p in optimised_points_dynamic]
                    efficient_frontier_returns_dynamic = [p[1] for p in optimised_points_dynamic]

            # 2. Find Tangency Portfolio (Maximum Sharpe Ratio Portfolio) - Dynamic
            if RUN_SHARPE_OPTIMISATION:
                tangency_portfolio_results_dynamic = minimize(negative_sharpe_ratio, initial_guess,
                                                      args=(annual_returns_array, dynamic_annualised_covariance_matrix, RISK_FREE_RATE),
                                                      method='SLSQP', bounds=bounds, constraints=constraints)

                tangency_volatility_dynamic = portfolio_volatility(tangency_portfolio_results_dynamic.x, dynamic_annualised_covariance_matrix)
                tangency_return_dynamic = portfolio_return(tangency_portfolio_results_dynamic.x, annual_returns_array)
                tangency_weights_dynamic = tangency_portfolio_results_dynamic.x
                tangency_sharpe_ratio_dynamic = (tangency_return_dynamic - RISK_FREE_RATE) / tangency_volatility_dynamic

                print(f"\nTangency Portfolio (Max Sharpe Ratio = {tangency_sharpe_ratio_dynamic:.4f}) - Dynamic:")
                print(f"Risk-Free Rate: {RISK_FREE_RATE:.2%}")
                print(f"Return: {tangency_return_dynamic:.2%}")
                print(f"Volatility: {tangency_volatility_dynamic:.2%}")
                print("Weights:")
                for i, ticker in enumerate(portfolio_tickers):
                    print(f"  {ticker}: {tangency_weights_dynamic[i]:.2%}")

            # 3. Find Sortino Ratio Optimised Portfolio - Dynamic
            if RUN_SORTINO_OPTIMISATION:
                sortino_portfolio_results_dynamic = minimize(negative_sortino_ratio, initial_guess,
                                                     args=(annual_returns_array, daily_returns, dynamic_annualised_covariance_matrix, RISK_FREE_RATE),
                                                     method='SLSQP', bounds=bounds, constraints=constraints)

                sortino_volatility_dynamic = portfolio_volatility(sortino_portfolio_results_dynamic.x, dynamic_annualised_covariance_matrix)
                sortino_return_dynamic = portfolio_return(sortino_portfolio_results_dynamic.x, annual_returns_array)
                sortino_weights_dynamic = sortino_portfolio_results_dynamic.x
                sortino_ratio_dynamic = (sortino_return_dynamic - RISK_FREE_RATE) / downside_deviation(sortino_weights_dynamic, daily_returns, RISK_FREE_RATE)

                print(f"\nSortino Portfolio (Max Sortino Ratio = {sortino_ratio_dynamic:.4f}) - Dynamic:")
                print(f"Risk-Free Rate: {RISK_FREE_RATE:.2%}")
                print(f"Return: {sortino_return_dynamic:.2%}")
                print(f"Volatility: {sortino_volatility_dynamic:.2%}")
                print("Weights:")
                for i, ticker in enumerate(portfolio_tickers):
                    print(f"  {ticker}: {sortino_weights_dynamic[i]:.2%}")

            # 4. Find MVSK Optimised Portfolio - Dynamic
            if RUN_MVSK_OPTIMISATION:
                mvsk_portfolio_results_dynamic = minimize(negative_mvsk_utility, initial_guess,
                                                 args=(annual_returns_array, daily_returns, dynamic_annualised_covariance_matrix, RISK_FREE_RATE, LAMBDA_S, LAMBDA_K),
                                                 method='SLSQP', bounds=bounds, constraints=constraints)

                mvsk_return_dynamic = portfolio_return(mvsk_portfolio_results_dynamic.x, annual_returns_array)
                mvsk_volatility_dynamic = portfolio_volatility(mvsk_portfolio_results_dynamic.x, dynamic_annualised_covariance_matrix)
                mvsk_skewness_dynamic = portfolio_skewness(mvsk_portfolio_results_dynamic.x, daily_returns)
                mvsk_kurtosis_dynamic = portfolio_kurtosis(mvsk_portfolio_results_dynamic.x, daily_returns)
                mvsk_weights_dynamic = mvsk_portfolio_results_dynamic.x

                print(f"\nMean-Variance-Skewness-Kurtosis Portfolio - Dynamic:")
                print(f"Return: {mvsk_return_dynamic:.2%}")
                print(f"Volatility: {mvsk_volatility_dynamic:.2%}")
                print(f"Skewness: {mvsk_skewness_dynamic:.4f}")
                print(f"Kurtosis: {mvsk_kurtosis_dynamic:.4f}")
                print("Weights:")
                for i, ticker in enumerate(portfolio_tickers):
                    print(f"  {ticker}: {mvsk_weights_dynamic[i]:.2%}")


    # --- Monte Carlo Simulation (for context of static frontier) ---
    if RUN_MONTE_CARLO_SIMULATION:
        NUM_PORTFOLIO_MC = 100000
        static_portfolio_points_raw_mc = []
        dynamic_portfolio_points_raw_mc = []
        for _ in range(NUM_PORTFOLIO_MC):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights) # Normalize weights to sum to 1

            # Check constraints for MC points (optional, but makes MC region more realistic)
            is_valid_mc_point = True
            if np.any(weights > effective_max_stock_weight):
                is_valid_mc_point = False
            
            # Check sector constraints for MC points
            for sector_name, stock_indices in sectors.items():
                sector_weight = np.sum(weights[stock_indices])
                if sector_weight > effective_max_sector_weight_for_constraint:
                    is_valid_mc_point = False
                    break
            
            if is_valid_mc_point:
                if RUN_STATIC_PORTFOLIO:
                    port_ret, port_std = calculate_portfolio_metrics(weights, annual_returns_array, static_annualised_covariance_matrix.values)
                    static_portfolio_points_raw_mc.append({
                    'std_dev': port_std,
                    'return': port_ret,
                    'weights': {ticker: w for ticker, w in zip(portfolio_tickers, weights)}
                })
                if RUN_DYNAMIC_PORTFOLIO:
                    port_ret, port_std = calculate_portfolio_metrics(weights, annual_returns_array, dynamic_annualised_covariance_matrix)
                    dynamic_portfolio_points_raw_mc.append({
                    'std_dev': port_std,
                    'return': port_ret,
                    'weights': {ticker: w for ticker, w in zip(portfolio_tickers, weights)}
                })
        if RUN_STATIC_PORTFOLIO:
            print(f"\nGenerated {len(static_portfolio_points_raw_mc)} valid static random portfolios for Monte Carlo visualisation (out of {NUM_PORTFOLIO_MC} attempts).")
        if RUN_DYNAMIC_PORTFOLIO:
            print(f"\nGenerated {len(dynamic_portfolio_points_raw_mc)} valid dynamic random portfolios for Monte Carlo visualisation (out of {NUM_PORTFOLIO_MC} attempts).")


    # --- Plotting the portfolios ---
    plt.figure(figsize=(14, 8)) # Larger figure for more elements

    # Plot all Monte-Carlo-simulated portfolio combinations (lighter color, background)
    if RUN_MONTE_CARLO_SIMULATION:
        if RUN_STATIC_PORTFOLIO and static_portfolio_points_raw_mc:
            plt.scatter([p['std_dev'] * 100 for p in static_portfolio_points_raw_mc],
                        [p['return'] * 100 for p in static_portfolio_points_raw_mc],
                        color='blue', marker='o', s=10, alpha=0.5, # More transparent
                        label='Monte Carlo portfolio combinations (Static)')
        if RUN_DYNAMIC_PORTFOLIO and dynamic_portfolio_points_raw_mc:
            plt.scatter([p['std_dev'] * 100 for p in dynamic_portfolio_points_raw_mc],
                        [p['return'] * 100 for p in dynamic_portfolio_points_raw_mc],
                        color='red', marker='o', s=10, alpha=0.5, # More transparent
                        label='Monte Carlo portfolio combinations (Dynamic)')
    
    # Plot the Efficient Frontier line (Static Covariance)
    #print("efficient_frontier_std_devs_static: ", efficient_frontier_std_devs_static)
    if RUN_STATIC_PORTFOLIO and RUN_MVO_OPTIMISATION and num_assets>20:
        plt.plot([s * 100 for s in efficient_frontier_std_devs_static],
                 [r * 100 for r in efficient_frontier_returns_static],
                 color='blue', linestyle='-', linewidth=2, label='Efficient frontier (Static)')

    # Plot the Efficient Frontier line (Dynamic Covariance)
    #print("efficient_frontier_std_devs_dynamic: ", efficient_frontier_std_devs_dynamic) 
    if RUN_DYNAMIC_PORTFOLIO and RUN_MVO_OPTIMISATION and num_assets>20 and 'efficient_frontier_std_devs_dynamic' in locals() and efficient_frontier_std_devs_dynamic:
        plt.plot([s * 100 for s in efficient_frontier_std_devs_dynamic],
                 [r * 100 for r in efficient_frontier_returns_dynamic],
                 color='red', linestyle='-', linewidth=2, label='Efficient frontier (Dynamic)')


    # Plot individual stocks
#    individual_stock_colors_palette = [
#        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
#        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
#        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
#        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
#        '#5254a3', '#6b6ecf', '#9c9ede', '#cedb9c', '#b5cf6b'
#    ]
    # Using Seaborn's colorblind palette
    individual_stock_colors_palette = sns.color_palette("deep", n_colors=len(portfolio_tickers)).as_hex()


    texts = []
    for i, stock in enumerate(individual_stock_metrics): # Changed from individual_stock_data to individual_stock_metrics
        plot_color = individual_stock_colors_palette[i % len(individual_stock_colors_palette)]
        plt.scatter(stock['annualised_std'] * 100, stock['annual_return'] * 100,
        marker='o', s=100, color=plot_color, edgecolor='black', linewidth=1.5,
        label='_nolegend_'
    )
        x = stock['annualised_std'] * 100
        y = stock['annual_return'] * 100
        texts.append(plt.text(x, y, stock['ticker'], color=plot_color, fontsize=11, ha='center', va='bottom'))
    
    plt.scatter([], [], s=100, color='grey', label='Stock\'s annualised performance')  # dummy point
    adjust_text(texts,
            only_move={'points': 'xy', 'text': 'xy'},
            expand_text=(2.0, 2.0),
            expand_points=(2.5, 2.5))


    if RUN_STATIC_PORTFOLIO:
        # Plot the MVP (Static)
        if RUN_MVO_OPTIMISATION:
            plt.scatter(mvp_volatility_static * 100, mvp_return_static * 100,
                        marker='*', s=200, color='darkblue', edgecolor='darkblue', alpha=0.3, linewidth=1.5,
                        label='MV (Static)')

        # Plot the Tangency Portfolio (Static)
        if RUN_SHARPE_OPTIMISATION:
            plt.scatter(tangency_volatility_static * 100, tangency_return_static * 100,
                        marker='P', s=200, color='darkblue', edgecolor='darkblue', alpha=0.3, linewidth=1.5,
                        label=f'Tangency (Static), Sharpe ratio={tangency_sharpe_ratio_static:.2}')
#            plt.annotate('Tangency',
#                         (tangency_volatility_static * 100, tangency_return_static * 100),
#                         textcoords="offset points", xytext=(10,10), ha='left', va='bottom', fontsize=10,
#                         bbox=dict(boxstyle="round,pad=0.1", fc="red", ec="none", alpha=0.7))

        # Plot the Sortino Portfolio (Static)
        if RUN_SORTINO_OPTIMISATION:
            plt.scatter(sortino_volatility_static * 100, sortino_return_static * 100,
                        marker='o', s=200, color='darkblue', edgecolor='darkblue', alpha=0.3, linewidth=1.5,
                        label=f'Sortino (Static), Sortino ratio={sortino_ratio_static:.2}')
#            plt.annotate('Sortino (Static)',
#                         (sortino_volatility_static * 100, sortino_return_static * 100),
#                         textcoords="offset points", xytext=(10,-10), ha='left', va='top', fontsize=10,
#                         bbox=dict(boxstyle="round,pad=0.1", fc="blue", ec="none", alpha=0.7))
        
        # Plot the MVSK Portfolio (Static)
        if RUN_MVSK_OPTIMISATION:
            plt.scatter(mvsk_volatility_static * 100, mvsk_return_static * 100,
                        marker='^', s=200, color='darkblue', edgecolor='darkblue', alpha=0.3, linewidth=1.5,
                        label=f'MVSK (Static)')
#            plt.annotate('MVSK (Static)',
#                         (mvsk_volatility_static * 100, mvsk_return_static * 100),
#                         textcoords="offset points", xytext=(10,10), ha='left', va='bottom', fontsize=10,
#                         bbox=dict(boxstyle="round,pad=0.1", fc="darkorange", ec="none", alpha=0.7))


    # Plot the MVP (Dynamic) if available and enabled
    if RUN_DYNAMIC_PORTFOLIO and dynamic_covariance_available:
        # Plot the MVP (Static)
        if RUN_MVO_OPTIMISATION:
            plt.scatter(mvp_volatility_dynamic * 100, mvp_return_dynamic * 100,
                        marker='*', s=200, color='red', edgecolor='red', alpha=0.3, linewidth=1.5,
                        label='MV (Dynamic)')
#            plt.annotate('MVP (Dynamic)',
#                         (mvp_volatility_dynamic * 100, mvp_return_dynamic * 100),
#                         textcoords="offset points", xytext=(10,-10), ha='right', va='top', fontsize=10,
#                         bbox=dict(boxstyle="round,pad=0.1", fc="lime", ec="none", alpha=0.7))

        # Plot the Tangency Portfolio (Dynamic)
        if RUN_SHARPE_OPTIMISATION:
            plt.scatter(tangency_volatility_dynamic * 100, tangency_return_dynamic * 100,
                        marker='P', s=200, color='red', edgecolor='red', alpha=0.3, linewidth=1.5,
                        label=f'Tangency (Dynamic), Sharpe ratio={tangency_sharpe_ratio_dynamic:.2}')
#            plt.annotate('Tangency (Dynamic)',
#                         (tangency_volatility_dynamic * 100, tangency_return_dynamic * 100),
#                         textcoords="offset points", xytext=(10,10), ha='right', va='bottom', fontsize=10,
#                         bbox=dict(boxstyle="round,pad=0.1", fc="purple", ec="none", alpha=0.7))

        # Plot the Sortino Portfolio (Dynamic)
        if RUN_SORTINO_OPTIMISATION:
            plt.scatter(sortino_volatility_dynamic * 100, sortino_return_dynamic * 100,
                        marker='o', s=200, color='red', edgecolor='red', alpha=0.3, linewidth=1.5,
                        label=f'Sortino (Dynamic), Sortino ratio={sortino_ratio_dynamic:.2}')
#            plt.annotate('Sortino (Dynamic)',
#                         (sortino_volatility_dynamic * 100, sortino_return_dynamic * 100),
#                         textcoords="offset points", xytext=(10,-10), ha='right', va='top', fontsize=10,
#                         bbox=dict(boxstyle="round,pad=0.1", fc="cyan", ec="none", alpha=0.7))

        # Plot the MVSK Portfolio (Dynamic)
        if RUN_MVSK_OPTIMISATION:
            plt.scatter(mvsk_volatility_dynamic * 100, mvsk_return_dynamic * 100,
                        marker='^', s=200, color='red', edgecolor='red', alpha=0.3, linewidth=1.5,
                        label=f'MVSK (Dynamic)')
#            plt.annotate('MVSK (Dynamic)',
#                         (mvsk_volatility_dynamic * 100, mvsk_return_dynamic * 100),
#                         textcoords="offset points", xytext=(10,10), ha='right', va='bottom', fontsize=10,
#                         bbox=dict(boxstyle="round,pad=0.1", fc="brown", ec="none", alpha=0.7))


    if (RUN_STATIC_PORTFOLIO or RUN_DYNAMIC_PORTFOLIO) and (RUN_MONTE_CARLO_SIMULATION or RUN_MVO_OPTIMISATION or RUN_SHARPE_OPTIMISATION or RUN_SORTINO_OPTIMISATION or RUN_MVSK_OPTIMISATION):
        plt.title('Optimised portfolios', fontsize=16)
        plt.xlabel('Annualised Standard Deviation (Volatility) (%)', fontsize=12)
        plt.ylabel('Annualised Return (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',labelspacing=2)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.savefig(os.path.join(OUTPUT_DIR, "optimised_portfolios.png"), bbox_inches='tight', dpi=100)
        print(f"\nOptimised portfolios saved in {os.path.join(OUTPUT_DIR, 'optimised_portfolios.png')}")
        plt.show()

        #print("\nOptimised portfolios plot displayed successfully.")
    else:
        print("\nYou must choose the type (static or dynamic) and at least one portfolio (Monte Carlo, Mean-Variance, Tangency, Sortino, MVSK to display anything!")

    # --- Save weights for optimised portfolios ---
    if RUN_STATIC_PORTFOLIO:
        if RUN_MVO_OPTIMISATION:
            #print("\n--- Weights for Key Optimised Portfolios (Static Covariance) ---")
            #print("\nMinimum Variance Portfolio (MVP):")
            write_portfolio_weights_to_csv(
                filename=OUTPUT_FILENAME,
                portfolio_type="Static Covariance",
                optimisation_type="Minimum Variance",
                metrics={"Return": mvp_return_static, "Volatility": mvp_volatility_static},
                weights=mvp_weights_static,
                portfolio_tickers=portfolio_tickers
            )

        if RUN_SHARPE_OPTIMISATION:
            #print(f"\nTangency Portfolio (Max Sharpe Ratio = {tangency_sharpe_ratio_static:.4f}):")
            write_portfolio_weights_to_csv(
                filename=OUTPUT_FILENAME,
                portfolio_type="Static Covariance",
                optimisation_type="Tangency (Max Sharpe Ratio)",
                metrics={
                    "Return": tangency_return_static,
                    "Volatility": tangency_volatility_static,
                    "Sharpe Ratio": tangency_sharpe_ratio_static
                },
                weights=tangency_weights_static,
                portfolio_tickers=portfolio_tickers
            )

        if RUN_SORTINO_OPTIMISATION:
            #print(f"\nSortino Portfolio (Max Sortino Ratio = {sortino_ratio_static:.4f}):")
            write_portfolio_weights_to_csv(
                filename=OUTPUT_FILENAME,
                portfolio_type="Static Covariance",
                optimisation_type="Sortino (Max Sortino Ratio)",
                metrics={
                    "Return": sortino_return_static,
                    "Volatility": sortino_volatility_static,
                    "Sortino Ratio": sortino_ratio_static
                },
                weights=sortino_weights_static,
                portfolio_tickers=portfolio_tickers
            )

        if RUN_MVSK_OPTIMISATION:
            #print(f"\nMean-Variance-Skewness-Kurtosis Portfolio:")
            write_portfolio_weights_to_csv(
                filename=OUTPUT_FILENAME,
                portfolio_type="Static Covariance",
                optimisation_type="Mean-Variance-Skewness-Kurtosis",
                metrics={
                    "Return": mvsk_return_static,
                    "Volatility": mvsk_volatility_static,
                    "Skewness": mvsk_skewness_static,
                    "Kurtosis": mvsk_kurtosis_static
                },
                weights=mvsk_weights_static,
                portfolio_tickers=portfolio_tickers
            )

    if RUN_DYNAMIC_PORTFOLIO and dynamic_covariance_available:
        if RUN_MVO_OPTIMISATION:
            #print("\n--- Weights for Key Optimised Portfolios (Dynamic Covariance) ---")
            #print("\nMinimum Variance Portfolio (MVP) - Dynamic:")
            write_portfolio_weights_to_csv(
                filename=OUTPUT_FILENAME,
                portfolio_type="Dynamic Covariance",
                optimisation_type="Minimum Variance",
                metrics={"Return": mvp_return_dynamic, "Volatility": mvp_volatility_dynamic},
                weights=mvp_weights_dynamic,
                portfolio_tickers=portfolio_tickers
            )

        if RUN_SHARPE_OPTIMISATION:
            #print(f"\nTangency Portfolio (Max Sharpe Ratio = {tangency_sharpe_ratio_dynamic:.4f}):")
            write_portfolio_weights_to_csv(
                filename=OUTPUT_FILENAME,
                portfolio_type="Dynamic Covariance",
                optimisation_type="Tangency (Max Sharpe Ratio)",
                metrics={
                    "Return": tangency_return_dynamic,
                    "Volatility": tangency_volatility_dynamic,
                    "Sharpe Ratio": tangency_sharpe_ratio_dynamic
                },
                weights=tangency_weights_dynamic,
                portfolio_tickers=portfolio_tickers
            )

        if RUN_SORTINO_OPTIMISATION:
            #print(f"\nSortino Portfolio (Max Sortino Ratio = {sortino_ratio_dynamic:.4f}):")
            write_portfolio_weights_to_csv(
                filename=OUTPUT_FILENAME,
                portfolio_type="Dynamic Covariance",
                optimisation_type="Sortino (Max Sortino Ratio)",
                metrics={
                    "Return": sortino_return_dynamic,
                    "Volatility": sortino_volatility_dynamic,
                    "Sortino Ratio": sortino_ratio_dynamic
                },
                weights=sortino_weights_dynamic,
                portfolio_tickers=portfolio_tickers
            )

        if RUN_MVSK_OPTIMISATION:
            #print(f"\nMean-Variance-Skewness-Kurtosis Portfolio - Dynamic:")
            write_portfolio_weights_to_csv(
                filename=OUTPUT_FILENAME,
                portfolio_type="Dynamic Covariance",
                optimisation_type="Mean-Variance-Skewness-Kurtosis",
                metrics={
                    "Return": mvsk_return_dynamic,
                    "Volatility": mvsk_volatility_dynamic,
                    "Skewness": mvsk_skewness_dynamic,
                    "Kurtosis": mvsk_kurtosis_dynamic
                },
                weights=mvsk_weights_dynamic,
                portfolio_tickers=portfolio_tickers
            )

    print(f"\nOptimisation data saved to {output_filepath}")
