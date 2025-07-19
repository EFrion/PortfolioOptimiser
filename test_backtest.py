import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from utils import run_backtest


# --- Unit Test Implementation ---

def test_fixed_returns_equal_weights():
    print("\n--- Running test: Fixed 1% daily returns with equal weights ---")

    # Test Parameters
    num_assets_test = 5
    daily_return_rate = 0.01
    test_start_date = pd.Timestamp('2020-01-01')
    test_end_date = pd.Timestamp('2020-01-31') # A month of data
    rebalance_freq = 'D' # Daily rebalancing for simplicity to ensure smooth curve
    hist_data_window = 2 # Minimal lookback for test, just needs to allow first rebalance

    # Generate synthetic all_stock_prices where each asset returns 1% daily
    dates = pd.date_range(start=test_start_date - pd.Timedelta(days=hist_data_window + 5), end=test_end_date + BDay(1), freq='B') # Include pre-backtest data
    initial_price = 100
    prices_data = np.cumprod(np.ones((len(dates), num_assets_test)) * (1 + daily_return_rate), axis=0) * initial_price
    prices_data[0, :] = initial_price # Set first row to initial price
    all_stock_prices_mock = pd.DataFrame(prices_data, index=dates, columns=[f'Asset_{i}' for i in range(num_assets_test)])
    
    # Ensure no prices are NaN or inf 
    if all_stock_prices_mock.isnull().any().any() or np.isinf(all_stock_prices_mock).any().any():
        raise ValueError("Generated prices contain NaN or Inf values, check test setup.")

    # Create dummy daily_returns (will be re-calculated inside run_backtest anyway)
    daily_returns_mock = all_stock_prices_mock.pct_change().dropna()

    # Configure the backtest to run only Buy & Hold (or primarily focus on it)
    config_mock = {
        'feature_toggles': {
            'RUN_STATIC_PORTFOLIO': False,
            'RUN_DYNAMIC_PORTFOLIO': False,
            'RUN_MVO_OPTIMISATION': False,
            'RUN_SHARPE_OPTIMISATION': False,
            'RUN_SORTINO_OPTIMISATION': False,
            'RUN_MVSK_OPTIMISATION': False,
        },
        'portfolio_parameters': {
            'NUM_FRONTIER_POINTS': 100, # Dummy
            'ROLLING_WINDOW_DAYS': 180,
        }
    }

    # Other dummy parameters for run_backtest
    initial_guess_mock = np.array([1.0 / num_assets_test] * num_assets_test)
    bounds_mock = tuple((0.0, 1.0) for _ in range(num_assets_test))
    constraints_mock = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    risk_free_rate_mock = 0.01
    lambda_s_mock = 0.0
    lambda_k_mock = 0.0

    # Run the backtest
    cumulative_returns_history = run_backtest(
        config=config_mock,
        all_stock_prices=all_stock_prices_mock,
        daily_returns=daily_returns_mock,
        portfolio_tickers=[f'Asset_{i}' for i in range(num_assets_test)],
        num_assets=num_assets_test,
        initial_guess=initial_guess_mock,
        bounds=bounds_mock,
        constraints=constraints_mock,
        risk_free_rate=risk_free_rate_mock,
        lambda_s=lambda_s_mock,
        lambda_k=lambda_k_mock,
        backtest_start_date=test_start_date,
        backtest_end_date=test_end_date,
        rebalancing_frequency=rebalance_freq,
        historical_data_window_days=hist_data_window
    )

    # Determine the actual PnL start date from the run_backtest output
    pnl_start_date = cumulative_returns_history.index[0]
    num_trading_days = len(cumulative_returns_history)

    # Calculate expected cumulative returns: (1 + daily_return_rate)^t
    expected_cumulative_returns = np.cumprod(np.ones(num_trading_days) * (1 + daily_return_rate))
    
    # The first value should be 1.0 after normalization
    expected_cumulative_returns = (1 + daily_return_rate) ** np.arange(num_trading_days)

    for i in range(num_trading_days):
        print(f"Day {i}: Expected={expected_cumulative_returns[i]:.6f}, Actual={cumulative_returns_history['Buy_and_Hold'].iloc[i]:.6f}")


    print(f"Test starting date: {test_start_date.strftime('%Y-%m-%d')}")
    print(f"P&L tracking starts: {pnl_start_date.strftime('%Y-%m-%d')}")
    print(f"Number of trading days in test: {num_trading_days}")
    print(f"Expected final cumulative return: {expected_cumulative_returns[-1]:.4f}")
    print(f"Actual final 'Buy_and_Hold' return: {cumulative_returns_history['Buy_and_Hold'].iloc[-1]:.4f}")

    # Assert that the 'Buy_and_Hold' strategy's cumulative returns match the expectation
    # Using np.testing.assert_allclose for floating-point comparisons
    try:
        np.testing.assert_allclose(
            cumulative_returns_history['Buy_and_Hold'].values,
            expected_cumulative_returns,
            rtol=1e-5,
            atol=1e-8
        )
        print("Test Passed: 'Buy_and_Hold' cumulative returns match (1.01)^t")
    except AssertionError as e:
        print(f"Test Failed: {e}")
        # Print differences for debugging
        print("\nExpected vs Actual (first 10, last 10):")
        for i in range(min(10, num_trading_days)):
            print(f"Day {i}: Expected={expected_cumulative_returns[i]:.6f}, Actual={cumulative_returns_history['Buy_and_Hold'].iloc[i]:.6f}")
        if num_trading_days > 20:
            print("...")
        for i in range(max(0, num_trading_days-10), num_trading_days):
            print(f"Day {i}: Expected={expected_cumulative_returns[i]:.6f}, Actual={cumulative_returns_history['Buy_and_Hold'].iloc[i]:.6f}")
        raise # Re-raise the exception to indicate test failure
        
def test_all_strategies_converge_identical_inputs():
    print("\n--- Running test: Sanity Test - All Strategies Converge if Inputs are Identical ---")

    # Test Parameters
    num_assets_test = 5
    daily_return_rate = 0.005 # A small positive daily return
    test_start_date = pd.Timestamp('2020-01-01')
    test_end_date = pd.Timestamp('2020-01-31') # A month of data
    rebalance_freq = 'D' # Daily rebalancing
    hist_data_window = num_assets_test+1 # Minimal lookback

    # Generate synthetic all_stock_prices where all assets have identical returns
    dates = pd.date_range(start=test_start_date - pd.Timedelta(days=hist_data_window + 5), end=test_end_date + BDay(1), freq='B')
    initial_price = 100

    # Create slightly different daily return rates for each asset
    # This ensures the covariance matrix is not singular, allowing optimization
    # The differences are extremely small (e.g., 0.0050000001 vs 0.005)
    return_rates_for_assets = daily_return_rate + np.linspace(0, 1e-10, num_assets_test) 
    
    prices_data = np.zeros((len(dates), num_assets_test))
    for i in range(num_assets_test):
        # Apply a slightly different return rate to each asset
        prices_data[:, i] = np.cumprod(np.ones(len(dates)) * (1 + return_rates_for_assets[i])) * initial_price
        prices_data[0, i] = initial_price # Ensure first price is initial
    all_stock_prices_mock = pd.DataFrame(prices_data, index=dates, columns=[f'Asset_{i}' for i in range(num_assets_test)])
    
    # Ensure no prices are NaN or inf
    if all_stock_prices_mock.isnull().any().any() or np.isinf(all_stock_prices_mock).any().any():
        raise ValueError("Generated prices contain NaN or Inf values, check test setup.")

    # Create dummy daily_returns
    daily_returns_mock = all_stock_prices_mock.pct_change().dropna()

    # Configure the backtest to run ALL strategies
    config_mock = {
        'feature_toggles': {
            'RUN_STATIC_PORTFOLIO': True,
            'RUN_DYNAMIC_PORTFOLIO': True,
            'RUN_MVO_OPTIMISATION': True,
            'RUN_SHARPE_OPTIMISATION': True,
            'RUN_SORTINO_OPTIMISATION': True,
            'RUN_MVSK_OPTIMISATION': True,
        },
        'portfolio_parameters': {
            'NUM_FRONTIER_POINTS': 10, # Dummy value
            'ROLLING_WINDOW_DAYS' : 180
        }
    }

    # Other dummy parameters for run_backtest
    initial_guess_mock = np.array([1.0 / num_assets_test] * num_assets_test)
    bounds_mock = tuple((0.0, 1.0) for _ in range(num_assets_test))
    constraints_mock = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    risk_free_rate_mock = 0.01 # Dummy
    lambda_s_mock = 0.0 # Dummy
    lambda_k_mock = 0.0 # Dummy

    # Run the backtest
    cumulative_returns_history = run_backtest(
        config=config_mock,
        all_stock_prices=all_stock_prices_mock,
        daily_returns=daily_returns_mock,
        portfolio_tickers=[f'Asset_{i}' for i in range(num_assets_test)],
        num_assets=num_assets_test,
        initial_guess=initial_guess_mock,
        bounds=bounds_mock,
        constraints=constraints_mock,
        risk_free_rate=risk_free_rate_mock,
        lambda_s=lambda_s_mock,
        lambda_k=lambda_k_mock,
        backtest_start_date=test_start_date,
        backtest_end_date=test_end_date,
        rebalancing_frequency=rebalance_freq,
        historical_data_window_days=hist_data_window
    )

    # Filter for the strategies that were actually run and compare them
    reference_strategy = 'Buy_and_Hold'
    
    # Get all strategy columns that were enabled and returned
    strategy_columns = [col for col in cumulative_returns_history.columns if config_mock['feature_toggles'].get(f'RUN_{col.upper()}', False) or col == reference_strategy]
    
    # Make sure we have at least two strategies to compare
    if len(strategy_columns) < 2:
        print("Warning: Less than two strategies ran. Cannot perform convergence check effectively.")
        return

    reference_returns = cumulative_returns_history[reference_strategy].values

    print(f"P&L tracking starts: {cumulative_returns_history.index[0].strftime('%Y-%m-%d')}")
    print(f"Number of trading days in test: {len(cumulative_returns_history)}")
    
    # Assert that all other strategy returns are close to the reference strategy
    try:
        for strategy in strategy_columns:
            if strategy == reference_strategy:
                continue
            
            print(f"Comparing {strategy} with {reference_strategy}...")
            np.testing.assert_allclose(
                cumulative_returns_history[strategy].values,
                reference_returns,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Cumulative returns for {strategy} do not converge to {reference_strategy}"
            )
            print(f"  {strategy} returns match {reference_strategy} returns.")
            
        print("Test Passed: All strategies converged to identical cumulative returns.")

    except AssertionError as e:
        print(f"Test Failed: {e}")
        print("\n--- Divergence Details ---")
        for strategy in strategy_columns:
            if strategy == reference_strategy:
                continue
            diff = np.abs(cumulative_returns_history[strategy].values - reference_returns)
            max_abs_diff = np.max(diff)
            max_rel_diff = np.max(diff / np.abs(reference_returns))
            print(f"Strategy: {strategy}")
            print(f"  Max Absolute Difference: {max_abs_diff:.8f}")
            print(f"  Max Relative Difference: {max_rel_diff:.8f}")
        raise # Re-raise the exception to indicate test failure
