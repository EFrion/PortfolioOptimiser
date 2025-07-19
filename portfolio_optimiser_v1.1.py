import pandas as pd
import os
import utils


# --- Main script execution ---
if __name__ == "__main__":   
    # --- Load configuration and initialise settings ---
    config, output_filepath, backtest_output_filepath, usd_column_mapping, eur_column_mapping, exchange_rate_column_mapping = \
        utils.load_and_initialise_settings('config.json')

    # --- Feature toggles ---
    RUN_STATIC_PORTFOLIO = config['feature_toggles']['RUN_STATIC_PORTFOLIO']
    RUN_DYNAMIC_PORTFOLIO = config['feature_toggles']['RUN_DYNAMIC_PORTFOLIO']
    RUN_EQUAL_WEIGHTED_PORTFOLIO = config['feature_toggles']['RUN_EQUAL_WEIGHTED_PORTFOLIO']
    RUN_MONTE_CARLO_SIMULATION = config['feature_toggles']['RUN_MONTE_CARLO_SIMULATION']
    RUN_MVO_OPTIMISATION = config['feature_toggles']['RUN_MVO_OPTIMISATION']
    RUN_SHARPE_OPTIMISATION = config['feature_toggles']['RUN_SHARPE_OPTIMISATION']
    RUN_SORTINO_OPTIMISATION = config['feature_toggles']['RUN_SORTINO_OPTIMISATION']
    RUN_MVSK_OPTIMISATION = config['feature_toggles']['RUN_MVSK_OPTIMISATION']
    RUN_BACKTEST = config['feature_toggles']['RUN_BACKTEST']
    # -------------------------------------------------------------------------

    # --- Data Paths ---
    STOCK_ROOT_FOLDER = config['data_paths']['STOCK_ROOT_FOLDER']
    USD_FOLDER = os.path.join(STOCK_ROOT_FOLDER, 'USD') # Dynamically construct
    EUR_FOLDER = os.path.join(STOCK_ROOT_FOLDER, 'EUR') # Dynamically construct
    EXCHANGE_RATE_FILE = os.path.join(STOCK_ROOT_FOLDER, config['data_paths']['EXCHANGE_RATE_FILE_NAME'])
    YAHOO_FINANCE_CACHE_DIR = config['data_source']['YAHOO_FINANCE_CACHE_DIR']
    # -------------------------------------------------------------------------

    # --- Portfolio Parameters --- 
    RISK_FREE_RATE = config['portfolio_parameters']['RISK_FREE_RATE'] # Define a risk-free rate for Sharpe Ratio calculation (e.g., U.S. 3-Month Treasury Bill Rate)
    LAMBDA_S = config['portfolio_parameters']['LAMBDA_S'] # A positive lambda_s rewards higher (more positive) skewness.
    LAMBDA_K = config['portfolio_parameters']['LAMBDA_K'] # A positive lambda_k penalises higher (more positive) kurtosis (fat tails).
    NUM_FRONTIER_POINTS = config['portfolio_parameters']['NUM_FRONTIER_POINTS'] # For efficient frontier. Fine-grain return percentage with more points
    CONFIGURED_MAX_STOCK_WEIGHT = config['portfolio_parameters']['CONFIGURED_MAX_STOCK_WEIGHT'] # Stock constraint
    CONFIGURED_MAX_SECTOR_WEIGHT = config['portfolio_parameters']['CONFIGURED_MAX_SECTOR_WEIGHT'] # Sector constraint
    ROLLING_WINDOW_DAYS = config['portfolio_parameters']['ROLLING_WINDOW_DAYS'] # For rolling covariance (Default: 252, 1 year of trading days)
    # -------------------------------------------------------------------------
    
    # --- Backtesting Parameters ---
    BACKTEST_START_DATE = config['backtesting_parameters']['BACKTEST_START_DATE']
    BACKTEST_END_DATE = config['backtesting_parameters']['BACKTEST_END_DATE']
    REBALANCING_FREQUENCY = config['backtesting_parameters']['REBALANCING_FREQUENCY'] # 'M' for monthly, 'Q' for quarterly, 'A' for annually
    HISTORICAL_DATA_WINDOW_DAYS = config['backtesting_parameters']['HISTORICAL_DATA_WINDOW_DAYS']
    # -------------------------------------------------------------------------

    # Output Settings
    OUTPUT_DIR = config['output_settings']['OUTPUT_DIR']
    OUTPUT_FILENAME = config['output_settings']['OUTPUT_FILENAME']
    BACKTEST_OUTPUT_FILENAME = config['output_settings']['BACKTEST_OUTPUT_FILENAME']
    # -------------------------------------------------------------------------
    
    # This dictionary maps each stock ticker (in uppercase) to its sector.
    # You MUST populate this accurately for sector constraints to work.
    STOCK_SECTORS = config['stock_sectors'] 
    
    # --- Load and preprocess stock data ---
    daily_returns, portfolio_tickers, individual_stock_metrics, annual_returns_array, all_stock_prices, BACKTEST_START_DATE, BACKTEST_END_DATE = \
        utils.load_and_preprocess_stock_data(config, usd_column_mapping, eur_column_mapping, exchange_rate_column_mapping, YAHOO_FINANCE_CACHE_DIR)
   
    if daily_returns is None:
        print("Data loading and preprocessing failed. Exiting script.")
        exit() # Exit if data loading/preprocessing failed

    num_assets = len(portfolio_tickers) # Ensure num_assets is correctly set after preprocessing


    
    # --- Optimisation setup for constraints---
    
    bounds, constraints, initial_guess = utils.setup_optimisation_constraints(num_assets, portfolio_tickers, STOCK_SECTORS, CONFIGURED_MAX_STOCK_WEIGHT, CONFIGURED_MAX_SECTOR_WEIGHT, RUN_MVO_OPTIMISATION)

#    print("bounds: ", bounds)
#    print("constraints: ", constraints)
#    print("initial_guess: ", initial_guess)
    

    # --- Perform static optimisation ---

    static_annualised_covariance_matrix = daily_returns.cov() * 252
    static_annualised_correlation_matrix = daily_returns.corr()


    static_results = utils.perform_static_optimisation(
        annual_returns_array=annual_returns_array,
        static_annualised_covariance_matrix=static_annualised_covariance_matrix.values,
        initial_guess=initial_guess,
        bounds=bounds,
        constraints=constraints,
        daily_returns=daily_returns,
        risk_free_rate=RISK_FREE_RATE,
        num_assets=num_assets,
        num_frontier_points=NUM_FRONTIER_POINTS,
        lambda_s=LAMBDA_S,
        lambda_k=LAMBDA_K,
        feature_toggles=config['feature_toggles']
    )
    # Extract results for plotting/saving later
    static_mvp_results = static_results['mvp']
    static_sharpe_results = static_results['sharpe']
    static_sortino_results = static_results['sortino']
    static_mvsk_results = static_results['mvsk']
    efficient_frontier_std_devs_static = static_results['efficient_frontier_std_devs']
    efficient_frontier_returns_static = static_results['efficient_frontier_returns']

    # --- Perform dynamic optimisation ---
    dynamic_results = utils.perform_dynamic_optimisation(
        annual_returns_array=annual_returns_array,
        daily_returns=daily_returns,
        initial_guess=initial_guess,
        bounds=bounds,
        constraints=constraints,
        risk_free_rate=RISK_FREE_RATE,
        num_assets=num_assets,
        num_frontier_points=NUM_FRONTIER_POINTS,
        rolling_window_days=ROLLING_WINDOW_DAYS,
        lambda_s=LAMBDA_S,
        lambda_k=LAMBDA_K,
        feature_toggles=config['feature_toggles']
    )
    # Extract results for plotting/saving later
    dynamic_covariance_available = dynamic_results['dynamic_covariance_available']
    dynamic_mvp_results = dynamic_results['mvp']
    dynamic_sharpe_results = dynamic_results['sharpe']
    dynamic_sortino_results = dynamic_results['sortino']
    dynamic_mvsk_results = dynamic_results['mvsk']
    efficient_frontier_std_devs_dynamic = dynamic_results['efficient_frontier_std_devs']
    efficient_frontier_returns_dynamic = dynamic_results['efficient_frontier_returns']


    # --- Run Monte Carlo Simulation ---
    static_portfolio_points_raw_mc, dynamic_portfolio_points_raw_mc = utils.run_monte_carlo_simulation(
        num_portfolio_mc=config['portfolio_parameters']['NUM_PORTFOLIO_MC'], # Assuming NUM_PORTFOLIO_MC is in config
        num_assets=num_assets,
        configured_max_stock_weight=CONFIGURED_MAX_STOCK_WEIGHT,
        configured_max_sector_weight=CONFIGURED_MAX_SECTOR_WEIGHT,
        stock_sectors=STOCK_SECTORS,
        portfolio_tickers=portfolio_tickers,
        annual_returns_array=annual_returns_array,
        daily_returns=daily_returns,
        static_annualised_covariance_matrix=static_annualised_covariance_matrix.values, # Pass as numpy array
        rolling_window_days=ROLLING_WINDOW_DAYS,
        risk_free_rate=RISK_FREE_RATE,
        feature_toggles=config['feature_toggles']
    )

    # --- Run Backtest ---
    cumulative_returns_history = pd.DataFrame()
    if RUN_BACKTEST:
        cumulative_returns_history = utils.run_backtest(
            config=config,
            all_stock_prices=all_stock_prices,
            daily_returns=daily_returns,
            portfolio_tickers=portfolio_tickers,
            num_assets=num_assets,
            initial_guess=initial_guess,
            bounds=bounds,
            constraints=constraints,
            risk_free_rate=RISK_FREE_RATE,
            lambda_s=LAMBDA_S,
            lambda_k=LAMBDA_K,
            backtest_start_date=BACKTEST_START_DATE,
            backtest_end_date=BACKTEST_END_DATE,
            rebalancing_frequency=REBALANCING_FREQUENCY,
            historical_data_window_days=HISTORICAL_DATA_WINDOW_DAYS
        )
        
        # Quantify and print performance metrics
        if cumulative_returns_history is not None and not cumulative_returns_history.empty:
            print("\n--- Backtest Performance Metrics ---")
            backtest_metrics = utils.calculate_backtest_metrics(
                cumulative_returns_history,
                RISK_FREE_RATE,
                BACKTEST_START_DATE,
                BACKTEST_END_DATE
            )

            # Print a formatted table
            print(f"{'Strategy':<25} | {'CAGR':>8} | {'Ann. Vol.':>11} | {'Sharpe':>8} | {'Sortino':>9} | {'Max Drawdown':>14}")
            print("-" * 90)
            for strategy, metrics in backtest_metrics.items():
                cagr_str = f"{metrics['CAGR']:.2%}" if pd.notna(metrics['CAGR']) else "N/A"
                vol_str = f"{metrics['Annualised Volatility']:.2%}" if pd.notna(metrics['Annualised Volatility']) else "N/A"
                sharpe_str = f"{metrics['Sharpe Ratio']:.2f}" if pd.notna(metrics['Sharpe Ratio']) else "N/A"
                sortino_str = f"{metrics['Sortino Ratio']:.2f}" if pd.notna(metrics['Sortino Ratio']) else "N/A"
                drawdown_str = f"{metrics['Max Drawdown']:.2%}" if pd.notna(metrics['Max Drawdown']) else "N/A"
                
                print(f"{strategy.replace('_', ' '):<25} | {cagr_str:>8} | {vol_str:>11} | {sharpe_str:>8} | {sortino_str:>9} | {drawdown_str:>14}")
            print("-" * 90)
            
            utils.write_backtest_metrics_to_csv(backtest_output_filepath, backtest_metrics)
        else:
            print("\nBacktest did not produce valid cumulative returns history. Skipping performance quantification and backtest plots.")


   # --- Plotting all results ---
    utils.plot_all_results(
        static_annualised_correlation_matrix=static_annualised_correlation_matrix,
        output_dir=OUTPUT_DIR,
        static_results=static_results,
        dynamic_results=dynamic_results,
        individual_stock_metrics=individual_stock_metrics,
        portfolio_tickers=portfolio_tickers,
        static_portfolio_points_raw_mc=static_portfolio_points_raw_mc,
        dynamic_portfolio_points_raw_mc=dynamic_portfolio_points_raw_mc,
        feature_toggles=config['feature_toggles'],
        num_assets=num_assets,
        cumulative_returns_history=cumulative_returns_history if RUN_BACKTEST else None,
        backtest_end_date=BACKTEST_END_DATE if RUN_BACKTEST else None
    )

    # --- Save weights for optimised portfolios ---
    utils.save_optimisation_results(
        output_filepath=output_filepath,
        portfolio_tickers=portfolio_tickers,
        static_results=static_results,
        dynamic_results=dynamic_results,
        feature_toggles=config['feature_toggles']
    )
