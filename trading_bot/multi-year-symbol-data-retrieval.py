import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Tuple
import numpy as np
import math
from tqdm import tqdm
from trading_bot.MultiSymbolDataRetrieval import retrieve_multi_symbol_data
from trading_bot.TouchDetection import BacktestTouchDetectionParameters

def retrieve_multi_year_symbol_data(symbols: List[str], years: List[int], symbols_batch_size: int, interval_months: int, params: BacktestTouchDetectionParameters, 
                                    first_seconds_sample: int = np.inf, last_seconds_sample: int = np.inf):
    """
    Retrieve and save quotes data for multiple symbols and years in specified month intervals.
    
    :param symbols: List of stock symbols
    :param years: List of years to retrieve data for
    :param symbols_batch_size: Number of symbols to process in each batch
    :param interval_months: Number of months for each interval (must be a divisor of 12)
    :param params: BacktestTouchDetectionParameters object
    :param sample_size: Sample size for quote data retrieval
    """
    assert 12 % interval_months == 0, "interval_months must be a divisor of 12"
    
    # Group symbols into batches of symbols_batch_size
    symbol_batches = [symbols[i:i+symbols_batch_size] for i in range(0, len(symbols), symbols_batch_size)]
    
    for year in tqdm(years, desc="Processing years"):
        for symbol_batch in tqdm(symbol_batches, desc=f"Processing symbol batches for {year}", leave=False):
            print(f"Processing symbol batch: {symbol_batch}")
            for interval_start in range(0, 12, interval_months):
                start_date = datetime(year, interval_start + 1, 1)
                end_date = start_date + relativedelta(months=interval_months)
                
                # Ensure we don't go into the next year
                if end_date.year > year:
                    end_date = datetime(year + 1, 1, 1)
                
                # Update params with the current date range
                params.start_date = start_date.strftime("%Y-%m-%d")
                params.end_date = end_date.strftime("%Y-%m-%d")# - timedelta(days=1)  # Subtract one day to get the last day of the previous month
                
                try:
                    print(f"Retrieving data from {params.start_date} to {params.end_date}")
                    # Retrieve quote data for the current batch of symbols
                    retrieve_multi_symbol_data(params, symbol_batch, first_seconds_sample=first_seconds_sample, last_seconds_sample=last_seconds_sample, return_data=False)
                except Exception as e:
                    print(f"multi-year-symbol-data-retrieval - Error retrieving data for {symbol_batch} from {params.start_date} to {params.end_date}: {str(e)}")

if __name__=="__main__":
    # Example usage:
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA']
    # symbols = ['NVDA', 'GOOGL', 'AAPL', 'AMZN', 'MSFT', 'META'] # ordered  by quotes activity, roughly
    # symbols = ['AAPL', 'GOOGL', 'NVDA', 'META']
    years = [2022, 2023]
    symbols_batch_size = 1 #
    interval_months = 4

    # Usage example (most params are just placeholders for this module):
    params = BacktestTouchDetectionParameters(
        symbol='',
        start_date='',
        end_date='',
        atr_period=15,
        level1_period=15,
        multiplier=1.4,
        min_touches=3,
        start_time=None,
        end_time='15:55',
        use_median=True,
        touch_area_width_agg=None,
        
        ema_span=15,
        price_ema_span=26,
        
        export_bars_path=f'bars/',
        export_quotes_path=f'quotes/'
    )

    # retrieve_multi_year_symbol_data(symbols, years, symbols_batch_size, interval_months, params, first_seconds_sample=100, last_seconds_sample=200)
    retrieve_multi_year_symbol_data(symbols, years, symbols_batch_size, interval_months, params)