import os
import zipfile
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from datetime import datetime, time, timedelta
from tqdm import tqdm
import time as t2
from typing import List, Tuple, Optional, Dict
import math
import numpy as np
from TouchDetection import BacktestTouchDetectionParameters, clean_quotes_data, fill_missing_data

from zoneinfo import ZoneInfo

import toml
from dotenv import load_dotenv

ny_tz = ZoneInfo("America/New_York")
STANDARD_DATETIME_STR = '%Y-%m-%d %H:%M:%S'
ROUNDING_DECIMAL_PLACES = 10  # Choose an appropriate number of decimal places

load_dotenv(override=True)
livepaper = os.getenv('LIVEPAPER')
config = toml.load('../config.toml')

# Replace with your Alpaca API credentials
API_KEY = config[livepaper]['key']
API_SECRET = config[livepaper]['secret']


# 60/200 = 0.3 second if we're using free account
# 60/1000 = 0.06 sec if we have Elite Smart Router
# 60/10000 = 0.006 sec if we have Algo Trader Plus

# SLEEP_TIME = 0.3
# SLEEP_TIME = 0.06
SLEEP_TIME = 0.006





import logging
def setup_logger(log_level=logging.INFO):
    logger = logging.getLogger('MultiSymbolDataRetrieval')
    logger.setLevel(log_level)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add a new handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger(logging.INFO)

def log(message, level=logging.INFO):
    logger.log(level, message)



def get_segment_sizes(dfs: List[pd.DataFrame]) -> List[int]:
    """
    Calculate the size of each DataFrame in the list.
    
    :param dfs: List of DataFrames
    :return: List of sizes (number of rows) for each DataFrame
    """
    return [df.shape[0] for df in dfs]

def divide_segments(dfs: List[pd.DataFrame], sizes: List[int], num_groups: int) -> List[List[int]]:
    """
    Divide the segments into groups for concatenation.
    
    :param dfs: List of DataFrames
    :param sizes: List of sizes for each DataFrame
    :param num_groups: Number of groups to divide into
    :return: List of lists containing indices for each group
    """
    total_size = sum(sizes)
    target_size = total_size / num_groups if num_groups > 0 else 0
    
    groups = [[] for _ in range(num_groups)]
    group_sizes = [0] * num_groups
    current_group = 0
    
    for i, size in enumerate(sizes):
        if current_group < num_groups - 1 and group_sizes[current_group] + size > target_size:
            current_group += 1
        groups[current_group].append(i)
        group_sizes[current_group] += size
    
    return [group for group in groups if group]  # Remove empty groups

def custom_concat(dfs: List[pd.DataFrame], group_size: int = 10) -> pd.DataFrame:
    """
    Custom concatenation function using a flexible grouping approach.
    
    :param dfs: List of DataFrames to concatenate
    :param group_size: Number of DataFrames to concatenate in each group
    :return: Concatenated DataFrame
    """
    # Handle edge cases
    if len(dfs) == 0:
        return pd.DataFrame()
    elif len(dfs) == 1:
        return dfs[0]
    elif len(dfs) <= group_size:
        return pd.concat(dfs, axis=0)
    
    sizes = get_segment_sizes(dfs)
    num_groups = math.ceil(len(dfs) / group_size)
    groups = divide_segments(dfs, sizes, num_groups)
    
    concatenated_groups = []
    for group in groups:
        group_dfs = [dfs[i] for i in group]
        if group_dfs:
            concatenated_groups.append(pd.concat(group_dfs, axis=0))
    
    if not concatenated_groups:
        return pd.DataFrame()
    elif len(concatenated_groups) == 1:
        return concatenated_groups[0]
    else:
        return custom_concat(concatenated_groups, group_size)



def retrieve_bar_data(client: StockHistoricalDataClient, symbol, params: BacktestTouchDetectionParameters):
    directory = os.path.dirname(params.export_bars_path)
    bars_zip_path = os.path.join(directory, f'bars_{symbol}_{params.start_date.strftime("%Y-%m-%d")}_{params.end_date.strftime("%Y-%m-%d")}.zip')
    os.makedirs(os.path.dirname(bars_zip_path), exist_ok=True)

    if params.use_saved_bars and os.path.isfile(bars_zip_path):
        with zipfile.ZipFile(bars_zip_path, 'r') as zip_file:
            with zip_file.open(os.path.basename(bars_zip_path).replace('.zip', '.csv')) as csv_file:
                df = pd.read_csv(csv_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(ny_tz)
                df.set_index(['symbol', 'timestamp'], inplace=True)
                log(f'Retrieved bars from {bars_zip_path}')
    else:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=params.start_date.tz_convert('UTC'),
            end=params.end_date.tz_convert('UTC'),
            adjustment=Adjustment.ALL,
        )
        df = client.get_stock_bars(request_params).df
        df.index = df.index.set_levels(
            df.index.get_level_values('timestamp').tz_convert(ny_tz),
            level='timestamp'
        )
        df.sort_index(inplace=True)
        df = fill_missing_data(df)
        
        with zipfile.ZipFile(bars_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            with zip_file.open(os.path.basename(bars_zip_path).replace('.zip', '.csv'), 'w') as csv_file:
                df.reset_index().to_csv(csv_file, index=False)
        log(f'Saved bars to {bars_zip_path}')

    return df

from numpy.random import MT19937, RandomState, SeedSequence
import hashlib
def retrieve_quote_data(client: StockHistoricalDataClient, symbols: List[str], minute_intervals_dict: Dict[str, pd.Index], params: BacktestTouchDetectionParameters, sample_size: int):
    quotes_data = {symbol: [] for symbol in symbols}
    n = len(symbols)
    
    after_sec = 1
    before_sec_search = [float(round(a,2)) for a in np.logspace( 0, np.log2(60-after_sec), 6, base=2)]
    log(f'for each minute, searching previous {before_sec_search} seconds to {after_sec} after')
        
    for minute in tqdm(minute_intervals_dict[symbols[0]], desc='Fetching quotes'):
        minute_end = minute + timedelta(seconds=after_sec)
        minute_of_day = minute.hour * 60 + minute.minute # for seed
        symbols_to_process = set(symbols)

        for before_sec in before_sec_search:
            if not symbols_to_process:
                break
            # start_time = t2.time()
            minute_start = minute - timedelta(seconds=before_sec)

            request_params = StockQuotesRequest(
                symbol_or_symbols=list(symbols_to_process),
                start=minute_start.tz_convert('UTC'),
                end=minute_end.tz_convert('UTC'),
            )
            qdf0 = client.get_stock_quotes(request_params).df

            if not qdf0.empty:
                for symbol in list(symbols_to_process):
                    symbol_df = qdf0.loc[qdf0.index.get_level_values('symbol') == symbol]
                    if not symbol_df.empty:
                        symbol_df = clean_quotes_data(symbol_df)
                        t = symbol_df.index.get_level_values('timestamp')
                        latest_before_earlier = symbol_df[t < minute].index.get_level_values('timestamp').max()
                        if not pd.isna(latest_before_earlier):
                            symbol_df = symbol_df[t >= latest_before_earlier]
                            if not symbol_df.empty:
                                seed = int(hashlib.sha256(f"{symbol}_{minute_of_day}".encode()).hexdigest(), 16) % (2**32)
                                rs = RandomState(MT19937(SeedSequence(seed)))
                                if len(symbol_df) > sample_size:
                                    symbol_df = symbol_df.sample(n=sample_size, replace=False, random_state=rs) # uniform sampling without replacement
                                quotes_data[symbol].append(symbol_df)
                            symbols_to_process.remove(symbol)

            # elapsed_time = t2.time() - start_time
            # remaining_sleep_time = max(0, 0.3+(0.02*n) - elapsed_time)
            # remaining_sleep_time = max(0.02*n, 0.3 - elapsed_time)
            # t2.sleep(remaining_sleep_time)        # NOTE: may get throttled if len(symbols) > 1
            
            # t2.sleep(SLEEP_TIME)

        if symbols_to_process:
            log(f"Not all symbols processed. Remaining: {symbols_to_process}", level=logging.WARNING)
            
    # Concatenate DataFrames for each symbol
    for symbol in symbols:
        try:
            if quotes_data[symbol]:
                log(f'{symbol} concat...')
                # quotes_data[symbol] = pd.concat(quotes_data[symbol])
                quotes_data[symbol] = custom_concat(quotes_data[symbol])
                # log('sort_index...')
                log(f'...done ({quotes_data[symbol].shape[0]} total rows)')
                quotes_data[symbol].sort_index(inplace=True)
                # log('saving...')
                save_quote_data(symbol, quotes_data[symbol], params)
                del quotes_data[symbol]
            else:
                log(f'{symbol} data not found')
        except Exception as e:
            log(f"{type(e).__qualname__}: {e}", level=logging.ERROR)
            raise e
        
    # return quotes_data

def save_quote_data(symbol, qdf: pd.DataFrame, params: BacktestTouchDetectionParameters):
    directory = os.path.dirname(params.export_quotes_path)
    quotes_zip_path = os.path.join(directory, f'quotes_{symbol}_{params.start_date.strftime("%Y-%m-%d")}_{params.end_date.strftime("%Y-%m-%d")}.zip')
    os.makedirs(os.path.dirname(quotes_zip_path), exist_ok=True)
    
    with zipfile.ZipFile(quotes_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        with zip_file.open(os.path.basename(quotes_zip_path).replace('.zip', '.csv'), 'w') as csv_file:
            qdf.reset_index().to_csv(csv_file, index=False)
    log(f'Saved quotes to {quotes_zip_path}')

def retrieve_multi_symbol_data(params: BacktestTouchDetectionParameters, symbols: List[str], sample_size: int):
    assert params.end_date > params.start_date
    
    if isinstance(params.start_date, str):
        params.start_date = pd.to_datetime(params.start_date).tz_localize(ny_tz)
    if isinstance(params.end_date, str):
        params.end_date = pd.to_datetime(params.end_date).tz_localize(ny_tz)

    client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

    # Retrieve bar data and build minute_intervals_dict
    minute_intervals_dict = {}
    for symbol in tqdm(symbols, desc="Retrieving bar data"):
        # start_time = t2.time()
        df = retrieve_bar_data(client, symbol, params)
        minute_intervals = df.index.get_level_values('timestamp')
        minute_intervals = minute_intervals[(minute_intervals.time >= time(9, 31)) & (minute_intervals.time <= time(15, 59))]
        minute_intervals_dict[symbol] = minute_intervals
        
        # elapsed_time = t2.time() - start_time
        # remaining_sleep_time = max(0, SLEEP_TIME - elapsed_time)
        # t2.sleep(remaining_sleep_time)
        t2.sleep(SLEEP_TIME)
        
    assert len(set([a.size for a in minute_intervals_dict.values()])) <= 1 # make sure len(minute_intervals) are the same

    # Retrieve quote data
    # quotes_data = 
    retrieve_quote_data(client, symbols, minute_intervals_dict, params, sample_size)

    # # Save quote data
    # for symbol, qdf in quotes_data.items():
    #     save_quote_data(symbol, qdf, params)

    log("Data retrieval complete for all symbols.")



start_date = "2024-08-01 00:00:00"
end_date =   "2024-10-01 00:00:00"

symbol = '' # placeholder

# Usage example:
params = BacktestTouchDetectionParameters(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    atr_period=15,
    level1_period=15,
    multiplier=1.4,
    min_touches=3,
    start_time=None,
    end_time='15:55',
    use_median=True,
    touch_area_width_agg=None,
    use_saved_bars=True,
    rolling_avg_decay_rate=0.85,
    export_bars_path=f'bars/bars_{symbol}_{start_date.split()[0]}_{end_date.split()[0]}.csv',
    export_quotes_path=f'quotes/quotes_{symbol}_{start_date.split()[0]}_{end_date.split()[0]}.csv'
)


# symbols = ['AAPL', 'GOOGL', 'META', 'NVDA']  # Add your list of symbols here
symbols = ['AAPL', 'GOOGL', 'NVDA']
# symbols = ['AAPL']
# TODO: implement parallel processing for concat before trying something like NVDA
# symbols = ['NVDA']  # Add your list of symbols here



retrieve_multi_symbol_data(params, symbols, sample_size=100)