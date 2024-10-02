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
                print(f'Retrieved bars from {bars_zip_path}')
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
        print(f'Saved bars to {bars_zip_path}')

    return df

def retrieve_quote_data(client: StockHistoricalDataClient, symbols: List[str], minute_intervals_dict: Dict[str, pd.Index], params: BacktestTouchDetectionParameters):
    quotes_data = {symbol: [] for symbol in symbols}
    n = len(symbols)
    for minute in tqdm(minute_intervals_dict[symbols[0]], desc='Fetching quotes'):
        minute_end = minute + timedelta(seconds=1)
        
        symbols_to_process = set(symbols)
        i = 3
        while symbols_to_process and i < 60:
            start_time = t2.time()
            minute_start = minute - timedelta(seconds=i)

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
                                quotes_data[symbol].append(symbol_df)
                            symbols_to_process.remove(symbol)

            i += 4
            elapsed_time = t2.time() - start_time
            
            # Calculate remaining sleep time to maintain 0.3 seconds between requests
            # 60/1000 = 0.06 sec if we have Elite Smart Router
            # 60/10000 = 0.006 sec if we have Algo Trader Plus
            # remaining_sleep_time = max(0, 0.3+(0.02*n) - elapsed_time)
            remaining_sleep_time = max(0.02*n, 0.3 - elapsed_time)
            
            t2.sleep(remaining_sleep_time)        # NOTE: may get throttled if len(symbols) > 1
            
            # t2.sleep(0.3)

    # Concatenate DataFrames for each symbol
    for symbol in tqdm(symbols, desc='convert quotes_data to df'):
        try:
            if quotes_data[symbol]:
                quotes_data[symbol] = pd.concat(quotes_data[symbol])
                print('sort_index...')
                quotes_data[symbol].sort_index(inplace=True)
            else:
                quotes_data[symbol] = pd.DataFrame()
        except Exception as e:
            print(f"{type(e).__qualname__}: {e}")
            raise e
        
    return quotes_data

def save_quote_data(symbol, qdf: pd.DataFrame, params: BacktestTouchDetectionParameters):
    directory = os.path.dirname(params.export_quotes_path)
    quotes_zip_path = os.path.join(directory, f'quotes_{symbol}_{params.start_date.strftime("%Y-%m-%d")}_{params.end_date.strftime("%Y-%m-%d")}.zip')
    os.makedirs(os.path.dirname(quotes_zip_path), exist_ok=True)
    
    with zipfile.ZipFile(quotes_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        with zip_file.open(os.path.basename(quotes_zip_path).replace('.zip', '.csv'), 'w') as csv_file:
            qdf.reset_index().to_csv(csv_file, index=False)
    print(f'Saved quotes to {quotes_zip_path}')

def retrieve_multi_symbol_data(params: BacktestTouchDetectionParameters, symbols: List[str]):
    assert params.end_date > params.start_date
    
    if isinstance(params.start_date, str):
        params.start_date = pd.to_datetime(params.start_date).tz_localize(ny_tz)
    if isinstance(params.end_date, str):
        params.end_date = pd.to_datetime(params.end_date).tz_localize(ny_tz)

    client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

    # Retrieve bar data and build minute_intervals_dict
    minute_intervals_dict = {}
    for symbol in tqdm(symbols, desc="Retrieving bar data"):
        start_time = t2.time()
        df = retrieve_bar_data(client, symbol, params)
        minute_intervals = df.index.get_level_values('timestamp')
        minute_intervals = minute_intervals[(minute_intervals.time >= time(9, 30)) & (minute_intervals.time <= time(16, 0))]
        minute_intervals_dict[symbol] = minute_intervals
        
        elapsed_time = t2.time() - start_time
        # Calculate remaining sleep time to maintain 0.3 seconds between requests
        # 60/1000 = 0.06 sec if we have Elite Smart Router
        # 60/10000 = 0.006 sec if we have Algo Trader Plus
        remaining_sleep_time = max(0, 0.3 - elapsed_time)
        t2.sleep(remaining_sleep_time)
        
    assert len(set([a.size for a in minute_intervals_dict.values()])) <= 1 # make sure len(minute_intervals) are the same

    # Retrieve quote data
    quotes_data = retrieve_quote_data(client, symbols, minute_intervals_dict, params)

    # Save quote data
    for symbol, qdf in quotes_data.items():
        save_quote_data(symbol, qdf, params)

    print("Data retrieval complete for all symbols.")



start_date = "2024-08-19 00:00:00"
end_date =   "2024-08-21 00:00:00"

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


symbols = ['AAPL', 'GOOGL', 'META', 'NVDA']  # Add your list of symbols here
retrieve_multi_symbol_data(params, symbols)