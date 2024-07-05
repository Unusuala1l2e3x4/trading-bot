import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import toml
import os
from dotenv import load_dotenv


load_dotenv(override=True)
livepaper = os.getenv('LIVEPAPER')
config = toml.load('../config.toml')

# Initialize the Alpaca API
api = tradeapi.REST(config[livepaper]['key'], config[livepaper]['secret'], config[livepaper]['endpoint'], api_version='v2')

def is_market_open():
    clock = api.get_clock()
    return clock.is_open

# Fetch historical data
def get_historical_data(symbol, start_date, end_date, timeframe='day'):
    """
    Fetch historical data for a given symbol and timeframe using the Alpaca API.
    
    :param symbol: The stock symbol to fetch data for.
    :param start_date: The start date for fetching historical data.
    :param end_date: The end date for fetching historical data.
    :param timeframe: The timeframe for the bars ('minute', '1Min', '5Min', '15Min', 'day', '1D').
    :return: A pandas DataFrame containing the historical data.
    """
    timeframe_map = {
        'minute': '1Min',
        '1Min': '1Min',
        '5Min': '5Min',
        '15Min': '15Min',
        'day': '1D',
        '1D': '1D'
    }
    
    tf = timeframe_map.get(timeframe, '1D')
    
    bars = api.get_bars(
        symbol,
        tf,
        start=start_date,
        end=end_date
    ).df

    # Ensure the DataFrame is sorted by time
    bars = bars.sort_index()
    
    return bars


def submit_market_order(symbol, qty, side='buy'):
    """
    Submit a market order.
    :param symbol: The stock symbol to trade.
    :param qty: The number of shares to trade.
    :param side: 'buy' or 'sell' (default: 'buy')
    """
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        return {
            'id': order.id,
            'symbol': order.symbol,
            'qty': order.qty,
            'status': order.status
        }
    except Exception as e:
        print(f"Error submitting market order: {e}")
        return None