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