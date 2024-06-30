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


def get_account_details():
    """
    Fetch account details such as equity, cash balance, buying power, etc.
    """
    try:
        account = api.get_account()
        return {
            'equity': account.equity,
            'cash': account.cash,
            'buying_power': account.buying_power,
            'portfolio_value': account.portfolio_value,
            'status': account.status
        }
    except Exception as e:
        print(f"Error fetching account details: {e}")
        return None

def get_current_positions():
    """
    Retrieve current positions held in the account.
    """
    try:
        positions = api.list_positions()
        positions_list = []
        for position in positions:
            positions_list.append({
                'symbol': position.symbol,
                'qty': position.qty,
                'market_value': position.market_value,
                'cost_basis': position.cost_basis,
                'unrealized_pl': position.unrealized_pl
            })
        return positions_list
    except Exception as e:
        print(f"Error fetching current positions: {e}")
        return None

def get_order_history(status='all'):
    """
    Fetch the order history.
    :param status: Order status filter (default: 'all', options: 'open', 'closed', 'all')
    """
    try:
        orders = api.list_orders(status=status, limit=100)
        orders_list = []
        for order in orders:
            orders_list.append({
                'id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'filled_qty': order.filled_qty,
                'status': order.status,
                'submitted_at': order.submitted_at
            })
        return orders_list
    except Exception as e:
        print(f"Error fetching order history: {e}")
        return None
    
    