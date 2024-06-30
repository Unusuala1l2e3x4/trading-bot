from trading_bot.account import get_account_details, get_current_positions
from trading_bot.trading import submit_market_order

# import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import toml
import os
from dotenv import load_dotenv

# load_dotenv(override=True)
# livepaper = os.getenv('LIVEPAPER')
# config = toml.load('../config.toml')

# # Initialize the Alpaca API
# api = tradeapi.REST(config[livepaper]['key'], config[livepaper]['secret'], config[livepaper]['endpoint'], api_version='v2')


if __name__ == '__main__':
    # Get account details
    account_details = get_account_details()
    if account_details:
        print("Account Details:", account_details)

    # Get current positions
    positions = get_current_positions()
    if positions:
        print("Current Positions:", positions)

    # Submit a market order
    order_response = submit_market_order('AAPL', 1, 'buy')
    if order_response:
        print("Order Response:", order_response)
