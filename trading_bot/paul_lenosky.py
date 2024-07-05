import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import toml
import os
from dotenv import load_dotenv

import time

load_dotenv(override=True)
livepaper = os.getenv('LIVEPAPER')
config = toml.load('../config.toml')

# # Initialize the Alpaca API
api = tradeapi.REST(config[livepaper]['key'], config[livepaper]['secret'], config[livepaper]['endpoint'], api_version='v2')


from trading import is_market_open, get_historical_data


# Touch Detection
def detect_support_resistance(df, min_touches=3, tolerance=0.02):
    levels = []
    touch_counts = {}

    for i in range(len(df)):
        for j in range(i+1, len(df)):
            level = (df['close'][i] + df['close'][j]) / 2
            
            if abs(df['close'][i] - df['close'][j]) < tolerance * df['close'][i]:
                if level not in touch_counts:
                    touch_counts[level] = 0
                touch_counts[level] += 1

    for level, touches in touch_counts.items():
        if touches >= min_touches:
            levels.append(level)
    
    levels = sorted(levels)
    return levels


# Volatility Analysis
def calculate_volatility(df, period=14):
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=period).std() * np.sqrt(252)
    return df['volatility'].iloc[-1]

# Alert/Logging System
def log_alert(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

# Placing Orders
def place_order(symbol, buy_price, stop_loss_price):
    try:
        buy_order = api.submit_order(
            symbol=symbol,
            qty=1,
            side='buy',
            type='stop',
            stop_price=buy_price,
            time_in_force='gtc',
        )
        stop_order = api.submit_order(
            symbol=symbol,
            qty=1,
            side='sell',
            type='stop',
            stop_price=stop_loss_price,
            time_in_force='gtc',
        )
        log_alert(f"Order placed: Buy at {buy_price}, Stop at {stop_loss_price}")
        return buy_order.id, stop_order.id
    except Exception as e:
        log_alert(f"Failed to place order: {e}")
        return None, None


def check_order_status(order_id):
    try:
        order = api.get_order(order_id)
        return order.status
    except Exception as e:
        log_alert(f"Failed to check order status: {e}")
        return None




# Main function
def run_trading_bot(symbol, min_touches=3, tolerance=0.02):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    df = get_historical_data(symbol, start_date, end_date)
    
    levels = detect_support_resistance(df, min_touches, tolerance)
    log_alert(f"Detected levels: {levels}")
    
    volatility = calculate_volatility(df)
    log_alert(f"Calculated volatility: {volatility}")

    for level in levels:
        high_price = level
        low_price = level - (level * 0.02)
        
        buy_price = high_price + 0.0001

        while not is_market_open():
            log_alert("Market is closed. Waiting for market to open...")
            time.sleep(60)  # Sleep for a minute before checking again

        buy_order_id, stop_order_id = place_order(symbol, buy_price, low_price)
        
        if buy_order_id and stop_order_id:
            while True:
                stop_order_status = check_order_status(stop_order_id)
                
                if stop_order_status == 'filled':
                    log_alert("Stop order executed. Placing new buy order.")
                    buy_order_id, stop_order_id = place_order(symbol, buy_price, low_price)
                
                time.sleep(60)  # Sleep for a minute before checking again
            break



# # Example usage
# run_trading_bot('AAPL')