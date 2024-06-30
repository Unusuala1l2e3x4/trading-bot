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



# Fetch historical data
def get_historical_data(symbol, start_date, end_date, timeframe='day'):
    barset = api.get_barset(symbol, timeframe, start=start_date, end=end_date)
    return barset[symbol].df

# Touch Detection
def detect_support_resistance(df, min_touches=2, tolerance=0.02):
    levels = {}
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if abs(df['close'][i] - df['close'][j]) < tolerance:
                level = round(df['close'][i], 2)
                if level not in levels:
                    levels[level] = 0
                levels[level] += 1

    return {level: count for level, count in levels.items() if count >= min_touches}

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
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='buy',
            type='stop',
            stop_price=buy_price,
            time_in_force='gtc',
        )
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='sell',
            type='stop',
            stop_price=stop_loss_price,
            time_in_force='gtc',
        )
        log_alert(f"Order placed: Buy {symbol} at {buy_price}, Stop Loss at {stop_loss_price}")
    except Exception as e:
        log_alert(f"Error placing order: {e}")

# Main function
def run_trading_bot(symbol):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    df = get_historical_data(symbol, start_date, end_date)
    
    # Detect support and resistance levels
    levels = detect_support_resistance(df)
    log_alert(f"Detected levels: {levels}")
    
    # Calculate volatility
    volatility = calculate_volatility(df)
    log_alert(f"Calculated volatility: {volatility}")

    # Determine the trading action
    for level, touches in levels.items():
        high_price = level
        low_price = level - (level * 0.02)  # Example stop loss 2% below level
        
        if touches >= 2:
            buy_price = high_price + 0.0001
            place_order(symbol, buy_price, low_price)
            break

# # Example usage
# run_trading_bot('AAPL')