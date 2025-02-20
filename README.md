# Alpaca Trading System

A Python-based algorithmic trading system built on Alpaca's API, focusing on support/resistance level breakout strategies.

## Overview

This project implements an automated trading system that attempts to identify and trade support/resistance levels using a touch-based detection approach. While ultimately not achieving consistent profitability in backtesting, the codebase demonstrates:

- Integration with Alpaca's trading APIs
- Implementation of complex trading mechanics and order management
- Clean code organization and separation of concerns
- Detailed handling of market data processing

The system was inspired by [this article on support/resistance trading](https://medium.com/@paullenosky/i-have-created-an-indicator-that-actually-makes-money-unlike-any-other-indicator-i-have-ever-seen-fd7b36aba975) but attempts to fully automate the strategy. Through development, it became apparent that technical analysis patterns often require human discretion and may not translate well to fully automated systems.

## Project Structure

The main components are:

### Core Trading Logic
- `TradingStrategy.py` - Main strategy implementation and position management
- `TouchDetection.py` - Support/resistance level detection using price touches
- `TouchArea.py` - Represents and manages support/resistance zones
- `TradePosition.py` - Tracks individual trading positions and their metrics

### Data Management
- `TypedBarData.py` - Strongly-typed wrapper for OHLCV bar data
- `VolumeProfile.py` - Volume distribution analysis
- `MultiSymbolDataRetrieval.py` - Historical data fetching and processing
- `PositionMetrics.py` - Position performance tracking and analysis

### Live Trading (In Progress)
- `LiveTrader.py` - Live trading implementation (incomplete)
- `OrderTracker.py` - Order state management (draft)
- `SlippageData.py` - Execution quality analysis (draft)

## Key Features

- Support/resistance detection through price touch analysis
- Volume profile analysis for level validation
- Position sizing based on volatility and volume
- Comprehensive position metrics and performance tracking
- Detailed handling of quote data for execution
- Framework for analyzing trading costs and slippage

## Technical Implementation Details

### Trading Mechanics
- Proper handling of margin requirements and buying power
- Accurate position cost basis tracking
- Support for both long and short positions
- Detailed P&L calculations including fees
- Price-time priority simulation

### Data Processing
- Real-time and historical data integration
- Quote aggregation and VWAP calculations
- Volatility measurements (ATR/MTR)
- Volume and trade count analysis

### Architecture
- Clear separation between strategy and execution
- Strong typing for critical data structures
- Comprehensive logging and error handling
- Efficient data storage and retrieval

## Development Status

This project is primarily a demonstration of trading system development and is not currently achieving profitable results in backtesting. Key learnings include:

- Technical analysis patterns that appear clear to human traders can be difficult to fully automate
- Support/resistance trading may require discretionary judgment that's hard to codify
- Market behavior around technical levels is more complex than simple breakout patterns suggest

The live trading components remain incomplete as development was paused after backtesting results did not justify deployment.

## Installation

```bash
pip install -r requirements.txt
```

Primary dependencies:
- alpaca-py
- pandas
- numpy
- numba

## Configuration

Create a `config.toml` file with your Alpaca API credentials:
```toml
[paper]
key = "YOUR_API_KEY"
secret = "YOUR_API_SECRET"
```

Create a `.env` file and specify the account name:
```env
ACCOUNTNAME=paper
```

## Usage

Refer to Jupyter Notebooks (*.ipynb) in trading_bot folder for examples and usage.

## Contributing

While this project is not actively maintained for trading purposes, contributions that improve code quality or documentation are welcome.

## Disclaimer

This software is for educational purposes only. It is not financial advice and should not be used for actual trading without extensive testing and modification.