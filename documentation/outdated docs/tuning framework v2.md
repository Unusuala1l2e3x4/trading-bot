I'm developing an advanced backtesting framework in Python for a sophisticated intraday trading strategy. The framework needs to test multiple stocks, timeframes, and parameter combinations while recording various stock metrics over time (i.e. every day, week, month). My strategy uses 1-minute bars and includes complex features such as variable leverage, sub-position management, accurate transaction cost modeling, partial entries/exits, and dynamic risk management.

I plan to use historical data from both Alpaca's market data API and FirstRate Data, which provides 1-minute bars for over 13,000 US stocks from 2000 to present, including both active and delisted stocks. (paid)

Key requirements for the framework:

1. Efficient data handling and storage for large datasets (13,000+ stocks, 24+ years of 1-minute data).
2. Parallelization for faster backtesting across multiple stocks and parameter combinations.
3. Flexible parameter tuning capabilities.
4. Comprehensive metrics calculation and recording, including stock-specific metrics (e.g., volatility, volume, etc) and strategy performance metrics (e.g., average ROE %, winrate, etc).
5. Ability to simulate realistic market conditions, including stock borrowing for shorts and variable liquidity.
6. Integration with my existing strategy logic, which includes leverage handling, sub-position management, transaction cost calculation, partial entries/exits, and risk management.
7. Results analysis and visualization tools to identify optimal parameters and suitable stocks for the strategy.
8. Capability to generate a data-driven stock screener based on historical performance and stock characteristics.

I have existing code for the strategy logic, which I can provide relevant parts of. The goal is to create a scalable, efficient backtesting framework that can handle this complex strategy across a large universe of stocks and timeframes, ultimately leading to a robust stock selection and parameter optimization system.

Please provide a high-level design for this backtesting framework, including recommendations for data structures, libraries, and architecture to meet these requirements efficiently. Also, suggest any potential optimizations or considerations specific to intraday strategies using 1-minute data.