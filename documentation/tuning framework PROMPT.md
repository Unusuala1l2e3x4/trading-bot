i want to create a larger backtesting framework in Python to tune parameters and test many more stocks and timeframes. while backtesting, i also intend to record stock metrics (i.e. volatility, etc) over time within my backtesting timeframes. as long as i record stock metrics, best parameter value combinations, and backtest results (i.e. based on profit_loss_pct for overall, only longs/shorts, only winning/losing trades, etc), i should be able to make a data-driven stock screener used to identify stocks and parameters that my strategy may most likely profit from in a given moment.

alpaca's market data API only goes back a handful of years so i might pay for some data. Info about the data source i found suitable:
`FirstRate Data
Bundle: Stocks Complete (10,000+ Tickers)
Jan 2000 - Aug 2024, 13306 Tickers
This bundle contains 1-minute, 5-minute, 30-minute, 1-hour, and 1-day historical intraday data for :
6306 most liquid US stocks (includes all active Russell 3000, S&P500, Nasdaq 100, and DJI stocks)
Over 7000 delisted stocks
All prices and volumes are adjusted for dividends and splits (unadjusted data is also included). Out-of-hours trades are included.
New tickers (from IPOs, spinoffs etc) are added at the end of each week`


my strategy solely uses 1-minute bars. once i fully adapt my code for a live setting, I will stream 1-minute bars in real time using alpaca and trade based on that.
without needing to show you my full code, here is further documentation/notes about my approach (already developed):
`1. Leverage handling:
   - Implements variable leverage (times_buying_power) up to 4x
   - Adjusts position sizing based on marginability of securities
   - Handles cases where times_buying_power < 1, effectively limiting balance usage

2. Sub-position management:
   - Uses 2 sub-positions when leverage > 2x in order to follow initial margin requirements while maximizing usage of buying power
   - Implements proportional entry/exit for sub-positions, maintaining an even number of total shares

3. Transaction cost calculation:
   - Accurately models FINRA TAF and SEC fees
   - Estimates entry costs to prevent insufficient balance for fees

4. Partial entries and exits:
   - Implements sophisticated partial entry/exit logic
   - Adjusts positions based on market movements and strategy rules

5. OHLC vs Close price strategy:
   - Found that using only close prices for exits yielded better results
   - Still use stop market buys for initial entries

6. Rebalancing and P&L calculation:
   - Accurately tracks realized and unrealized P&L
   - Implements rebalancing after partial entries/exits

7. Risk management:
   - Implements trailing stops and take-profit mechanisms
   - Adjusts position sizes based on available balance and estimated transaction costs

8. Flexible parameters:
   - Implements flexible backtesting with parameters for leverage, margin usage, and trading directions (long/short), soft and hard sell times, etc.`

using both previous and new info ive provided as context, please write me an effective prompt (assuming i also attach the relevants parts of my code) to use in a fresh chat to jumpstart my efforts to construct my intended backtesting framework.
