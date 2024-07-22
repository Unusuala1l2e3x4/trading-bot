#### Q
turns out excluding open/high/low logic for selling actually gave much better results. including them caused an overall decrease to happen much more frequently

#### A
This is a very interesting finding, and it highlights the importance of thorough testing and analysis in trading strategy development. Let's explore why excluding the open/high/low (OHLC) logic for selling might be yielding better results:

1. Reduced Noise:
   - By only using close prices for exits, you're potentially filtering out intraday noise that could trigger premature exits.
   - This approach might be better aligned with the overall trend, rather than reacting to short-term fluctuations.

2. Momentum Capture:
   - Holding positions until the close might allow the strategy to capture more of a trending move, especially if the asset tends to trend throughout the day.

3. Avoiding Whipsaws:
   - OHLC data can sometimes lead to more frequent exits due to intraday price swings, which might result in missing out on larger moves.

4. Strategy Alignment:
   - Your entry conditions and overall strategy might be better suited to a close-to-close approach, rather than reacting to intraday price movements.

5. Market Characteristics:
   - The specific assets or markets you're trading might have characteristics that make close prices more reliable for decision-making than intraday highs and lows.

6. Reduced Transaction Costs:
   - Fewer exits likely mean lower transaction costs, which can significantly impact overall profitability, especially when compounded over many trades.

7. Time Frame Alignment:
   - If your strategy is based on daily data or longer-term trends, using close prices for exits might be more consistent with this time frame.

Given these observations, here are some recommendations:

1. Validate Across Different Market Conditions:
   - Ensure that this improved performance is consistent across different market conditions (bull, bear, sideways markets).

2. Time Frame Analysis:
   - Test if this close-only exit strategy performs consistently across different time frames (hourly, daily, weekly).

3. Entry Optimization:
   - Since the exit strategy has changed, revisit your entry criteria to ensure they're optimally paired with the close-only exit approach.

4. Stop Loss and Take Profit Levels:
   - Adjust your stop loss and take profit levels to align with the close-only exit strategy. They might need to be wider to accommodate intraday fluctuations.

5. Risk Management:
   - With potentially longer hold times, ensure your risk management strategy is robust enough to handle larger adverse moves.

6. Partial Exit Strategy:
   - Consider implementing a partial exit strategy based on closes if you haven't already. This could help lock in some profits while still allowing for further upside.

7. After-Hours Movements:
   - Be aware of how after-hours movements might affect your strategy, especially if you're trading instruments that have significant after-hours activity.

8. Backtesting and Forward Testing:
   - Conduct extensive backtesting over longer periods and consider forward testing or paper trading to validate the strategy's performance.

9. Monitoring and Adjustment:
   - Implement a system to monitor the strategy's performance and be prepared to make adjustments if market conditions change.

10. Hybrid Approach:
    - Consider testing a hybrid approach where you use OHLC data for entries but close prices for exits, or vice versa.

The fact that excluding OHLC logic for exits has improved your results is a valuable insight. It demonstrates the importance of not over-complicating a strategy and the potential benefits of a more patient, trend-following approach. Continue to refine and test this approach, always keeping in mind the balance between capturing larger moves and managing risk effectively.