To set effective limit prices that maximize the likelihood of fills within 10 to 50 seconds—without straying too far from the market price—you can adopt the following strategy, which is both practical for real trading and backtesting given your data constraints:

### 1. Utilize Current Best Bid and Ask Prices

**For Buy Orders (Long Positions):**

- **Place Limit Orders at the Current Ask Price:**
  - By setting your buy limit order at the current **ask_price**, you're willing to pay the lowest price sellers are currently accepting. This increases the likelihood of immediate or quick fills.
  - **Example:** If the current ask price is \$50.00, set your buy limit order at \$50.00.

- **Add a Small Buffer if Necessary:**
  - In fast-moving markets or when immediate execution is critical, consider adding a small increment to the ask price (e.g., \$0.01 or a small percentage of the spread).
  - **Example:** Set your limit price at \$50.01 instead of \$50.00 to account for rapid price changes.

**For Sell Orders (Short Positions):**

- **Place Limit Orders at the Current Bid Price:**
  - By setting your sell limit order at the current **bid_price**, you're willing to sell at the highest price buyers are currently offering.
  - **Example:** If the current bid price is \$50.00, set your sell limit order at \$50.00.

- **Subtract a Small Buffer if Necessary:**
  - In volatile markets, consider setting your limit price slightly below the bid price to increase fill probability.
  - **Example:** Set your limit price at \$49.99.

### 2. Leverage Recent Quotes Data

- **Analyze the Last Few Seconds of Data:**
  - Before placing orders at the start of each minute, examine the recent **bid_price** and **ask_price** changes to gauge market momentum.
  - **Compute Averages and Volatility:**
    - Calculate the average bid and ask prices over the last few seconds.
    - Assess the spread (ask_price - bid_price) to understand market tightness.

- **Adjust Limit Prices Based on Market Conditions:**
  - **In Tight Markets (Small Spread):**
    - Placing orders at the bid or ask prices is often sufficient.
  - **In Volatile Markets (Large Spread or Rapid Price Changes):**
    - Adjust your limit prices more aggressively by increasing the buffer.

### 3. Account for Liquidity Constraints

- **Consider Bid and Ask Sizes:**
  - Use **bid_size** and **ask_size** to estimate available liquidity at the current prices.
  - **Example:** A large ask_size at the current ask_price suggests ample liquidity for buy orders.

- **Avoid Overtrading Relative to Market Volume:**
  - Continue using your safeguard of not trading more than 1% of the moving average of market volume, but enhance it by incorporating real-time **bid_size** and **ask_size**.

### 4. Backtesting Approach

- **Simulate Fills Based on Quotes Data:**
  - Since you have quotes data with timestamps, assume that your limit order is filled if the market reaches your limit price within your time window (10 to 50 seconds).
  - **Example:** If your buy limit order is at \$50.00 and the **ask_price** drops to \$50.00 within the next 30 seconds, assume the order is filled.

- **Use 1-Minute Bars for Approximation:**
  - If quotes data is limited, use the high and low prices from the 1-minute bars to estimate if your limit price was reachable.

### 5. Adjust for Data Limitations

- **Acknowledge Missing Dark Pool Data:**
  - Recognize that your data may not reflect all market activity, especially from dark pools or undisplayed liquidity.
  - **Treat Data as a Representative Sample:**
    - While not exhaustive, Level 1 data from liquid stocks provides a reasonable basis for setting limit prices.

### 6. Practical Tips for Implementation

- **Set GTC (Good 'Til Canceled) Limit Orders:**
  - Since you're placing orders at the beginning of each minute, GTC orders ensure they remain active until filled or canceled.
  
- **Monitor Order Execution:**
  - Implement logic to cancel and adjust orders if not filled within your desired time frame.
  - **Example:** If a buy order isn't filled within 50 seconds, reevaluate market conditions before reissuing.

- **Minimize Slippage:**
  - By placing limit orders close to the market price, you reduce the risk of slippage compared to market orders.

### 7. Example Strategy Workflow

1. **At the Start of Each Minute:**
   - Collect the latest quotes data for your target stock.

2. **Calculate Relevant Metrics:**
   - Current **bid_price**, **ask_price**, **bid_size**, **ask_size**.
   - Average prices and spreads over the last few seconds.

3. **Set Limit Prices:**
   - **Buys:** Limit price = Current ask_price (+ small buffer if necessary).
   - **Sells:** Limit price = Current bid_price (- small buffer if necessary).

4. **Place GTC Limit Orders:**
   - Submit orders with the calculated limit prices.

5. **Order Monitoring:**
   - If the order isn't filled within 50 seconds, consider canceling and reassessing.

6. **Backtesting Execution:**
   - Use historical quotes to determine if the limit price was met within the time frame.
   - Record fills based on simulated order execution.

### 8. Considerations for Shorts and Sells

- **Regulatory Compliance:**
  - Ensure compliance with short-selling regulations, such as the uptick rule.

- **Availability of Shares:**
  - Verify that shares are available to borrow for short selling.

### Conclusion

By setting your limit orders at or near the current bid and ask prices and adjusting slightly based on recent market data, you strike a balance between execution speed and price accuracy. This method leverages the most immediate market information available to you and fits within your backtesting framework, providing a practical approach to refine your trading system.