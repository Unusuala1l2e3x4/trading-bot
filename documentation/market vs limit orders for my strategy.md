## Q:

my algorithmic strategy aims to partially lock in profits immediately after receiving the 1-minute close price if the price had moved towards the trailing sell price, selling an amount proportional to (price change / (max close price encountered - trailing sell price)), and similarly re-buying shares if the price moves favorably again. based on my backtesting results, my strategy seems to work far better (consistently profitable over time) when only going off of close prices at each minute. if there are stop/limit orders sent that may execute within the 60 seconds between close prices (simulated by looking at the high and low prices), my positions often exit prematurely (hit the stop price) before being able to partially lock in profits or wait for favorable movements, causing poor performance (consistently losing over time). so it seems like the close prices remove a lot of the noise, and this allows my strategy to perform best. this suggests i should still only act based on close prices when trading for real.

however, i have not considered bracket orders with multiple legs (see OrderClass in the API. Valid values: simple, bracket, oco or oto); it looks like it's impossible to simulate them accurately using 1-minute bars. they might be good for partially locking in profits if i set up leg prices between the max price and the trailing sell price, but again this probably also leads to premature actions (and more transaction costs) given the amount of noise.

assuming i stick to close prices only, the question still remains - should i use market orders or limit orders (focusing on high-liquidity stocks)? market orders sound better since they are faster, with the tradeoff being slippage. limit orders seem undesirable since partial fills and missed opportunities could be more detrimental than slippage. 

it also helps to split investments across multiple high-liquidity stocks, which not only follows a traditional risk management strategy but also reduces slippage per trade.


## A:


Given your strategy and the insights from your backtesting results, it seems that market orders are likely the better choice for several reasons:

### Key Considerations for Market Orders

1. **Execution Speed**: 
   - **Crucial for Your Strategy**: Since your strategy involves reacting quickly to 1-minute close prices to lock in profits or re-buy shares, the immediate execution provided by market orders is essential. Market orders ensure that your trades are executed as soon as the signal is received, aligning with your need for fast action.
   
2. **Reduced Complexity**:
   - **Simplicity**: Market orders simplify the trading process by avoiding the need to constantly monitor and adjust limit prices. This can be particularly beneficial when dealing with multiple high-liquidity stocks.

3. **Slippage Management**:
   - **High-Liquidity Stocks**: In high-liquidity stocks, the bid-ask spread is typically narrow, which helps minimize slippage. While some slippage is inevitable, it is usually less problematic than the risks associated with partial fills or missed opportunities from limit orders.
   - **Strategic Diversification**: By spreading investments across multiple high-liquidity stocks, you can further mitigate the impact of slippage on your overall portfolio.

### Disadvantages of Limit Orders

1. **Execution Uncertainty**:
   - **Partial Fills**: Limit orders can lead to partial fills, where only a portion of your order is executed at the desired price. This can disrupt your strategy, especially if you need to adjust positions quickly.
   - **Missed Opportunities**: If the market price does not reach your limit price, your order may not be executed at all, causing you to miss trading opportunities.

2. **Premature Actions**:
   - **Noise and Volatility**: Limit orders can be affected by short-term noise and volatility, leading to premature exits or entries that do not align with your strategy's reliance on 1-minute close prices.

### Strategic Approach

1. **Stick to Close Prices**:
   - **Noise Reduction**: Your strategy performs best when based on 1-minute close prices, as this approach reduces noise and prevents premature actions. Continuing to use close prices will likely maintain the consistency and profitability observed in your backtesting.

2. **Use Market Orders**:
   - **Aligns with Strategy**: Market orders align with your need for quick execution after receiving close prices. They help ensure that your trades are executed promptly, capturing the intended price action.
   - **Manageable Slippage**: Given the high liquidity of the stocks you trade, the slippage associated with market orders should be minimal and manageable, especially when balanced against the risks of partial fills and missed opportunities with limit orders.

3. **Diversify Across Stocks**:
   - **Risk Management**: Diversifying your investments across multiple high-liquidity stocks not only follows traditional risk management principles but also reduces the impact of slippage on any single trade.
   - **Enhanced Execution**: Smaller trade sizes per stock help ensure that each market order has a minimal impact on the stock's price, further reducing slippage.

### Conclusion

Based on your strategy's reliance on 1-minute close prices and the need for rapid execution, market orders appear to be the most suitable choice. They offer the speed and certainty required for your trading approach, with the tradeoff of manageable slippage, especially when dealing with high-liquidity stocks. By sticking to market orders and diversifying across multiple stocks, you can maintain the effectiveness of your strategy while minimizing potential drawbacks.