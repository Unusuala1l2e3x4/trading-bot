## Q
recall i said this about my current strategy: "i use [trailing] stop losses in a conceptual sense, but i manage it myself at 1-minute intervals rather than actually submit/cancel stop orders." in other words, i manually adjust these stop prices at every minute (the stop distance depends on a rolling 15-minute window of median true range), and place market orders once the threshold has been reached within the past minute. in this way, i avoid using stop and limit orders. and in fact this works better in many situations (i found this out by simulating conditions for stop orders using the high and low prices of each 1-minute bar, then comparing this to just using close prices), since the close prices are a lot less noisy than the intra-minute data.

if i were to submit and modify stop market orders instead, this creates a some risk of getting flagged, given the adjustments every minute. fortunately this is counteracted by my position rescaling that happens whenever there's a price movement confined between the stop price and trailing distance. thus it is extremely rare for there to be only stop price adjustments without being paired with a rescaling action for 100 minutes in a row.

thus, i think it makes the most sense to use a stop market order that is modified every minute to the intended stop price, plus some buffer distance if i find that using market orders at the close price (like before) yields better results. to clarify, this only applies to the stop price at the "final" sell, not any position rescaling that happens before.

to still handle the extreme case, i'll stop adjusting the stop market order after, say, ~80 consecutive minutes without a market order.

so overall, implementing this would adequately safeguard my system against extreme events, like a rapid crash, while still closesly following my original strategy.


## A
Your detailed explanation of your strategy and proposed modifications shows a thoughtful approach to automated trading. Let's analyze your current method and the proposed changes:



Current Strategy:
1. Conceptual stop losses managed at 1-minute intervals
2. Market orders placed when thresholds are reached
3. No actual stop or limit orders used
4. Position rescaling based on price movements

Proposed Changes:
1. Implement actual stop market orders
2. Modify stop prices every minute
3. Add a buffer to the stop price
4. Limit consecutive stop order modifications to ~80 minutes

Analysis:

1. Use of Stop Market Orders:
   - Pros: 
     * Provides protection against rapid price movements between your 1-minute checks
     * Aligns more closely with traditional risk management practices
   - Cons: 
     * Potentially increases order modifications, which could be flagged

2. Minute-by-Minute Modifications:
   - This approach is less likely to be flagged compared to sub-second modifications
   - The pairing with rescaling actions further reduces the risk of appearing as excessive modifications

3. Buffer on Stop Price:
   - This is a good approach to balance between your strategy's effectiveness (based on close prices) and protection against extreme events

4. Limiting Consecutive Modifications:
   - Excellent safeguard against potential flagging for excessive modifications
   - 80 minutes seems reasonable, but you might want to test different thresholds

Overall Assessment:
Your proposed changes strike a good balance between maintaining the core of your strategy (which you've found to be effective) and implementing safeguards against extreme events and potential flagging. Here are some additional thoughts:

1. Fill Ratio: 
   - Your approach should maintain a good fill-to-modification ratio, as most modifications will be paired with market orders for rescaling

2. Risk Management:
   - The implementation of actual stop orders enhances your risk management, particularly for black swan events

3. Strategy Integrity:
   - By adding a buffer to the stop price, you're preserving the essence of your strategy while adding an extra layer of protection

4. Compliance:
   - This approach is less likely to be flagged as problematic compared to high-frequency strategies

Recommendations:

1. Testing:
   - Thoroughly backtest this new approach, comparing its performance and behavior to your current strategy
   - Pay particular attention to how often the stop orders are triggered vs. your current method of using market orders based on close prices

2. Monitoring:
   - Implement a system to monitor the frequency of stop order modifications and executions
   - Track the ratio of stop order executions to modifications

3. Gradual Implementation:
   - Consider implementing this change gradually, perhaps starting with a small portion of your trades

4. Buffer Size:
   - Experiment with different buffer sizes to find the optimal balance between protection and strategy performance

5. Modification Limit:
   - The 80-minute limit on consecutive modifications is a good start, but be prepared to adjust this based on observed performance and any feedback from Alpaca

6. Documentation:
   - Keep clear documentation of your strategy and rationale for these changes, in case you need to explain your approach to Alpaca

Conclusion:
Your proposed changes appear to be a well-thought-out evolution of your strategy. They add important safeguards while maintaining the core elements that you've found to be effective. By implementing these changes carefully and monitoring their impact, you should be able to enhance your strategy's robustness without significantly increasing the risk of being flagged as a non-retail trader.