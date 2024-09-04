# End-of-Day Trading Strategy Recommendations

## Current Strategy
- Soft stop time: 15:50 (no new buys, only sells)
- Hard stop time: 15:57 (sell all remaining shares)
- No volume consideration for hard stop sells

## Proposed Improvements
1. Apply volume-based limits to all sells, including at hard stop time
2. Adjust soft and hard stop times

## Detailed Recommendations

### 1. Volume-Based Sell Limits

- Apply the 1% of rolling 15-minute volume rule to all sells, including during the hard stop period.
- Implement a gradual increase in the percentage as you approach market close. For example:
  - 1% up to 15:30
  - 1.25% from 15:30 to 15:45
  - 1.5% from 15:45 to 15:55
  - 2% from 15:55 to close

### 2. Adjusted Stop Times

- Soft stop time: Move to 15:45
  - Cease new buys
  - Begin more aggressive selling (within volume limits)
- Hard stop time: Move to 15:55
  - Initiate final liquidation process

### 3. Liquidation Process

- At hard stop time (15:55):
  - Calculate total shares to sell
  - Estimate required time based on current volume and percentage limits
  - If estimated time exceeds available time, increase percentage limit incrementally
  - Set a maximum percentage limit (e.g., 5%) to avoid extreme market impact

### 4. Partial Position Overnight

- If unable to fully liquidate by 15:59:
  - Assess risk of holding overnight vs. potential market impact
  - Consider setting a maximum acceptable overnight position size
  - For positions exceeding this size, use a more aggressive selling strategy in the final minute

### 5. Morning Liquidation

- For any positions held overnight:
  - Prepare to sell at market open
  - Use a similar volume-based approach, but with higher initial percentages

## Implementation Notes

1. Test the strategy thoroughly in paper trading before live implementation.
2. Monitor slippage carefully during the end-of-day period.
3. Regularly review and adjust percentages based on observed market impact and execution quality.
4. Consider implementing price checks to avoid selling into significant short-term price drops.

By implementing these recommendations, you can maintain the core strengths of your strategy while significantly reducing the risks associated with end-of-day liquidation.

