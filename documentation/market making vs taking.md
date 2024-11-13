
## A:
you said "Market makers might step in to provide liquidity if they see market order flow." so it isnt worth being a market maker?

## Q:
Let me clarify my thinking about market making vs taking:

1. **Individual vs Professional Market Makers**:
   - Professional MMs have advantages:
     * Ultra-low latency
     * Advanced risk management
     * Regulatory privileges/obligations
     * Better data/order flow information
     * Can handle many symbols simultaneously
   - Individual traders lack these advantages

2. **Your Strategy Context**:
   - Already has a working logic (rescaling based on price movements)
   - Tested in backtesting with assumed fills
   - Primary goal is to maintain strategy performance
   - Time-based (minute intervals) rather than tick-based

3. **Market Making Challenges**:
   - Requires continuous quote management
   - Higher operational complexity
   - More vulnerable to adverse selection
   - Needs very sophisticated risk controls
   - May conflict with your existing strategy's timing

I was incorrect to casually mention market makers stepping in - that's their specialized role and they're equipped for it. For your strategy, being a taker with market orders likely provides:
- More reliable execution
- Simpler implementation
- Better alignment with your tested strategy
- Less operational risk

Would you like to explore this further or return to discussing the trailing stop implementation?