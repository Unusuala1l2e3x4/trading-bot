# PositionMetrics Documentation

## Overview
The PositionMetrics module provides time-series tracking and analysis of trading positions. It maintains metrics for both entered and unentered positions, tracking price movements, P&L, VWAP, and timing information.

## Key Classes

### PositionSnapshot
A dataclass that captures the complete state of a position at a specific minute. Used by PositionMetrics to maintain a historical record.

Key attributes:
- Price data via TypedBarData
- Position sizing information (shares, max_shares)
- P&L metrics (running_pl, realized_pl, etc.)
- State flags (has_entered, has_exited)

### PositionMetrics
Manages a collection of PositionSnapshots and calculates derived metrics.

Key features:
- Price movement tracking relative to reference points
- P&L peaks and timing
- VWAP tracking from entry
- Normalization methods (R-multiples, percentages)

## Integration with Other Modules

### TradePosition
- Creates PositionSnapshots via record_snapshot()
- Passes P&L calculations from calculate_exit_pl_values()
- Uses metrics for position management decisions

### TouchArea
- Provides area_width for R-multiple calculations
- Supplies reference prices for normalization
- Used for area-based metrics (above_buy_price)

### TypedBarData 
- Embedded in PositionSnapshot
- Provides price and indicator data
- Used for technical analysis metrics

## Key Concepts

### Reference Values
Two primary reference points used for normalization:
1. R-unit (reference_area_width):
   - For entered positions: area_width at entry
   - For unentered: area_width at first bar
   - Used for risk-based normalization

2. Reference price (reference_price):
   - For entered positions: close price at entry
   - For unentered: first bar's close
   - Used for percentage-based normalization

### Timing Metrics
- Tracks when significant events occur (first_pospl_time, etc.)
- All times are minutes relative to entry (0-based)
- Separate tracking for body and wick-based events

### VWAP Tracking
- Maintains position-anchored VWAP from entry
- Tracks price distances from VWAP
- Uses bar.vwap (intrabar) and volume for calculations

### P&L Metrics
Three categories tracked:
1. Body-based (using close prices)
2. Wick-based (using high/low)
3. VWAP-based (volume-weighted)

Each category tracks:
- Absolute values
- Percentage returns
- Timing information

## Data Flow

1. TradePosition calls record_snapshot for each bar
2. PositionMetrics.add_snapshot processes each snapshot:
   - Updates area width extremes
   - Calculates price differences
   - Updates P&L peaks
   - Maintains VWAP metrics
3. Metrics exposed through get_metrics_dict():
   - Normalizes values (R-unit, percentage)
   - Groups related metrics
   - Used for analysis and export

## Usage Notes

### Normalization
Two methods available:
1. normalize_by_r(): Divides by reference_area_width
   - Better for risk/reward analysis
   - Adapts to volatility

2. normalize_by_price(): Calculates percentage of reference_price
   - More intuitive interpretation
   - Good for performance comparison

### Position States
Metrics handle three states:
1. Pre-entry: After area identified, before trades
2. Active: Between first entry and final exit
3. Post-exit: After position closed

### Edge Cases
- Handles zero-share positions
- Manages multiple entries/exits
- Accounts for position reversals
- Tracks partial fills

## Performance Considerations
- Maintains running calculations where possible
- Uses NumPy for array operations
- Minimizes recalculations
- Efficient snapshot storage