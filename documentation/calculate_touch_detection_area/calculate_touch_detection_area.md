# Function Documentation: `calculate_touch_detection_area`

## Overview

- The `calculate_touch_detection_area` function performs custom volatility analysis to identify strong support and resistance levels in 1-minute stock bars data. It uses Average True Range (ATR) or Median True Range (MTR) to define touch areas and considers only market hours for its analysis. The function incorporates flexibility to switch between mean and median calculations for levels and volatility measures.
- This documentation provides a comprehensive overview of the `calculate_touch_detection_area` function, including its helper functions, key concepts, and usage. 

## Parameters

- `symbol (str|list)`: The stock symbol(s) to analyze.
- `start_date (str|datetime)`: The start datetime for historical data (format: 'YYYY-MM-DD HH:MM:SS').
- `end_date (str|datetime)`: The end datetime for historical data (format: 'YYYY-MM-DD HH:MM:SS').
- `atr_period (int)`: The period for calculating ATR or MTR. Default is 10.
- `multiplier (float)`: The multiplier for ATR/MTR to define the touch detection area width. Default is 2.
- `min_touches (int)`: The minimum number of touches to consider a level as strong support or resistance. Default is 2.
- `bid_buffer_pct (float)`: The percentage above the high or below the low to place a stop market buy. Default is 0.005.
- `sell_time (str|None)`: The time to sell all positions before (format: 'HH:MM'). If None, defaults to market close.
- `use_median (bool)`: Flag to switch between using mean (False) or median (True) for levels and touch areas. Default is False.
- `touch_area_width_agg (function)`: Aggregation function for calculating touch area width. Default is np.median.

## Returns

- `dict`: A dictionary containing:
  - `symbol`: The stock symbol(s) to analyze.
  - `long_touch_area`: List of TouchArea objects for resistance levels.
  - `short_touch_area`: List of TouchArea objects for support levels.
  - `market_hours`: Dictionary of market hours for each date.
  - `bars`: DataFrame of historical price data.
  - `mask`: Boolean mask for filtering data within market hours.
  - `bid_buffer_pct`: The bid buffer percentage used.
  - `min_touches`: The minimum number of touches used.
  - `sell_time`: The sell time used.
  - `use_median`: Boolean indicating whether median was used for calculations.

## Helper Functions

### `calculate_touch_area`

This function processes the identified levels and creates TouchArea objects.

#### Parameters
- `levels_by_date`: Dictionary of levels grouped by date.
- `is_long`: Boolean indicating if it's for long (resistance) or short (support) positions.
- `df`: DataFrame of historical price data.
- `symbol`: The stock symbol.
- `market_hours`: Dictionary of market hours for each date.
- `min_touches`: Minimum number of touches required.
- `bid_buffer_pct`: Bid buffer percentage.
- `use_median`: Boolean for using median instead of mean.
- `touch_area_width_agg`: Function for aggregating touch area width.
- `multiplier`: Multiplier for ATR/MTR.
- `sell_time`: Time to sell positions before.

#### Returns
- List of TouchArea objects.
- List of touch area widths.

### `process_touches`

This function processes the touches for a specific level and determines if it qualifies as a touch area.

#### Parameters
- `touches`: Array of touch indices.
- `prices`: Array of price values.
- `touch_area_lower`: Lower bound of the touch area.
- `touch_area_upper`: Upper bound of the touch area.
- `level`: The price level being analyzed.
- `level_lower_bound`: Lower bound of the original level.
- `level_upper_bound`: Upper bound of the original level.
- `is_long`: Boolean indicating if it's for long (resistance) or short (support) positions.
- `min_touches`: Minimum number of touches required.

#### Returns
- NumPy array of consecutive touch indices if criteria are met, otherwise an empty array.

## Key Concepts and Calculations

1. **True Range (TR) Calculation**:
   $$TR = \max(High - Low, |High - PreviousClose|, |Low - PreviousClose|)$$

2. **ATR/MTR Calculation**:
   $$ATR = \frac{1}{n}\sum_{i=1}^n TR_i$$
   $$MTR = \text{median}(TR_1, TR_2, ..., TR_n)$$
   where $n$ is the `atr_period`.

3. **Touch Area Width**:
   $$TouchAreaWidth = (ATR \text{ or } MTR) \times multiplier$$

4. **Touch Area Bounds**:
   For long positions (resistance):
   $$UpperBound = Level + \frac{TouchAreaWidth}{3}$$
   $$LowerBound = Level - \frac{2 \times TouchAreaWidth}{3}$$
   
   For short positions (support):
   $$UpperBound = Level + \frac{2 \times TouchAreaWidth}{3}$$
   $$LowerBound = Level - \frac{TouchAreaWidth}{3}$$

5. **Level Identification**:
   Levels are identified based on price movements and clustering of price points. The process involves:
   
   a. Calculating a price range for each candle:
      $$w = \frac{High - Low}{2}$$
   
   b. Defining potential level bounds for each candle:
      $$x = Close - w$$
      $$y = Close + w$$
   
   c. Grouping similar price levels:
      - A price point is added to an existing level if it falls within the level's bounds.
      - If a price point doesn't fall within any existing level, it creates a new potential level.

   d. Filtering levels:
      - Levels with fewer touches than `min_touches` are discarded.
      - Levels are classified as support or resistance based on their relation to the central value:
        $$\text{Classification} = \begin{cases} 
        \text{Resistance} & \text{if } Level > CentralValue \\
        \text{Support} & \text{if } Level \leq CentralValue
        \end{cases}$$

6. **Touch Criteria**:
   A touch is counted when one of the following conditions is met:
   
   a. Price crosses the level from below (for resistance) or above (for support):
      $$\text{For Resistance: } PreviousClose < Level \leq CurrentClose$$
      $$\text{For Support: } PreviousClose > Level \geq CurrentClose$$
   
   b. Price exactly equals the level:
      $$CurrentClose = Level$$
   
   c. Price is within the level bounds:
      $$LowerBound \leq CurrentClose \leq UpperBound$$
   
   Consecutive touches are tracked, and a touch area is created when the number of consecutive touches reaches `min_touches`.

7. **Touch Area Creation**:
   When a level accumulates `min_touches` consecutive touches, a TouchArea object is created with the following properties:
   - `id`: A unique identifier for the touch area
   - `level`: The price level of the touch area
   - `upper_bound` and `lower_bound`: Calculated as per the Touch Area Bounds formulas
   - `touches`: List of timestamps when touches occurred
   - `is_long`: Boolean indicating if it's a resistance (long) or support (short) level
   - `min_touches`: The minimum number of touches required (as per input parameter)
   - `bid_buffer_pct`: The bid buffer percentage (as per input parameter)


## Function Logic

1. Fetch historical stock data using Alpaca API.
2. Calculate TR, ATR, and MTR.
3. Identify potential levels for each trading day.
4. Filter levels based on the minimum number of touches.
5. Create touch areas for qualifying levels.
6. Apply market hours filtering.
7. Return the results including touch areas and related data.

## Example Usage

```python
touch_detection_areas = calculate_touch_detection_area(
    symbol='AAPL',
    start_date='2024-06-01 08:00:00',
    end_date='2024-06-30 23:59:59',
    atr_period=15,
    multiplier=1.6,
    min_touches=3,
    sell_time='16:00',
    use_median=True,
    touch_area_width_agg=np.median
)
```
This call analyzes AAPL stock from June 1 to June 30, 2024, using a 15-period MTR, an MTR multiplier of 1.6, requiring at least 3 touches for strong levels, selling positions before 4:00 PM, and using median calculations for touch areas and volatility measures.
