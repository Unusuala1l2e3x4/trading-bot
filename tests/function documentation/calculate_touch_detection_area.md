# Function Documentation: `calculate_touch_detection_area`

## Overview

The `calculate_touch_detection_area` function is designed to perform custom volatility analysis and identify strong support and resistance levels based on historical stock data. It leverages the Average True Range (ATR) or Median True Range (MTR) to define touch areas and considers only market hours for its analysis. This function also incorporates flexibility to switch between using the mean or median for level calculations and true range calculations.

## Goals

1. **Identify Support and Resistance Levels**: Detect significant support and resistance levels in historical stock price data.
2. **Volatility Analysis**: Utilize ATR or MTR to define the width of touch areas around these levels.
3. **Market Hours Filtering**: Ensure analysis considers only market hours.
4. **Customizable Level Calculation**: Allow switching between mean and median calculations for touch areas and true range calculations.

## Parameters

- `symbol (str)`: The stock symbol to analyze.
- `start_date (str|datetime)`: The start datetime for historical data (format: 'YYYY-MM-DD HH:MM:SS.%f').
- `end_date (str|datetime)`: The end datetime for historical data (format: 'YYYY-MM-DD HH:MM:SS.%f').
- `atr_period (int)`: The period for calculating Average True Range (ATR) or Median True Range (MTR).
- `multiplier (float)`: The multiplier for ATR or MTR to define the touch detection area width.
- `min_touches (int)`: The minimum number of touches to consider a level as strong support or resistance.
- `bid_buffer_pct (float)`: The percentage above the high or below the low to place a stop market buy.
- `sell_time (str)`: The time to sell all positions before (format: 'HH:MM'). If None, defaults to market close time.
- `use_median (bool)`: Flag to switch between using mean or median for levels and touch areas, and for true range

 calculation.

## Returns

- `dict`: A dictionary containing:
  - `long_touch_area`: List of tuples with resistance touch areas.
  - `short_touch_area`: List of tuples with support touch areas.
  - `bars`: Filtered historical data bars.
  - `atr`: ATR values if `use_median` is `False`, otherwise MTR values.
  - `bid_buffer_pct`: The bid buffer percentage.
  - `min_touches`: The minimum number of touches.
  - `sell_time`: The sell time.
  - `central_value`: The central value (mean or median) used in calculations.

## Key Concepts and Calculations

### Average True Range (ATR) and Median True Range (MTR) Calculation

1. **True Range (TR)**:
   \[
   \text{TR}_t = \max \left( \text{High}_t - \text{Low}_t, \left| \text{High}_t - \text{Close}_{t-1} \right|, \left| \text{Low}_t - \text{Close}_{t-1} \right| \right)
   \]
   where:
   - \(\text{High}_t\): High price of the current period.
   - \(\text{Low}_t\): Low price of the current period.
   - \(\text{Close}_{t-1}\): Closing price of the previous period.

2. **Average True Range (ATR)**:
   \[
   \text{ATR}_t = \frac{1}{n} \sum_{i=0}^{n-1} \text{TR}_{t-i}
   \]
   where \(n\) is the number of periods (e.g., 10).

3. **Median True Range (MTR)**:
   \[
   \text{MTR}_t = \text{Median} (\text{TR}_{t-i} \text{ for } i \text{ from } 0 \text{ to } n-1)
   \]

### Touch Detection Area

1. **Touch Area Width**:
   \[
   \text{Touch Area Width} = \text{(ATR or MTR)} \times \text{Multiplier}
   \]

2. **Bounds of Touch Area**:
   - **Upper Bound**:
     \[
     \text{Upper Bound} = \text{Level} + \frac{\text{Touch Area Width}}{2}
     \]
   - **Lower Bound**:
     \[
     \text{Lower Bound} = \text{Level} - \frac{\text{Touch Area Width}}{2}
     \]

3. **Support and Resistance Levels**:
   - Identified as local minima and maxima in the price data.

### Central Value Calculation

- **Mean**:
  \[
  \text{Central Value (Mean)} = \text{Level}
  \]
- **Median**:
  \[
  \text{Central Value (Median)} = \text{Median of Close Prices within the Touch Area Bounds}
  \]

### Identifying Potential Levels

1. **Local Minima (Support Levels)**:
   - If \( \text{Low}_i \leq \text{Low}_{i-1} \) and \( \text{Low}_i \leq \text{Low}_{i+1} \), it's a potential support level.

2. **Local Maxima (Resistance Levels)**:
   - If \( \text{High}_i \geq \text{High}_{i-1} \) and \( \text{High}_i \geq \text{High}_{i+1} \), it's a potential resistance level.

### Function Logic

1. **Historical Data Retrieval**:
   - Fetch historical stock data using Alpaca API.

2. **TR, ATR or MTR Calculation**:
   - Compute the TR and then calculate ATR or MTR based on the `use_median` flag.

3. **Identify Potential Levels**:
   - Loop through the data to find local minima and maxima.

4. **Filter Strong Levels**:
   - Filter out levels that do not meet the minimum touch criteria.

5. **Calculate Central Value**:
   - Based on the `use_median` flag, calculate either the mean or median central value for touch areas.

6. **Define Touch Areas**:
   - Calculate upper and lower bounds and central value for each touch area.

7. **Market Hours Filtering**:
   - Ensure only data within market hours is considered.

8. **Return Results**:
   - Return the calculated touch areas, filtered data, ATR or MTR values, and other parameters.

## Example Usage

```python
result = calculate_touch_detection_area(
    symbol='AAPL',
    start_date='2024-06-27 08:00:00',
    end_date='2024-06-28 23:59:59',
    atr_period=10,
    multiplier=2,
    min_touches=2,
    bid_buffer_pct=0.005,
    sell_time='15:30',
    use_median=True
)
```

This function call would analyze AAPL stock data from June 27, 2024, to June 28, 2024, using a 10-period MTR, an MTR multiplier of 2, requiring at least 2 touches for strong levels, a bid buffer percentage of 0.5%, and selling all positions before 3:30 PM, using the median for touch area calculations.