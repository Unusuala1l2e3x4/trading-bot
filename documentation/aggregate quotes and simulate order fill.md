## adjusted quote retrieval strategy:
- for each minute at a time, retrieve ALL data in the minute (remove the before_sec and after_sec function. just retrieve all 60 seconds in the same minute). I will perform aggregation on it into a new dataframe before saving it. Apply aggregation to each minute chunk at a time, appending to the new dataframe after each minute is processed.
- for each minute, keep the first 1 second of raw data (this adequately covers the timeframe in which i plan to submit orders) and the last 5 seconds of raw data. after cleaning with clean_quotes_data:
  - for the first 1 second of raw data: use the same sampling method as before.
  - for the last 5 seconds of raw data: keep the LATEST 200 points, omitting any remaining. this would capture the effect of fast-moving markets.
- remember to keep the raw datapoints and aggregated datapoints in seperate dataframes.

## my question for aggregation part:
what if i just want to aggregate data in a way that is useful for approximating whether an order is fully/partially filled? 

## chatgpt answer:
Let's simplify the approach to make it both practical and effective for your backtesting needs, focusing on whether an order is fully or partially filled. Given your constraints and the nature of your data, Time-Based Aggregation at 1-second intervals is a sensible choice.

### **Simplified Approach Overview**

**Objective:** Aggregate quotes data in 1-second intervals to capture essential market information that helps approximate order fills during backtesting.

**Key Metrics to Calculate per Interval:**

- **Minimum Ask Price (`min_ask_price`)** and the **associated Ask Size (`min_ask_size`)**.
- **Maximum Bid Price (`max_bid_price`)** and the **associated Bid Size (`max_bid_size`)**.
- **Volume-Weighted Average Prices (VWAP)** for bids and asks (optional).
- **Total Bid and Ask Sizes** (optional).

By focusing on these metrics, you can approximate whether your limit orders would have been filled based on the available prices and sizes during each interval.

### **Implementing the Simplified Aggregation**

Assuming you have a pandas DataFrame `quotes_df` with a multi-index of `(symbol, timestamp)` and the following columns:

- `bid_price`
- `ask_price`
- `bid_size`
- `ask_size`

Here's how you can aggregate the data:

```python
import pandas as pd
import numpy as np

def aggregate_quotes_time_based(quotes_df, interval_seconds=1):
    """
    Aggregates quotes data into fixed time intervals.

    Parameters:
    - quotes_df: pandas DataFrame with multi-index (symbol, timestamp), containing fields:
        bid_price, ask_price, bid_size, ask_size, etc.
    - interval_seconds: The interval in seconds for aggregation.

    Returns:
    - aggregated_df: pandas DataFrame with the aggregated statistics per interval.
    """
    # Reset index to access 'timestamp' as a column
    quotes_df = quotes_df.reset_index()
    
    # Ensure the DataFrame is sorted by 'timestamp'
    quotes_df = quotes_df.sort_values('timestamp').copy()
    
    # Create a new column 'interval_start' representing the start time of each interval
    quotes_df['interval_start'] = quotes_df['timestamp'].dt.floor(f'{interval_seconds}S')
    
    # Group by 'symbol' and 'interval_start'
    grouped = quotes_df.groupby(['symbol', 'interval_start'])
    
    # Define aggregation functions
    agg_funcs = {
        'bid_price': ['max', 'mean'],
        'ask_price': ['min', 'mean'],
        'bid_size': ['max', 'sum'],
        'ask_size': ['max', 'sum'],
    }
    
    # Perform aggregation
    aggregated = grouped.agg(agg_funcs)
    
    # Flatten the MultiIndex columns
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    
    # Rename columns for clarity
    aggregated.rename(columns={
        'bid_price_max': 'max_bid_price',
        'bid_price_mean': 'mean_bid_price',
        'ask_price_min': 'min_ask_price',
        'ask_price_mean': 'mean_ask_price',
        'bid_size_max': 'max_bid_size',
        'bid_size_sum': 'total_bid_size',
        'ask_size_max': 'max_ask_size',
        'ask_size_sum': 'total_ask_size',
    }, inplace=True)
    
    # Reset index to get a flat DataFrame
    aggregated_df = aggregated.reset_index()
    
    # Set multi-index (symbol, interval_start)
    aggregated_df.set_index(['symbol', 'interval_start'], inplace=True)
    
    return aggregated_df
```

### **Explanation of the Aggregation Steps**

1. **Reset Index and Sort:**
   - Reset the index to turn `timestamp` into a column.
   - Sort the DataFrame by `timestamp` to ensure proper chronological order.

2. **Create Interval Start Times:**
   - Use `dt.floor(f'{interval_seconds}S')` to floor the timestamps to the nearest interval (e.g., every 1 second).
   - This groups all quotes within each interval together.

3. **Group By Symbol and Interval:**
   - Group the DataFrame by `symbol` and `interval_start` to perform aggregations on each interval separately.

4. **Define Aggregation Functions:**
   - For **bid prices**:
     - `max` to get the **maximum bid price** (useful for sell orders).
     - `mean` to get the average bid price (optional).
   - For **ask prices**:
     - `min` to get the **minimum ask price** (useful for buy orders).
     - `mean` to get the average ask price (optional).
   - For **bid and ask sizes**:
     - `max` to get the **maximum size** observed (size at the best price).
     - `sum` to get the **total size** available during the interval (optional).

5. **Perform Aggregation:**
   - Apply the aggregation functions to the grouped data.

6. **Flatten MultiIndex Columns:**
   - The aggregation results in MultiIndex columns; flatten them for easier access.

7. **Rename Columns:**
   - Rename columns to meaningful names for clarity.

8. **Set Multi-Index:**
   - Set `symbol` and `interval_start` as the index of the aggregated DataFrame.

### **Using the Aggregated Data for Backtesting**

With the aggregated data, you can simulate order fills during backtesting as follows:

1. **Determine the Time Window:**
   - For each order placed, define the time window after the order placement during which you want to check for fills (e.g., up to 10 seconds).

2. **Retrieve Relevant Intervals:**
   - Extract the intervals from `aggregated_df` that fall within the time window after the order placement.

3. **Simulate Fills:**
   - **Buy Orders:**
     - Check if the **minimum ask price (`min_ask_price`)** in any interval is less than or equal to your **limit price**.
     - If so, assume the order could be filled up to the **maximum ask size (`max_ask_size`)** at that price.
     - If your order size is less than or equal to `max_ask_size`, assume a full fill; otherwise, assume a partial fill.
   - **Sell Orders:**
     - Check if the **maximum bid price (`max_bid_price`)** in any interval is greater than or equal to your **limit price**.
     - If so, assume the order could be filled up to the **maximum bid size (`max_bid_size`)** at that price.
     - Similar to buy orders, determine if the fill is full or partial based on your order size.

4. **Record Fill Details:**
   - For each filled order, record the fill price (could be your limit price or the market price), fill time, and fill size.

### **Example of Simulating an Order Fill**

```python
def simulate_order_fill(order, aggregated_df, time_window_seconds=10):
    """
    Simulates whether an order would have been filled using aggregated data.

    Parameters:
    - order: A dictionary containing order details:
        'timestamp', 'symbol', 'side', 'limit_price', 'quantity'
    - aggregated_df: The aggregated DataFrame returned by aggregate_quotes_time_based()
    - time_window_seconds: The time window after order placement to check for fills.

    Returns:
    - fill_details: A dictionary containing fill information or None if not filled.
    """
    import datetime

    symbol = order['symbol']
    order_time = order['timestamp']
    side = order['side']
    limit_price = order['limit_price']
    quantity = order['quantity']

    # Define the end time of the time window
    end_time = order_time + datetime.timedelta(seconds=time_window_seconds)

    # Filter the aggregated_df for the relevant intervals
    intervals = aggregated_df.loc[symbol]
    intervals = intervals[(intervals.index >= order_time.floor('S')) & (intervals.index <= end_time.floor('S'))]

    fill_price = None
    fill_time = None
    fill_quantity = 0

    if side == 'buy':
        for interval_start, row in intervals.iterrows():
            if row['min_ask_price'] <= limit_price:
                fill_price = min(limit_price, row['min_ask_price'])
                fill_time = interval_start
                available_size = row['max_ask_size']
                fill_quantity = min(quantity, available_size)
                break  # Order is filled
    elif side == 'sell':
        for interval_start, row in intervals.iterrows():
            if row['max_bid_price'] >= limit_price:
                fill_price = max(limit_price, row['max_bid_price'])
                fill_time = interval_start
                available_size = row['max_bid_size']
                fill_quantity = min(quantity, available_size)
                break  # Order is filled

    if fill_quantity > 0:
        return {
            'fill_time': fill_time,
            'fill_price': fill_price,
            'fill_quantity': fill_quantity,
            'status': 'filled' if fill_quantity == quantity else 'partially filled'
        }
    else:
        return {
            'status': 'unfilled'
        }
```

### **Adjustments and Considerations**

- **Size Limitations:**
  - Since you have only Level 1 data, you may not have complete visibility into the available sizes beyond the best bid and ask.
  - The `max_bid_size` and `max_ask_size` represent the largest sizes observed at the best prices during the interval.

- **Assumptions:**
  - **Liquidity Assumption:** We assume that the available size at the best price is sufficient to fill the order up to `max_bid_size` or `max_ask_size`.
  - **Price Time Priority:** Orders are filled based on the sequence of intervals; the earliest interval where the price condition is met is used.

- **Order Size vs. Available Size:**
  - If your order size exceeds `max_bid_size` or `max_ask_size`, you may choose to:
    - Assume a partial fill.
    - Use `total_bid_size` or `total_ask_size` as an upper bound (though this may overestimate the available size).

- **Interval Length:**
  - You can adjust `interval_seconds` to balance between data granularity and storage constraints.
  - Shorter intervals (e.g., 1 second) provide more precise simulation but result in more data points.

- **Storage Efficiency:**
  - This approach significantly reduces data size compared to storing raw quotes.
  - You only store essential aggregated metrics per interval.

### **Example Usage in Backtesting Loop**

```python
# Assuming you have a list of orders and the aggregated_df
for order in orders:
    fill_details = simulate_order_fill(order, aggregated_df)
    if fill_details['status'] == 'filled' or fill_details['status'] == 'partially filled':
        # Update order with fill details
        order['fill_time'] = fill_details['fill_time']
        order['fill_price'] = fill_details['fill_price']
        order['filled_quantity'] = fill_details['fill_quantity']
        order['status'] = fill_details['status']
    else:
        order['status'] = 'unfilled'
```

### **Handling Partial Fills and Remaining Quantities**

- If an order is partially filled, you can decide how to handle the remaining quantity:
  - **Cancel the Remaining Order:** Assume the unfilled portion is canceled after the time window.
  - **Keep the Order Open:** Continue to check for fills in subsequent intervals (may require adjusting your time window and storage considerations).

### **Advantages of This Approach**

- **Simplicity:** Easy to implement and understand.
- **Efficiency:** Reduces data size by aggregating essential information.
- **Practicality:** Provides a reasonable approximation for order fills using available Level 1 data.
- **Flexibility:** Allows you to adjust interval lengths and incorporate additional metrics if needed.

### **Limitations**

- **Level 1 Data Constraints:** Without Level 2 data, you cannot accurately model depth beyond the best bid and ask.
- **Assumptions on Available Sizes:** The available sizes at the best prices are estimates and may not reflect actual market conditions.
- **Price Movements Within Intervals:** Rapid price movements within intervals may not be fully captured, potentially affecting fill accuracy.

### **Potential Enhancements**

- **Include Price Ranges:**
  - Record the range of bid and ask prices within each interval (e.g., `bid_price_min`, `bid_price_max`).
- **Weighted Averages:**
  - Use volume-weighted average prices for more accurate representations.
- **Adjusting for Market Conditions:**
  - Modify your fill assumptions during periods of high volatility or low liquidity.

### **Conclusion**

By aggregating your quotes data into fixed time intervals and calculating key metrics like minimum ask price, maximum bid price, and associated sizes, you create a practical and storage-efficient dataset for backtesting your strategy. This method simplifies the simulation of order fills and aligns well with your data constraints.

Remember to validate this approach by comparing backtesting results with any available historical execution data or by performing sensitivity analyses. Adjust the parameters as needed to better fit your specific strategy and market conditions.

If you have further questions or need assistance with specific aspects of the implementation, feel free to ask!