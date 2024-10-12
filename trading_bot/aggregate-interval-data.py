import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta

def aggregate_interval_data(symbol: str, start_date: datetime, end_date: datetime, data_type: str):
    """
    Aggregate data for a given symbol and date range from stored interval files.
    
    :param symbol: The stock symbol
    :param start_date: Start date for the backtesting period
    :param end_date: End date for the backtesting period
    :param data_type: Either 'bars' or 'quotes'
    :return: Aggregated DataFrame
    """
    assert data_type in ['bars', 'quotes'], "data_type must be either 'bars' or 'quotes'"
    
    # Define the directory where the data is stored
    data_dir = f"{data_type}/{symbol}"
    
    # List all zip files in the directory
    zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
    
    # Sort zip files by date
    zip_files.sort()
    
    # Filter zip files based on the date range
    relevant_files = []
    for zip_file in zip_files:
        file_start, file_end = zip_file.split('_')[-2:]
        file_start = datetime.strptime(file_start, "%Y-%m-%d")
        file_end = datetime.strptime(file_end.split('.')[0], "%Y-%m-%d")
        
        if file_start <= end_date and file_end >= start_date:
            relevant_files.append((zip_file, file_start, file_end))
    
    if not relevant_files:
        raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}")
    
    # Check for gaps
    relevant_files.sort(key=lambda x: x[1])  # Sort by file start date
    for i in range(len(relevant_files) - 1):
        current_end = relevant_files[i][2]
        next_start = relevant_files[i+1][1]
        if next_start - current_end > timedelta(days=1):
            raise ValueError(f"Gap detected between {current_end} and {next_start}")
    
    # Read and concatenate data from relevant files
    dfs = []
    for zip_file, file_start, file_end in relevant_files:
        file_path = os.path.join(data_dir, zip_file)
        with zipfile.ZipFile(file_path, 'r') as zf:
            csv_file = zf.namelist()[0]  # Assume there's only one CSV file in each zip
            with zf.open(csv_file) as f:
                df = pd.read_csv(f, parse_dates=['timestamp'])
                df.set_index(['symbol', 'timestamp'], inplace=True)
                
                # Filter the DataFrame to the requested date range
                df = df.loc[(slice(None), slice(max(start_date, file_start), min(end_date, file_end))), :]
                
                dfs.append(df)
    
    # Concatenate all DataFrames
    if dfs:
        combined_df = pd.concat(dfs)
        combined_df.sort_index(inplace=True)
        
        return combined_df
    else:
        raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}")

# Example usage:
# start_date = datetime(2023, 1, 15)  # Mid-month start
# end_date = datetime(2023, 2, 15)    # Crosses month boundary
# symbol = "AAPL"
# bars_data = aggregate_interval_data(symbol, start_date, end_date, "bars")


# Example usage for TradingStrategy:
# def run_backtest(symbol, start_date, end_date):
#     # Aggregate bar and quote data
#     bars_data = aggregate_interval_data(symbol, start_date, end_date, "bars")
#     quotes_data = aggregate_interval_data(symbol, start_date, end_date, "quotes")
    
#     # Your backtesting logic here, using bars_data and quotes_data
#     # ...

# # Run a backtest for a full year
# start_date = datetime(2023, 1, 1)
# end_date = datetime(2023, 12, 31)
# symbol = "AAPL"
# run_backtest(symbol, start_date, end_date)