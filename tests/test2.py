def backtest_strategy(touch_detection_areas, initial_investment=10000, do_longs=True, do_shorts=True, use_margin=False, times_buying_power=4):
    # ... (existing code)

    def debug_print(message):
        if debug:
            print(message)

    debug_print(f"Strategy: {'Long' if do_longs else ''}{'&' if do_longs and do_shorts else ''}{'Short' if do_shorts else ''}")
    debug_print(f"Initial Investment: {initial_investment}, Times Buying Power: {times_buying_power}")

    # ... (existing code)

    def place_stop_market_buy(area: TouchArea, timestamp: datetime, open_price: float, high_price: float, low_price: float, close_price: float, prev_close: float):
        nonlocal balance, current_id, total_account_value

        if open_positions or balance <= 0:
            return False

        # ... (existing code)

        debug_print(f"{timestamp} - Attempting order: {'Long' if area.is_long else 'Short'} at {execution_price:.4f}")
        debug_print(f"  Balance: {balance:.2f}, Total Account Value: {total_account_value:.2f}")
        debug_print(f"  Shares: {total_shares:.2f}, Invest Amount: {invest_amount:.2f}")
        debug_print(f"  Margin Multiplier: {actual_margin_multiplier:.2f}, Sub-positions: {num_sub_positions}")

        if actual_cash_used > balance:
            debug_print(f"  Order rejected: Insufficient balance")
            return False

        # ... (existing code)

    def update_positions(timestamp, open_price, close_price, high_price, low_price):
        nonlocal trades_executed

        for area_id, position in open_positions.items():
            old_stop_price = position.current_stop_price
            position.update_stop_price(open_price)
            
            debug_print(f"{timestamp} - Updating position {position.id}")
            debug_print(f"  Current price: {close_price:.4f}, Stop price: {position.current_stop_price:.4f}")
            
            if position.should_exit(open_price):
                debug_print(f"  Exiting at open: {open_price:.4f}")
                perform_exit(area_id, position)
                continue
            
            # ... (rest of the update logic)

        update_total_account_value(close_price)
        debug_print(f"  Updated Total Account Value: {total_account_value:.2f}")

    # ... (rest of the function)

    for i in tqdm(range(1, len(timestamps))):
        current_time = timestamps[i]
        
        if current_time.date() != current_date:
            debug_print(f"\nNew trading day: {current_time.date()}")
            # ... (existing new day logic)

        if market_open <= current_time < market_close and i != len(df)-1:
            debug_print(f"\n{current_time} - Market Open")
            # ... (existing trading logic)

        elif current_time >= market_close or i >= len(df)-1:
            debug_print(f"\n{current_time} - Market Close")
            close_all_positions(current_time, df['close'].iloc[i])

    # ... (existing end-of-backtest code)

    debug_print("\nBacktest Complete")
    debug_print(f"Final Balance: {balance:.2f}")
    debug_print(f"Total Trades Executed: {trades_executed}")
    debug_print(f"Win Rate: {winning_trades / len(trades) * 100:.2f}%")

    return balance, trades_executed, percent_increase, winning_trades / len(trades) * 100, entry_costs, exit_costs, borrow_costs, total_costs