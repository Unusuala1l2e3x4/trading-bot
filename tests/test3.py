def backtest_strategy(touch_detection_areas, initial_investment: float=10000, do_longs=True, do_shorts=True, use_margin=False, times_buying_power=4, \
    min_stop_dist_relative_change_for_partial:Optional[int]=0, soft_start_time:Optional[str]=None, soft_end_time:Optional[str]=None, export_trades_path:Optional[str]=None):

    long_touch_area = touch_detection_areas['long_touch_area']
    short_touch_area = touch_detection_areas['short_touch_area']
    df = touch_detection_areas['bars']
    min_touches = touch_detection_areas['min_touches']
...

    if soft_start_time:
        soft_start_time = pd.to_datetime(soft_start_time, format='%H:%M').time()
    if soft_end_time:
        soft_end_time = pd.to_datetime(soft_end_time, format='%H:%M').time()
        
    debug = False
    debug2 = False
    debug3 = True
    def debug_print(*args, **kwargs):
        if debug:
            print(*args, **kwargs)
    def debug2_print(*args, **kwargs):
        if debug2:
            print(*args, **kwargs)
    def debug3_print(*args, **kwargs):
        if debug3:
            print(*args, **kwargs)
            
    assert do_longs or do_shorts
    assert 0 <= min_stop_dist_relative_change_for_partial <= 1
    
    POSITION_OPENED = True
    NO_POSITION_OPENED = False
    
    all_touch_areas = []
    if do_longs:
        all_touch_areas.extend(long_touch_area)
    if do_shorts:
        all_touch_areas.extend(short_touch_area)
    touch_area_collection = TouchAreaCollection(all_touch_areas, min_touches)

    # df = df[mask]
    df = df.sort_index(level='timestamp')
    timestamps = df.index.get_level_values('timestamp')


    def update_total_account_value(current_price, name):
        nonlocal total_account_value, balance
        for position in open_positions.values():
            position.update_market_value(current_price)
        
        market_value = sum(position.market_value for position in open_positions.values())
        cash_committed = sum(position.cash_committed for position in open_positions.values())
        total_account_value = balance + cash_committed
...
    def rebalance(cash_change: float, current_price: float = None):
        nonlocal balance, total_account_value
        old_balance = balance
        new_balance = balance + cash_change

        assert new_balance >= 0, f"Negative balance encountered: {new_balance}"
        
        balance = new_balance

        if current_price is not None:
            update_total_account_value(current_price, 'REBALANCE')
        
        s = sum(pos.cash_committed for pos in open_positions.values())
        assert abs(total_account_value - (balance + s)) < 1e-8, \
            f"Total account value mismatch: {total_account_value:.2f} != {balance + s:.2f} ({balance:.2f} + {s:.2f})"

        debug2_print(f"Rebalance: Old balance: {old_balance:.4f}, Change: {cash_change:.4f}, New balance: {balance:.4f}, Total Account Value: {total_account_value:.4f}")
            
        
    def exit_action(area_id, position):
        nonlocal trades
        debug2_print(f"{'res' if position.area.is_long else 'sup'} area {area_id}:\t{position.id} {position.exit_time} - Exit {'Long ' if position.is_long else 'Short'} at {position.exit_price:.4f}")
...
        trades.append(position)
            
            
    def close_all_positions(timestamp, exit_price):
        nonlocal trades_executed
        positions_to_remove = []
        
        debug2_print('CLOSING ALL POSITIONS...')

        for area_id, position in list(open_positions.items()):
            realized_pnl, cash_released, fees = position.partial_exit(timestamp, exit_price, position.shares)
            debug2_print(f"  Partial exit complete - Realized PnL: {realized_pnl:.2f}, Cash released: {cash_released:.2f}")
            rebalance(cash_released + realized_pnl - fees, exit_price)
            debug2_print(f"  Partial exit: Sold {position.shares} shares at {exit_price:.4f}, Realized PnL: {realized_pnl:.2f}, Cash released: {cash_released:.4f}")

            position.close(timestamp, exit_price)
            trades_executed += 1
            position.area.record_entry_exit(position.entry_time, position.entry_price, 
                                            timestamp, exit_price)
            position.area.terminate(touch_area_collection)
            positions_to_remove.append(area_id)

        temp = {}
        for area_id in positions_to_remove:
            temp[area_id] = open_positions[area_id]
            del open_positions[area_id]
        for area_id in positions_to_remove:
            exit_action(area_id, temp[area_id])

        assert not open_positions
        open_positions.clear()

                                
    def calculate_position_details(current_price, times_buying_power, existing_sub_positions=None, target_shares=None):
        nonlocal balance, use_margin, is_marginable

        max_shares, max_additional_shares, actual_margin_multiplier, overall_margin_multiplier, estimated_entry_cost = calculate_max_shares(current_price, times_buying_power, is_marginable)
        if target_shares is None:
            target_shares = max_shares

        if existing_sub_positions is None:
            # Initial entry
            debug3_print(f"  initial: max_shares: {max_shares}")
            assert max_additional_shares == max_shares
            max_additional_shares = max_shares
        else:
            # Partial entry
            assert max_additional_shares <= max_shares
            current_shares = sum(existing_sub_positions)
            max_additional_shares = min(target_shares - current_shares, max_additional_shares)
            debug3_print(f"  BEFORE adjust - partial: Current shares: {current_shares}, Max additional shares: {max_additional_shares}, Target shares: {target_shares}")
            max_shares = current_shares + max_additional_shares
            debug3_print(f"  AFTER adjust - partial: Current shares: {current_shares}, Max additional shares: {max_additional_shares}, Target shares: {target_shares}")

        invest_amount = max_additional_shares * current_price
        actual_cash_used = invest_amount / overall_margin_multiplier
        estimated_entry_cost = TradePosition.estimate_entry_cost(max_shares, overall_margin_multiplier, existing_sub_positions)
        debug3_print(f"Final max_shares: {max_shares}, Invest amount: {invest_amount:.4f}, Actual cash used: {actual_cash_used:.4f}, Estimated entry cost: {estimated_entry_cost:.4f}")

        return max_shares, actual_margin_multiplier, overall_margin_multiplier, estimated_entry_cost, actual_cash_used, max_additional_shares, invest_amount

    
    def create_new_position(area: TouchArea, timestamp: datetime, open_price: float, high_price: float, low_price: float, close_price: float, prev_close: float):
        nonlocal balance, current_id, total_account_value, open_positions, trades_executed, is_marginable

        if open_positions or balance <= 0:
            return NO_POSITION_OPENED
        # Check if the stop buy would have executed based on high/low. close price unnecessary (i think)
        if area.is_long:
            if prev_close > area.get_buy_price:
                return NO_POSITION_OPENED
            if high_price < area.get_buy_price or close_price > high_price:
                return NO_POSITION_OPENED
        else:  # short
            if prev_close < area.get_buy_price:
                return NO_POSITION_OPENED
            if low_price > area.get_buy_price or close_price < low_price:
                return NO_POSITION_OPENED

        # Stop buy (placed at time of min_touches) would have executed ############### IMPORTANT
        execution_price = area.get_buy_price
        
        debug3_print(f"\n{timestamp}\texecution_price: {execution_price:.4f}\t{'long' if area.is_long else 'short'}")
        debug3_print(f"  position.id {current_id}")

        # Calculate position size, etc...
        max_shares, actual_margin_multiplier, overall_margin_multiplier, estimated_entry_cost, actual_cash_used, _, invest_amount = \
            calculate_position_details( execution_price, times_buying_power)

        debug3_print(f"  Calculated position details: max_shares={max_shares}, actual_margin_multiplier={actual_margin_multiplier:.4f}, overall_margin_multiplier={overall_margin_multiplier:.4f}")
        debug3_print(f"  Estimated entry cost: {estimated_entry_cost:.4f}, Actual cash used: {actual_cash_used:.4f}")
    
        if actual_cash_used + estimated_entry_cost * overall_margin_multiplier > balance:
            debug3_print(f"  Order rejected: Insufficient balance ({actual_cash_used:.4f} > {balance:.4f})")
            return NO_POSITION_OPENED
        
        debug3_print(f"  Invest amount: {invest_amount:.4f}")

        # Create the position
        position = TradePosition(
            id=current_id,
            area=area,
            is_long=area.is_long,
            entry_time=timestamp,
            initial_balance=actual_cash_used,
            initial_shares=max_shares,
            entry_price=execution_price,
            use_margin=use_margin,
            is_marginable=is_marginable,
            times_buying_power=overall_margin_multiplier,
            actual_margin_multiplier=actual_margin_multiplier,
            current_stop_price=high_price - area.get_range if area.is_long else low_price + area.get_range,
            max_price=high_price if area.is_long else None,
            min_price=low_price if not area.is_long else None,
        )
        
        debug3_print(f'Balance {balance:.4f}, invest_amount {invest_amount:.4f}, actual_cash_used {actual_cash_used:.4f}')
        cash_needed, fees = position.initial_entry()
        
        debug3_print(f'INITIAL entry fees estimated {estimated_entry_cost:.4f}, actual {fees:.4f}')
        debug3_print(f'  cash needed {cash_needed:.4f}')
        
        assert estimated_entry_cost >= fees
        
        current_id += 1
        
        # Add to open positions
        open_positions[area.id] = position
        
        debug2_print(f"{'res' if area.is_long else 'sup'} area {area.id}: {position.id} {timestamp} - Enter {'Long ' if area.is_long else 'Short'} at {execution_price:.4f}. "
              f"Shares: {max_shares}, Amount: ${invest_amount:.4f} (Margin: {actual_margin_multiplier:.2f}x, Overall: {overall_margin_multiplier:.2f}x, Sub-positions: {len(position.sub_positions)})")
        # debug2_print(cash_needed + fees, fees)
        rebalance(-cash_needed - fees, close_price)
        return POSITION_OPENED
    
    def calculate_max_shares(current_price, times_buying_power, is_marginable, existing_sub_positions: Optional[List[int]] = None):
        nonlocal balance, use_margin
        
        current_shares = 0
        if existing_sub_positions:
            current_shares = sum(existing_sub_positions)
        
        if use_margin and is_marginable:
            initial_margin_requirement = 0.5  # 50% for marginable securities
            overall_margin_multiplier = min(times_buying_power, 4.0)
        else:
            initial_margin_requirement = 1.0  # 100% for non-marginable securities
            overall_margin_multiplier = min(times_buying_power, 1.0)
        
        actual_margin_multiplier = min(overall_margin_multiplier, 1.0/initial_margin_requirement)
        available_balance = balance * overall_margin_multiplier
        
        max_additional_shares = math.floor(available_balance / current_price)
        estimated_entry_cost = TradePosition.estimate_entry_cost(max_additional_shares, overall_margin_multiplier, existing_sub_positions)
        
        if max_additional_shares * current_price + estimated_entry_cost * overall_margin_multiplier > available_balance:
            max_additional_shares = math.floor((available_balance - estimated_entry_cost * overall_margin_multiplier) / current_price)
        
        max_shares = max_additional_shares + current_shares
        
        if times_buying_power > 2 and is_marginable and max_shares % 2 == 1:
            max_shares -= 1
            max_additional_shares -= 1
        
        return max_shares, max_additional_shares, actual_margin_multiplier, overall_margin_multiplier, estimated_entry_cost

    def calculate_target_shares(position: TradePosition, current_price, max_shares):
        if position.is_long:
            price_movement = current_price - position.current_stop_price
        else:
            price_movement = position.current_stop_price - current_price
        target_pct = min(max(0, price_movement / position.area.get_range), 1.0)
        target_shares = math.floor(target_pct * max_shares)
        
        if position.times_buying_power > 2 and target_shares % 2 != 0:
            target_shares -= 1
        
        return target_shares

    def update_positions(timestamp:datetime, open_price, high_price, low_price, close_price):
        nonlocal trades_executed, warning_count_insuf_none, warning_count_insuf_some
        positions_to_remove = []

        # if using trailing stops, exit_price = None
        def perform_exit(area_id, position, exit_price=None):
            nonlocal trades_executed
            position.close(timestamp, position.current_stop_price if exit_price is None else exit_price)
            trades_executed += 1
            position.area.record_entry_exit(position.entry_time, position.entry_price, 
                                            timestamp, position.current_stop_price if exit_price is None else exit_price)
            position.area.terminate(touch_area_collection)
            positions_to_remove.append(area_id)

        for area_id, position in open_positions.items():
            existing_sub_positions = [sp.shares for sp in position.sub_positions if sp.shares > 0]
            price_at_action = None
            
            max_shares, max_additional_shares, _, _, _ = calculate_max_shares(close_price, position.times_buying_power, position.is_marginable, existing_sub_positions)
            target_shares = calculate_target_shares(position, close_price, max_shares)
            if not price_at_action and (position.update_stop_price(close_price) or target_shares == 0):
                debug2_print(f"  Trailing Stop - Exiting at close: {close_price:.4f} {'<=' if position.is_long else '>='} {position.current_stop_price:.4f}")
                perform_exit(area_id, position, close_price) # pass price into function since NOT using trailing stops
                price_at_action = close_price
            
            if not price_at_action:
                price_at_action = close_price
            assert target_shares <= max_shares

            target_pct = target_shares / max_shares
            current_pct = min(1.0, position.shares / max_shares)
            assert 0.0 <= target_pct <= 1.0, target_pct
            assert 0.0 <= current_pct <= 1.0, current_pct
...
            if target_shares < position.shares:
                shares_to_adjust = position.shares - target_shares
                if shares_to_adjust > 0:
                    realized_pnl, cash_released, fees = position.partial_exit(timestamp, price_at_action, shares_to_adjust)
                    rebalance(cash_released + realized_pnl - fees, price_at_action)

            elif target_shares > position.shares:
                debug3_print(f"\n{timestamp}\tprice_at_action: {price_at_action:.4f}\t{'long' if position.is_long else 'short'}")
                debug3_print(f"  position.id {position.id}")
                shares_to_adjust = target_shares - position.shares
                if shares_to_adjust > 0:
                    debug3_print(f"    Current -> Target percentage: {current_pct*100:.2f}% ({position.shares}) -> {target_pct*100:.2f}% ({target_shares})")
                    
                    max_shares, _, _, estimated_entry_cost, actual_cash_used, max_additional_shares, invest_amount = calculate_position_details(
                        price_at_action,
                        position.times_buying_power,
                        existing_sub_positions,
                        target_shares
                    )

                    shares_to_buy = min(shares_to_adjust, max_additional_shares)
                    
                    if shares_to_buy > 0:
                        debug3_print(f'Balance: {balance:.4f}')
                        cash_needed, fees = position.partial_entry(timestamp, price_at_action, shares_to_buy)
                        debug3_print(f'PARTIAL entry fees estimated {estimated_entry_cost:.4f}, actual {fees:.4f}')
                        debug3_print(f'  cash needed {cash_needed:.4f}')
                        
                        debug3_print(f"  Partial entry complete - Shares bought: {shares_to_buy}, Cash used: {cash_needed:.2f}")
                        rebalance(-cash_needed - fees, price_at_action)

                        if shares_to_buy < shares_to_adjust:
                            warning_count_insuf_some += 1
                    else:
                        debug3_print(f'PARTIAL entry SKIPPED.')
                        debug3_print(f'  shares_to_adjust {shares_to_adjust}, max_additional_shares {max_additional_shares}, shares_to_buy {shares_to_buy}.')
                        debug3_print(f'  fees estimated {estimated_entry_cost:.4f}')
                        warning_count_insuf_none += 1

        temp = {}
        for area_id in positions_to_remove:
            temp[area_id] = open_positions[area_id]
            del open_positions[area_id]
        for area_id in positions_to_remove:
            exit_action(area_id, temp[area_id])
        
        update_total_account_value(close_price, 'AFTER removing exited positions')
    ...
    is_marginable = is_security_marginable(symbol)
    print(f'{symbol} is {'NOT ' if not is_marginable else ''}marginable.')

    balance = initial_investment
    total_account_value = initial_investment

    trades = []  # List to store all trades
    trades_executed = 0
    open_positions = {}
    current_id = 0
    
    warning_count_insuf_some = 0
    warning_count_insuf_none = 0
    
    daily_data = None
    current_date, market_open, market_close = None, None, None
    for i in tqdm(range(1, len(timestamps))):
        current_time = timestamps[i].tz_convert(ny_tz)
        
        if current_time.date() != current_date:
            # New day, reset daily data
            current_date = current_time.date()
            daily_data = df[timestamps.date == current_date]
            
            market_open, market_close = market_hours.get(str(current_date), (None, None))
            if market_open and market_close:
                date_obj = pd.Timestamp(current_date).tz_localize(ny_tz)
                if start_time:
                    day_start_time = date_obj.replace(hour=start_time.hour, minute=start_time.minute)
                else:
                    day_start_time = market_open
                if end_time:
                    day_end_time = min(date_obj.replace(hour=end_time.hour, minute=end_time.minute), market_close - pd.Timedelta(minutes=3))
                else:
                    day_end_time = market_close - pd.Timedelta(minutes=3)

                if soft_start_time:
                    day_soft_start_time = max(market_open, day_start_time, date_obj.replace(hour=soft_start_time.hour, minute=soft_start_time.minute))
                else:
                    day_soft_start_time = max(market_open, day_start_time)
                    
            daily_index = 1 # start at 2nd position
            assert not open_positions
            
            soft_end_triggered = False
            
        
        if not market_open or not market_close:
            continue
        
        if day_soft_start_time <= current_time < day_end_time and daily_index < len(daily_data)-1 and i < len(df)-1 :
            if soft_end_time and not soft_end_triggered:
                if current_time >= pd.Timestamp.combine(current_date, soft_end_time).tz_localize(ny_tz):
                    soft_end_triggered = True

            prev_close = daily_data['close'].iloc[daily_index - 1]
            open_price = daily_data['open'].iloc[daily_index]
            high_price = daily_data['high'].iloc[daily_index]
            low_price = daily_data['low'].iloc[daily_index]
            close_price = daily_data['close'].iloc[daily_index]

            update_positions(current_time, open_price, close_price, high_price, low_price)
            
            if not soft_end_triggered:
                active_areas = touch_area_collection.get_active_areas(current_time)
                for area in active_areas:
                    if balance <= 0:
                        break
                    if open_positions:
                        break
                    
                    if (area.is_long and do_longs) or (not area.is_long and do_shorts):

                        if create_new_position(area, current_time, open_price, high_price, low_price, close_price, prev_close):
                            break  # Exit the loop after placing a position
        elif current_time >= day_end_time:
            close_all_positions(current_time, df['close'].iloc[i])
        elif i >= len(df)-1:
            close_all_positions(current_time, df['close'].iloc[i]) # only for testing. not in live environment
            
        daily_index += 1

    if current_time >= day_end_time:
        assert not open_positions
...