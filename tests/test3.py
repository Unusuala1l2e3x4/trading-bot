    # You're right, that's a good optimization. We can calculate max_shares earlier in the process and use it in calculate_target_shares. Here's how we can modify the code to implement this change:

    # 1. First, let's create a new function to calculate max_shares independently:

    # ```python
    def calculate_max_shares(current_price, balance, times_buying_power, is_marginable):
        if is_marginable:
            initial_margin_requirement = 0.5  # 50% for marginable securities
            overall_margin_multiplier = min(times_buying_power, 4.0)
        else:
            initial_margin_requirement = 1.0  # 100% for non-marginable securities
            overall_margin_multiplier = min(times_buying_power, 1.0)
        
        available_balance = balance * overall_margin_multiplier
        max_shares = math.floor(available_balance / current_price)
        estimated_entry_cost = TradePosition.estimate_entry_cost(max_shares, overall_margin_multiplier)
        
        # Adjust max_shares if necessary
        if max_shares * current_price + estimated_entry_cost * overall_margin_multiplier > available_balance:
            max_shares = math.floor((available_balance - estimated_entry_cost * overall_margin_multiplier) / current_price)
        
        # Ensure max_shares is a multiple of 2 when times_buying_power > 2 and is_marginable
        if times_buying_power > 2 and is_marginable and max_shares % 2 == 1:
            max_shares -= 1
        
        return max_shares
    # ```

    # 2. Now, let's modify the calculate_target_shares function to use max_shares:

    # ```python
    def calculate_target_shares(position: TradePosition, current_price, max_shares):
        if position.is_long:
            price_movement = current_price - position.current_stop_price
        else:
            price_movement = position.current_stop_price - current_price
        target_pct = min(max(0, price_movement / position.area.get_range), 1.0)
        target_shares = math.floor(target_pct * max_shares)
        
        # Ensure target_shares is even when times_buying_power > 2
        if position.times_buying_power > 2 and target_shares % 2 != 0:
            target_shares -= 1
        
        return target_shares
    # ```

    # 3. Update the update_positions function to calculate max_shares before calling calculate_target_shares:

    # ```python
    def update_positions(timestamp:datetime, open_price, high_price, low_price, close_price):
        # ... (existing code)

        for area_id, position in open_positions.items():
            # ... (existing code)

            price_at_action = close_price
            max_shares = calculate_max_shares(price_at_action, balance, position.times_buying_power, position.is_marginable)
            target_shares = calculate_target_shares(position, price_at_action, max_shares)

            # ... (rest of the function remains the same)
    # ```

    # 4. Finally, update the calculate_position_details function to use the pre-calculated max_shares:

    # ```python
    def calculate_position_details(current_price, times_buying_power, existing_sub_positions, target_shares, max_shares):
        # ... (existing code, remove the max_shares calculation for initial entry)

        if existing_sub_positions is None:
            # Initial entry
            debug3_print(f"  initial: max_shares: {max_shares}")
        else:
            # Partial entry
            current_shares = sum(existing_sub_positions)
            max_additional_shares = min(target_shares - current_shares, max_shares - current_shares)
            
            debug3_print(f"  BEFORE adjust - partial: Current shares: {current_shares}, Max additional shares: {max_additional_shares}, Target shares: {target_shares}")
            
            max_shares = current_shares + max_additional_shares
            
            debug3_print(f"  AFTER adjust - partial: Current shares: {current_shares}, Max additional shares: {max_additional_shares}")

        # ... (rest of the function remains the same)
    # ```

    # These changes will optimize the process by calculating max_shares once and using it throughout the update_positions function, including in the calculate_target_shares function. This should improve efficiency and consistency in your calculations.