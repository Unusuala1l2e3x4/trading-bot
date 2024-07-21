    # Certainly. I'll provide enhanced debug statements and assertions for the key functions involved in this process. We'll focus on the `partial_exit` method of the `TradePosition` class and the `update_positions` function in the main strategy loop. Here's how we can modify these:

    # 1. In the `TradePosition` class, update the `partial_exit` method:

    # ```python
    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int):
        print(f"DEBUG: Entering partial_exit - Time: {exit_time}, Price: {exit_price}, Shares to sell: {shares_to_sell}")
        print(f"DEBUG: Current position - Shares: {self.shares}, Cash committed: {self.cash_committed:.2f}")
        
        initial_shares = self.shares
        initial_cash_committed = self.cash_committed
        cash_released = 0
        realized_pnl = 0
        remaining_shares_to_sell = shares_to_sell

        for sp in self.sub_positions:
            if sp.shares > 0 and remaining_shares_to_sell > 0:
                shares_sold = min(sp.shares, remaining_shares_to_sell)
                
                sub_cash_released = (shares_sold / sp.shares) * sp.cash_committed
                sp_realized_pnl = (exit_price - sp.entry_price) * shares_sold if self.is_long else (sp.entry_price - exit_price) * shares_sold
                
                print(f"DEBUG: Selling from sub-position - Entry price: {sp.entry_price:.4f}, Shares sold: {shares_sold}, Realized PnL: {sp_realized_pnl:.2f}")
                
                sp.shares -= shares_sold
                self.shares -= shares_sold
                sp.cash_committed -= sub_cash_released
                cash_released += sub_cash_released
                realized_pnl += sp_realized_pnl
                
                self.add_transaction(exit_time, shares_sold, exit_price, is_entry=False, sub_position=sp, sp_realized_pnl=sp_realized_pnl)
                remaining_shares_to_sell -= shares_sold

                if sp.shares == 0:
                    sp.exit_time = exit_time
                    sp.exit_price = exit_price

        self.cash_committed -= cash_released
        self.market_value = self.shares * exit_price
        self.partial_exit_count += 1

        assert self.shares == initial_shares - shares_to_sell, f"Share mismatch after partial exit: {self.shares} != {initial_shares - shares_to_sell}"
        assert abs(self.cash_committed - (initial_cash_committed - cash_released)) < 1e-8, f"Cash committed mismatch: {self.cash_committed:.2f} != {initial_cash_committed - cash_released:.2f}"

        print(f"DEBUG: Partial exit complete - New shares: {self.shares}, Cash released: {cash_released:.2f}, Realized PnL: {realized_pnl:.2f}")
        print(f"DEBUG: Remaining sub-positions:")
        for i, sp in enumerate(self.sub_positions):
            if sp.shares > 0:
                print(f"  Sub-position {i}: Shares: {sp.shares}, Entry price: {sp.entry_price:.4f}")

        return realized_pnl, realized_pnl / self.times_buying_power, cash_released
    # ```

    # 2. In the main strategy loop, update the `update_positions` function:

    # ```python
    def update_positions(timestamp, open_price, high_price, low_price, close_price):
        print(f"\nDEBUG: Updating positions at {timestamp}, Close price: {close_price:.4f}")
        
        for area_id, position in list(open_positions.items()):
            print(f"\nDEBUG: Processing position {position.id} for area {area_id}")
            print(f"  Current position - Shares: {position.shares}, Cash committed: {position.cash_committed:.2f}")
            
            if position.update_stop_price(close_price):
                print(f"  Stop price triggered: {position.current_stop_price:.4f}")
                perform_exit(area_id, position)
            else:
                target_shares = calculate_target_shares(position, close_price)
                print(f"  Target shares: {target_shares}, Current shares: {position.shares}")
                
                if target_shares < position.shares:
                    shares_to_adjust = position.shares - target_shares
                    print(f"  Initiating partial exit - Shares to sell: {shares_to_adjust}")
                    realized_pnl, adjusted_realized_pnl, cash_released = position.partial_exit(timestamp, close_price, shares_to_adjust)
                    print(f"  Partial exit complete - Realized PnL: {realized_pnl:.2f}, Adjusted PnL: {adjusted_realized_pnl:.2f}, Cash released: {cash_released:.2f}")
                    rebalance(cash_released + adjusted_realized_pnl, close_price)
                elif target_shares > position.shares:
                    shares_to_adjust = target_shares - position.shares
                    print(f"  Initiating partial entry - Shares to buy: {shares_to_adjust}")
                    max_shares, actual_margin_multiplier, overall_margin_multiplier = calculate_max_position_size(close_price)
                    shares_to_buy = min(shares_to_adjust, max_shares)
                    if shares_to_buy > 0:
                        cash_needed = position.partial_entry(timestamp, close_price, shares_to_buy)
                        print(f"  Partial entry complete - Shares bought: {shares_to_buy}, Cash used: {cash_needed:.2f}")
                        rebalance(-cash_needed, close_price)
                    else:
                        print(f"  WARNING: Insufficient balance to buy any shares. Desired: {shares_to_adjust}, Max possible: {max_shares}")
            
            print(f"  Final position - Shares: {position.shares}, Cash committed: {position.cash_committed:.2f}")
            
        update_total_account_value(close_price, 'AFTER removing exited positions')
    # ```

    # 3. Add an assertion in the `rebalance` function:

    # ```python
    def rebalance(cash_change: float, current_price: float = None):
        nonlocal balance, total_account_value
        old_balance = balance
        new_balance = balance + cash_change
        
        assert new_balance >= 0, f"Negative balance encountered: {new_balance:.2f}"
        
        balance = new_balance

        if current_price is not None:
            update_total_account_value(current_price, 'REBALANCE')
        
        print(f"Rebalance: Old balance: {old_balance:.4f}, Change: {cash_change:.4f}, New balance: {balance:.4f}, Total Account Value: {total_account_value:.4f}")

        assert abs(total_account_value - (balance + sum(pos.market_value for pos in open_positions.values()))) < 1e-8, \
            f"Total account value mismatch: {total_account_value:.2f} != {balance + sum(pos.market_value for pos in open_positions.values()):.2f}"
    # ```

    # These debug statements and assertions will help track the flow of the partial exits and entries, providing more detailed information about the state of the positions and the changes being made. They will also help catch any inconsistencies in the calculations.

    # When you run the strategy with these modifications, you should get a clearer picture of what's happening during the critical moments, especially around the 11:46:00 timestamp where the deviation occurs. This will help identify any discrepancies in how the 2x and 2.1x leverage cases are being handled.