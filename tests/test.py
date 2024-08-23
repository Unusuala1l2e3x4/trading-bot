from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

        
        
class TradingStrategy:
    # ... (existing methods)
    def __init__(self, ..., is_live_trading: bool):
        self.is_live_trading = is_live_trading
        # ... other initialization code ...

    def process_live_data(self, current_timestamp: datetime):
        # ... other code ...
        if self.should_close_all_positions(current_timestamp, day_end_time, df_index, self.is_live_trading):
            # Close all positions
        
        if self.is_trading_time(current_timestamp, day_soft_start_time, day_end_time, daily_index, daily_data, i, self.is_live_trading):
            # Proceed with trading logic
        
        # ... rest of the method ...
        
        
    def process_live_data(self, current_timestamp: datetime):
        """
        Process the latest data point for live trading.
        
        :param current_timestamp: The timestamp of the latest data point
        :return: MarketOrderRequest if an order should be placed, None otherwise
        """
        try:
            # Get the latest data point
            latest_data = self.df.loc[self.df.index.get_level_values('timestamp') == current_timestamp].iloc[-1]
            prev_data = self.df.loc[self.df.index.get_level_values('timestamp') < current_timestamp].iloc[-1]

            # Update positions based on the latest data
            self.update_positions(current_timestamp, latest_data)

            # Process active areas
            if self.balance > 0 and not self.open_positions:
                active_areas = self.touch_area_collection.get_active_areas(current_timestamp)
                for area in active_areas:
                    if ((area.is_long and (self.params.do_longs or self.params.sim_longs)) or 
                        (not area.is_long and (self.params.do_shorts or self.params.sim_shorts))):
                        
                        result = self.place_stop_market_buy(area, current_timestamp, latest_data, prev_data['close'])
                        if result == POSITION_OPENED:
                            # Prepare order details
                            position = self.open_positions[area.id]
                            order_side = OrderSide.BUY if position.is_long else OrderSide.SELL
                            
                            order_request = MarketOrderRequest(
                                symbol=self.symbol,
                                qty=position.shares,
                                side=order_side,
                                time_in_force=TimeInForce.DAY
                            )
                            return order_request

            # Check if we need to close all positions (e.g., end of day)
            if self.should_close_all_positions(current_timestamp):
                for area_id, position in list(self.open_positions.items()):
                    order_side = OrderSide.SELL if position.is_long else OrderSide.BUY
                    order_request = MarketOrderRequest(
                        symbol=self.symbol,
                        qty=position.shares,
                        side=order_side,
                        time_in_force=TimeInForce.DAY
                    )
                    return order_request

        except Exception as e:
            self.log(f"Error in process_live_data: {e}", logging.ERROR)

        return None

    # ... (other methods remain the same)

    
    
class LiveTrader:
    # ... (other methods)
    
    def __init__(self, api_key, secret_key, symbol, initial_balance, touch_detection_params: LiveTouchDetectionParameters, strategy_params: StrategyParameters, simulation_mode=False):
        # ... (other initializations)
        self.trading_strategy = TradingStrategy(touch_detection_params, strategy_params)
        # ... (rest of the initialization)
        
    # ... (other methods)
        
    async def execute_trading_logic(self):
        try:
            if not self.is_ready:
                self.log("Data not ready for trading")
                return

            current_timestamp = self.data.index.get_level_values('timestamp').max()
            order_request = self.trading_strategy.process_live_data(current_timestamp)

            if order_request:
                await self.place_order(order_request)

        except Exception as e:
            self.log(f"Error in execute_trading_logic: {e}", logging.ERROR)

    async def place_order(self, order_request: MarketOrderRequest):
        try:
            order = self.trading_client.submit_order(order_request)
            self.log(f"Placed {order_request.side} order for {order_request.qty} shares of {order_request.symbol}")
            return order
        except Exception as e:
            self.log(f"Error placing order: {e}", logging.ERROR)
            
            