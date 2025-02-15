from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from trading_bot.VolumeProfile import VolumeProfile
from zoneinfo import ZoneInfo

ny_tz = ZoneInfo("America/New_York")


@dataclass
class PreMarketBar:
    """Lightweight bar class for pre-market volume profile updates."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    ATR: float

    @classmethod
    def from_row(cls, row: pd.Series) -> 'PreMarketBar':
        timestamp = row.name[1] if isinstance(row.name, tuple) else row.name
        return cls(
            timestamp=timestamp,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            vwap=row['vwap'],
            ATR=row['ATR'],
        )


@dataclass
class AnchoredVWAPMetrics:
    """Container for VWAP metrics calculated from a sequence of bars"""
    vwap: float = np.nan  # Anchored VWAP value
    vwap_dist: float = np.nan  # Distance from close to VWAP
    vwap_std: float = np.nan  # Standard deviation of prices from VWAP
    vwap_std_close: float = np.nan  # Close price in standard deviations from VWAP
    
    
@dataclass
class TypedBarData:
    """Wrapper for bar data with proper type hints."""
    # Index fields
    symbol: str
    timestamp: datetime
    
    # Bar data fields
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: float
    
    vwap: float
    
    MACD: float
    MACD_signal: float
    MACD_hist: float
    MACD_hist_roc: float
    RSI: float
    RSI_roc: float
    MFI: float
    MFI_roc: float
        
    VWAP: float
    VWAP_dist: float
    VWAP_std: float
    VWAP_std_close: float
    
    central_value: float
    central_value_dist: float

    exit_ema: float
    exit_ema_dist: float

    is_res: bool
    
    # New bar data fields
    H_L: float  # High-Low range
    # H_PC: float  # High-Previous Close
    # L_PC: float  # Low-Previous Close
    # TR: float  # True Range
    
    ATR: float  # Average True Range (EMA)
    MTR: float  # Median True Range    
    
    # shares_per_trade: float
    avg_volume: float # (EMA)
    avg_trade_count: float # (EMA)
    log_return: float
    volatility: float # (rolling)

    rolling_range_min_4: float  # 4-bar minimum range
    rolling_range_min_7: float  # 7-bar minimum range
    rolling_ATR: float  # rolling ATR average

    # New trend metrics
    ADX: float                      # Overall trend strength (0-100)
    trend_strength: float           # Directional trend strength (-100 to +100)

    
    # Store original row for access to any additional columns
    _row: pd.Series = field(repr=False)
    
    # Overall volume distribution metrics (using all volume)
    buy_vol_balance: float = 0.0      # Sign indicates if volume is mostly above/below price
    buy_vol_concentration: float = 0.0 # How clustered volume is near price
    buy_vol_kurtosis: float = 0.0     # Peakedness of distribution
    sell_vol_balance: float = 0.0
    sell_vol_concentration: float = 0.0
    sell_vol_kurtosis: float = 0.0
    
    # HVN-specific metrics (using only detected peaks)
    buy_hvn_balance: float = 0.0      # Sign indicates if HVNs are mostly above/below price
    buy_hvn_concentration: float = 0.0 # How clustered HVNs are near price
    buy_hvn_avg_prominence: float = 0.0 # Average strength of detected HVNs
    sell_hvn_balance: float = 0.0 
    sell_hvn_concentration: float = 0.0
    sell_hvn_avg_prominence: float = 0.0
    
    
    # Add threshold parameters as class attributes with defaults
    doji_ratio_threshold: float = 0.1
    wick_ratio_threshold: float = 2.0
    atr_compression_threshold: float = 0.9
    volume_compression_threshold: float = 0.7
    min_indecision_signals: int = 2


    def __post_init__(self):
        self.rsi_overbought = 65
        self.rsi_oversold = 35
        self.mfi_overbought = 75
        self.mfi_oversold = 25

    @property 
    def market_phase_inc(self) -> float:
        """
        Market phase normalized to [0,1] where:
        - 0.0 = market open (9:30)
        - 0.25 = pre-lunch (11:00) 
        - 0.5 = lunch (12:30)
        - 0.75 = post-lunch (2:00)
        - 1.0 = market close (16:00)
        """
        minutes_from_open = (self.timestamp - pd.Timestamp.combine(self.timestamp.date(), pd.Timestamp("9:30").time()).tz_localize(ny_tz)).total_seconds() / 60
        total_market_minutes = 390  # 6.5 hours * 60
        return np.clip(minutes_from_open / total_market_minutes, 0, 1)

    @property 
    def market_phase_dec(self) -> float:
        """
        Inverse of market_phase_inc (1 at open, 0 at close)
        """
        return 1 - self.market_phase_inc
    

    def update_volume_metrics(self, volume_profile: VolumeProfile, atr: Optional[float] = None):
        """Update all volume profile related attributes and add to _row."""
        if volume_profile.buy_profile is None or volume_profile.sell_profile is None:
            return
                
        # Calculate moments for both profiles
        (
            (self.buy_vol_balance, self.buy_vol_concentration, self.buy_vol_kurtosis),
            (self.sell_vol_balance, self.sell_vol_concentration, self.sell_vol_kurtosis)
        ) = volume_profile.calculate_profile_moments(self.close)
        
        # Calculate HVN metrics for both profiles
        (
            (self.buy_hvn_balance, self.buy_hvn_concentration, self.buy_hvn_avg_prominence),
            (self.sell_hvn_balance, self.sell_hvn_concentration, self.sell_hvn_avg_prominence)
        ) = volume_profile.get_hvn_metrics(self.close, self.ATR if atr is None else atr)
        
        # Add all metrics to _row for DataFrame conversion
        # Buy volume metrics
        self._row['buy_vol_balance'] = self.buy_vol_balance
        self._row['buy_vol_concentration'] = self.buy_vol_concentration
        self._row['buy_vol_kurtosis'] = self.buy_vol_kurtosis
        self._row['buy_hvn_balance'] = self.buy_hvn_balance
        self._row['buy_hvn_concentration'] = self.buy_hvn_concentration
        self._row['buy_hvn_avg_prominence'] = self.buy_hvn_avg_prominence
        
        # Sell volume metrics
        self._row['sell_vol_balance'] = self.sell_vol_balance
        self._row['sell_vol_concentration'] = self.sell_vol_concentration
        self._row['sell_vol_kurtosis'] = self.sell_vol_kurtosis
        self._row['sell_hvn_balance'] = self.sell_hvn_balance
        self._row['sell_hvn_concentration'] = self.sell_hvn_concentration
        self._row['sell_hvn_avg_prominence'] = self.sell_hvn_avg_prominence


    @property
    def shares_per_trade(self) -> float:
        return self.volume / self.trade_count
    
    @property
    def body_size(self) -> float:
        """Calculate the absolute size of the candle's body."""
        return abs(self.close - self.open)
    
    @property
    def range_size(self) -> float:
        """Calculate the total range of the candle (high minus low)."""
        return self.high - self.low
    
    @property
    def doji_ratio(self) -> float:
        """Calculate the ratio of the body size to the total range."""
        if self.range_size == 0:
            return 0
        return (self.close - self.open) / self.range_size
    
    @property
    def doji_ratio_abs(self) -> float:
        """Calculate the ratio of the body size to the total range."""
        return abs(self.doji_ratio)
    
    @property
    def wick_ratio(self) -> float:
        """Calculate the ratio of the total range to the body size."""
        if self.body_size == 0:
            # return float('inf')
            return self.range_size / 0.01
        return self.range_size / self.body_size

    @property
    def is_doji(self) -> bool:
        """
        Determine if the bar is a doji candle.

        For 1-minute bars, a threshold of 0.1 may be more appropriate to capture more doji patterns,
        as small price movements can be significant.

        :param threshold: The maximum ratio of body size to range size to consider as a doji.
        :return: True if the bar is a doji, False otherwise.
        """
        return self.doji_ratio_abs < self.doji_ratio_threshold
    
    @property
    def has_long_wicks(self) -> bool:
        """
        Determine if the bar has long wicks relative to its body.

        For 1-minute bars, a lower threshold like 1.5-2.0 may be more suitable,
        as wicks can be proportionally larger on shorter timeframes.

        :param threshold: The minimum ratio of range size to body size to consider as having long wicks.
        :return: True if the bar has long wicks, False otherwise.
        """
        return self.wick_ratio > self.wick_ratio_threshold
    
    
    @property
    def nr4_hl_diff(self) -> bool:
        return self.H_L - self.rolling_range_min_4
    
    @property
    def nr7_hl_diff(self) -> bool:
        return self.H_L - self.rolling_range_min_7
    
    @property
    def is_nr4(self) -> bool:
        """Check if the current bar is a Narrow Range 4 (NR4) bar."""
        return self.nr4_hl_diff <= 0
    
    @property
    def is_nr7(self) -> bool:
        """Check if the current bar is a Narrow Range 7 (NR7) bar."""
        return self.nr7_hl_diff <= 0
    
    @property
    def ATR_ratio(self) -> float:
        """ATR relative to rolling avg."""
        if self.rolling_ATR == 0:
            return 0
        return self.ATR / self.rolling_ATR
    
    @property
    def is_atr_compressed(self) -> bool:
        """
        Check if the ATR is compressed relative to the recent average ATR.

        For 1-minute bars, adjusting the threshold to 0.7-0.8 may better capture periods of reduced volatility.

        :param threshold: The maximum ratio of current ATR to rolling ATR to consider as compressed.
        :return: True if the ATR is compressed, False otherwise.
        """
        return self.ATR_ratio < self.atr_compression_threshold

    @property
    def volume_ratio(self) -> float:
        """Volume relative to recent average."""
        if self.avg_volume == 0:
            return 0
        return self.volume / self.avg_volume
    
    @property
    def is_volume_compressed(self) -> bool:
        """Check if volume shows indecision (lower than average)."""
        return self.volume_ratio < self.volume_compression_threshold

    def get_indecision_signals(self): # -> List[Tuple[str, bool]]:
        """
        Get list of tuples containing (description, boolean) for each indecision signal.
        """
        return [
            (f"Doji Abs (ratio={self.doji_ratio_abs:.3f} < {self.doji_ratio_threshold})", 
             self.is_doji),
            
            (f"Long Wicks (ratio={self.wick_ratio:.2f} > {self.wick_ratio_threshold})", 
             self.has_long_wicks),
            
            (f"NR4 (range={self.H_L:.3f} <= {self.rolling_range_min_4:.3f})", 
             self.is_nr4),
            
            (f"NR7 (range={self.H_L:.3f} <= {self.rolling_range_min_7:.3f})", 
             self.is_nr7),
            
            (f"ATR Compressed (ratio={self.ATR_ratio:.2f} < {self.atr_compression_threshold})", 
             self.is_atr_compressed),
            
            (f"Volume Compressed (ratio={self.volume_ratio:.2f} < {self.volume_compression_threshold})", 
             self.is_volume_compressed)
        ]

    @property
    def indecision_count(self) -> int:
        """Count how many indecision indicators are present."""
        return sum(signal[1] for signal in self.get_indecision_signals())
    
    @property
    def describe_indecision(self) -> str:
        """Get a readable description of all indecision signals."""
        signals = self.get_indecision_signals()
        active_signals = [desc for desc, is_active in signals if is_active]
        return f"{len(active_signals)} signals active:\n" + "\n".join(
            f"  - {desc}" for desc in active_signals
        )

    @property
    def shows_indecision(self) -> bool:
        """
        Check if enough indecision signals are present.

        For 1-minute bars, reducing the minimum required signals to 2-3 may capture more opportunities.

        :param min_signals: The minimum number of indecision indicators to consider as indecision.
        :return: True if enough indecision signals are present, False otherwise.
        """
        return self.indecision_count >= self.min_indecision_signals
    
    @property
    def has_divergence(self) -> bool:
        """Check if price and indicators are showing divergence."""
        price_trending_up = self.close > self.open
        rsi_trending_down = self.RSI_roc < 0
        mfi_trending_down = self.MFI_roc < 0
        
        price_trending_down = self.close < self.open
        rsi_trending_up = self.RSI_roc > 0
        mfi_trending_up = self.MFI_roc > 0
        
        return (price_trending_up and rsi_trending_down and mfi_trending_down) or \
            (price_trending_down and rsi_trending_up and mfi_trending_up)
    
    
    @property
    def rsi_divergence(self) -> float:
        """Calculate RSI price divergence strength.
        
        Returns:
            float: Range [-1, 1] where:
                - Positive values indicate bullish divergence (price down, RSI up)
                - Negative values indicate bearish divergence (price up, RSI down)
                - Magnitude indicates strength of divergence
                - 0 indicates no divergence
        """
        price_change = (self.close - self.open) / self.open
        rsi_change = self.RSI_roc / 100  # Convert to decimal
        
        # No divergence if price and RSI moving same direction
        if (price_change >= 0 and rsi_change >= 0) or \
        (price_change <= 0 and rsi_change <= 0):
            return 0.0
            
        # Calculate divergence strength 
        divergence = rsi_change - price_change
        return np.tanh(divergence)

    @property
    def mfi_divergence(self) -> float:
        """Calculate MFI price divergence strength.
        
        Returns:
            float: Range [-1, 1] where:
                - Positive values indicate bullish divergence (price down, MFI up)
                - Negative values indicate bearish divergence (price up, MFI down)
                - Magnitude indicates strength of divergence
                - 0 indicates no divergence
        """
        price_change = (self.close - self.open) / self.open
        mfi_change = self.MFI_roc / 100  # Convert to decimal
        
        # No divergence if price and MFI moving same direction  
        if (price_change >= 0 and mfi_change >= 0) or \
        (price_change <= 0 and mfi_change <= 0):
            return 0.0
            
        # Calculate divergence strength
        divergence = mfi_change - price_change
        return np.tanh(divergence)
        
        
        
    def shows_reversal_potential(self, area_is_long, std_threshold = 1, pre_position: bool = False) -> bool:
        """
        Check if we should switch sides based on VWAP extension and price action.
        
        For long areas (looking to short):
        - Price significantly above daily VWAP (>1.5-2 std)
        - Signs of upward momentum exhaustion
        
        For short areas (looking to long):
        - Price significantly below daily VWAP (<-1.5-2 std)
        - Signs of downward momentum exhaustion
        """
        # Negative trend_strength means bearish trend
        trend_strength_favors_reversal = (
            (area_is_long and self.trend_strength < 0) or  # Bearish trend for long area
            (not area_is_long and self.trend_strength > 0)  # Bullish trend for short area
        )
        
        # Check for price extension from VWAP
        vwap_extension = abs(self.VWAP_std_close) >= std_threshold
        
        # Check if extended in right direction for area
        properly_extended = (
            (area_is_long and self.VWAP_std_close > 0) or  # Above VWAP for long area
            (not area_is_long and self.VWAP_std_close < 0)  # Below VWAP for short area
        )
        
        # Check for momentum exhaustion using MACD
        momentum_waning = (
            (area_is_long and self.MACD_hist_roc < 0) or  # Declining momentum in long area
            (not area_is_long and self.MACD_hist_roc > 0)  # Rising momentum in short area
        )
        
        # Volume characteristics
        volume_confirms = self.volume_ratio > 1.0  # Above average volume
        
        # Core conditions
        basic_conditions = (
            vwap_extension and
            properly_extended and 
            momentum_waning
        )
        
        # Additional confirmation
        supporting_conditions = (
            trend_strength_favors_reversal and
            volume_confirms
        )
        
        return basic_conditions and supporting_conditions

    def describe_reversal_potential(self, area_was_long, std_threshold = 1) -> str:
        """Enhanced description separating core and supporting conditions."""
        basic_conditions_list = []
        supporting_conditions_list = []
        
        # Basic Conditions
        # 1. VWAP extension
        vwap_extension = abs(self.VWAP_std_close) >= std_threshold
        if vwap_extension:
            basic_conditions_list.append(
                f"Extended {abs(self.VWAP_std_close):.2f} std {'above' if self.VWAP_std_close > 0 else 'below'} VWAP"
            )
        
        # 2. Extended in right direction
        properly_extended = (
            (area_was_long and self.VWAP_std_close < 0) or  # Below VWAP to reverse long
            (not area_was_long and self.VWAP_std_close > 0)  # Above VWAP to reverse short
        )
        if properly_extended:
            basic_conditions_list.append(
                f"Extension direction matches potential reversal"
            )
        
        # 3. Momentum
        momentum_waning = (
            (area_was_long and self.MACD_hist_roc < 0) or 
            (not area_was_long and self.MACD_hist_roc > 0)
        )
        if momentum_waning:
            supporting_conditions_list.append(
                f"{'Bullish' if not area_was_long else 'Bearish'} momentum waning (MACD hist ROC={self.MACD_hist_roc:.3f})"
            )

        # Supporting Conditions
        # 1. Trend strength
        trend_strength_favors_reversal = (
            (area_was_long and self.trend_strength < 0) or 
            (not area_was_long and self.trend_strength > 0)
        )
        if trend_strength_favors_reversal:
            supporting_conditions_list.append(
                f"Trend strength favors reversal ({self.trend_strength:.1f})"
            )

        # 2. Volume
        volume_confirms = self.volume_ratio > 1.0 
        if volume_confirms:
            supporting_conditions_list.append(
                f"Higher than average volume ({self.volume_ratio:.1f}x)"
            )

        # Summarize conditions
        if not basic_conditions_list and not supporting_conditions_list:
            return "No reversal conditions detected"


        # Core conditions
        basic_conditions = (
            vwap_extension and
            properly_extended
        )
        
        # Additional confirmation
        supporting_conditions = (
            momentum_waning and
            trend_strength_favors_reversal and
            volume_confirms
        )


        # has_potential = len(basic_conditions_list) >= 3  # All basic conditions met
        # has_support = len(supporting_conditions_list) > 0  # At least one supporting condition
        
        
        has_potential = basic_conditions
        has_support = supporting_conditions
        
    

        summary = []
        summary.append(f"Basic Conditions ({len(basic_conditions_list)}/2 met):")
        summary.extend(f"  - {cond}" for cond in basic_conditions_list)
        
        summary.append(f"\nSupporting Conditions ({len(supporting_conditions_list)}/3 met):")
        summary.extend(f"  - {cond}" for cond in supporting_conditions_list)

        status = "POTENTIAL REVERSAL" if has_potential and has_support else "Partial conditions"
        
        return f"{status} detected:\n" + "\n".join(summary)
            
            
    
        
    
    def update_thresholds(self,
                         doji_ratio_threshold: Optional[float] = None,
                         wick_ratio_threshold: Optional[float] = None,
                         atr_compression_threshold: Optional[float] = None,
                         volume_compression_threshold: Optional[float] = None,
                         min_indecision_signals: Optional[int] = None):
        """Update indecision thresholds."""
        if doji_ratio_threshold is not None:
            self.doji_ratio_threshold = doji_ratio_threshold
        if wick_ratio_threshold is not None:
            self.wick_ratio_threshold = wick_ratio_threshold
        if atr_compression_threshold is not None:
            self.atr_compression_threshold = atr_compression_threshold
        if volume_compression_threshold is not None:
            self.volume_compression_threshold = volume_compression_threshold
        if min_indecision_signals is not None:
            self.min_indecision_signals = min_indecision_signals
            
    @classmethod
    def from_row(cls, row: pd.Series) -> 'TypedBarData':
        """Create TypedBarData instance from DataFrame row."""
        # Extract symbol and timestamp from index
        symbol = row.name[0] if isinstance(row.name, tuple) else None
        timestamp = row.name[1] if isinstance(row.name, tuple) else row.name
        
        # Create kwargs dict with all available fields
        kwargs = {
            'symbol': symbol,
            'timestamp': timestamp,
            '_row': row
        }
        
        # Add all known fields, handling missing or NaN values
        known_fields = {k: v for k, v in cls.__annotations__.items() 
                    if not k.endswith('_threshold') and k != 'min_indecision_signals'}
        
        for field in known_fields:
            if field in ['symbol', 'timestamp', '_row']:
                continue
            
            value = row.get(field)
            kwargs[field] = None if pd.isna(value) else value

        return cls(**kwargs)
    
    def to_dict(self):
        """Convert bar data to dictionary."""
        return {
            field: getattr(self, field) 
            for field in self.__annotations__ 
            if not field.startswith('_')
        }

    def __str__(self):
        """String representation showing all attributes."""
        fields = [
            f"{field}: {getattr(self, field)}" 
            for field in self.__annotations__ 
            if not field.startswith('_')
        ]
        return f"TypedBarData({', '.join(fields)})"
    
    @staticmethod
    def get_field_names() -> list[str]:
        """Get list of field names from class annotations, excluding _row."""
        return [field for field in TypedBarData.__annotations__ if field != '_row']

    @staticmethod
    def to_dataframe(bars: list['TypedBarData']) -> pd.DataFrame:
        """Convert a list of TypedBarData objects to a DataFrame.
        
        Args:
            bars: List of TypedBarData objects
            
        Returns:
            pd.DataFrame: DataFrame where index attributes become columns
        """
        # Create list of dictionaries from each bar's Series
        data = []
        for bar in bars:
            row_dict = bar._row.to_dict()
            # Add index values if they exist (for multi-index)
            if isinstance(bar._row.name, tuple):
                row_dict['symbol'] = bar._row.name[0]
                row_dict['timestamp'] = bar._row.name[1]
            else:
                row_dict['timestamp'] = bar._row.name
            data.append(row_dict)
        
        # Create DataFrame from list of dictionaries
        df = pd.DataFrame(data)
        
        # # Set the index if needed
        # if 'symbol' in df.columns:
        #     df.set_index(['symbol', 'timestamp'], inplace=True)
        # else:
        #     df.set_index('timestamp', inplace=True)
            
        return df





    # @staticmethod
    # def shows_reversal_potential(area_is_long, rsi_overbought: float = 65, rsi_oversold: float = 35,
    #                         mfi_overbought: float = 75, mfi_oversold: float = 25, 
    #                         pre_position: bool = False) -> bool:
    
    
    
    @staticmethod
    def calculate_anchored_vwap_metrics(bars: List['TypedBarData'], 
                                      is_long: bool) -> Optional[AnchoredVWAPMetrics]:
        """
        Calculate VWAP metrics anchored at first bar through last bar.
        Always calculates metrics regardless of bar count.
        
        Args:
            bars: List of TypedBarData objects
            is_long: Whether this is a long position 
            
        Returns:
            AnchoredVWAPMetrics object containing the calculated metrics
        """
        if not bars:
            return AnchoredVWAPMetrics()
            
        # Calculate cumulative values
        cumulative_volume = sum(bar.volume for bar in bars)
        if cumulative_volume == 0:
            return AnchoredVWAPMetrics()
            
        # Calculate VWAP using bar VWAPs (more accurate than typical price)
        cumulative_vwap_volume = sum(bar.vwap * bar.volume for bar in bars)
        vwap = cumulative_vwap_volume / cumulative_volume
        
        # Get last bar's close for distance calculation
        last_close = bars[-1].close
        
        # Calculate VWAP distance based on position direction
        if is_long:
            vwap_dist = last_close - vwap
        else:
            vwap_dist = vwap - last_close
            
        # Calculate volume-weighted standard deviation
        squared_devs = sum(
            bar.volume * (bar.vwap - vwap)**2 
            for bar in bars
        )
        vwap_std = np.sqrt(squared_devs / cumulative_volume)
        
        # Calculate close price in standard deviations from VWAP
        vwap_std_close = ((last_close - vwap) / vwap_std 
                         if vwap_std > 0 else 0.0)
        
        return AnchoredVWAPMetrics(
            vwap=vwap,
            vwap_dist=vwap_dist,
            vwap_std=vwap_std,
            vwap_std_close=vwap_std_close
        )

    @staticmethod
    def calculate_breakout_metrics(position_bars: List['TypedBarData'],
                                 prior_bars: List['TypedBarData'],
                                 is_long: bool) -> Tuple[Optional[AnchoredVWAPMetrics], Optional[AnchoredVWAPMetrics]]:
        """
        Calculate VWAP metrics for both pre-entry and post-entry periods.
        
        Args:
            position_bars: List of bars since entry (up to 5)
            prior_bars: List of exactly 5 bars ending at entry
            is_long: Whether this is a long position
            
        Returns:
            Tuple of (pre_entry_metrics, post_entry_metrics)
            Either may be None if no volume
        """
        # Calculate metrics for both periods
        pre_entry_metrics = TypedBarData.calculate_anchored_vwap_metrics(
            prior_bars, is_long
        )
        
        post_entry_metrics = TypedBarData.calculate_anchored_vwap_metrics(
            position_bars, is_long
        )
                
        return pre_entry_metrics, post_entry_metrics