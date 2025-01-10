from dataclasses import dataclass, field
from typing import Optional, Dict
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

    @classmethod
    def from_row(cls, row: pd.Series) -> 'PreMarketBar':
        timestamp = row.name[1] if isinstance(row.name, tuple) else row.name
        return cls(
            timestamp=timestamp,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        
    
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
    central_value: float
    is_res: bool
    # shares_per_trade: float
    avg_volume: float # (EMA)
    avg_trade_count: float # (EMA)
    log_return: float
    volatility: float # (rolling)
    
    # New bar data fields
    H_L: float  # High-Low range
    # H_PC: float  # High-Previous Close
    # L_PC: float  # Low-Previous Close
    # TR: float  # True Range
    ATR: float  # Average True Range (EMA)
    MTR: float  # Median True Range
    rolling_range_min_4: float  # 4-bar minimum range
    rolling_range_min_7: float  # 7-bar minimum range
    rolling_ATR: float  # rolling ATR average

    # New trend metrics
    central_value_dist: float       # Normalized distance to central value
    ADX: float                      # Overall trend strength (0-100)
    trend_strength: float           # Directional trend strength (-100 to +100)

    
    # Store original row for access to any additional columns
    _row: pd.Series = field(repr=False)
    
    # Overall volume distribution metrics (using all volume)
    vol_balance: float = 0.0      # Sign indicates if volume is mostly above/below price
    vol_concentration: float = 0.0 # How clustered volume is near price
    vol_kurtosis: float = 0.0     # Peakedness of distribution
    
    # HVN-specific metrics (using only detected peaks)
    hvn_balance: float = 0.0      # Sign indicates if HVNs are mostly above/below price
    hvn_concentration: float = 0.0 # How clustered HVNs are near price
    hvn_avg_prominence: float = 0.0 # Average strength of detected HVNs
    
    
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
        if volume_profile.profile is None:
            return
            
        # Calculate metrics using all volume
        self.vol_balance, self.vol_concentration, self.vol_kurtosis = \
            volume_profile.calculate_moments_relative_to_price(
                volume_profile.bin_centers,
                volume_profile.profile,
                self.close
            )
        
        # Calculate metrics using only HVNs
        self.hvn_balance, self.hvn_concentration, self.hvn_avg_prominence = \
            volume_profile.get_hvn_metrics(self.close, self.ATR if atr is None else atr)
        
        # Add to _row for DataFrame conversion
        self._row['vol_balance'] = self.vol_balance
        self._row['vol_concentration'] = self.vol_concentration 
        self._row['vol_kurtosis'] = self.vol_kurtosis
        self._row['hvn_balance'] = self.hvn_balance
        self._row['hvn_concentration'] = self.hvn_concentration
        self._row['hvn_avg_prominence'] = self.hvn_avg_prominence


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
        
    
    
    # TODO: needs to be adjusted based on data-driven analysis
    def shows_reversal_potential(self, area_is_long, rsi_overbought: float = 65, rsi_oversold: float = 35,
                            mfi_overbought: float = 75, mfi_oversold: float = 25, 
                            pre_position: bool = False) -> bool:
        """
        Check if conditions suggest trend reversal.
        """
        # Base momentum conditions - only need one indicator
        oversold_reversal = (
            not area_is_long and (
                # MFI oversold and improving
                (self.MFI <= mfi_oversold and self.MFI_roc > 0) or
                # RSI oversold and improving  
                (self.RSI <= rsi_oversold and self.RSI_roc > 0)
            )
        )
        
        overbought_reversal = (
            area_is_long and (
                # MFI overbought and deteriorating
                (self.MFI >= mfi_overbought and self.MFI_roc < 0) or
                # RSI overbought and deteriorating
                (self.RSI >= rsi_overbought and self.RSI_roc < 0)
            )
        )
        
        if not (oversold_reversal or overbought_reversal):
            return False
            
        # Get common metrics
        indecision_count = sum(signal[1] for signal in self.get_indecision_signals())
        close_to_high = (self.close - self.low) / (self.high - self.low) if self.high != self.low else 0.5
        close_to_low = (self.high - self.close) / (self.high - self.low) if self.high != self.low else 0.5
        volume_strength = self.volume_ratio > 1.0
        
        if pre_position:
            # For side switching, need ANY TWO of:
            # 1. Indecision signals (market struggling to continue trend)
            # 2. Volume confirmation
            # 3. Price moving in new direction
            
            confirmation_count = sum([
                indecision_count >= 1,  # Relaxed from 2 to 1
                volume_strength,
                oversold_reversal and close_to_high > 0.6,  # Relaxed from 0.7
                overbought_reversal and close_to_low > 0.6   # Relaxed from 0.7
            ])
            
            return confirmation_count >= 2  # Need any 2 confirmations
                
        else:
            # For position scaling warnings (keep as before)
            warning_count = sum([
                indecision_count >= 2,
                volume_strength,
                oversold_reversal and close_to_low > 0.7,  # Staying near low when oversold
                overbought_reversal and close_to_high > 0.7  # Staying near high when overbought
            ])
            
            return warning_count >= 2
        
    def describe_reversal_potential(self, area_was_long, rsi_overbought: float = 65, rsi_oversold: float = 35,
                            mfi_overbought: float = 75, mfi_oversold: float = 25, ) -> str:
        """Get a readable description of reversal potential conditions."""
        conditions = []
        
        # # Check oversold conditions
        if self.MFI <= mfi_oversold:
            # assert not area_was_long
            conditions.append(f"MFI oversold ({self.MFI:.1f} <= {mfi_oversold})")
        if not area_was_long and self.RSI <= rsi_oversold:
            # assert not area_was_long
            conditions.append(f"RSI oversold ({self.RSI:.1f} <= {rsi_oversold})")
        
        # # Check overbought conditions
        if self.MFI >= mfi_overbought:
            # assert area_was_long
            conditions.append(f"MFI overbought ({self.MFI:.1f} >= {mfi_overbought})")
        if area_was_long and self.RSI >= rsi_overbought:
            # assert area_was_long
            conditions.append(f"RSI overbought ({self.RSI:.1f} >= {rsi_overbought})")
        
        # # Check price action
        # if self.close >= self.open:
        #     conditions.append(f"Bullish price action (O:{self.open:.2f} -> C:{self.close:.2f})")
        # elif self.close <= self.open:
        #     conditions.append(f"Bearish price action (O:{self.open:.2f} -> C:{self.close:.2f})")
        
        # Check divergence
        if self.has_divergence:
            conditions.append(f"Divergence detected (Price: {'up' if self.close > self.open else 'down'}, " 
                            f"RSI: {'down' if self.RSI_roc < 0 else 'up'}, "
                            f"MFI: {'down' if self.MFI_roc < 0 else 'up'})"
                            )
        
        # Check indecision
        if self.shows_indecision:
            # conditions.append(f"Shows indecision ({self.indecision_count} signals)")
            conditions.append(f"Shows indecision - {self.describe_indecision}")
        
        if not conditions:
            return "describe_reversal_potential - No reversal conditions detected"
        
        # Determine overall reversal potential
        has_potential = self.shows_reversal_potential(area_was_long, rsi_overbought, rsi_oversold, 
                                                    mfi_overbought, mfi_oversold,
                                                    pre_position=True)
        
        return (f"describe_reversal_potential - {'POTENTIAL REVERSAL' if has_potential else 'Partial conditions'} detected:\n" + 
                "\n".join(f"- {cond}" for cond in conditions))
        
    
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
