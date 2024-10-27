You're right - let's break this down into distinct components. Given the complexity of combining price, size, and spread dynamics, we likely need both a feature calculation step and a separate limit price determination step.



```python
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from typing import Tuple, Optional

class BreakoutStrength(Enum):
    STRONG = "STRONG"
    STANDARD = "STANDARD"
    WEAK = "WEAK"
    CHOPPY = "CHOPPY"

@dataclass
class LimitPriceParameters:
    # Spread-based parameters
    spread_ema_alpha: float = 0.1
    max_spread_ratio: float = 3.0  # Current spread / EMA spread threshold
    
    # Size dynamics parameters
    min_size_changes: int = 3  # Minimum changes to consider size "active"
    size_bounce_threshold: float = 0.3  # Relative size change threshold
    
    # Price movement parameters
    price_movement_threshold: float = 0.2  # % move threshold for strong breakout
    
    # Limit price adjustment factors
    strong_adjustment: float = 0.0002  # How far above/below to place in strong moves
    weak_adjustment: float = 0.0001    # How far above/below in weak moves

def calculate_breakout_features(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate features needed for limit price determination from cleaned quote data.
    
    Args:
        quotes_df: DataFrame from clean_quotes_data
    Returns:
        DataFrame with additional features for limit price determination
    """
    # Start with the last row's basic metrics
    current_quote = quotes_df.iloc[-1]
    
    # Calculate spread EMA
    spread_ema = quotes_df['spread_last'].ewm(
        times=quotes_df['duration'].cumsum(),
        ignore_na=True,
        adjust=False
    ).mean()
    
    # Size dynamics features
    features = {}
    
    # Calculate size changes frequency and magnitude
    for side in ['bid', 'ask']:
        size_col = f'{side}_size_last'
        pos_changes = quotes_df[f'{side}_size_intra_pos'].sum()
        neg_changes = abs(quotes_df[f'{side}_size_intra_neg'].sum())
        
        features.update({
            f'{side}_size_total_changes': pos_changes + neg_changes,
            f'{side}_size_net_change': pos_changes - neg_changes,
            f'{side}_size_bounce_count': (
                quotes_df[f'{side}_size_intra_pos'] > 0).sum() + 
                (quotes_df[f'{side}_size_intra_neg'] < 0).sum()
        })
    
    # Price movement features
    for side in ['bid', 'ask']:
        price_col = f'{side}_price_last'
        features[f'{side}_price_move_pct'] = (
            (quotes_df[price_col].iloc[-1] - quotes_df[price_col].iloc[0]) / 
            quotes_df[price_col].iloc[0]
        )
    
    # Collect all features
    features.update({
        'spread_current': current_quote['spread_last'],
        'spread_ema': spread_ema.iloc[-1],
        'spread_ratio': current_quote['spread_last'] / spread_ema.iloc[-1],
        'bid_price': current_quote['bid_price_last'],
        'ask_price': current_quote['ask_price_last']
    })
    
    return pd.Series(features)

def determine_breakout_strength(
    features: pd.Series,
    params: LimitPriceParameters,
    is_long: bool
) -> Tuple[BreakoutStrength, float]:
    """
    Determine breakout strength and suggested price adjustment.
    
    Args:
        features: Features from calculate_breakout_features
        params: LimitPriceParameters instance
        is_long: True for long entry, False for short
    Returns:
        Tuple of (BreakoutStrength, price_adjustment)
    """
    side = 'ask' if is_long else 'bid'
    opposite = 'bid' if is_long else 'ask'
    
    # Check spread conditions
    spread_healthy = features['spread_ratio'] < params.max_spread_ratio
    
    # Check size dynamics
    size_active = features[f'{side}_size_bounce_count'] >= params.min_size_changes
    size_increasing = features[f'{side}_size_net_change'] > 0
    
    # Check price movement
    price_move = features[f'{side}_price_move_pct']
    strong_move = abs(price_move) > params.price_movement_threshold
    
    # Determine strength
    if (strong_move and spread_healthy and size_active and size_increasing):
        strength = BreakoutStrength.STRONG
        adjustment = params.strong_adjustment
    elif not spread_healthy or not size_active:
        strength = BreakoutStrength.WEAK
        adjustment = -params.weak_adjustment
    elif size_active and spread_healthy:
        strength = BreakoutStrength.STANDARD
        adjustment = 0.0
    else:
        strength = BreakoutStrength.CHOPPY
        adjustment = -params.weak_adjustment
    
    # Flip adjustment sign for shorts
    if not is_long:
        adjustment = -adjustment
        
    return strength, adjustment

def calculate_limit_price(
    quotes_df: pd.DataFrame,
    params: LimitPriceParameters,
    is_long: bool
) -> Tuple[float, BreakoutStrength]:
    """
    Calculate suggested limit price based on market conditions.
    
    Args:
        quotes_df: DataFrame from clean_quotes_data
        params: LimitPriceParameters instance
        is_long: True for long entry, False for short
    Returns:
        Tuple of (limit_price, breakout_strength)
    """
    features = calculate_breakout_features(quotes_df)
    strength, adjustment = determine_breakout_strength(features, params, is_long)
    
    # Base price is ask for longs, bid for shorts
    base_price = features['ask_price'] if is_long else features['bid_price']
    
    # Calculate limit price
    limit_price = base_price + adjustment
    
    return limit_price, strength

```

Key aspects of this implementation:

1. **Feature Calculation**:
- Spread dynamics (current vs EMA)
- Size activity (changes frequency and magnitude)
- Price movement percentages
- No linear regression (complex and possibly unnecessary)

2. **Breakout Strength Determination**:
- Combines spread, size, and price metrics
- Uses enumerated strength levels
- Provides adjustment factors based on conditions

3. **Limit Price Setting**:
- Base price at bid/ask
- Adjustments based on market conditions
- Signs flipped appropriately for shorts

Key improvements we could consider:

1. Add volume and trade frequency metrics
2. Consider relative size changes (% of total size)
3. Add momentum indicators for price movement
4. Include more sophisticated spread analysis

Would you like me to elaborate on any of these aspects or suggest alternative approaches?