Given that you're trying to extrapolate near-future bid/ask prices based on real-time stock quote data, where you want to incorporate bid/ask sizes as weights, here’s an approach:

1. **Weighting by Bid/Ask Size**:
   You can modify the linear regression to weight the data points based on the sizes of the bid/ask orders. This is known as weighted least squares (WLS). The weights would be proportional to the size of the bid or ask orders, giving more importance to prices associated with larger sizes.

2. **Extrapolation**:
   Once you’ve fitted a weighted linear regression model, you can use it to extrapolate bid/ask prices at a near-future time (seconds in the future).

3. **Extrapolation Strategy Based on Aggressiveness**:
   Depending on whether you want more aggressive or less aggressive execution, you’ll adjust the extrapolated prices:
   - **More Aggressive**: For buys, set the limit price at or above the ask price; for sells, set the limit price at or below the bid price.
   - **Less Aggressive**: For buys, set the limit price below the ask price; for sells, set the limit price above the bid price.

### Implementation

Here’s how you can modify the previous function to handle weighted data based on bid/ask sizes and then use it for extrapolation:

```python
import numpy as np
import scipy.stats as stats
from numba import jit

@jit(nopython=True)
def weighted_linear_fit(x, y, weights, confidence=0.95):
    # Step 1: Weighted linear fit (degree 1)
    w = np.asarray(weights)
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Weighted means
    mean_x = np.sum(w * x) / np.sum(w)
    mean_y = np.sum(w * y) / np.sum(w)
    
    # Weighted least squares calculations
    s_xx = np.sum(w * (x - mean_x)**2)
    s_xy = np.sum(w * (x - mean_x) * (y - mean_y))
    
    slope = s_xy / s_xx
    intercept = mean_y - slope * mean_x
    
    # Step 2: Calculate residuals and standard error
    y_pred = slope * x + intercept
    residuals = y - y_pred
    residual_std_error = np.sqrt(np.sum(w * residuals**2) / (np.sum(w) - 2))  # Weighted residual std error
    
    # Step 3: Get t-value for confidence interval
    dof = len(x) - 2
    t_value = stats.t.ppf((1 + confidence) / 2.0, dof)
    
    return slope, intercept, residual_std_error, t_value, mean_x, s_xx

# Example usage:
# Let's say your bid prices are 'y', the sizes are 'weights', and time is 'x'
x = np.array([0, 1, 2, 3, 4])  # Time in seconds
y = np.array([1, 3, 7, 9, 11])  # Bid prices (example data)
weights = np.array([10, 20, 15, 5, 25])  # Bid sizes (example data)

# Call the function to get weighted linear regression params
slope, intercept, residual_std_error, t_value, mean_x, s_xx = weighted_linear_fit(x, y, weights)

# To extrapolate bid/ask price at a near-future time:
x_new = 6  # Future time (in seconds)
y_new = slope * x_new + intercept

# Confidence interval for the new point
conf_interval_width = t_value * residual_std_error * np.sqrt(1/len(x) + (x_new - mean_x)**2 / s_xx)

# Upper and lower bounds of the confidence interval
y_new_upper = y_new + conf_interval_width
y_new_lower = y_new - conf_interval_width

print(f"Extrapolated y value at x = {x_new}: {y_new}")
print(f"95% confidence interval: [{y_new_lower}, {y_new_upper}]")

# Strategy for setting limit prices based on your extrapolated value
# - More aggressive (faster execution): Set limit price at or above y_new_upper for buys, below y_new_lower for sells
# - Less aggressive (better pricing): Set limit price closer to y_new (adjust based on aggressiveness)
```

### Key Points:
- **Weighted Fit**: This function performs a weighted linear regression where the weights are the bid/ask sizes. Larger sizes have more influence on the trend line.
- **Extrapolation**: You can use the `slope` and `intercept` to extrapolate prices for a future time.
- **Confidence Intervals**: The confidence interval helps you understand the uncertainty in your prediction. You can set your limit prices based on the aggressiveness of the trade (either at the extrapolated price or adjusting based on the confidence interval).
  
Would you like to further customize the aggressiveness strategy or refine the confidence interval handling?