# Strategy Suite

`strategy_suite.py` contains three ready-to-run strategy implementations that
mirror the requirements you shared.  Each class ingests any CSV containing a
`date` column plus a `price_usd` column (e.g. `btc_price_last_1y.csv`) and
exposes a `run()` helper that returns a dictionary with:

- `executions`: chronological list of trade events
- `equity_curve`: simplified equity series you can chart
- `max_drawdown` / `final_equity` plus strategy specific fields

Below is a concise reference for the three strategies alongside their tags and
defaults.

## LEVERAGETRENDSTRATEGY

- **Tags**: `leverage`, `trend`, `long-short`
- **Class**: `LeverageTrendStrategy`
- **Idea**: Uses N-day momentum to open leveraged long/short positions and
  enforces discrete take-profit/stop-loss thresholds while incorporating a
  daily funding charge.

| Parameter | Default | Notes |
| --- | --- | --- |
| `lookback_days` | 14 | Momentum lookback window in days |
| `leverage_multiple` | 2.0 | Applied to the allocated margin |
| `margin_allocation` | 0.7 | Fraction of equity that can be posted as margin |
| `daily_funding_rate` | 0.0005 | Applied to absolute notional each day |
| `long_trigger_pct` | 0.35 | Momentum threshold (e.g. 0.35 == +35%) |
| `long_take_profit_pct` | 0.30 | Long TP relative to entry price |
| `long_stop_loss_pct` | 1.0 | Long SL relative to entry |
| `enable_long` | `True` | Toggle long leg |
| `enable_short` | `False` | Toggle short leg |
| `short_trigger_pct` | 0.35 | Momentum threshold for shorts |
| `short_take_profit_pct` | 0.30 | Short TP relative to entry |
| `short_stop_loss_pct` | 0.30 | Short SL relative to entry |

## SIMPLEBUYDIPSTRATEGY

- **Tags**: `spot`, `dca`, `rolling-monthly`
- **Class**: `SimpleBuyDipStrategy`
- **Idea**: Rolling dip-buying engine that executes an initial allocation,
  follows with user-configurable DCA layers, optional bottom fishing and a
  take-profit exit anchored to either the initial or averaged cost basis.

| Parameter | Default | Notes |
| --- | --- | --- |
| `initial_buy_trigger` | 0.7 | Ratio vs. cycle high that unlocks the first buy |
| `initial_buy_amount` | 0.4 | Portion of starting capital used for the first buy |
| `dca_triggers` | `[0.6, 0.5, 0.4]` | Additional drawdown ratios |
| `dca_amounts` | Even split of remaining capital | Can override with custom list |
| `sell_trigger` | 1.12 | Multiple applied to chosen cost basis to exit |
| `sell_cost_mode` | `"average"` | `"initial"` uses first fill price |
| `enable_bottom_fishing` | `False` | Unlocks a final allocation if drop is deep |
| `bottom_fishing_drawdown` | 0.5 | Activation ratio vs. cycle high |

## BUYDIPHOLDSTRATEGY

- **Tags**: `spot`, `hold`, `accumulation`
- **Class**: `BuyDipHoldStrategy`
- **Idea**: Pure accumulation.  Executes initial + staged DCA buys based on
  drawdown levels and keeps redeploying capital every additional
  `depth_percentage` drop without ever selling until the observation window
  ends.

| Parameter | Default | Notes |
| --- | --- | --- |
| `initial_buy_trigger` | 0.7 | Kick-off ratio vs. cycle high |
| `initial_buy_amount` | 0.3 | Starting allocation |
| `dca_triggers` | `[0.6, 0.5, 0.3]` | Ladder of additional drawdowns |
| `dca_amounts` | Equal split of remaining capital | Override if needed |
| `depth_percentage` | 0.05 | Additional ladder step used for repeat buys |

## Quick Usage Example

```python
import pandas as pd
from strategy_suite import (
    LeverageTrendStrategy,
    SimpleBuyDipStrategy,
    BuyDipHoldStrategy,
)

prices = pd.read_csv("btc_price_last_1y.csv")

lev = LeverageTrendStrategy("BTC", prices)
simple = SimpleBuyDipStrategy("BTC", prices, enable_bottom_fishing=True)
hodl = BuyDipHoldStrategy("BTC", prices)

print(lev.run()["final_equity"])
print(simple.run()["executions"][:3])
print(hodl.run()["holdings"])
```
