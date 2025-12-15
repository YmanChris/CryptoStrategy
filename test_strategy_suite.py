import pandas as pd
import pytest
from pathlib import Path

from strategy_suite import BuyDipHoldStrategy, LeverageTrendStrategy, SimpleBuyDipStrategy


DATA_PATH = Path(__file__).resolve().parent / "ada_usd_last_1y.csv"


def _load_ada_prices():
    """Helper to normalise the ADA dataset for the strategy interfaces."""
    df = pd.read_csv(DATA_PATH)
    return df.rename(columns={"datetime": "date"})


def test_leverage_trend_strategy_with_ada_history():
    strategy = LeverageTrendStrategy(
        "ADA",
        _load_ada_prices(),
        lookback_days=10,
        leverage_multiple=2.0,
        margin_allocation=0.6,
        long_trigger_pct=0.1,
        long_take_profit_pct=0.1,
        long_stop_loss_pct=0.3,
    )

    result = strategy.run()

    assert result["final_equity"] == pytest.approx(2500.92, rel=1e-4)
    assert len(result["executions"]) == 20
    assert result["max_drawdown"] >= 0


def test_simple_buy_dip_strategy_accumulates_positions():
    strategy = SimpleBuyDipStrategy(
        "ADA",
        _load_ada_prices(),
        initial_buy_trigger=0.9,
        initial_buy_amount=0.3,
        dca_triggers=[0.85, 0.75],
        dca_amounts=[0.3, 0.2],
        sell_trigger=1.15,
    )

    result = strategy.run()

    assert result["final_equity"] == pytest.approx(10876.07, rel=1e-4)
    assert result["remaining_cash"] == pytest.approx(6360.52, rel=1e-4)
    assert result["holdings"] == pytest.approx(11379.440392, rel=1e-6)
    assert result["executions"][0].action == "BUY_INITIAL"


def test_buy_dip_hold_strategy_depth_ladders():
    strategy = BuyDipHoldStrategy(
        "ADA",
        _load_ada_prices(),
        initial_buy_trigger=0.9,
        initial_buy_amount=0.3,
        dca_triggers=[0.85, 0.75],
        dca_amounts=[0.3, 0.2],
    )

    result = strategy.run()

    assert result["final_equity"] == pytest.approx(4610.65, rel=1e-4)
    assert result["holdings"] == pytest.approx(11619.117149, rel=1e-6)
    assert any(ex.action == "BUY_DEPTH" for ex in result["executions"])
