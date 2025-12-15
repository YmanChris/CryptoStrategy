"""
High level strategy implementations for different crypto investment styles.

The module keeps things self contained so the strategies can be used directly
with any CSV that follows the ``date, price`` convention that already exists in
the workspace.  Each strategy exposes a ``run`` method that accepts no
arguments and returns a dictionary containing trade executions, equity curve
information and a couple of helper statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd


PriceInput = Union[str, pd.DataFrame]


@dataclass
class Execution:
    """
    Lightweight trade/execution log mainly used for debugging and reporting.
    """

    date: pd.Timestamp
    symbol: str
    action: str
    quantity: float
    price: float
    notional: float
    pnl: float = 0.0
    notes: str = ""
    metadata: Dict[str, float] = field(default_factory=dict)


def _load_price_frame(price_data: PriceInput, price_column: str) -> pd.DataFrame:
    """
    Normalise price input so every strategy can rely on a clean dataframe.
    """
    if isinstance(price_data, pd.DataFrame):
        df = price_data.copy()
    else:
        df = pd.read_csv(price_data)

    if "date" not in df.columns:
        raise ValueError("price_data must contain a 'date' column")

    df["date"] = pd.to_datetime(df["date"])
    if price_column not in df.columns:
        raise ValueError(f"price_data must contain a '{price_column}' column")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def _max_drawdown(equity_curve: List[float]) -> float:
    """
    Simple max drawdown helper based on a running peak observation.
    """
    drawdown = 0.0
    peak = float("-inf")
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            drawdown = min(drawdown, (value - peak) / peak)
    return abs(drawdown)


class BaseStrategy:
    """
    Lightweight base object used to keep bookkeeping consistent.
    """

    def __init__(
        self,
        symbol: str,
        price_data: PriceInput,
        account_equity: float = 10_000.0,
        price_column: str = "price_usd",
    ) -> None:
        self.symbol = symbol.upper()
        self.price_column = price_column
        self.price_df = _load_price_frame(price_data, price_column)
        self.initial_equity = float(account_equity)
        self.executions: List[Execution] = []

    def run(self) -> Dict:
        raise NotImplementedError


class LeverageTrendStrategy(BaseStrategy):
    """
    Trend following strategy with leverage, TP/SL gates and daily funding costs.
    """

    def __init__(
        self,
        symbol: str,
        price_data: PriceInput,
        account_equity: float = 10_000.0,
        lookback_days: int = 14,
        leverage_multiple: float = 2.0,
        margin_allocation: float = 0.7,
        daily_funding_rate: float = 0.0005,
        long_trigger_pct: float = 0.35,
        long_take_profit_pct: float = 0.3,
        long_stop_loss_pct: float = 1.0,
        enable_long: bool = True,
        enable_short: bool = False,
        short_trigger_pct: float = 0.35,
        short_take_profit_pct: float = 0.3,
        short_stop_loss_pct: float = 0.3,
    ) -> None:
        super().__init__(symbol, price_data, account_equity)
        self.lookback_days = lookback_days
        self.leverage_multiple = leverage_multiple
        self.margin_allocation = margin_allocation
        self.daily_funding_rate = daily_funding_rate
        self.long_trigger_pct = long_trigger_pct
        self.long_take_profit_pct = long_take_profit_pct
        self.long_stop_loss_pct = long_stop_loss_pct
        self.enable_long = enable_long
        self.enable_short = enable_short
        self.short_trigger_pct = short_trigger_pct
        self.short_take_profit_pct = short_take_profit_pct
        self.short_stop_loss_pct = short_stop_loss_pct

    def run(self) -> Dict:
        df = self.price_df
        equity = self.initial_equity
        equity_curve: List[float] = []
        position: Optional[Dict] = None

        for idx in range(self.lookback_days, len(df)):
            row = df.iloc[idx]
            price = float(row[self.price_column])
            date = row["date"]
            lookback_price = float(df.iloc[idx - self.lookback_days][self.price_column])
            momentum = (price / lookback_price) - 1.0

            if position:
                notional = position["size"] * price
                funding = abs(notional) * self.daily_funding_rate
                equity -= funding
                position["funding_paid"] += funding

                entry_price = position["entry_price"]
                direction = position["direction"]

                take_profit_hit = False
                stop_loss_hit = False

                if direction == 1:
                    if price >= entry_price * (1 + self.long_take_profit_pct):
                        take_profit_hit = True
                    if price <= entry_price * (1 - self.long_stop_loss_pct):
                        stop_loss_hit = True
                else:
                    if price <= entry_price * (1 - self.short_take_profit_pct):
                        take_profit_hit = True
                    if price >= entry_price * (1 + self.short_stop_loss_pct):
                        stop_loss_hit = True

                if take_profit_hit or stop_loss_hit:
                    gross_pnl = position["size"] * (price - entry_price) * direction
                    equity += gross_pnl
                    net_pnl = gross_pnl - position["funding_paid"]
                    action = "CLOSE_LONG" if direction == 1 else "CLOSE_SHORT"
                    self.executions.append(
                        Execution(
                            date=date,
                            symbol=self.symbol,
                            action=action,
                            quantity=position["size"],
                            price=price,
                            notional=position["size"] * price,
                            pnl=net_pnl,
                            notes="TP" if take_profit_hit else "SL",
                            metadata={
                                "entry_price": entry_price,
                                "funding_paid": position["funding_paid"],
                            },
                        )
                    )
                    position = None

            if position is None:
                margin = equity * self.margin_allocation
                if self.enable_long and momentum >= self.long_trigger_pct:
                    size = (margin * self.leverage_multiple) / price
                    position = {
                        "direction": 1,
                        "entry_price": price,
                        "size": size,
                        "funding_paid": 0.0,
                    }
                    self.executions.append(
                        Execution(
                            date=date,
                            symbol=self.symbol,
                            action="OPEN_LONG",
                            quantity=size,
                            price=price,
                            notional=size * price,
                            notes=f"momentum {momentum:.2%}",
                        )
                    )
                elif self.enable_short and momentum <= -self.short_trigger_pct:
                    size = (margin * self.leverage_multiple) / price
                    position = {
                        "direction": -1,
                        "entry_price": price,
                        "size": size,
                        "funding_paid": 0.0,
                    }
                    self.executions.append(
                        Execution(
                            date=date,
                            symbol=self.symbol,
                            action="OPEN_SHORT",
                            quantity=size,
                            price=price,
                            notional=size * price,
                            notes=f"momentum {momentum:.2%}",
                        )
                    )

            equity_curve.append(equity)

        if position:
            final_price = float(df[self.price_column].iloc[-1])
            final_date = df["date"].iloc[-1]
            direction = position["direction"]
            entry_price = position["entry_price"]
            gross_pnl = position["size"] * (final_price - entry_price) * direction
            equity += gross_pnl
            net_pnl = gross_pnl - position["funding_paid"]
            action = "CLOSE_LONG" if direction == 1 else "CLOSE_SHORT"
            self.executions.append(
                Execution(
                    date=final_date,
                    symbol=self.symbol,
                    action=action,
                    quantity=position["size"],
                    price=final_price,
                    notional=position["size"] * final_price,
                    pnl=net_pnl,
                    notes="forced exit at end",
                    metadata={"entry_price": entry_price, "funding_paid": position["funding_paid"]},
                )
            )
            if equity_curve:
                equity_curve[-1] = equity
            else:
                equity_curve.append(equity)

        return {
            "symbol": self.symbol,
            "final_equity": equity,
            "equity_curve": equity_curve,
            "max_drawdown": _max_drawdown(equity_curve),
            "executions": self.executions,
        }


class SimpleBuyDipStrategy(BaseStrategy):
    """
    Enhanced spot buy-the-dip strategy with optional rolling DCA legs.
    """

    def __init__(
        self,
        symbol: str,
        price_data: PriceInput,
        account_equity: float = 10_000.0,
        initial_buy_trigger: float = 0.7,
        initial_buy_amount: float = 0.4,
        dca_triggers: Optional[Sequence[float]] = None,
        dca_amounts: Optional[Sequence[float]] = None,
        sell_trigger: float = 1.12,
        sell_cost_mode: str = "average",
        enable_bottom_fishing: bool = False,
        bottom_fishing_drawdown: float = 0.5,
    ) -> None:
        super().__init__(symbol, price_data, account_equity)
        self.initial_buy_trigger = initial_buy_trigger
        self.initial_buy_amount = initial_buy_amount
        raw_triggers = list(dca_triggers) if dca_triggers is not None else [0.6, 0.5, 0.4]
        if dca_amounts is None:
            if raw_triggers:
                residual = max(0.0, 1.0 - initial_buy_amount)
                equal_share = residual / len(raw_triggers)
                raw_amounts = [equal_share] * len(raw_triggers)
            else:
                raw_amounts = []
        else:
            if len(dca_amounts) != len(raw_triggers):
                raise ValueError("dca_amounts and dca_triggers must match in size")
            raw_amounts = list(dca_amounts)

        pairs = sorted(zip(raw_triggers, raw_amounts), key=lambda item: item[0], reverse=True)
        if pairs:
            self.dca_triggers, self.dca_amounts = map(list, zip(*pairs))
        else:
            self.dca_triggers, self.dca_amounts = [], []
        self.sell_trigger = sell_trigger
        self.sell_cost_mode = sell_cost_mode
        self.enable_bottom_fishing = enable_bottom_fishing
        self.bottom_fishing_drawdown = bottom_fishing_drawdown

    def run(self) -> Dict:
        df = self.price_df
        cash = self.initial_equity
        holdings = 0.0
        avg_cost = 0.0
        initial_cost_basis = 0.0
        cycle_high = float(df[self.price_column].iloc[0])
        pending_dca = list(zip(self.dca_triggers, self.dca_amounts))
        bottom_fished = False
        equity_curve: List[float] = []

        for _, row in df.iterrows():
            price = float(row[self.price_column])
            date = row["date"]
            cycle_high = max(cycle_high, price)
            equity_curve.append(cash + holdings * price)

            drawdown = price / cycle_high

            if holdings == 0 and drawdown <= self.initial_buy_trigger and self.initial_buy_amount > 0:
                spend = min(cash, self.initial_equity * self.initial_buy_amount)
                if spend > 0:
                    quantity = spend / price
                    holdings += quantity
                    cash -= spend
                    avg_cost = price
                    initial_cost_basis = price
                    self.executions.append(
                        Execution(
                            date=date,
                            symbol=self.symbol,
                            action="BUY_INITIAL",
                            quantity=quantity,
                            price=price,
                            notional=spend,
                            notes=f"drawdown {drawdown:.2%}",
                        )
                    )
                    pending_dca = list(zip(self.dca_triggers, self.dca_amounts))
                    bottom_fished = False
                    continue

            if holdings > 0:
                while pending_dca and drawdown <= pending_dca[-1][0]:
                    trigger, allocation = pending_dca.pop()
                    spend = min(cash, self.initial_equity * allocation)
                    if spend <= 0:
                        continue
                    quantity = spend / price
                    new_cost = avg_cost * holdings + spend
                    holdings += quantity
                    cash -= spend
                    avg_cost = new_cost / holdings
                    self.executions.append(
                        Execution(
                            date=date,
                            symbol=self.symbol,
                            action="BUY_DCA",
                            quantity=quantity,
                            price=price,
                            notional=spend,
                            notes=f"trigger {trigger:.2f}",
                        )
                    )

                if (
                    self.enable_bottom_fishing
                    and not bottom_fished
                    and drawdown <= self.bottom_fishing_drawdown
                    and cash > 0
                ):
                    spend = cash
                    quantity = spend / price if price else 0.0
                    prev_holdings = holdings
                    self.executions.append(
                        Execution(
                            date=date,
                            symbol=self.symbol,
                            action="BUY_BOTTOM",
                            quantity=quantity,
                            price=price,
                            notional=cash,
                            notes=f"drawdown {drawdown:.2%}",
                        )
                    )
                    holdings += quantity
                    cash = 0.0
                    total_cost = (avg_cost * prev_holdings) + spend
                    avg_cost = total_cost / holdings if holdings else 0.0
                    bottom_fished = True

                base_cost = initial_cost_basis if self.sell_cost_mode == "initial" else avg_cost
                if base_cost > 0 and price >= base_cost * self.sell_trigger:
                    proceeds = holdings * price
                    pnl = proceeds - (avg_cost * holdings)
                    cash += proceeds
                    self.executions.append(
                        Execution(
                            date=date,
                            symbol=self.symbol,
                            action="SELL",
                            quantity=holdings,
                            price=price,
                            notional=proceeds,
                            pnl=pnl,
                            notes="take profit",
                        )
                    )
                    holdings = 0.0
                    avg_cost = 0.0
                    initial_cost_basis = 0.0
                    cycle_high = price
                    pending_dca = list(zip(self.dca_triggers, self.dca_amounts))
                    bottom_fished = False

        final_equity = cash + holdings * float(df[self.price_column].iloc[-1])
        return {
            "symbol": self.symbol,
            "final_equity": final_equity,
            "equity_curve": equity_curve,
            "max_drawdown": _max_drawdown(equity_curve),
            "executions": self.executions,
            "remaining_cash": cash,
            "holdings": holdings,
        }


class BuyDipHoldStrategy(BaseStrategy):
    """
    Pure accumulation strategy that never sells and keeps stacking dips.
    """

    def __init__(
        self,
        symbol: str,
        price_data: PriceInput,
        account_equity: float = 10_000.0,
        initial_buy_trigger: float = 0.7,
        initial_buy_amount: float = 0.3,
        dca_triggers: Optional[Sequence[float]] = None,
        dca_amounts: Optional[Sequence[float]] = None,
        depth_percentage: float = 0.05,
    ) -> None:
        super().__init__(symbol, price_data, account_equity)
        self.initial_buy_trigger = initial_buy_trigger
        self.initial_buy_amount = initial_buy_amount
        raw_triggers = list(dca_triggers) if dca_triggers is not None else [0.6, 0.5, 0.3]
        if dca_amounts is None:
            if raw_triggers:
                residual = max(0.0, 1.0 - initial_buy_amount)
                equal_share = residual / len(raw_triggers)
                raw_amounts = [equal_share] * len(raw_triggers)
            else:
                raw_amounts = []
        else:
            if len(dca_amounts) != len(raw_triggers):
                raise ValueError("dca_amounts and dca_triggers must match in size")
            raw_amounts = list(dca_amounts)

        pairs = sorted(zip(raw_triggers, raw_amounts), key=lambda item: item[0], reverse=True)
        if pairs:
            self.dca_triggers, self.dca_amounts = map(list, zip(*pairs))
        else:
            self.dca_triggers, self.dca_amounts = [], []
        self.depth_percentage = depth_percentage
        self.depth_allocation_pct = (
            self.dca_amounts[-1] if self.dca_amounts else max(0.0, 1.0 - initial_buy_amount)
        )

    def run(self) -> Dict:
        df = self.price_df
        cash = self.initial_equity
        holdings = 0.0
        avg_cost = 0.0
        cycle_high = float(df[self.price_column].iloc[0])
        next_depth_price: Optional[float] = None
        pending_dca = list(zip(self.dca_triggers, self.dca_amounts))
        equity_curve: List[float] = []

        for _, row in df.iterrows():
            price = float(row[self.price_column])
            date = row["date"]
            cycle_high = max(cycle_high, price)
            drawdown = price / cycle_high
            equity_curve.append(cash + holdings * price)

            if holdings == 0 and drawdown <= self.initial_buy_trigger:
                allocation = min(cash, self.initial_equity * self.initial_buy_amount)
                if allocation > 0:
                    qty = allocation / price
                    holdings += qty
                    cash -= allocation
                    avg_cost = price
                    self.executions.append(
                        Execution(
                            date=date,
                            symbol=self.symbol,
                            action="BUY_INITIAL",
                            quantity=qty,
                            price=price,
                            notional=allocation,
                            notes=f"drawdown {drawdown:.2%}",
                        )
                    )
                    next_depth_price = price * (1 - self.depth_percentage)
                continue

            if holdings > 0:
                while pending_dca and drawdown <= pending_dca[-1][0]:
                    trigger, allocation = pending_dca.pop()
                    spend = min(cash, self.initial_equity * allocation)
                    if spend <= 0:
                        continue
                    qty = spend / price
                    prev_holdings = holdings
                    holdings += qty
                    cash -= spend
                    avg_cost = ((avg_cost * prev_holdings) + spend) / holdings
                    self.executions.append(
                        Execution(
                            date=date,
                            symbol=self.symbol,
                            action="BUY_DCA",
                            quantity=qty,
                            price=price,
                            notional=spend,
                            notes=f"trigger {trigger:.2f}",
                        )
                    )
                    next_depth_price = price * (1 - self.depth_percentage)

                if next_depth_price and price <= next_depth_price and cash > 0:
                    spend = min(cash, self.initial_equity * self.depth_allocation_pct)
                    if spend > 0:
                        qty = spend / price
                        prev_holdings = holdings
                        holdings += qty
                        cash -= spend
                        avg_cost = ((avg_cost * prev_holdings) + spend) / holdings
                        self.executions.append(
                            Execution(
                                date=date,
                                symbol=self.symbol,
                                action="BUY_DEPTH",
                                quantity=qty,
                                price=price,
                                notional=spend,
                                notes=f"depth ladder {self.depth_percentage:.2%}",
                            )
                        )
                        next_depth_price = price * (1 - self.depth_percentage)

        final_price = float(df[self.price_column].iloc[-1])
        final_equity = cash + holdings * final_price
        return {
            "symbol": self.symbol,
            "final_equity": final_equity,
            "equity_curve": equity_curve,
            "max_drawdown": _max_drawdown(equity_curve),
            "executions": self.executions,
            "holdings": holdings,
            "avg_cost": avg_cost,
        }
