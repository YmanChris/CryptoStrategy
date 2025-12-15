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
    if isinstance(price_data, pd.DataFrame):
        df = price_data.copy()
    else:
        df = pd.read_csv(price_data)

    if "date" not in df.columns:
        raise ValueError("price_data must contain 'date' column")

    df["date"] = pd.to_datetime(df["date"])
    if price_column not in df.columns:
        raise ValueError(f"price_data must contain '{price_column}' column")

    return df.sort_values("date").reset_index(drop=True)


def _max_drawdown(equity_curve: List[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        max_dd = min(max_dd, (v - peak) / peak)
    return abs(max_dd)


class BaseStrategy:
    def __init__(
        self,
        symbol: str,
        price_data: PriceInput,
        account_equity: float = 10_000.0,
        price_column: str = "price_usd",
    ):
        self.symbol = symbol.upper()
        self.price_column = price_column
        self.price_df = _load_price_frame(price_data, price_column)
        self.initial_equity = float(account_equity)
        self.executions: List[Execution] = []

    def run(self) -> Dict:
        raise NotImplementedError


# ==========================================================
# 1️⃣ LeverageTrendStrategy（修正版）
# ==========================================================

class LeverageTrendStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        price_data: PriceInput,
        account_equity: float = 10_000.0,
        lookback_days: int = 14,
        leverage_multiple: float = 2.0,
        margin_allocation: float = 0.7,
        daily_funding_rate: float = 0.0005,
        long_trigger_pct: float = 0.3,
        long_take_profit_pct: float = 0.3,
        long_stop_loss_pct: float = 0.3,
    ):
        super().__init__(symbol, price_data, account_equity)
        self.lookback_days = lookback_days
        self.leverage_multiple = leverage_multiple
        self.margin_allocation = margin_allocation
        self.daily_funding_rate = daily_funding_rate
        self.long_trigger_pct = long_trigger_pct
        self.long_take_profit_pct = long_take_profit_pct
        self.long_stop_loss_pct = long_stop_loss_pct

    def run(self) -> Dict:
        df = self.price_df

        cash = self.initial_equity
        margin_locked = 0.0
        position = None
        equity_curve: List[float] = []
        cooldown = False

        for i in range(self.lookback_days, len(df)):
            price = float(df.iloc[i][self.price_column])
            date = df.iloc[i]["date"]

            # ===== 更新浮动盈亏 =====
            floating_pnl = 0.0
            if position:
                if position["direction"] == 1:
                    floating_pnl = position["size"] * (price - position["entry_price"])
                else:
                    floating_pnl = position["size"] * (position["entry_price"] - price)

                funding = abs(position["size"] * price) * self.daily_funding_rate
                cash -= funding
                position["funding_paid"] += funding

            equity = cash + margin_locked + floating_pnl

            # ===== 平仓判断 =====
            if position:
                entry = position["entry_price"]
                if (
                    price >= entry * (1 + self.long_take_profit_pct)
                    or price <= entry * (1 - self.long_stop_loss_pct)
                ):
                    realized_pnl = floating_pnl - position["funding_paid"]
                    cash += margin_locked + floating_pnl
                    self.executions.append(
                        Execution(
                            date=date,
                            symbol=self.symbol,
                            action="CLOSE_LONG",
                            quantity=position["size"],
                            price=price,
                            notional=position["size"] * price,
                            pnl=realized_pnl,
                        )
                    )
                    position = None
                    margin_locked = 0.0
                    cooldown = True

            # ===== 开仓判断 =====
            if not position and not cooldown:
                lookback_price = float(df.iloc[i - self.lookback_days][self.price_column])
                momentum = price / lookback_price - 1

                if momentum >= self.long_trigger_pct:
                    margin_locked = cash * self.margin_allocation
                    cash -= margin_locked
                    size = (margin_locked * self.leverage_multiple) / price
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
                        )
                    )

            cooldown = False
            equity_curve.append(cash + margin_locked + floating_pnl)

        return {
            "symbol": self.symbol,
            "final_equity": equity_curve[-1],
            "equity_curve": equity_curve,
            "max_drawdown": _max_drawdown(equity_curve),
            "executions": self.executions,
        }


# ==========================================================
# 2️⃣ SimpleBuyDipStrategy（修正版）
# ==========================================================

class SimpleBuyDipStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        price_data: PriceInput,
        account_equity: float = 10_000.0,
        initial_buy_trigger: float = 0.8,
        initial_buy_amount: float = 0.4,
        dca_triggers: Sequence[float] = (0.7, 0.6),
        dca_amounts: Sequence[float] = (0.3, 0.3),
        sell_trigger: float = 1.15,
    ):
        super().__init__(symbol, price_data, account_equity)
        self.initial_buy_trigger = initial_buy_trigger
        self.initial_buy_amount = initial_buy_amount
        self.dca_triggers = list(dca_triggers)
        self.dca_amounts = list(dca_amounts)
        self.sell_trigger = sell_trigger

    def run(self) -> Dict:
        df = self.price_df

        cash = self.initial_equity
        holdings = 0.0
        avg_cost = 0.0
        cycle_high = float(df.iloc[0][self.price_column])
        pending_dca = list(zip(self.dca_triggers, self.dca_amounts))
        equity_curve: List[float] = []

        for _, row in df.iterrows():
            price = float(row[self.price_column])
            date = row["date"]
            cycle_high = max(cycle_high, price)
            drawdown = price / cycle_high

            # ===== 初始买入 =====
            if holdings == 0 and drawdown <= self.initial_buy_trigger:
                spend = min(cash, self.initial_equity * self.initial_buy_amount)
                if spend > 0:
                    qty = spend / price
                    holdings += qty
                    cash -= spend
                    avg_cost = price
                    pending_dca = list(zip(self.dca_triggers, self.dca_amounts))
                    self.executions.append(
                        Execution(date, self.symbol, "BUY_INITIAL", qty, price, spend)
                    )

            # ===== DCA =====
            while holdings > 0 and pending_dca and drawdown <= pending_dca[-1][0]:
                trigger, alloc = pending_dca.pop()
                spend = min(cash, self.initial_equity * alloc)
                if spend <= 0:
                    continue
                qty = spend / price
                avg_cost = (avg_cost * holdings + spend) / (holdings + qty)
                holdings += qty
                cash -= spend
                self.executions.append(
                    Execution(date, self.symbol, "BUY_DCA", qty, price, spend)
                )

            # ===== 止盈卖出 =====
            if holdings > 0 and price >= avg_cost * self.sell_trigger:
                proceeds = holdings * price
                pnl = proceeds - avg_cost * holdings
                cash += proceeds
                self.executions.append(
                    Execution(date, self.symbol, "SELL", holdings, price, proceeds, pnl)
                )
                holdings = 0
                avg_cost = 0
                cycle_high = price
                pending_dca = list(zip(self.dca_triggers, self.dca_amounts))

            equity_curve.append(cash + holdings * price)

        return {
            "symbol": self.symbol,
            "final_equity": equity_curve[-1],
            "equity_curve": equity_curve,
            "max_drawdown": _max_drawdown(equity_curve),
            "executions": self.executions,
            "remaining_cash": cash,   # ✅ 新增
            "holdings": holdings,
            "avg_cost": avg_cost,      # （可选，但强烈建议）
        }



# ==========================================================
# 3️⃣ BuyDipHoldStrategy（修正版）
# ==========================================================

class BuyDipHoldStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        price_data: PriceInput,
        account_equity: float = 10_000.0,
        initial_buy_trigger: float = 0.8,
        initial_buy_amount: float = 0.3,
        dca_triggers: Sequence[float] = (0.7, 0.6),
        dca_amounts: Sequence[float] = (0.3, 0.4),
        depth_percentage: float = 0.05,
    ):
        super().__init__(symbol, price_data, account_equity)
        self.initial_buy_trigger = initial_buy_trigger
        self.initial_buy_amount = initial_buy_amount
        self.dca_triggers = list(dca_triggers)
        self.dca_amounts = list(dca_amounts)
        self.depth_percentage = depth_percentage

    def run(self) -> Dict:
        df = self.price_df

        cash = self.initial_equity
        holdings = 0.0
        avg_cost = 0.0
        cycle_high = float(df.iloc[0][self.price_column])
        pending_dca = list(zip(self.dca_triggers, self.dca_amounts))
        next_depth_price = None
        equity_curve: List[float] = []

        for _, row in df.iterrows():
            price = float(row[self.price_column])
            date = row["date"]

            # 新高 → 重置 DCA
            if price > cycle_high:
                cycle_high = price
                pending_dca = list(zip(self.dca_triggers, self.dca_amounts))

            drawdown = price / cycle_high

            if holdings == 0 and drawdown <= self.initial_buy_trigger:
                spend = min(cash, self.initial_equity * self.initial_buy_amount)
                if spend > 0:
                    qty = spend / price
                    holdings += qty
                    cash -= spend
                    avg_cost = price
                    next_depth_price = price * (1 - self.depth_percentage)
                    self.executions.append(
                        Execution(date, self.symbol, "BUY_INITIAL", qty, price, spend)
                    )

            while holdings > 0 and pending_dca and drawdown <= pending_dca[-1][0]:
                trigger, alloc = pending_dca.pop()
                spend = min(cash, self.initial_equity * alloc)
                if spend <= 0:
                    continue
                qty = spend / price
                avg_cost = (avg_cost * holdings + spend) / (holdings + qty)
                holdings += qty
                cash -= spend
                next_depth_price = price * (1 - self.depth_percentage)
                self.executions.append(
                    Execution(date, self.symbol, "BUY_DCA", qty, price, spend)
                )

            if next_depth_price and price <= next_depth_price and cash > 0:
                spend = cash
                qty = spend / price
                avg_cost = (avg_cost * holdings + spend) / (holdings + qty)
                holdings += qty
                cash = 0
                next_depth_price = price * (1 - self.depth_percentage)
                self.executions.append(
                    Execution(date, self.symbol, "BUY_DEPTH", qty, price, spend)
                )

            equity_curve.append(cash + holdings * price)

        return {
            "symbol": self.symbol,
            "final_equity": equity_curve[-1],
            "equity_curve": equity_curve,
            "max_drawdown": _max_drawdown(equity_curve),
            "executions": self.executions,
            "holdings": holdings,
            "avg_cost": avg_cost,
        }



def main(csv_path: str = "ada_usd_last_1y.csv") -> None:
    """
    Run all strategies against the ADA 1-year dataset and print summary stats.
    """
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        else:
            raise ValueError("Input CSV must contain a 'date' or 'datetime' column")

    leverage = LeverageTrendStrategy(
        "ADA",
        df,
        lookback_days=10,
        leverage_multiple=2.0,
        margin_allocation=0.6,
        long_trigger_pct=0.1,
        long_take_profit_pct=0.1,
        long_stop_loss_pct=0.3,
    ).run()

    buy_dip = SimpleBuyDipStrategy(
        "ADA",
        df,
        initial_buy_trigger=0.9,
        initial_buy_amount=0.3,
        dca_triggers=[0.85, 0.75],
        dca_amounts=[0.3, 0.2],
        sell_trigger=1.15,
    ).run()

    buy_hold = BuyDipHoldStrategy(
        "ADA",
        df,
        initial_buy_trigger=0.9,
        initial_buy_amount=0.3,
        dca_triggers=[0.85, 0.75],
        dca_amounts=[0.3, 0.2],
    ).run()

    initial_equity = 10_000.0

    def _fmt_return(final_equity: float) -> str:
        return f"{((final_equity / initial_equity) - 1) * 100:.2f}%"

    print("== LeverageTrendStrategy ==")
    print(f"final_equity: {leverage['final_equity']:.2f}")
    print(f"return: {_fmt_return(leverage['final_equity'])}")
    print(f"executions: {len(leverage['executions'])}")
    print(f"max_drawdown: {leverage['max_drawdown']:.4f}")

    print("\n== SimpleBuyDipStrategy ==")
    print(f"final_equity: {buy_dip['final_equity']:.2f}")
    print(f"return: {_fmt_return(buy_dip['final_equity'])}")
    print(f"remaining_cash: {buy_dip['remaining_cash']:.2f}")
    print(f"holdings: {buy_dip['holdings']:.6f}")
    print(f"executions: {len(buy_dip['executions'])}")

    print("\n== BuyDipHoldStrategy ==")
    print(f"final_equity: {buy_hold['final_equity']:.2f}")
    print(f"return: {_fmt_return(buy_hold['final_equity'])}")
    print(f"holdings: {buy_hold['holdings']:.6f}")
    print(f"avg_cost: {buy_hold['avg_cost']:.6f}")
    print(f"executions: {len(buy_hold['executions'])}")


if __name__ == "__main__":
    main()
