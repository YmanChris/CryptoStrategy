from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union, Literal

import requests
import pandas as pd

DT = Union[datetime, int, float]  # datetime 或 unix秒


def _to_unix_seconds(t: DT) -> int:
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return int(t.timestamp())
    return int(t)


@dataclass
class CoinGeckoClient:
    plan: Literal["demo", "pro"] = "demo"
    api_key: Optional[str] = None

    # Demo: https://api.coingecko.com/api/v3
    # Pro : https://pro-api.coingecko.com/api/v3
    base_url: Optional[str] = None

    auth_in: Literal["header", "query"] = "header"  # 推荐 header
    timeout: int = 20
    max_retries: int = 5
    backoff_base: float = 0.8
    min_interval_sec: float = 0.2

    _last_call_ts: float = 0.0

    def __post_init__(self):
        if self.base_url is None:
            self.base_url = (
                "https://api.coingecko.com/api/v3"
                if self.plan == "demo"
                else "https://pro-api.coingecko.com/api/v3"
            )

    def _sleep_if_needed(self):
        gap = time.time() - self._last_call_ts
        if gap < self.min_interval_sec:
            time.sleep(self.min_interval_sec - gap)

    def _attach_auth(self, headers: dict, params: dict) -> tuple[dict, dict]:
        if not self.api_key:
            # 现在很多端点不带 key 会 401（你遇到的就是这种）
            return headers, params

        if self.auth_in == "header":
            # 官方推荐 Header 方式
            if self.plan == "demo":
                headers["x-cg-demo-api-key"] = self.api_key
            else:
                headers["x-cg-pro-api-key"] = self.api_key
        else:
            # query 方式（文档也给了示例，但不如 header 安全）
            if self.plan == "demo":
                params["x_cg_demo_api_key"] = self.api_key
            else:
                params["x_cg_pro_api_key"] = self.api_key

        return headers, params

    def _get(self, path: str, params: dict) -> dict:
        url = f"{self.base_url}{path}"
        headers = {"accept": "application/json"}

        headers, params = self._attach_auth(headers, params)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                self._sleep_if_needed()
                resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
                self._last_call_ts = time.time()

                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    time.sleep(float(retry_after) if retry_after else self.backoff_base * (2 ** attempt))
                    continue

                resp.raise_for_status()
                return resp.json()

            except Exception as e:
                last_err = e
                time.sleep(self.backoff_base * (2 ** attempt))

        raise RuntimeError(f"CoinGecko request failed after retries: {last_err}") from last_err


def fetch_price_range(
    coin_id: str,
    vs_currency: str,
    start: DT,
    end: DT,
    *,
    client: CoinGeckoClient,
    include: tuple[str, ...] = ("prices",),
    tz: Literal["utc", "local"] = "utc",
) -> pd.DataFrame:
    if client is None:
        raise ValueError("client is required (pass CoinGeckoClient with api_key).")

    from_ts = _to_unix_seconds(start)
    to_ts = _to_unix_seconds(end)
    if to_ts <= from_ts:
        raise ValueError("end must be greater than start")

    data = client._get(
        f"/coins/{coin_id}/market_chart/range",
        params={"vs_currency": vs_currency, "from": from_ts, "to": to_ts},
    )

    frames = []
    for key in include:
        if key not in data:
            raise KeyError(f"Response has no field '{key}'. Available: {list(data.keys())}")

        arr = data[key]
        col = {
            "prices": f"price_{vs_currency.lower()}",
            "market_caps": f"market_cap_{vs_currency.lower()}",
            "total_volumes": f"volume_{vs_currency.lower()}",
        }.get(key, key)

        df = pd.DataFrame(arr, columns=["timestamp_ms", col])
        frames.append(df)

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="timestamp_ms", how="outer")

    ts = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True)
    if tz == "local":
        ts = ts.dt.tz_convert(None)
    else:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    out.insert(0, "datetime", ts)
    out = out.drop(columns=["timestamp_ms"]).sort_values("datetime").reset_index(drop=True)
    return out

import pandas as pd
from typing import Literal

def fetch_top_marketcap(
    *,
    client: CoinGeckoClient,
    vs_currency: str = "usd",
    top_n: int = 100,
    page_size: int = 250,
) -> pd.DataFrame:
    """
    获取按市值排序的 TopN 币种（CoinGecko coins/markets）。
    返回包含 rank/id/symbol/name/market_cap/market_cap_rank 等字段的 DataFrame。
    """
    if top_n <= 0:
        raise ValueError("top_n must be > 0")
    if page_size <= 0 or page_size > 250:
        raise ValueError("page_size must be in (1..250)")

    rows = []
    page = 1

    while len(rows) < top_n:
        per_page = min(page_size, top_n - len(rows))
        data = client._get(
            "/coins/markets",
            params={
                "vs_currency": vs_currency,
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": "false",
                "price_change_percentage": "24h",
            },
        )
        if not isinstance(data, list) or len(data) == 0:
            break

        rows.extend(data)
        page += 1

    df = pd.DataFrame(rows)

    # 统一字段 & 排序
    keep = [
        "market_cap_rank", "id", "symbol", "name",
        "current_price", "market_cap", "total_volume",
        "price_change_percentage_24h"
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = None

    df = df[keep].sort_values("market_cap_rank").head(top_n).reset_index(drop=True)
    return df


def print_top_marketcap(df: pd.DataFrame, top_n: int = 100) -> None:
    """
    漂亮打印 TopN（控制台输出）。
    """
    show = df.head(top_n).copy()
    show["symbol"] = show["symbol"].str.upper()

    # 避免科学计数法太难读
    pd.set_option("display.max_rows", top_n)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print(show.to_string(index=False))



from datetime import datetime, timedelta, timezone

client = CoinGeckoClient(
    plan="demo",                 # 或 "pro"
    api_key="CG-Sf7QTtyqta3DcMr2x2ncV4BA",     # 换成你的 key
    auth_in="header",
)

# 1) 打印市值 Top100
top_df = fetch_top_marketcap(client=client, vs_currency="usd", top_n=100)
print_top_marketcap(top_df, top_n=100)

# 2) 下载过去一年 BTC 日/小时粒度（由 CoinGecko 自动采样）
end = datetime.now(timezone.utc) - timedelta(days=365)
start = end - timedelta(days=365)

btc_df = fetch_price_range("cardano", "usd", start, end, client=client, include=("prices",))
btc_df.to_csv("ada_usd_last_2y.csv", index=False)
print("\nSaved ada_usd_last_2y.csv, rows=", len(btc_df))

