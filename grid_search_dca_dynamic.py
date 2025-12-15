import pandas as pd
import itertools
from datetime import datetime


def simulate_dynamic_strategy(df_range, invest_interval_days,
                              invest_amount,
                              price_up_pct, sell_pct,
                              price_down_pct, buy_pct):
    """
    在已经筛选好的 df_range（按 date index 排好序）上按规则模拟逐日交易。
    返回 final_value, total_invested, btc_holding, usdt_balance, roi_percent
    """
    btc_holding = 0.0
    usdt_balance = 0.0
    total_invested = 0.0

    # invest_counter counts交易日索引，从0开始，每当索引 % invest_interval_days == 0 则定投
    prev_price = None

    for i, (current_date, row) in enumerate(df_range.iterrows()):
        price = float(row['priceClose'])

        # 定投（按可用交易日间隔）
        if i % invest_interval_days == 0:
            # 直接用 invest_amount 买入 BTC（不计手续费）
            btc_bought = invest_amount / price
            btc_holding += btc_bought
            total_invested += invest_amount

        # 当日相较于前一日的涨跌幅判断
        if prev_price is not None and prev_price > 0:
            change_pct = (price - prev_price) / prev_price * 100.0

            # 涨幅止盈：涨 >= price_up_pct -> 卖出 sell_pct% 的 BTC
            if price_up_pct is not None and change_pct >= price_up_pct and btc_holding > 0:
                sell_amount_btc = btc_holding * (sell_pct / 100.0)
                usdt_balance += sell_amount_btc * price
                btc_holding -= sell_amount_btc

            # 跌幅止损：跌 <= -price_down_pct -> 用 usdt_balance 的 buy_pct% 买入 BTC（若有余额）
            if price_down_pct is not None and change_pct <= -price_down_pct and usdt_balance > 0:
                buy_usdt = usdt_balance * (buy_pct / 100.0)
                btc_bought = buy_usdt / price
                btc_holding += btc_bought
                usdt_balance -= buy_usdt

        prev_price = price

    # 结束：按最后一个交易日收盘价估值
    final_price = float(df_range['priceClose'].iloc[-1])
    final_value = btc_holding * final_price + usdt_balance

    # 防止 divide-by-zero
    if total_invested <= 0:
        roi = float('nan')
    else:
        roi = (final_value - total_invested) / total_invested * 100.0

    return {
        "final_value": final_value,
        "total_invested": total_invested,
        "btc_holding": btc_holding,
        "usdt_balance": usdt_balance,
        "roi_percent": roi
    }

def grid_search_dynamic_strategy(file_path,
                                 start_date, end_date,
                                 invest_amount=100,
                                 invest_interval_days_space=(1,7,30),
                                 price_up_space=range(1,11),    # 1%..10% step1
                                 sell_pct_space=range(10,101,10), # 10%..100% step10
                                 price_down_space=range(1,11),  # 1%..10%
                                 buy_pct_space=range(10,101,10), # 10%..100%
                                 top_k=10,
                                 out_csv='grid_search_results.csv'):
    """
    对给定参数空间执行网格搜索，返回最佳 top_k 组合（按 roi 降序）。
    可调整参数空间为更细/更粗的网格。
    """
    # 1. 读取并准备数据
    df = pd.read_excel(file_path)
    if df.empty:
        raise ValueError("Excel 文件为空")
    if 'timeClose' not in df.columns or 'priceClose' not in df.columns:
        raise KeyError("Excel 文件必须包含 'timeClose' 和 'priceClose' 列")

    df['timeClose'] = pd.to_datetime(df['timeClose'], unit='ms')
    df['date'] = df['timeClose'].dt.date
    df = df.set_index('date').sort_index()

    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()
    df_range = df.loc[(df.index >= start_dt) & (df.index <= end_dt)].copy()

    if df_range.empty:
        raise ValueError("指定时间范围内没有交易数据")

    # 2. 生成参数组合（注意：组合数量 = len(intervals)*len(price_up)*len(sell_pct)*...）
    combos = list(itertools.product(invest_interval_days_space,
                                    price_up_space,
                                    sell_pct_space,
                                    price_down_space,
                                    buy_pct_space))
    total = len(combos)
    print(f"[GridSearch] 共 {total} 个组合，将逐一模拟（请耐心等待）")

    results = []
    # 3. 遍历组合
    for idx, (interval_days, price_up, sell_pct, price_down, buy_pct) in enumerate(combos, start=1):
        # 进度提示（简单）
        if idx % max(1, total//20) == 0 or idx==1:
            print(f"[{idx}/{total}] interval={interval_days}, up={price_up}%, sell={sell_pct}%, down={price_down}%, buy={buy_pct}%")

        sim = simulate_dynamic_strategy(df_range,
                                        invest_interval_days=interval_days,
                                        invest_amount=invest_amount,
                                        price_up_pct=price_up,
                                        sell_pct=sell_pct,
                                        price_down_pct=price_down,
                                        buy_pct=buy_pct)

        results.append({
            "interval_days": interval_days,
            "price_up_pct": price_up,
            "sell_pct": sell_pct,
            "price_down_pct": price_down,
            "buy_pct": buy_pct,
            "total_invested": sim["total_invested"],
            "btc_holding": sim["btc_holding"],
            "usdt_balance": sim["usdt_balance"],
            "final_value": sim["final_value"],
            "roi_percent": sim["roi_percent"]
        })

    # 4. 转成 DataFrame，排序，保存 CSV
    res_df = pd.DataFrame(results)
    # 排序时把 NaN roi 放到末尾
    res_df = res_df.sort_values(by='roi_percent', ascending=False, na_position='last').reset_index(drop=True)
    res_df.to_csv(out_csv, index=False)

    # 5. 返回 top_k
    topk_df = res_df.head(top_k).copy()
    return topk_df, res_df

def pretty_print_topk(topk_df):
    print("\n=== 网格搜索最优结果（前几名） ===")
    for i, row in topk_df.iterrows():
        print(f"Rank {i+1}: interval={int(row['interval_days'])} days | up={row['price_up_pct']}% sell={row['sell_pct']}% | down={row['price_down_pct']}% buy={row['buy_pct']}%")
        print(f"    总投入: {row['total_invested']:,.2f} USDT  | 持币: {row['btc_holding']:.6f} BTC  | 账户USDT余额: {row['usdt_balance']:,.2f} USDT")
        print(f"    最终总价值: {row['final_value']:,.2f} USDT  | ROI: {row['roi_percent']:.2f}%")
    print("=================================\n")

# ---------------------------
# 使用示例（把路径与参数按需改）
# ---------------------------
if __name__ == "__main__":
    # 可根据需要缩小/扩大搜索空间以节省时间
    topk, alldf = grid_search_dynamic_strategy(
        file_path='/Users/wangzai/Downloads/bitcoin.xlsx',
        start_date='2020-09-01',
        end_date='2025-09-01',
        invest_amount=300,
        invest_interval_days_space=(30),    # 交易日间隔可自定义
        price_up_space=range(1, 11),              # 1%..10%
        sell_pct_space=range(10, 101, 10),        # 10%..100%
        price_down_space=range(1, 11),            # 1%..10%
        buy_pct_space=range(10, 101, 10),         # 10%..100%
        top_k=10,
        out_csv='grid_search_results.csv'
    )

    pretty_print_topk(topk)
    print("所有组合结果已保存为 grid_search_results.csv")
