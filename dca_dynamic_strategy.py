import pandas as pd

### 动态定投+止盈止损策略

import pandas as pd
from tqdm import tqdm

def simulate_dynamic_strategy(
    file_path, start_date, end_date,
    invest_interval_days=7, invest_amount=100,
    price_up_pct=5, sell_pct=50,
    price_down_pct=5, buy_pct=50,
    initial_usdt=0.0
):
    """
    修正后：动态定投 + 止盈止损（平均成本法）
    - total_deposited: 外部注入的 USDT（仅通过定投注入）
    - usdt_balance: 账户内现金（可能来自卖出或 initial_usdt）
    - cost_basis: 当前 BTC 持仓的累计成本（USDT）
    - realized_pnl: 累计已实现盈亏（USDT）
    """
    df = pd.read_excel(file_path)
    # 假设 timeClose 单位是 ms
    df['timeClose'] = pd.to_datetime(df['timeClose'], unit='ms')
    # 使用日期索引（注意若有多条同日数据，需要按需求去重或取收盘）
    df['date'] = df['timeClose'].dt.date
    df = df.set_index('date').sort_index()

    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()
    df_range = df.loc[(df.index >= start_dt) & (df.index <= end_dt)].copy()

    if df_range.empty:
        raise ValueError("指定时间范围内没有交易数据")

    # 初始化
    btc_holding = 0.0  # 当前持仓的 BTC 数量
    usdt_balance = float(initial_usdt) # 账户内现金
    total_deposited = 0.0  # 外部注入资金（只在定投时增加）
    cost_basis = 0.0       # 当前持仓的累计成本（USDT）
    realized_pnl = 0.0     # 累计已实现盈亏（USDT）

    invest_counter = 0

    for current_date, row in df_range.iterrows():
        price = float(row['priceClose'])

        # 定投（把 invest_amount 视为外部注入资金，用于购买 BTC）
        if invest_counter % invest_interval_days == 0:
            # 直接用外部资金买入 BTC
            btc_bought = invest_amount / price
            btc_holding += btc_bought
            cost_basis += invest_amount          # 成本基数增加
            total_deposited += invest_amount     # 外部注入增加

        # 只有在持有 BTC 时计算平均成本与涨跌百分比
        avg_cost = cost_basis / btc_holding if btc_holding > 0 else None

        if avg_cost is not None:
            change_pct = (price - avg_cost) / avg_cost * 100.0

            # 涨幅止盈（按平均成本计算）：卖出比例 sell_pct
            if change_pct >= price_up_pct and btc_holding > 0:
                sell_btc = btc_holding * (sell_pct / 100.0) # 卖出BTC个数
                sell_proceeds = sell_btc * price # 获得USDT金额
                # 按平均成本扣减 cost_basis
                sold_cost = sell_btc * avg_cost # 卖出BTC的对应的USDT成本
                realized = sell_proceeds - sold_cost # 卖出BTC获得的USDT收益

                usdt_balance += sell_proceeds  # 账户内现金增加
                btc_holding -= sell_btc  # 持仓减少
                cost_basis -= sold_cost  # BTC购入成本基数减少
                realized_pnl += realized  # 已实现盈亏增加

                # 容错：若持仓接近 0，则清零
                if btc_holding <= 1e-12:
                    btc_holding = 0.0
                    cost_basis = 0.0

            # 跌幅买入（用账户内 USDT 买入）
            if change_pct <= -price_down_pct and usdt_balance > 0:
                spend_usdt = usdt_balance * (buy_pct / 100.0) # 用账户内USDT购买BTC
                # 防止极小值或浮点问题
                if spend_usdt > 0:
                    bought_btc = spend_usdt / price # 购买BTC的数量
                    btc_holding += bought_btc  # 持仓增加
                    cost_basis += spend_usdt  # 成本基数增加
                    usdt_balance -= spend_usdt  # 账户内现金减少

        invest_counter += 1

    final_price = float(df_range['priceClose'].iloc[-1])
    final_value = btc_holding * final_price + usdt_balance
    roi = (final_value - total_deposited) / total_deposited * 100.0 if total_deposited > 0 else float('nan')
    avg_cost_final = cost_basis / btc_holding if btc_holding > 0 else None

    return {
        "total_deposited_usdt": total_deposited,
        "btc_holding": btc_holding,
        "usdt_balance": usdt_balance,
        "cost_basis": cost_basis,
        "avg_cost_final": avg_cost_final,
        "realized_pnl": realized_pnl,
        "final_value_usdt": final_value,
        "roi_percent": roi
    }

# # 使用示例
# if __name__ == "__main__":
#     result = simulate_dynamic_strategy(
#         file_path='/Users/wangzai/Downloads/bitcoin.xlsx',
#         start_date='2020-09-01',
#         end_date='2025-09-01',
#         invest_interval_days=30,   # 每隔30个交易日定投
#         invest_amount=300,
#         price_up_pct=50,   # 较均价的涨幅百分比卖出部分BTC
#         sell_pct=20,      # 卖掉持仓比例的BTC
#         price_down_pct=1, # 较均价的跌幅百分比买入BTC
#         buy_pct=10       # 用该比例USDT买入
#     )
    
#     # 美化输出
#     print("=== 动态定投策略结果 ===")
#     print(f"总投入 USDT       : {result['total_deposited_usdt']:,} USDT")
#     print(f"持有 BTC         : {result['btc_holding']:.6f} BTC")
#     print(f"账户 USDT 余额    : {result['usdt_balance']:,} USDT")
#     print(f"最终总价值 USDT   : {result['final_value_usdt']:,} USDT")
#     print(f"策略收益率       : {result['roi_percent']:.2f}%")
#     print("==========================")

def find_optimal_parameters():
    """
    对 price_up_pct、sell_pct、price_down_pct、buy_pct 分别按照 step=5 的幅度进行遍历，
    找到使策略收益率最大化的参数组合。
    """
    # 定义参数范围和步长
    step = 10
    price_up_pct_range = range(0, 51, step)
    sell_pct_range = range(0, 51, step)
    price_down_pct_range = range(0, 51, step)
    buy_pct_range = range(0, 51, step)

    max_roi = float('-inf')
    optimal_params = {}

    # 使用 tqdm 显示进度条
    total_iterations = len(price_up_pct_range) * len(sell_pct_range) * len(price_down_pct_range) * len(buy_pct_range)
    with tqdm(total=total_iterations, desc="寻找最优参数") as pbar:
        for price_up_pct in price_up_pct_range:
            for sell_pct in sell_pct_range:
                for price_down_pct in price_down_pct_range:
                    for buy_pct in buy_pct_range:
                        try:
                            result = simulate_dynamic_strategy(
                                file_path='/Users/wangzai/Downloads/bitcoin.xlsx',
                                start_date='2020-09-01',
                                end_date='2025-09-01',
                                invest_interval_days=30,
                                invest_amount=300,
                                price_up_pct=price_up_pct,
                                sell_pct=sell_pct,
                                price_down_pct=price_down_pct,
                                buy_pct=buy_pct
                            )
                            roi = result['roi_percent']
                            if roi > max_roi:
                                max_roi = roi
                                optimal_params = {
                                    'price_up_pct': price_up_pct,
                                    'sell_pct': sell_pct,
                                    'price_down_pct': price_down_pct,
                                    'buy_pct': buy_pct
                                }
                        except Exception as e:
                            print(f"参数组合 {price_up_pct}, {sell_pct}, {price_down_pct}, {buy_pct} 出错: {str(e)}")
                        pbar.update(1)

    print("=== 最优参数组合 ===")
    print(f"price_up_pct: {optimal_params['price_up_pct']}")
    print(f"sell_pct: {optimal_params['sell_pct']}")
    print(f"price_down_pct: {optimal_params['price_down_pct']}")
    print(f"buy_pct: {optimal_params['buy_pct']}")
    print(f"最大收益率: {max_roi:.2f}%")
    return optimal_params

if __name__ == "__main__":
    # 调用寻找最优参数的函数
    find_optimal_parameters()
