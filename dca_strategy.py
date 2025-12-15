import pandas as pd

### 定投策略模拟

def simulate_strategy_on_available_dates(file_path, start_date, end_date, interval_days=7, invest_amount=100):
    # 读取数据
    df = pd.read_excel(file_path)
    if df.empty:
        raise ValueError("Excel 文件为空")
    if 'timeClose' not in df.columns or 'priceClose' not in df.columns:
        raise KeyError("Excel 文件必须包含 'timeClose' 和 'priceClose' 列")

    # 转换时间戳为日期
    df['timeClose'] = pd.to_datetime(df['timeClose'], unit='ms')
    df['date'] = df['timeClose'].dt.date
    df = df.set_index('date').sort_index()
    print(f"总交易数据行数: {len(df)}")


    # 筛选时间范围内的交易日
    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()
    df_range = df.loc[(df.index >= start_dt) & (df.index <= end_dt)].copy()
    
    if df_range.empty:
        raise ValueError("指定时间范围内没有交易数据")

    print(f"总交易数据行数: {len(df_range)}")
    
    # 按间隔选择交易日
    invest_dates = df_range.iloc[::interval_days]

    # 计算总投入和持币
    total_usdt = invest_amount * len(invest_dates)
    total_coin = (invest_amount / invest_dates['priceClose']).sum()

    # 最终价值按最后一个交易日收盘价计算
    final_price = df_range['priceClose'].iloc[-1]
    total_value = total_coin * final_price
    roi = (total_value - total_usdt) / total_usdt * 100

    # 返回结果
    result = {
        "total_invested_usdt": total_usdt,
        "total_coin": total_coin,
        "final_value_usdt": total_value,
        "roi_percent": roi,
        "invested_dates_count": len(invest_dates)
    }
    
    return result

def print_strategy_result(result):
    print("=== 定投策略结果 ===")
    print(f"投资次数          : {result['invested_dates_count']} 次")
    print(f"总投入 USDT       : {result['total_invested_usdt']:,} USDT")
    print(f"持有币数量        : {result['total_coin']:.6f} BTC")
    print(f"最终价值 USDT     : {result['final_value_usdt']:,} USDT")
    print(f"收益率            : {result['roi_percent']:.2f}%")
    print("===================")

# 使用示例
if __name__ == "__main__":
    result = simulate_strategy_on_available_dates(
        file_path='/Users/wangzai/Downloads/bitcoin.xlsx',  
        start_date='2020-09-01',
        end_date='2025-09-01',
        interval_days=30,  
        invest_amount=300
    )

    print_strategy_result(result)
