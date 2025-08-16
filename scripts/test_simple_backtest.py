"""Test simple backtest."""

import pandas as pd

# Load a metrics file
df = pd.read_parquet('data/metrics/Trump Election Rally 2016_metrics.parquet')

print(f"Data shape: {df.shape}")
print(f"Has close: {'close' in df.columns}")

# Test the simple backtest logic
if not df.empty and 'close' in df.columns:
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    total_return = (end_price - start_price) / start_price
    print(f"Start price: {start_price}")
    print(f"End price: {end_price}")
    print(f"Total return: {total_return:.2%}")
else:
    print("No close data!")