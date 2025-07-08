# %%
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
# Add Option_Data to sys.path to import the loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Option_Data')))
from load_merged_data import load_merged_option_ohlcv

# %%
def prepare_price_direction_data(df):
    # Ensure datetime for date and expiration
    df['expiration'] = pd.to_datetime(df['expiration'])
    df['date'] = pd.to_datetime(df['date'])
    # Add time to expiration in days
    df['tte_days'] = (df['expiration'] - df['date']).dt.days
    # Drop the original expiration column
    df = df.drop(columns=['expiration'])
    # Ensure sorted by date for each contract
    df = df.sort_values(['date', 'tte_days', 'strike', 'type'])
    # Use close price to create target: 1 if next day's close > today's, else 0
    df['close_next'] = df.groupby('date')['close'].transform(lambda x: x.shift(-1))
    df['target'] = (df['close_next'] > df['close']).astype(int)
    # Drop rows where next day's close is not available
    df = df.dropna(subset=['close_next'])
    return df

# %%
# Load merged data
df = load_merged_option_ohlcv()

# %%
# Prepare features and target
df = prepare_price_direction_data(df)

# %%
# Example: use all numeric columns except 'target' as features
feature_cols = df.select_dtypes(include='number').columns.difference(['target', 'close_next'])
X = df[feature_cols]
y = df['target']

# %%
# Chronological train/test split for time series
total_len = len(X)
split_idx = int(total_len * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Sample features: {X_train.head()}")
print(f"Sample targets: {y_train.head()}")


