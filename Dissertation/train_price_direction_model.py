# %%
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
# Add Option_Data to sys.path to import the loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Option_Data')))
from load_merged_data import load_merged_option_ohlcv

# %%
# Load merged daily data (one row per date)
df = load_merged_option_ohlcv()

# Drop rows with missing return (first day or missing data)
df = df.dropna(subset=['return'])

# Set target: 1 if return > 0 else 0
y = (df['return'] > 0).astype(int)

# Use all numeric columns except 'return' as features
feature_cols = df.select_dtypes(include='number').columns.difference(['return'])
X = df[feature_cols]

# %%
# Chronological train/test split for time series
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Sample features: {X_train.head()}")
print(f"Sample targets: {y_train.head()}")

# %%
# Train a simple logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# %%
# Evaluate the model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model accuracy: {accuracy:.4f}")

# %%
df.head()
# %%
