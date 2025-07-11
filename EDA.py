#%%
#get data from toption/test_data/ .json

import json
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 中文：
# 從 Option_Data/Test_data 目錄獲取所有 JSON 文件並合併成一個 pd DataFrame
#%%
folder = "/Users/nigel/Documents/GitHub/Dynamic_Option_Hedging/Option_Data/Test_data"
all_data = []

# 遍歷文件夾中的所有 JSON 文件
for filename in os.listdir(folder):
    if filename.endswith(".json") and not filename.startswith("._"):
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            content = json.load(f)
            if "data" in content:
                all_data.extend(content["data"])

# 從所有期權合約創建 DataFrame
df = pd.DataFrame(all_data)

#%%
# 生成到期時間 (TTE) 特徵
# 將日期列轉換為 datetime
df['expiration'] = pd.to_datetime(df['expiration'])
df['date'] = pd.to_datetime(df['date'])

# 計算 TTE（天數）
df['TTE_days'] = (df['expiration'] - df['date']).dt.days

# 計算 TTE（年數，用於期權定價模型）
df['TTE_years'] = df['TTE_days'] / 365.25

# 將字符串列轉換為數值以便過濾
df['bid_size'] = pd.to_numeric(df['bid_size'], errors='coerce')
df['ask_size'] = pd.to_numeric(df['ask_size'], errors='coerce')
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')

# 顯示原始數據形狀
print(f"原始數據形狀: {df.shape}")

# 過濾掉交易量極小的數據
# 刪除 bid_size 或 ask_size 為 0 的行
df_filtered = df[(df['bid_size'] > 10) & (df['ask_size'] > 10)]

# 額外的過濾選項（根據需要取消註釋）：
# 刪除成交量很低的行（例如，volume = 0）
df_filtered = df_filtered[df_filtered['volume'] >= 10]

# 刪除未平倉合約很少的行（例如，open_interest < 5）
df_filtered = df_filtered[df_filtered['open_interest'] >= 5]

# 刪除買賣價都為 0 的行（無流動性）
df_filtered = df_filtered[~((df_filtered['bid'] == '0.00') & (df_filtered['ask'] == '0.00'))]

print(f"過濾後數據形狀: {df_filtered.shape}")
print(f"刪除了 {df.shape[0] - df_filtered.shape[0]} 行")

# %%

import seaborn as sns
import matplotlib.pyplot as plt

# Set style for better visualizations
plt.style.use('default')  # Use default style instead of seaborn-v0_8
sns.set_theme(style="whitegrid")  # Use seaborn's set_theme instead
sns.set_palette("husl")

# 1. Basic Data Overview
print("=== BASIC DATA OVERVIEW ===")
print(f"Total number of option contracts: {len(df_filtered)}")
print(f"Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")
print(f"Expiration range: {df_filtered['expiration'].min()} to {df_filtered['expiration'].max()}")
print(f"Number of unique strike prices: {df_filtered['strike'].nunique()}")
print(f"Number of unique expiration dates: {df_filtered['expiration'].nunique()}")

# 2. Option Type Distribution
print("\n=== OPTION TYPE DISTRIBUTION ===")
option_type_counts = df_filtered['type'].value_counts()
print(option_type_counts)

# Create a pie chart for option types
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.pie(option_type_counts.values, labels=option_type_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Option Types')

# 3. Strike Price Analysis
plt.subplot(2, 3, 2)
plt.hist(pd.to_numeric(df_filtered['strike']), bins=30, alpha=0.7, edgecolor='black')
plt.title('Distribution of Strike Prices')
plt.xlabel('Strike Price')
plt.ylabel('Frequency')

# 4. Time to Expiry Analysis
plt.subplot(2, 3, 3)
plt.hist(df_filtered['TTE_days'], bins=30, alpha=0.7, edgecolor='black')
plt.title('Distribution of Time to Expiry')
plt.xlabel('Days to Expiry')
plt.ylabel('Frequency')

# 5. Volume Analysis
plt.subplot(2, 3, 4)
plt.hist(df_filtered['volume'], bins=30, alpha=0.7, edgecolor='black')
plt.title('Distribution of Trading Volume')
plt.xlabel('Volume')
plt.ylabel('Frequency')

# 6. Open Interest Analysis
plt.subplot(2, 3, 5)
plt.hist(df_filtered['open_interest'], bins=30, alpha=0.7, edgecolor='black')
plt.title('Distribution of Open Interest')
plt.xlabel('Open Interest')
plt.ylabel('Frequency')

# 7. Implied Volatility Analysis
plt.subplot(2, 3, 6)
plt.hist(pd.to_numeric(df_filtered['implied_volatility']), bins=30, alpha=0.7, edgecolor='black')
plt.title('Distribution of Implied Volatility')
plt.xlabel('Implied Volatility')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 8. Greeks Analysis
print("\n=== GREEKS ANALYSIS ===")
greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
for greek in greeks:
    df_filtered[greek] = pd.to_numeric(df_filtered[greek], errors='coerce')

plt.figure(figsize=(15, 10))

for i, greek in enumerate(greeks, 1):
    plt.subplot(2, 3, i)
    plt.hist(df_filtered[greek].dropna(), bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of {greek.upper()}')
    plt.xlabel(greek.upper())
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 9. Strike Price vs Greeks Scatter Plots
plt.figure(figsize=(15, 10))

# Delta vs Strike
plt.subplot(2, 3, 1)
plt.scatter(pd.to_numeric(df_filtered['strike']), df_filtered['delta'], alpha=0.5)
plt.title('Delta vs Strike Price')
plt.xlabel('Strike Price')
plt.ylabel('Delta')

# Gamma vs Strike
plt.subplot(2, 3, 2)
plt.scatter(pd.to_numeric(df_filtered['strike']), df_filtered['gamma'], alpha=0.5)
plt.title('Gamma vs Strike Price')
plt.xlabel('Strike Price')
plt.ylabel('Gamma')

# Theta vs Strike
plt.subplot(2, 3, 3)
plt.scatter(pd.to_numeric(df_filtered['strike']), df_filtered['theta'], alpha=0.5)
plt.title('Theta vs Strike Price')
plt.xlabel('Strike Price')
plt.ylabel('Theta')

# Vega vs Strike
plt.subplot(2, 3, 4)
plt.scatter(pd.to_numeric(df_filtered['strike']), df_filtered['vega'], alpha=0.5)
plt.title('Vega vs Strike Price')
plt.xlabel('Strike Price')
plt.ylabel('Vega')

# Rho vs Strike
plt.subplot(2, 3, 5)
plt.scatter(pd.to_numeric(df_filtered['strike']), df_filtered['rho'], alpha=0.5)
plt.title('Rho vs Strike Price')
plt.xlabel('Strike Price')
plt.ylabel('Rho')

# Implied Volatility vs Strike
plt.subplot(2, 3, 6)
plt.scatter(pd.to_numeric(df_filtered['strike']), pd.to_numeric(df_filtered['implied_volatility']), alpha=0.5)
plt.title('Implied Volatility vs Strike Price')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.tight_layout()
plt.show()

# 10. Correlation Matrix
print("\n=== CORRELATION ANALYSIS ===")
numeric_columns = ['strike', 'volume', 'open_interest', 'TTE_days', 'delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility']
numeric_df = df_filtered[numeric_columns].apply(pd.to_numeric, errors='coerce')

correlation_matrix = numeric_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()
plt.show()

# 11. Summary Statistics
print("\n=== SUMMARY STATISTICS ===")
print(df_filtered.describe())


# %% 