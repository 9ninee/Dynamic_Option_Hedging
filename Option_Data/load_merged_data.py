#%%

import os
import json
import pandas as pd
import numpy as np
from scipy.stats import skew

def generate_option_chain_features(contracts_df):
    # Calculate mid, bid-ask spread, bid-ask ratio
    contracts_df['mid'] = (contracts_df['bid'] + contracts_df['ask']) / 2
    contracts_df['bid_ask_spread'] = contracts_df['ask'] - contracts_df['bid']
    contracts_df['bid_ask_ratio'] = contracts_df['bid_ask_spread'] / contracts_df['mid'].replace(0, np.nan)
    contracts_df['tte_days'] = (pd.to_datetime(contracts_df['expiration']) - pd.to_datetime(contracts_df['date'])).dt.days

    # Aggregation functions
    agg = {
        'iv': ['mean', 'std'],
        'bid_ask_spread': ['mean', 'median', 'std'],
        'bid_ask_ratio': ['mean', 'median', 'std'],
        'open_interest': ['sum', 'mean'],
        'volume': ['sum', 'mean'],
        'tte_days': ['mean', 'std'],
        'mid': ['mean', 'std'],
        'strike': ['mean', 'std'],
    }
    all_opts = contracts_df.groupby('date').agg(agg)
    all_opts.columns = ['all_' + '_'.join(col) for col in all_opts.columns]

    # By type (call/put)
    features = [all_opts]
    for opt_type in ['call', 'put']:
        sub = contracts_df[contracts_df['type'] == opt_type]
        sub_agg = sub.groupby('date').agg(agg)
        sub_agg.columns = [f'{opt_type}_' + '_'.join(col) for col in sub_agg.columns]
        features.append(sub_agg)

    # IV Spread (call - put)
    daily_iv = contracts_df.groupby(['date', 'type'])['iv'].mean().unstack()
    daily_iv['iv_spread'] = daily_iv['call'] - daily_iv['put']

    # IV skew: std of IV by strike for each day
    iv_skew = contracts_df.groupby('date').apply(lambda x: x.groupby('strike')['iv'].mean().std())
    iv_skew.name = 'iv_skew'

    # Call/put open interest ratio
    call_oi = contracts_df[contracts_df['type'] == 'call'].groupby('date')['open_interest'].sum()
    put_oi = contracts_df[contracts_df['type'] == 'put'].groupby('date')['open_interest'].sum()
    call_put_oi_ratio = (call_oi / put_oi).replace([np.inf, -np.inf], np.nan).rename('call_put_oi_ratio')

    # Merge all features
    daily_features = features[0].join(features[1:], how='outer')
    daily_features = daily_features.join(daily_iv[['iv_spread']], how='left')
    daily_features = daily_features.join(iv_skew, how='left')
    daily_features = daily_features.join(call_put_oi_ratio, how='left')
    daily_features = daily_features.reset_index()
    return daily_features

def generate_single_option_feature_row(contracts_df):
    # Calculate mid, bid-ask spread, bid-ask ratio
    contracts_df['mid'] = (contracts_df['bid'] + contracts_df['ask']) / 2
    contracts_df['bid_ask_spread'] = contracts_df['ask'] - contracts_df['bid']
    contracts_df['bid_ask_ratio'] = contracts_df['bid_ask_spread'] / contracts_df['mid'].replace(0, np.nan)
    contracts_df['tte_days'] = (pd.to_datetime(contracts_df['expiration']) - pd.to_datetime(contracts_df['date'])).dt.days

    # Aggregation functions (no groupby)
    agg_funcs = {
        'iv': ['mean', 'std'],
        'bid_ask_spread': ['mean', 'median', 'std'],
        'bid_ask_ratio': ['mean', 'median', 'std'],
        'open_interest': ['sum', 'mean'],
        'volume': ['sum', 'mean'],
        'tte_days': ['mean', 'std'],
        'mid': ['mean', 'std'],
        'strike': ['mean', 'std'],
    }
    all_features = {}
    for col, funcs in agg_funcs.items():
        for func in funcs:
            key = f'all_{col}_{func}'
            all_features[key] = getattr(contracts_df[col], func)()

    # By type (call/put)
    for opt_type in ['call', 'put']:
        sub = contracts_df[contracts_df['type'] == opt_type]
        for col, funcs in agg_funcs.items():
            for func in funcs:
                key = f'{opt_type}_{col}_{func}'
                all_features[key] = getattr(sub[col], func)() if not sub.empty else np.nan

    # IV Spread (call - put)
    call_iv_mean = contracts_df[contracts_df['type'] == 'call']['iv'].mean()
    put_iv_mean = contracts_df[contracts_df['type'] == 'put']['iv'].mean()
    all_features['iv_spread'] = call_iv_mean - put_iv_mean

    # IV skew: std of IV by strike
    iv_skew = contracts_df.groupby('strike')['iv'].mean().std()
    all_features['iv_skew'] = iv_skew

    # Call/put open interest ratio
    call_oi = contracts_df[contracts_df['type'] == 'call']['open_interest'].sum()
    put_oi = contracts_df[contracts_df['type'] == 'put']['open_interest'].sum()
    all_features['call_put_oi_ratio'] = (call_oi / put_oi) if put_oi != 0 else np.nan

    return pd.DataFrame([all_features])

def load_merged_option_ohlcv(merged_folder="Option_Data/Test_data/merged"):
    all_contracts = []
    all_ohlcv = {}

    # Read all merged files
    for filename in os.listdir(merged_folder):
        if filename.endswith(".json") and not filename.startswith("._"):
            with open(os.path.join(merged_folder, filename), "r", encoding="utf-8") as f:
                merged = json.load(f)
                all_contracts.extend(merged["contracts"])
                for date, ohlcv in merged["ohlcv"].items():
                    all_ohlcv[date] = ohlcv

    contracts_df = pd.DataFrame(all_contracts)
    ohlcv_df = pd.DataFrame.from_dict(all_ohlcv, orient="index").reset_index().rename(columns={"index": "date"})
    contracts_df['date'] = pd.to_datetime(contracts_df['date'])
    contracts_df['expiration'] = pd.to_datetime(contracts_df['expiration'])
    ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date'])

    # Always rename and convert columns for aggregation
    contracts_df = contracts_df.rename(columns={'implied_volatility': 'iv'})
    for col in ['bid', 'ask', 'open_interest', 'volume', 'strike', 'iv']:
        contracts_df[col] = pd.to_numeric(contracts_df[col], errors='coerce')

    # Generate daily option features
    option_features = generate_option_chain_features(contracts_df)

    # Merge with OHLCV (one row per date)
    daily_df = ohlcv_df.merge(option_features, on='date', how='left')
    daily_df = daily_df.sort_values('date').reset_index(drop=True)

    # Only return OHLCV columns and aggregated option features (no contract-level columns)
    # This is already the case, but let's be explicit:
    # Remove any columns that are not from OHLCV or option_features
    ohlcv_cols = set(ohlcv_df.columns)
    opt_cols = set(option_features.columns)
    keep_cols = [col for col in daily_df.columns if col in ohlcv_cols or col in opt_cols]
    return daily_df[keep_cols]

if __name__ == "__main__":
    df = load_merged_option_ohlcv()
    print(df.head())
    all_contracts = []
    for filename in os.listdir("Option_Data/Test_data/merged"):
        if filename.endswith(".json") and not filename.startswith("._"):
            with open(os.path.join("Option_Data/Test_data/merged", filename), "r", encoding="utf-8") as f:
                merged = json.load(f)
                all_contracts.extend(merged["contracts"])
    contracts_df = pd.DataFrame(all_contracts)
    # Ensure correct column names and types for single row feature generation
    contracts_df = contracts_df.rename(columns={'implied_volatility': 'iv'})
    for col in ['bid', 'ask', 'open_interest', 'volume', 'strike', 'iv']:
        contracts_df[col] = pd.to_numeric(contracts_df[col], errors='coerce')
    print("Single row of aggregated option features:")
    print(generate_single_option_feature_row(contracts_df).T)
# %%
