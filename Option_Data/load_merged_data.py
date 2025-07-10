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
    daily_features = daily_features.sort_values('date').reset_index(drop=True)
    return daily_features

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

    # Prepare DataFrames
    contracts_df = pd.DataFrame(all_contracts)
    ohlcv_df = pd.DataFrame.from_dict(all_ohlcv, orient="index").reset_index().rename(columns={"index": "date"})
    contracts_df['date'] = pd.to_datetime(contracts_df['date'])
    contracts_df['expiration'] = pd.to_datetime(contracts_df['expiration'])
    ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date'])

    # Rename and convert columns for aggregation
    contracts_df = contracts_df.rename(columns={'implied_volatility': 'iv'})
    for col in ['bid', 'ask', 'open_interest', 'volume', 'strike', 'iv']:
        contracts_df[col] = pd.to_numeric(contracts_df[col], errors='coerce')

    # Aggregate option features by date
    option_features = generate_option_chain_features(contracts_df)

    # Merge OHLCV and option features by date
    daily_df = ohlcv_df.merge(option_features, on='date', how='left')
    daily_df = daily_df.sort_values('date').reset_index(drop=True)

    # Return only OHLCV + option features (no contract-level columns, no repeated dates)
    return daily_df

if __name__ == "__main__":
    df = load_merged_option_ohlcv()
    print(df.head())
# %%
