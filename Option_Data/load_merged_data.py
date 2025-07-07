#%%

import os
import json
import pandas as pd

def load_merged_option_ohlcv(merged_folder="Option_Data/Test_data/merged"):
    all_contracts = []
    all_ohlcv = {}

    # Read all merged files
    for filename in os.listdir(merged_folder):
        if filename.endswith(".json") and not filename.startswith("._"):
            with open(os.path.join(merged_folder, filename), "r", encoding="utf-8") as f:
                merged = json.load(f)
                # Collect contracts
                all_contracts.extend(merged["contracts"])
                # Collect OHLCV (no duplicates)
                for date, ohlcv in merged["ohlcv"].items():
                    all_ohlcv[date] = ohlcv

    # Convert to DataFrame
    contracts_df = pd.DataFrame(all_contracts)
    ohlcv_df = pd.DataFrame.from_dict(all_ohlcv, orient="index").reset_index().rename(columns={"index": "date"})

    # Merge OHLCV into each contract by date
    contracts_df = contracts_df.merge(ohlcv_df, on="date", suffixes=("", "_ohlcv"))
    return contracts_df

if __name__ == "__main__":
    df = load_merged_option_ohlcv()
    print(df.head()) 
# %%
