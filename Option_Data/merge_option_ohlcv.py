import os
import json
import pandas as pd
from typing import Optional

def merge_option_chain_with_ohlcv(option_chain_folder: str, ohlcv_csv: str, output_folder: Optional[str] = None):
    """
    Merge option chain JSON files with OHLCV CSV data by date and output merged JSON files.
    OHLCV data for each date is saved only once per file, and contracts reference their date.

    Args:
        option_chain_folder (str): Folder containing option chain JSON files.
        ohlcv_csv (str): Path to OHLCV CSV file.
        output_folder (str, optional): Folder to save merged JSON files. Defaults to option_chain_folder + '/merged'.
    """
    # Load OHLCV data
    ohlcv_df = pd.read_csv(ohlcv_csv)
    if 'date' in ohlcv_df.columns:
        ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date']).dt.strftime('%Y-%m-%d')
    else:
        raise ValueError('OHLCV CSV must have a "date" column.')
    ohlcv_dict = ohlcv_df.set_index('date').to_dict(orient='index')

    # Prepare output folder
    if output_folder is None:
        output_folder = os.path.join(option_chain_folder, 'merged')
    os.makedirs(output_folder, exist_ok=True)

    # Process each option chain file
    for filename in os.listdir(option_chain_folder):
        if filename.endswith('.json') and not filename.startswith('._'):
            with open(os.path.join(option_chain_folder, filename), 'r', encoding='utf-8') as f:
                option_data = json.load(f)
            contracts = option_data.get('data', [])
            # Collect all unique dates in this file
            dates_in_file = set(contract.get('date') for contract in contracts if contract.get('date') in ohlcv_dict)
            # Build ohlcv sub-dictionary for only the dates present in this file
            ohlcv_for_file = {date: ohlcv_dict[date] for date in dates_in_file}
            # Save merged data
            merged_filename = os.path.join(output_folder, f"merged_{filename}")
            with open(merged_filename, 'w', encoding='utf-8') as f:
                json.dump({'ohlcv': ohlcv_for_file, 'contracts': contracts}, f, ensure_ascii=False, indent=4)
            print(f"Merged file saved: {merged_filename}")

if __name__ == "__main__":
    # Example usage
    merge_option_chain_with_ohlcv(
        option_chain_folder="Option_Data/Test_data",
        ohlcv_csv="Option_Data/Training_data/SPY_OHLCV_data.csv"
    ) 