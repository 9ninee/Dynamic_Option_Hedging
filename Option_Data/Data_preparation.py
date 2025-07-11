#%%
import requests
from datetime import datetime, timedelta
import json
import pandas as pd
import os
from typing import Optional

#%%
def get_option_chain_data(symbol, api_key, date):
    """
    Fetch option chain data for a given symbol from Alpha Vantage within a date range.

    Args:
        symbol (str): The stock symbol, e.g., 'AAPL'.
        api_key (str): Your Alpha Vantage API key.
        date (str): Date in 'YYYY-MM-DD' format.

    Returns:
        dict: Filtered option chain data within the date range.
    """
    # Build the full URL with all query parameters
    full_url = (
        f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS"
        f"&symbol={symbol}"
        f"&apikey={api_key}"
        f"&date={date}"
    )

    response = requests.get(full_url)
    data = response.json()

    return data

#%%
Api_key_list = [ "5XPODGOOJYWCJZ2V", "02JAVMIS5Z2AJSTL","HJQUS4XUZ3WZTA7B","YVN79CFRSBAO3U1R"]
api_key = Api_key_list[0]
Ticker = "SPY"
start_date = datetime.strptime("2017-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2025-07-10", "%Y-%m-%d")

#%%
# Track API key usage
api_key_index = 0
request_count = 0
max_requests_per_key = 25

for i in range((end_date - start_date).days + 1):
    date_obj = start_date + timedelta(days=i)
    # Skip weekends
    if date_obj.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        continue
    
    # Check if we need to switch API keys
    if request_count >= max_requests_per_key:
        print(f"\n=== API KEY SWITCH REQUIRED ===")
        print(f"API key {api_key_index + 1} has reached {max_requests_per_key} requests.")
        print(f"Please allocate the correct VPN for API key {api_key_index + 2}.")
        print("Press Enter when ready to continue...")
        input()  # Wait for user input
        
        # Switch to next API key
        api_key_index = (api_key_index + 1) % len(Api_key_list)
        request_count = 0
        print(f"Switched to API key {api_key_index + 1}: {Api_key_list[api_key_index]}")
        print("=" * 40)
    
    date = date_obj.strftime("%Y-%m-%d")
    current_api_key = Api_key_list[api_key_index]
    option_chain = get_option_chain_data(Ticker, current_api_key, date)
    
    # Increment request count
    request_count += 1

    # check if data present out of API request 
    
    # Check if data is present or if the response contains a "No data" message
    if (
        isinstance(option_chain, dict)
        and "data" in option_chain
        and isinstance(option_chain["data"], list)
        and len(option_chain["data"]) == 0
        and "No data for symbol" in option_chain.get("message", "")
    ):
        print(f"No data for {Ticker} on {date}. Skipping.")
        continue
    
    # Save the data if present
    output_filename = f"Training_data/{Ticker}_options_{date.replace('-', '')}.json"
    os.makedirs("Training_data", exist_ok=True)
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(option_chain, f, ensure_ascii=False, indent=4)
        print(f"Option chain data for {Ticker} on {date} saved to {output_filename} (API key {api_key_index + 1}, request {request_count}/{max_requests_per_key})")
    


#%%
api_key = 'c4509040f3424118929aba59e24c696b'
interval = '1day'

# Construct API URL with parameters for retrieving time series data
# dp=4: decimal places, outputsize=800: number of data points to retrieve
url = f"https://api.twelvedata.com/time_series?apikey={api_key}&interval={interval}&symbol={Ticker}&dp=4&outputsize=5000"

# Make API request and convert response to JSON
response = requests.get(url).json()

#%%
import talib
import numpy as np

# Extract relevant fields from JSON response and create list of daily values
# Each day contains datetime, open, high, low, close prices
values = [[day['datetime'], day['open'], day['high'], day['low'], day['close'], day['volume']] for day in response['values']]

# Create pandas DataFrame with extracted values and column names
df = pd.DataFrame(values, columns=['date','open', 'high', 'low', 'close', 'volume'])

# Convert date string to datetime object for proper date handling
df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d')

# Convert price columns from string to float for numerical operations
df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

# Filter data for specified date range and sort by date in ascending order
OHLCV_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)].sort_values(by="date").reset_index(drop=True)

# generate return data
OHLCV_data['return'] = OHLCV_data['close'].pct_change()

# generate volatility data
OHLCV_data['volatility'] = OHLCV_data['close'].pct_change().rolling(window=20).std()

# generate log return data
OHLCV_data['log_return'] = np.log(OHLCV_data['close'] / OHLCV_data['close'].shift(1))

# generate bollinger bands
OHLCV_data['upper_band'], OHLCV_data['middle_band'], OHLCV_data['lower_band'] = talib.BBANDS(OHLCV_data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

# generate RSI
OHLCV_data['RSI'] = talib.RSI(OHLCV_data['close'], timeperiod=14)

# generate MACD
OHLCV_data['MACD'], OHLCV_data['MACD_signal'], OHLCV_data['MACD_hist'] = talib.MACD(OHLCV_data['close'], fastperiod=12, slowperiod=26, signalperiod=9)

# generate ADX
OHLCV_data['ADX'] = talib.ADX(OHLCV_data['high'], OHLCV_data['low'], OHLCV_data['close'], timeperiod=14)

# generate ATR
OHLCV_data['ATR'] = talib.ATR(OHLCV_data['high'], OHLCV_data['low'], OHLCV_data['close'], timeperiod=14)

# generate OBV
OHLCV_data['OBV'] = talib.OBV(OHLCV_data['close'], OHLCV_data['volume'])

# generate EMA 14, 50, 200
OHLCV_data['EMA_14'] = talib.EMA(OHLCV_data['close'], timeperiod=14)
OHLCV_data['EMA_50'] = talib.EMA(OHLCV_data['close'], timeperiod=50)
OHLCV_data['EMA_200'] = talib.EMA(OHLCV_data['close'], timeperiod=200)

# generate SMA 5, 10, 20, 50, 100, 200
OHLCV_data['SMA_5'] = talib.SMA(OHLCV_data['close'], timeperiod=5)
OHLCV_data['SMA_10'] = talib.SMA(OHLCV_data['close'], timeperiod=10)
OHLCV_data['SMA_20'] = talib.SMA(OHLCV_data['close'], timeperiod=20)
OHLCV_data['SMA_50'] = talib.SMA(OHLCV_data['close'], timeperiod=50)
OHLCV_data['SMA_100'] = talib.SMA(OHLCV_data['close'], timeperiod=100)
OHLCV_data['SMA_200'] = talib.SMA(OHLCV_data['close'], timeperiod=200)

# generate WMA 10, 20, 50
OHLCV_data['WMA_10'] = talib.WMA(OHLCV_data['close'], timeperiod=10)
OHLCV_data['WMA_20'] = talib.WMA(OHLCV_data['close'], timeperiod=20)
OHLCV_data['WMA_50'] = talib.WMA(OHLCV_data['close'], timeperiod=50)

# generate DEMA 10, 20, 50
OHLCV_data['DEMA_10'] = talib.DEMA(OHLCV_data['close'], timeperiod=10)
OHLCV_data['DEMA_20'] = talib.DEMA(OHLCV_data['close'], timeperiod=20)
OHLCV_data['DEMA_50'] = talib.DEMA(OHLCV_data['close'], timeperiod=50)

# generate TEMA 10, 20, 50
OHLCV_data['TEMA_10'] = talib.TEMA(OHLCV_data['close'], timeperiod=10)
OHLCV_data['TEMA_20'] = talib.TEMA(OHLCV_data['close'], timeperiod=20)
OHLCV_data['TEMA_50'] = talib.TEMA(OHLCV_data['close'], timeperiod=50)

# generate MA cross signals
OHLCV_data['SMA_5_cross_SMA_20'] = np.where(OHLCV_data['SMA_5'] > OHLCV_data['SMA_20'], 1, 0)
OHLCV_data['SMA_10_cross_SMA_50'] = np.where(OHLCV_data['SMA_10'] > OHLCV_data['SMA_50'], 1, 0)
OHLCV_data['EMA_14_cross_EMA_50'] = np.where(OHLCV_data['EMA_14'] > OHLCV_data['EMA_50'], 1, 0)
OHLCV_data['EMA_50_cross_EMA_200'] = np.where(OHLCV_data['EMA_50'] > OHLCV_data['EMA_200'], 1, 0)
OHLCV_data['WMA_10_cross_WMA_50'] = np.where(OHLCV_data['WMA_10'] > OHLCV_data['WMA_50'], 1, 0)
OHLCV_data['DEMA_10_cross_DEMA_50'] = np.where(OHLCV_data['DEMA_10'] > OHLCV_data['DEMA_50'], 1, 0)
OHLCV_data['TEMA_10_cross_TEMA_50'] = np.where(OHLCV_data['TEMA_10'] > OHLCV_data['TEMA_50'], 1, 0)

# generate VWAP
# save the OHLCV data
OHLCV_data.to_csv(f"Test_data/{Ticker}_OHLCV_data.csv", index=False)

# %%

OHLCV_data

# %%

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


# Example usage
merge_option_chain_with_ohlcv(
    option_chain_folder="Training_data",
    ohlcv_csv="Training_data/SPY_OHLCV_data.csv"
) 
# %%
