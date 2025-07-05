#%%
import requests
from datetime import datetime, timedelta
import json
import pandas as pd
import os

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
start_date = datetime.strptime("2024-03-31", "%Y-%m-%d")
end_date = datetime.strptime("2024-4-10", "%Y-%m-%d")

#%%
# Track API key usage
api_key_index = 0
request_count = 0
max_requests_per_key = 25

for i in range((end_date - start_date).days + 1):
    # Check if we need to switch to next API key
    if request_count >= max_requests_per_key:
        api_key_index = (api_key_index + 1) % len(Api_key_list)
        request_count = 0
        print(f"Switching to API key {api_key_index + 1}")
    
    date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
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
    output_filename = f"Test_data/{Ticker}_options_{date.replace('-', '')}.json"
    os.makedirs("Test_data", exist_ok=True)
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(option_chain, f, ensure_ascii=False, indent=4)
        print(f"Option chain data for {Ticker} on {date} saved to {output_filename} (API key {api_key_index + 1}, request {request_count})")
    

#%%
api_key = 'c4509040f3424118929aba59e24c696b'
ticker_list = requests.get('https://api.twelvedata.com/stocks').json()
ticker_list = pd.DataFrame(ticker_list['data'])
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

# generate VWAP
# save the OHLCV data
OHLCV_data.to_csv(f"Option_Data/Test_data/{Ticker}_OHLCV_data.csv", index=False)

# %%

OHLCV_data

# %%
