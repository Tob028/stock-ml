import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import talib as ta
import finnhub as fh
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


TICKER = "AAPL"

ENV = os.getenv("ENV", "development")

if ENV == "development":
    from dotenv import load_dotenv
    load_dotenv()

# Init alpaca client

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
account = trading_client.get_account()

# Check traded symbol

asset = trading_client.get_asset(TICKER)

if not asset.tradable:
    raise ValueError(f"{TICKER} isn't tradable.")


# Build dataset

today = datetime.now().date()
year_ago = today.replace(year=today.year - 1)

stock_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
request_params = StockBarsRequest(
    symbol_or_symbols=TICKER,
    timeframe=TimeFrame.Day,
    start=year_ago,
    end=today
)

bars = stock_client.get_stock_bars(request_params)
df = bars.df

df.reset_index(inplace=True)
df.drop(columns=['symbol'], inplace=True)

def process_dataset(df):
    df["close_lag_1"] = df["close"].shift(1)
    df["close_lag_2"] = df["close"].shift(2)

    df["open_lag_1"] = df["open"].shift(1)
    df["open_lag_2"] = df["open"].shift(2)

    df["high_lag_1"] = df["high"].shift(1)
    df["high_lag_2"] = df["high"].shift(2)

    df["low_lag_1"] = df["low"].shift(1)
    df["low_lag_2"] = df["low"].shift(2)

    df["volume_lag_1"] = df["volume"].shift(1)
    df["volume_lag_2"] = df["volume"].shift(2)

    df["RSI_14"] = ta.RSI(df["close"], timeperiod=14)
    df["SMA_20"] = ta.SMA(df["close"], timeperiod=20)
    df["SMA_50"] = ta.SMA(df["close"], timeperiod=50)
    df["Bollinger_High"] = ta.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2)[0]
    df["Bollinger_Low"] = ta.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2)[2]
    df["MACD"] = ta.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)[0]
    df["RSI_14"] = ta.RSI(df["close"], timeperiod=14)

    fh_client = fh.Client(api_key=os.getenv("FINNHUB_API_KEY"))
    earnings = fh_client.company_earnings(symbol=TICKER)

    for earning in earnings:
        quarter = earning['quarter']
        year = earning['year']

        df["earnings_estimate"] = np.where(
            (df["timestamp"].dt.year == year) & (df["timestamp"].dt.quarter == quarter),
            earning['estimate'],
            np.nan
        )

        df["earnings_actual"] = np.where(
            (df["timestamp"].dt.year == year) & (df["timestamp"].dt.quarter == quarter),
            earning['actual'],
            np.nan
        )

        df["earnings_surprise"] = np.where(
            (df["timestamp"].dt.year == year) & (df["timestamp"].dt.quarter == quarter),
            earning['surprise'],
            np.nan
        )

    df["is_up"] = (df["close"] > df["open"]).astype(int)
    # df.dropna(inplace=True)
    return df

df = process_dataset(df)

X_train = df.drop(columns=["timestamp", "is_up"])
y_train = df["is_up"]

# Build model

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)

# Get todays data
snapshot_request = StockSnapshotRequest(
    symbol_or_symbols=TICKER,
    timeframe=TimeFrame.Day,
)
snapshot = stock_client.get_stock_snapshot(snapshot_request)
today_bar = snapshot[TICKER].daily_bar

today_data = pd.DataFrame({
    "open": [today_bar.open],
    "high": [today_bar.high],
    "low": [today_bar.low],
    "close": [today_bar.close],
    "volume": [today_bar.volume],
})

latest_df = pd.concat([df, today_data], ignore_index=True)
latest_df = process_dataset(latest_df)

X_today = latest_df.drop(columns=["timestamp", "is_up"]).tail(1)

y_pred = xgb.predict(X_today)

# Make a trade

if y_pred[0] == 1:
    print("going up, buying")

if y_pred[0] == 0:
    print("going down, selling")
