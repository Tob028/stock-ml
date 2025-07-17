from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.data import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame

import os

ENV = os.getenv("ENV", "development")

if ENV == "development":
    from dotenv import load_dotenv
    load_dotenv()

# Init alpaca client

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
account = trading_client.get_account()

# Check traded symblol

TRADED_SYMBOL = "AAPL"

asset = trading_client.get_asset(TRADED_SYMBOL)

if not asset.tradable:
    raise ValueError(f"{TRADED_SYMBOL} isn't tradable.")


# Get historical data

stock_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Model