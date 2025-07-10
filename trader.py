import os
from alpaca.trading.client import TradingClient

ENV = os.getenv("ENV", "development")

if ENV == "development":
    from dotenv import load_dotenv
    load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

client = TradingClient(API_KEY, SECRET_KEY, paper=True)
account = client.get_account()

print(f"Account ID: {account.id}")