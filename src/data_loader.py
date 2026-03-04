import yfinance as yf
import pandas as pd

def load_data():
    data=yf.download("AAPL", start="2015-01-01", end="2024-01-01")
    data=data[['Close']]
    data.to_csv("data/stock_data.csv")
    return data