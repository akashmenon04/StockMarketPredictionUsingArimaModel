import yfinance as yf
import pandas as pd
import os
# Create directory to store EOD files
os.makedirs("EOD", exist_ok=True)
# Load NSE equity symbols
equity_details = pd.read_csv("./Data/EQUITY_L.csv")
# Loop through each symbol
for name in equity_details.SYMBOL:
    try:
        print(f"Fetching {name}...")
        data = yf.download(f"{name}.NS", period="4y")
        if not data.empty:
            data.to_csv(f"./EOD/{name}.csv")
        else:
            print(f"No data for {name}")
    except Exception as e:
        print(f"{name} ===> {e}")