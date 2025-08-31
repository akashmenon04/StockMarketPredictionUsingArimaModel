import pandas as pd
import os

os.makedirs("Data/AllStocksData", exist_ok=True)

directory_path = 'EOD'  # Replace with your directory path
headers = ['Date','Close','High','Low','Open','Volume']
for entry_name in os.listdir(directory_path):
    full_path = os.path.join(directory_path, entry_name)
    if os.path.isfile(full_path):
        save_path = 'Data/AllStocksData/' + entry_name
        df = pd.read_csv(full_path, skiprows=3, names=headers)
        df.to_csv(save_path, index=False)