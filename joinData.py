import pandas as pd
import os

final_dataset_path = "Data/data.csv"

directory_path = "Data/AllStocksData/"

headers = ['Date','Close','High','Low','Open','Volume']

df_master = []
# Train Dataset

for entry_name in os.listdir(directory_path):
    full_path = os.path.join(directory_path, entry_name)
    if os.path.isfile(full_path):
        file_path = directory_path + entry_name
        df = pd.read_csv(file_path)
        print("data count: ", df.count())
        print(entry_name.split('.')[0])
        df.insert(loc=0, column='Symbol', value=entry_name.split('.')[0])
        df_master.append(df)

final_df = pd.concat(df_master, ignore_index=True)

final_df.to_csv(final_dataset_path, index=False)
