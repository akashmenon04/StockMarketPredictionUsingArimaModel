import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --------------------------
# 2) Feature engineering per stock
# --------------------------
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = ema(up, n) / (ema(down, n) + 1e-12)
    return 100 - (100 / (1 + rs))

def make_features(g):
    g = g.copy()
    g['log_close'] = np.log(g['Close'])
    g['ret_1'] = g['log_close'].diff()  # daily log return

    # Lags of returns
    for k in [1,2,3,5,10]:
        g[f'ret_lag{k}'] = g['ret_1'].shift(k)

    # Moving averages
    for w in [5,10,20]:
        g[f'ema_{w}'] = ema(g['Close'], w)
        g[f'ret_mean_{w}'] = g['ret_1'].rolling(w).mean()
        g[f'ret_std_{w}']  = g['ret_1'].rolling(w).std()

    # RSI
    g['rsi_14'] = rsi(g['Close'], 14)

    # Bollinger band %B
    g['bb_mid'] = g['Close'].rolling(20).mean()
    g['bb_std'] = g['Close'].rolling(20).std()
    g['bb_up']  = g['bb_mid'] + 2*g['bb_std']
    g['bb_dn']  = g['bb_mid'] - 2*g['bb_std']
    g['bb_percB'] = (g['Close'] - g['bb_dn']) / (g['bb_up'] - g['bb_dn'] + 1e-12)

    # Volume features
    g['vol_chg'] = g['Volume'].pct_change()
    g['vol_z20'] = (g['Volume'] - g['Volume'].rolling(20).mean()) / (g['Volume'].rolling(20).std() + 1e-12)

    # Calendar features
    g['dow'] = g['Date'].dt.dayofweek
    g['month'] = g['Date'].dt.month

    # Target = next-day log return
    g['target'] = g['log_close'].shift(-1) - g['log_close']

    return g

# Load new stock CSV
new_df = pd.read_csv(".\Data\predictData.csv")
new_df['Date'] = pd.to_datetime(new_df['Date'])
new_df = new_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

# Apply feature engineering
new_feat = (
    new_df.drop(columns=['Symbol'])                 # <— drop before groupby
      .groupby(new_df['Symbol'], group_keys=False)  # use df['Symbol'] as grouper
      .apply(make_features)
      .assign(Symbol=new_df['Symbol'])
      .reset_index(drop=True)
)
# print(new_feat)

le = LabelEncoder()
# Encode symbol with previously fitted LabelEncoder
new_feat['symbol_id'] = le.fit_transform(new_feat['Symbol'])

# Select features (same order as training!)

feature_cols = [c for c in new_feat.columns if c not in 
                ['Date','Symbol','log_close','target','bb_mid','bb_up','bb_dn']]

X_new = new_feat[feature_cols]

model__path = "lightbgm_model.txt"

loaded_model = lgb.Booster(model_file=model__path)

# Predict
preds = loaded_model.predict(X_new)
new_feat['pred_next_ret'] = preds
print(new_feat[['Symbol','Date','Close','pred_next_ret']].head())

#       Symbol       Date       Close  pred_next_ret
# 0  20MICRONS 2025-08-28  230.570007       0.001002

# Actual Data for 2025-08-29
# Symbol,Date,Close,High,Low,Open,Volume
# 20MICRONS,2025-08-29,230.8699951171875,234.6000061035156,228.0200042724609,230.75,69878

# 6.89% Difference in the Predicted Next Day Return value and the actual close value on next 