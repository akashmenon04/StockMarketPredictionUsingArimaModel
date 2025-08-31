import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------
# 1) Load your CSV
# --------------------------
# Example file structure: Symbol,Date,Close,High,Low,Open,Volume
df = pd.read_csv('./Data/data.csv')

# df = pd.read_csv("stocks.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Symbol','Date']).reset_index(drop=True)

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

# feat_df = df.groupby('Symbol', group_keys=False).apply(make_features)
feat_df = (
    df.drop(columns=['Symbol'])                 # <— drop before groupby
      .groupby(df['Symbol'], group_keys=False)  # use df['Symbol'] as grouper
      .apply(make_features)
      .assign(Symbol=df['Symbol'])
      .reset_index(drop=True)
)

feat_df = feat_df.dropna().reset_index(drop=True)

# --------------------------
# 3) Encode stock IDs
# --------------------------
le = LabelEncoder()
feat_df['symbol_id'] = le.fit_transform(feat_df['Symbol'])
# --------------------------
# 4) Train/validation/test split
# --------------------------
# Example: train <2022, valid=2022, test=2023+

cut1 = pd.Timestamp('2023-01-01')
cut2 = pd.Timestamp('2024-01-01')

train = feat_df[feat_df['Date'] < cut1]
valid = feat_df[(feat_df['Date'] >= cut1) & (feat_df['Date'] < cut2)]
test  = feat_df[feat_df['Date'] >= cut2]

feature_cols = [c for c in feat_df.columns if c not in 
                ['Date','Symbol','log_close','target','bb_mid','bb_up','bb_dn']]

X_train, y_train = train[feature_cols], train['target']
X_valid, y_valid = valid[feature_cols], valid['target']
X_test,  y_test  = test[feature_cols],  test['target']

# --------------------------
# 5) Train LightGBM
# --------------------------
cat_cols = ['symbol_id','dow','month']
cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_idx)
lgb_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_idx)

params = dict(
    objective='regression',
    metric='mae',
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_in_leaf=100,
)

callbacks = [
    lgb.early_stopping(stopping_rounds=200),
    lgb.log_evaluation(period=200)
]

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    num_boost_round=5000,
    callbacks=callbacks
)

# --------------------------
# 6) Evaluate
# --------------------------
def evaluate(model, X, y, name):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    dir_acc = np.mean(np.sign(preds) == np.sign(y))
    print(f"{name}: MAE={mae:.6f}, RMSE={rmse:.6f}, DirAcc={dir_acc:.3f}")

evaluate(model, X_valid, y_valid, "VALID")
evaluate(model, X_test, y_test, "TEST")

# import joblib
# Assuming 'model' is your trained LightGBM model
# joblib.dump(model, 'lightgbm_model.pkl')
model.save_model("lightbgm_model.txt")

# --------------------------
# 7) Predict next-day returns
# --------------------------
latest = feat_df.groupby('Symbol').tail(2)   # last row per stock
X_latest = latest[feature_cols]
preds_latest = model.predict(X_latest)

latest['pred_next_ret'] = preds_latest
print(latest[['Symbol','Date','Close','pred_next_ret']].head())
