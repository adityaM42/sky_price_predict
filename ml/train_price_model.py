# ml/train_price_model.py
import os, json
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Config (adjust as you use in your classification script) ---
DATA_CSV = "../data/historical_prices_timeseries.csv"
OUT_DIR = "../ml"
LOOKAHEAD_MAX_DAYS = 90             # same as classification
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_LOG_TARGET = True               # set True to train on log(price) for stability

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "encoders"), exist_ok=True)

print("Loading", DATA_CSV)
df = pd.read_csv(DATA_CSV, parse_dates=["query_date","departure_date"])
group_cols = ["origin","destination","airline","departure_date"]
df = df.sort_values(group_cols + ["query_date"]).reset_index(drop=True)

rows = []
for key, group in df.groupby(group_cols):
    group = group.sort_values("query_date").reset_index(drop=True)
    for i, row in group.iterrows():
        qd = row["query_date"]
        dep = row["departure_date"]
        if qd >= dep:
            continue
        max_look = min(dep - pd.Timedelta(days=1), qd + pd.Timedelta(days=LOOKAHEAD_MAX_DAYS))
        future_mask = (group["query_date"] >= qd) & (group["query_date"] <= max_look)
        future_prices = group.loc[future_mask, "price"].values
        if len(future_prices) == 0:
            continue
        current_price = float(row["price"])
        min_price = float(future_prices.min())
        # optional: day_of_min etc.
        days_left = (dep - qd).days

        # history features (same as classification)
        hist = group.loc[group["query_date"] < qd, "price"]
        hist_min_7 = hist.tail(7).min() if len(hist.tail(7)) > 0 else np.nan
        hist_mean_14 = hist.tail(14).mean() if len(hist.tail(14)) > 0 else np.nan
        hist_std_30 = hist.tail(30).std() if len(hist.tail(30)) > 0 else np.nan
        price_momentum_7 = current_price - (hist.tail(7).mean() if len(hist.tail(7))>0 else current_price)

        rows.append({
            "origin": row["origin"],
            "destination": row["destination"],
            "airline": row["airline"],
            "stops": row.get("stops",""),
            "departure_time": row.get("departure_time",""),
            "class": row.get("class",""),
            "days_left": days_left,
            "price": current_price,
            "hist_min_7": hist_min_7,
            "hist_mean_14": hist_mean_14,
            "hist_std_30": hist_std_30,
            "price_momentum_7": price_momentum_7,
            # TARGET: min future price (within the lookahead window)
            "min_future_price": min_price
        })

label_df = pd.DataFrame(rows)
print("Prepared rows:", len(label_df))
label_df = label_df.dropna(subset=["min_future_price"]).reset_index(drop=True)

# Impute history columns
for col in ["hist_min_7","hist_mean_14","hist_std_30","price_momentum_7"]:
    label_df[col] = label_df[col].fillna(-1.0)

# Encode categorical columns (reuse encoders if you already made them)
cat_cols = ["origin","destination","airline","stops","departure_time","class"]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    label_df[c] = label_df[c].astype(str)
    label_df[c + "_enc"] = le.fit_transform(label_df[c])
    encoders[c] = le
    joblib.dump(le, os.path.join(OUT_DIR, "encoders", f"{c}_encoder.joblib"))
print("Saved encoders for:", cat_cols)

# Choose features (must match what your server will send to price_model.predict)
feature_cols = ["airline_enc","origin_enc","destination_enc","days_left","price",
                "hist_min_7","hist_mean_14","hist_std_30","price_momentum_7",
                "stops_enc","departure_time_enc","class_enc"]

# safety: if some encodings were skipped earlier, only keep columns present
available_feats = [c for c in feature_cols if c in label_df.columns]
X = label_df[available_feats].astype(float).fillna(-1.0)
y = label_df["min_future_price"].astype(float)

# Optionally transform target (log)
if USE_LOG_TARGET:
    # avoid log(0)
    y_trans = np.log1p(y)
else:
    y_trans = y.values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_trans, test_size=TEST_SIZE,
                                                    random_state=RANDOM_STATE)

# LightGBM regression
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "verbosity": -1,
    "seed": RANDOM_STATE
}
callback = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=50)
]
bst = lgb.train(params, lgb_train, valid_sets=[lgb_valid], num_boost_round=1000, callbacks=callback)

model_path = os.path.join(OUT_DIR, "price_model.pkl")
joblib.dump(bst, model_path)
print("Saved price model:", model_path)

# Save list of feature columns actually used
with open(os.path.join(OUT_DIR, "price_feature_cols.json"), "w") as f:
    json.dump({"feature_cols": available_feats, "use_log_target": USE_LOG_TARGET}, f, indent=2)

# Evaluate
y_pred = bst.predict(X_test)
if USE_LOG_TARGET:
    y_pred_orig = np.expm1(y_pred)
    y_test_orig = np.expm1(y_test)
else:
    y_pred_orig = y_pred
    y_test_orig = y_test

mae = mean_absolute_error(y_test_orig, y_pred_orig)
mse = mean_squared_error(y_test_orig, y_pred_orig)          # mean squared error
rmse = float(np.sqrt(mse))                                 # root mean squared error
r2 = r2_score(y_test_orig, y_pred_orig)
print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.4f}")
