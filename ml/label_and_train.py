# ml/label_and_train.py
import os, json
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score

DATA_CSV = "../data/historical_prices_timeseries.csv"
OUT_DIR = "../ml"
LOOKAHEAD_MAX_DAYS = 90
MIN_DROP_PCT = 0.03
TEST_SIZE = 0.2
RANDOM_STATE = 42

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
        future_dates = group.loc[future_mask, "query_date"].values
        if len(future_prices) == 0:
            continue
        current_price = float(row["price"])
        min_price = float(future_prices.min())
        min_idx = future_prices.argmin()
        day_of_min = pd.to_datetime(future_dates[min_idx])
        days_to_min = (day_of_min - qd).days
        pct_drop = (current_price - min_price) / current_price if current_price>0 else 0.0

        if pct_drop < MIN_DROP_PCT:
            label = "buy_now"
        else:
            if days_to_min <= 7:
                label = "wait_1_7"
            elif days_to_min <= 21:
                label = "wait_8_21"
            else:
                label = "wait_22_plus"

        days_left = (dep - qd).days
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
            "label": label
        })

label_df = pd.DataFrame(rows)
print("Labeled rows:", len(label_df))
label_df = label_df.dropna(subset=["label"]).reset_index(drop=True)

# Impute
for col in ["hist_min_7","hist_mean_14","hist_std_30","price_momentum_7"]:
    label_df[col] = label_df[col].fillna(-1.0)

# Encode cats
cat_cols = ["origin","destination","airline","stops","departure_time","class"]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    label_df[c] = label_df[c].astype(str)
    label_df[c + "_enc"] = le.fit_transform(label_df[c])
    encoders[c] = le
    joblib.dump(le, os.path.join(OUT_DIR, "encoders", f"{c}_encoder.joblib"))

le_y = LabelEncoder()
label_df["y"] = le_y.fit_transform(label_df["label"])
joblib.dump(le_y, os.path.join(OUT_DIR, "booktime_label_encoder.joblib"))
print("Classes:", le_y.classes_)

feature_cols = ["airline_enc","origin_enc","destination_enc","days_left","price","hist_min_7","hist_mean_14","hist_std_30","price_momentum_7"]
for c in ["stops","departure_time","class"]:
    if c + "_enc" in label_df.columns:
        feature_cols.append(c + "_enc")

X = label_df[feature_cols].astype(float).fillna(-1)
y = label_df["y"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)
params = {"objective":"multiclass","num_class":len(le_y.classes_),"metric":"multi_logloss","learning_rate":0.05,"num_leaves":64,"verbosity":-1,"seed":RANDOM_STATE}
# bst = lgb.train(params, dtrain, valid_sets=[dvalid], num_boost_round=500, early_stopping_rounds=50, verbose_eval=50)
callback = [lgb.early_stopping(stopping_rounds=50, verbose=50)]
bst = lgb.train(params, dtrain, valid_sets=[dvalid], num_boost_round=500, callbacks=callback)

model_path = os.path.join(OUT_DIR, "booktime_model.pkl")
joblib.dump(bst, model_path)
print("Saved model:", model_path)
with open(os.path.join(OUT_DIR, "booktime_label_map.json"), "w") as f:
    json.dump({"classes": list(le_y.classes_)}, f)

y_pred = np.argmax(bst.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=list(le_y.classes_)))
print("Accuracy:", accuracy_score(y_test, y_pred))
