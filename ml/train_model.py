# ml/train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# ---------- CONFIG ----------
# Path to the raw CSV (update if your CSV is elsewhere)
ROOT = os.path.dirname(os.path.dirname(__file__))   # project root (one level above ml/)
DATA_PATH = os.path.join(ROOT, "data", "modified_dataset.csv")  # change filename if needed

# Output folder for model/encoders/mappings
OUT_DIR = os.path.join(ROOT, "ml")
os.makedirs(OUT_DIR, exist_ok=True)
# ----------------------------

print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Columns to encode (adjust to match your CSV column names)
enc_cols = ["airline", "source_city", "departure_time", "arrival_time", "destination_city", "stops", "class"]

encoders = {}
for c in enc_cols:
    print(f"Encoding column: {c}")
    le = LabelEncoder()
    # convert NaN to string to avoid errors
    df[c] = le.fit_transform(df[c].astype(str))
    encoders[c] = le
    joblib.dump(le, os.path.join(OUT_DIR, f"{c}_encoder.joblib"))

# Save mapping dict for readability
mappings = {}
for c, le in encoders.items():
    # classes_ is ordered; create mapping index->label as strings for JSON safety
    mappings[c] = {str(i): name for i, name in enumerate(le.classes_)}

with open(os.path.join(OUT_DIR, "mappings.json"), "w", encoding="utf-8") as f:
    json.dump(mappings, f, indent=2, ensure_ascii=False)

print("Mappings saved to:", os.path.join(OUT_DIR, "mappings.json"))

# Prepare features/target (adjust if your CSV column names differ)
drop_cols = ["price", "price_bucket", "price_bin"]
X = df.drop(drop_cols, axis=1, errors='ignore')
y = df["price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training RandomForestRegressor ...")
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Save model
model_path = os.path.join(OUT_DIR, "model.joblib")
joblib.dump(rf, model_path)
print("Model saved to:", model_path)

# Evaluate
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", rmse)
print("R2:", r2_score(y_test, y_pred))

print("Done.")
