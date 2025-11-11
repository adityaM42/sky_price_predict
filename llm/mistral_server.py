"""
SkyPredict - Mistral / Ollama + ML server

Endpoints:
 - GET  /health
 - POST /parse        -> parse free text into structured JSON (uses Ollama)
 - POST /chat_predict -> end-to-end: parse -> featurize -> classify & price -> recommendation
 - POST /predict      -> direct numeric prediction for price model (if available)
"""

import os
import json
import logging
import traceback
from typing import Any, Dict, Optional

import requests
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# -------------------------
# Logging & paths
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mistral_server")

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # one level above llm/
ML_DIR = os.path.join(PROJECT_ROOT, "ml")
ENCODERS_DIR = os.path.join(ML_DIR, "encoders")
HIST_CSV = os.path.join(PROJECT_ROOT, "data", "historical_prices_timeseries.csv")

# Ollama config (local)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"
OLLAMA_TIMEOUT = 30  # seconds

# Model artifact paths
BOOK_MODEL_PATH = os.path.join(ML_DIR, "booktime_model.pkl")
BOOK_LABEL_ENCODER_PATH = os.path.join(ML_DIR, "booktime_label_encoder.joblib")
PRICE_MODEL_PATH = os.path.join(ML_DIR, "price_model.pkl")
PRICE_META_JSON = os.path.join(ML_DIR, "price_feature_cols.json")

# Feature column order / fallback (if price meta missing)
FEATURE_COLS = [
    "airline_enc",
    "origin_enc",
    "destination_enc",
    "days_left",
    "price",
    "hist_min_7",
    "hist_mean_14",
    "hist_std_30",
    "price_momentum_7",
    "stops_enc",
    "departure_time_enc",
    "class_enc",
]

# -------------------------
# FastAPI app and schemas
# -------------------------
app = FastAPI(title="SkyPredict - Ollama + Models")

class QueryText(BaseModel):
    text: str

# class PredictFeatures(BaseModel):
#     # free-form dict expected; Pydantic will coerce - we'll accept dicts via request body
#     __root__: Dict[str, Any]

# -------------------------
# Helper: call Ollama
# -------------------------
def call_ollama(prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
    """
    Call Ollama /api/generate and try to extract textual completion.
    Returns either the textual response or a string starting with "ERROR:" on failure.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        logger.exception("Ollama request failed")
        return f"ERROR: {e}"

    # Try to decode JSON response
    try:
        data = r.json()
    except Exception:
        # If JSON parse failed, return raw text
        return r.text

    # Common shapes returned by Ollama: { "response": str } or { "results": [ ... ] } etc.
    if isinstance(data, dict):
        # preferred simple field
        if "response" in data and isinstance(data["response"], str):
            return data["response"]
        if "completion" in data and isinstance(data["completion"], str):
            return data["completion"]

        # nested results content
        if "results" in data and isinstance(data["results"], list) and data["results"]:
            r0 = data["results"][0]
            # try to find textual fields
            if isinstance(r0, dict):
                # Often content is list of dicts
                content = r0.get("content")
                if isinstance(content, list) and content:
                    c0 = content[0]
                    if isinstance(c0, dict):
                        for k in ("text", "content", "output_text", "completion"):
                            if k in c0 and isinstance(c0[k], str):
                                return c0[k]
                for k in ("text", "completion", "content"):
                    if k in r0 and isinstance(r0[k], str):
                        return r0[k]

        # choices style (openai-like)
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            ch0 = data["choices"][0]
            if isinstance(ch0, dict) and "text" in ch0:
                return ch0["text"]

    # fallback: return compacted JSON string
    return json.dumps(data)

# -------------------------
# JSON extractor (balanced braces)
# -------------------------
def extract_first_json(text: str) -> Any:
    """
    Extract the first complete JSON object from a string.
    Raises ValueError if no complete JSON object found.
    """
    if not isinstance(text, str):
        raise ValueError("Input is not a string")

    s = text.strip()
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON start found")

    level = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        c = s[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                level += 1
            elif c == "}":
                level -= 1
                if level == 0:
                    candidate = s[start:i+1]
                    return json.loads(candidate)
    raise ValueError("No complete JSON object found")

# -------------------------
# Load models, encoders, history CSV
# -------------------------
# booking model + label encoder
book_model = None
label_le = None
if os.path.exists(BOOK_MODEL_PATH):
    try:
        book_model = joblib.load(BOOK_MODEL_PATH)
        logger.info("Loaded booking model from %s", BOOK_MODEL_PATH)
    except Exception:
        logger.exception("Failed to load booking model")

if os.path.exists(BOOK_LABEL_ENCODER_PATH):
    try:
        label_le = joblib.load(BOOK_LABEL_ENCODER_PATH)
        logger.info("Loaded booking label encoder")
    except Exception:
        logger.exception("Failed to load booking label encoder")

# price model + meta
price_model = None
price_model_cols = []
price_use_log_target = False
if os.path.exists(PRICE_MODEL_PATH):
    try:
        price_model = joblib.load(PRICE_MODEL_PATH)
        logger.info("Loaded price model from %s", PRICE_MODEL_PATH)
    except Exception:
        logger.exception("Failed to load price model")

if os.path.exists(PRICE_META_JSON):
    try:
        with open(PRICE_META_JSON, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
            price_model_cols = meta.get("feature_cols", [])
            price_use_log_target = meta.get("use_log_target", False)
            logger.info("Price meta loaded: cols=%s, use_log_target=%s", price_model_cols, price_use_log_target)
    except Exception:
        logger.exception("Failed to load price_feature_cols.json")

# categorical encoders
encoders = {}
if os.path.exists(ENCODERS_DIR):
    for fname in os.listdir(ENCODERS_DIR):
        if fname.endswith(".joblib"):
            key = fname.replace("_encoder.joblib", "")
            try:
                encoders[key] = joblib.load(os.path.join(ENCODERS_DIR, fname))
                logger.info("Loaded encoder: %s", key)
            except Exception:
                logger.exception("Failed to load encoder: %s", fname)

# load history CSV (if available)
_hist_df = None
if os.path.exists(HIST_CSV):
    try:
        _hist_df = pd.read_csv(HIST_CSV, parse_dates=["query_date", "departure_date"])
        # keep sorted for fast slicing
        _hist_df = _hist_df.sort_values(["departure_date", "query_date"]).reset_index(drop=True)
        logger.info("Loaded history CSV rows: %d", len(_hist_df))
    except Exception:
        logger.exception("Failed to load history CSV")
else:
    logger.warning("History CSV not found at %s", HIST_CSV)

# -------------------------
# Encoding helpers & featurization
# -------------------------
def safe_transform_encoder(enc, val) -> int:
    """
    Encode a textual/categorical value using a LabelEncoder loaded earlier.
    Return 0 on failure (safe fallback).
    """
    if enc is None:
        return 0
    try:
        return int(enc.transform([str(val)])[0])
    except Exception:
        # fallback: try to match class names lowercased
        try:
            classes = [str(x).lower() for x in enc.classes_]
            key = str(val).strip().lower()
            if key in classes:
                return int(classes.index(key))
            # try variants
            key2 = key.replace(" ", "_").replace("-", "_")
            if key2 in classes:
                return int(classes.index(key2))
            # fuzzy-inclusion
            for i, name in enumerate(classes):
                if key in name or name in key:
                    return int(i)
        except Exception:
            pass
    return 0

def map_time_str(s: str) -> int:
    """
    Map time-of-day string to numeric encoding if encoder missing.
    Uses a fallback map: Afternoon=0, Early_Morning=1, Evening=2, Late_Night=3, Morning=4, Night=5
    """
    lookup = {"afternoon":0,"early_morning":1,"evening":2,"late_night":3,"morning":4,"night":5}
    if s is None:
        return lookup["morning"]
    key = str(s).strip().lower().replace(" ", "_")
    return lookup.get(key, lookup["morning"])

def map_stops_str(s: str) -> int:
    """
    Map stops string to numeric fallback: one -> 0, two_or_more -> 1, zero -> 2
    """
    if s is None:
        return 2
    key = str(s).lower()
    if "one" in key or key == "1":
        return 0
    if "two" in key or "2" in key or "more" in key:
        return 1
    if "zero" in key or "non" in key or "0" in key:
        return 2
    # try integer
    try:
        return int(key)
    except Exception:
        return 2

def compute_hist_features_from_csv(query_date, departure_date):
    """
    Compute hist_min_7, hist_mean_14, hist_std_30, price_momentum_7 for the given pair
    If CSV not available, return default -1s and 0 momentum.
    """
    if _hist_df is None:
        return -1.0, -1.0, -1.0, 0.0

    try:
        qd = pd.to_datetime(query_date)
        dd = pd.to_datetime(departure_date)
    except Exception:
        return -1.0, -1.0, -1.0, 0.0

    # filter rows for same departure_date
    part = _hist_df[_hist_df["departure_date"] == dd]
    if part.empty:
        return -1.0, -1.0, -1.0, 0.0

    prior = part[part["query_date"] < qd].sort_values("query_date")
    if prior.empty:
        return -1.0, -1.0, -1.0, 0.0

    series = prior["price"]
    hist_min_7 = float(series.tail(7).min()) if len(series.tail(7))>0 else -1.0
    hist_mean_14 = float(series.tail(14).mean()) if len(series.tail(14))>0 else -1.0
    hist_std_30 = float(series.tail(30).std()) if len(series.tail(30))>0 else -1.0
    hist_momentum = float(series.diff().tail(7).mean()) if len(series.tail(7))>0 else 0.0

    # sanitize
    hist_min_7 = hist_min_7 if np.isfinite(hist_min_7) else -1.0
    hist_mean_14 = hist_mean_14 if np.isfinite(hist_mean_14) else -1.0
    hist_std_30 = hist_std_30 if np.isfinite(hist_std_30) else -1.0

    return hist_min_7, hist_mean_14, hist_std_30, hist_momentum

def featurize_from_parsed(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn parsed fields (from LLM) into numeric features expected by models.
    Returns a dict with keys matching FEATURE_COLS (or price_model_cols).
    """
    # textual values
    airline_val = parsed.get("airline", parsed.get("airline_name", ""))
    origin_val = parsed.get("source_city", parsed.get("origin", ""))
    dest_val = parsed.get("destination_city", parsed.get("destination", ""))
    dep_time_val = parsed.get("departure_time", "")
    stops_val = parsed.get("stops", "")
    class_val = parsed.get("class", parsed.get("flight_class", ""))

    # days_left and dates
    days_left = None
    if parsed.get("days_left") is not None:
        try:
            days_left = int(parsed.get("days_left"))
        except Exception:
            days_left = None

    # If departure_date & query_date provided, compute dates; else try to infer
    query_date = parsed.get("query_date")
    departure_date = parsed.get("departure_date")
    if days_left is None:
        # try to infer from dates if present
        if query_date and departure_date:
            try:
                days_left = int((pd.to_datetime(departure_date) - pd.to_datetime(query_date)).days)
            except Exception:
                days_left = 1
        else:
            days_left = int(parsed.get("days_left", 1))

    # current price if provided
    price_val = parsed.get("price", parsed.get("current_price", None))
    try:
        price_val = float(price_val) if price_val is not None else 0.0
    except Exception:
        price_val = 0.0

    # encode using encoders if available
    airline_enc = safe_transform_encoder(encoders.get("airline"), airline_val)
    origin_enc = safe_transform_encoder(encoders.get("origin"), origin_val)
    dest_enc = safe_transform_encoder(encoders.get("destination"), dest_val)
    stops_enc = safe_transform_encoder(encoders.get("stops"), stops_val)
    dep_time_enc = safe_transform_encoder(encoders.get("departure_time"), dep_time_val)
    class_enc = safe_transform_encoder(encoders.get("class"), class_val)

    # if encoder didn't work, fallback mapping
    if dep_time_enc == 0 and dep_time_val:
        dep_time_enc = map_time_str(dep_time_val)
    if stops_enc == 0 and stops_val:
        stops_enc = map_stops_str(stops_val)

    # compute historical features (prefer explicit dates; if not, attempt using days_left)
    if query_date and departure_date:
        hist_min_7, hist_mean_14, hist_std_30, hist_momentum = compute_hist_features_from_csv(query_date, departure_date)
    else:
        # if only days_left provided, approximate by taking latest departure_date in CSV with same days_left
        hist_min_7, hist_mean_14, hist_std_30, hist_momentum = -1.0, -1.0, -1.0, 0.0
        if _hist_df is not None and isinstance(days_left, int):
            # find recent entries with same days_left and use their statistics
            try:
                # compute days_left column on csv if not present
                tmp = _hist_df.copy()
                tmp["days_left_calc"] = (tmp["departure_date"] - tmp["query_date"]).dt.days
                recent = tmp[tmp["days_left_calc"] == days_left]
                if not recent.empty:
                    s = recent["price"]
                    hist_min_7 = float(s.tail(7).min()) if len(s.tail(7))>0 else -1.0
                    hist_mean_14 = float(s.tail(14).mean()) if len(s.tail(14))>0 else -1.0
                    hist_std_30 = float(s.tail(30).std()) if len(s.tail(30))>0 else -1.0
                    hist_momentum = float(s.diff().tail(7).mean()) if len(s.tail(7))>0 else 0.0
            except Exception:
                hist_min_7, hist_mean_14, hist_std_30, hist_momentum = -1.0, -1.0, -1.0, 0.0

    features = {
        "airline_enc": int(airline_enc),
        "origin_enc": int(origin_enc),
        "destination_enc": int(dest_enc),
        "days_left": int(days_left),
        "price": float(price_val),
        "hist_min_7": float(hist_min_7),
        "hist_mean_14": float(hist_mean_14),
        "hist_std_30": float(hist_std_30),
        "price_momentum_7": float(hist_momentum),
        "stops_enc": int(stops_enc),
        "departure_time_enc": int(dep_time_enc),
        "class_enc": int(class_enc),
    }
    return features

# -------------------------
# Price prediction helper
# -------------------------
def predict_price_from_features(features: Dict[str, Any]) -> Optional[float]:
    """
    Use price_model and price_model_cols to predict the expected minimum/future price.
    Returns float price or None on failure / if price_model missing.
    """
    if price_model is None:
        return None

    # build DataFrame with required columns in order
    if price_model_cols:
        cols = price_model_cols
    else:
        cols = FEATURE_COLS

    X = pd.DataFrame([features])
    # ensure missing columns filled
    for c in cols:
        if c not in X.columns:
            X[c] = -1.0
    X = X[cols].astype(float).fillna(-1.0)

    try:
        raw = price_model.predict(X)[0]
        if price_use_log_target:
            pred = float(np.expm1(raw))
        else:
            pred = float(raw)
        pred = max(0.0, float(pred))
        return pred
    except Exception:
        logger.exception("price_model.predict failed")
        return None

# -------------------------
# Recommendation builder
# -------------------------
def build_recommendation_text(current_price: float, predicted_price: float, best_bucket: str, probs_vec: np.ndarray) -> str:
    """
    Build a human-friendly recommendation sentence using predicted price and classifier info.
    """
    if current_price is None or current_price <= 0:
        # If current price unknown, be conservative
        if predicted_price is None:
            return "Insufficient data to provide a recommendation."
        else:
            return f"Estimated best price: ₹{round(predicted_price):,}."

    pct_change = 0.0
    if predicted_price is not None:
        pct_change = (predicted_price - current_price) / current_price
    pct_display = round(abs(pct_change) * 100, 2)
    conf = float(np.max(probs_vec)) if probs_vec is not None else 0.0

    bucket_window_text = {
        "buy_now": "right now",
        "wait_1_7": "over the next 1–7 days",
        "wait_8_21": "over the next 8–21 days",
        "wait_22_plus": "over the next 22+ days"
    }
    window_text = bucket_window_text.get(best_bucket, "in the near future")

    # Compose message
    if predicted_price is None:
        # fallback to classifier suggestion only
        if best_bucket == "buy_now" and conf > 0.7:
            base = f"Model suggests booking {window_text}."
        else:
            base = f"Model suggests {best_bucket.replace('_',' ')} (low confidence: {conf:.2f})."
        return base

    # If predicted to fall meaningfully
    if pct_change < -0.03:
        return f"Prices are likely to fall by about {pct_display}% {window_text}; waiting a few days may be beneficial. (estimated best price: ₹{round(predicted_price):,})."
    elif pct_change > 0.03:
        return f"Prices are predicted to rise by about {pct_display}% {window_text}; consider booking now to avoid higher fares. (estimated price: ₹{round(predicted_price):,})."
    else:
        # small/no change
        if best_bucket == "buy_now" and conf > 0.6:
            return f"Model recommends booking {window_text}; price change expected to be small. (estimated best price: ₹{round(predicted_price):,})."
        return f"No significant change expected ({pct_display}%); monitor fares and buy if you see a good deal. (estimated best price: ₹{round(predicted_price):,})."

# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "booking_model_loaded": book_model is not None,
        "price_model_loaded": price_model is not None,
        "ollama_endpoint": OLLAMA_URL,
        "encoders_loaded": list(encoders.keys()),
        "history_rows": int(len(_hist_df)) if _hist_df is not None else 0
    }

import re

def clean_json_snippet(snippet: str) -> str:
    """
    Attempt to fix common LLM JSON mistakes:
      - Replace single quotes with double quotes (careful, only if safe)
      - Replace common unquoted tokens (Not_available, Not available, N/A, None) -> null
      - Remove trailing explanatory sentences after the JSON object
      - Remove JavaScript-style comments (// ...), /* ... */
      - Trim excessive commas like {...,}
    This is heuristic — always inspect raw_snippet for failure cases.
    """
    s = snippet.strip()

    # If there are newline-separated blocks, take the first block that looks like JSON or starts with {
    # keep original for debugging in server logs
    # attempt to find the first '{' ... '}' balanced substring
    try:
        candidate = extract_first_json(s)
        # if extract_first_json succeeded -> we actually get Python object, return dump to standardized JSON
        return json.dumps(candidate)
    except Exception:
        pass

    # Remove JS/C-style comments
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)

    # Replace common unquoted tokens with null
    s = re.sub(r"\b(Not_available|Not available|NotAvailable|N/A|None|NA)\b", "null", s, flags=re.IGNORECASE)

    # If single quotes used for keys/strings, convert to double quotes carefully
    # Only change when it looks like JSON-like (has { and } and quotes)
    if ("{" in s or "}" in s) and ("'" in s) and ('"' not in s[:min(len(s), 200)]):
        # naive single -> double quotes, but avoid touching apostrophes inside words by simple patterns
        s = re.sub(r"\'([^\']*?)\'", r'"\1"', s)

    # Remove trailing explanatory text after the closing brace of the first JSON object
    m = re.search(r"(\{.*\})", s, flags=re.S)
    if m:
        s = m.group(1)

    # Remove dangling commas before closing braces/brackets { ... , } -> { ... }
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # ensure keys are quoted (if something like {airline: "Air India"} -> "airline": ...)
    # simple heuristic: add quotes to bare keys (only safe for simple cases)
    def quote_keys(text):
        return re.sub(r'([{\s,])([A-Za-z0-9_]+)\s*:', r'\1"\2":', text)
    s = quote_keys(s)

    return s

def regex_field_extract(raw: str) -> dict:
    """
    As a last resort, find key:value pairs in the string using regex.
    Returns a small parsed dict (strings or numbers or nulls).
    Used only when JSON parsing fails.
    """
    fields = {}
    # basic patterns for expected keys
    keys = ["airline","source_city","departure_time","stops","arrival_time",
            "destination_city","class","duration","days_left","query_date","departure_date","price"]
    for k in keys:
        # look for "k": "value" or k: "value" or "k": value
        # capture values until comma or closing brace or newline
        # pattern = rf'["\']?{k}["\']?\s*[:=]\s*(?:(["\'])(.*?)\1|([^,\}\n]+))'
        pattern = rf'["\']?{k}["\']?\s*[:=]\s*(?:(["\'])(.*?)\1|([^,\}}\n]+))'
        m = re.search(pattern, raw, flags=re.IGNORECASE | re.S)
        if m:
            if m.group(2) is not None:
                val = m.group(2).strip()
            else:
                val = m.group(3).strip()
            # normalize common terms
            if re.match(r'^(null|none|na|n/a|not_available|not available)$', val, flags=re.I):
                fields[k] = None
            else:
                # try number
                try:
                    if "." in val:
                        fields[k] = float(val)
                    else:
                        fields[k] = int(val)
                except:
                    fields[k] = val.strip('"').strip("'").strip()
    return fields

@app.post("/parse")
def parse_text(q: QueryText):
    """
    Robust parse endpoint:
     - uses a strict prompt asking for pure JSON only (null for unknown)
     - tries extract_first_json()
     - falls back to cleaning heuristics + json.loads
     - final fallback: regex_field_extract
    """
    prompt = f"""
You are a flight booking assistant. Extract ONLY the following fields and return EXACTLY a single JSON object (no commentary, no explanation, nothing else).
Use these keys and these types. If a value is unknown, use null (JSON null).

Keys (exact names):
airline (string),
source_city (string),
departure_time (string or null),
stops (string or null),
arrival_time (string or null),
destination_city (string),
class (string or null),
duration (number in hours or null),
days_left (integer or null),
query_date (string in YYYY-MM-DD format or null),
departure_date (string in YYYY-MM-DD format or null),
price (number or null)

Return only the JSON object and nothing else.

Example valid output:
{{"airline":"Air India","source_city":"Delhi","departure_time":"Morning","stops":"zero","arrival_time":"Morning","destination_city":"Chennai","class":"Economy","duration":2.5,"days_left":7,"query_date":"2025-01-03","departure_date":"2025-05-03","price":10359.54}}

User request: {q.text}
"""
    raw = call_ollama(prompt)
    if isinstance(raw, str) and raw.startswith("ERROR:"):
        raise HTTPException(status_code=502, detail=f"Ollama error: {raw}")

    # 1) try to directly extract balanced JSON
    try:
        parsed = extract_first_json(raw)
        return {"parsed": parsed}
    except Exception:
        pass

    # 2) try cleaning heuristics and reparse
    cleaned = clean_json_snippet(raw)
    try:
        parsed2 = json.loads(cleaned)
        return {"parsed": parsed2, "raw_snippet": raw, "cleaned_snippet": cleaned}
    except Exception as e:
        # 3) final regex fallback
        parsed3 = regex_field_extract(raw)
        if parsed3:
            return {"parsed": parsed3, "raw_snippet": raw, "note": "used regex fallback"}
        # 4) failed entirely, return debug info
        return {"parsed": {"error": "Could not parse response", "raw_snippet": raw, "cleaned_snippet": cleaned, "parse_error": str(e)}}

@app.post("/chat_predict")
def chat_predict(q: QueryText):
    """
    Full flow: natural language -> parse (LLM) -> featurize -> predict booking-time class + price -> recommendation
    """
    try:
        # 1) parse
        parse_result = parse_text(q)
        parsed = parse_result.get("parsed")
        if not isinstance(parsed, dict) or "error" in parsed:
            return {"error": "Parsing failed", "parse_result": parsed}

        # 2) featurize
        features = featurize_from_parsed(parsed)
        # prepare DataFrame for classifier/predictors if needed
        X_df = pd.DataFrame([features])

        # 3) booking classifier predict
        if book_model is None or label_le is None:
            logger.warning("Booking model or label encoder not available")
            probs_vec = None
            best_bucket = "unknown"
            conf = 0.0
        else:
            try:
                raw_probs = book_model.predict(X_df[FEATURE_COLS].astype(float).fillna(-1.0))
                arr = np.array(raw_probs)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                probs_vec = arr[0]
                classes = list(label_le.classes_)
                if len(probs_vec) != len(classes):
                    # fallback softmax
                    exps = np.exp(probs_vec - np.max(probs_vec))
                    probs_vec = exps / exps.sum()
                proba_map = {classes[i]: float(probs_vec[i]) for i in range(len(classes))}
                best_bucket = classes[int(np.argmax(probs_vec))]
                conf = float(np.max(probs_vec))
            except Exception:
                logger.exception("Booking model prediction failed")
                probs_vec = None
                best_bucket = "unknown"
                proba_map = {}
                conf = 0.0

        # 4) price prediction
        predicted_price = predict_price_from_features(features)
        # fallback heuristic if price_model not present or returned None
        if predicted_price is None:
            hist_min = features.get("hist_min_7", -1.0)
            hist_mean = features.get("hist_mean_14", -1.0)
            current_price = float(features.get("price", 0.0) or 0.0)
            if hist_min and hist_min > 0:
                predicted_price = float(hist_min)
                predicted_source = "hist_min_7"
            elif hist_mean and hist_mean > 0:
                predicted_price = float(hist_mean)
                predicted_source = "hist_mean_14"
            else:
                # use classifier bucket-based heuristic
                bucket_expected_drop = {
                    "buy_now": 0.0,
                    "wait_1_7": 0.03,
                    "wait_8_21": 0.06,
                    "wait_22_plus": 0.10
                }
                expected_drop = bucket_expected_drop.get(best_bucket, 0.02)
                predicted_price = max(0.0, current_price * (1.0 - expected_drop))
                predicted_source = "heuristic_bucket"
        else:
            predicted_source = "price_model"

        # 5) build recommendation
        current_price = float(features.get("price", 0.0) or 0.0)
        rec_text = build_recommendation_text(current_price, predicted_price, best_bucket, probs_vec)

        # assemble response
        response = {
            "user_input": q.text,
            "parsed_request": parsed,
            "encoded_features": features,
            "predicted_price": round(float(predicted_price), 2) if predicted_price is not None else None,
            "predicted_price_source": predicted_source,
            "recommendation": rec_text,
            "probabilities": proba_map if 'proba_map' in locals() else {},
            "best_bucket": best_bucket,
            "confidence": round(conf, 3)
        }
        return response

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("chat_predict failed")
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": tb})

@app.post("/predict")
def predict_price_raw(payload: dict = Body(...)):
    """
    Direct numeric prediction for price model.
    Accepts a JSON object with numeric feature keys (matching price_model_cols or FEATURE_COLS).
    """
    features = payload
    # convert to DataFrame
    try:
        df = pd.DataFrame([features])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    if price_model is None:
        raise HTTPException(status_code=404, detail="Price model not available on server")

    # reorder to expected columns if available
    cols = price_model_cols if price_model_cols else FEATURE_COLS
    for c in cols:
        if c not in df.columns:
            df[c] = -1.0
    X = df[cols].astype(float).fillna(-1.0)

    try:
        raw = price_model.predict(X)[0]
        if price_use_log_target:
            pred = float(np.expm1(raw))
        else:
            pred = float(raw)
        pred = max(0.0, pred)
        return {"predicted_price": round(pred, 2)}
    except Exception as e:
        logger.exception("Price prediction failed")
        raise HTTPException(status_code=500, detail=f"Price model prediction failed: {e}")

# -------------------------
# If run directly
# -------------------------
if __name__ == "__main__":
    print("Run with uvicorn: uvicorn llm.mistral_server:app --reload --port 8001")
