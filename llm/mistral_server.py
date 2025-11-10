# # llm/mistral_server.py
# import os
# import json
# import re
# import requests
# import joblib
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel

# # ---------------------
# # Config
# # ---------------------
# MODEL_PATH = os.path.join("ml", "booktime_model.pkl")
# LABEL_ENCODER_PATH = os.path.join("ml", "booktime_label_encoder.joblib")
# ENCODERS_DIR = os.path.join("ml", "encoders")

# # Load model and label encoder
# book_model = joblib.load(MODEL_PATH)
# label_le = joblib.load(LABEL_ENCODER_PATH)

# # Load all categorical encoders
# encoders = {}
# for name in ["airline", "origin", "destination", "stops", "departure_time", "class"]:
#     path = os.path.join(ENCODERS_DIR, f"{name}_encoder.joblib")
#     if os.path.exists(path):
#         encoders[name] = joblib.load(path)
#     else:
#         print(f"⚠️ Warning: encoder not found for {name}")

# OLLAMA_URL = "http://127.0.0.1:11434/api/generate"   # your working endpoint
# OLLAMA_MODEL = "mistral:latest"

# # ---------------------
# # Load model + mappings
# # ---------------------
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
# model = joblib.load(MODEL_PATH)

# if not os.path.exists(MAPPINGS_PATH):
#     raise FileNotFoundError(f"Mappings not found at {MAPPINGS_PATH}")
# with open(MAPPINGS_PATH, "r", encoding="utf-8") as f:
#     mappings = json.load(f)

# # Build reverse maps: text -> code
# reverse_maps = {
#     field: {v.lower(): int(k) for k, v in fmap.items()}
#     for field, fmap in mappings.items()
# }

# # ---------------------
# # FastAPI
# # ---------------------
# app = FastAPI(title="SkyPredict - Simple Mistral Server")

# class Query(BaseModel):
#     text: str

# # ---------------------
# # Helpers
# # ---------------------
# def call_ollama(prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
#     """
#     Simple call to Ollama /api/generate. Return textual completion or an error string.
#     """
#     payload = {
#         "model": OLLAMA_MODEL,
#         "prompt": prompt,
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#         "stream": False
#     }
#     try:
#         r = requests.post(OLLAMA_URL, json=payload, timeout=30)
#         r.raise_for_status()
#     except Exception as e:
#         return f"ERROR: {e}"

#     try:
#         data = r.json()
#     except Exception:
#         return r.text

#     # Prefer 'response' then 'completion'
#     if isinstance(data, dict):
#         if "response" in data and isinstance(data["response"], str):
#             return data["response"]
#         if "completion" in data and isinstance(data["completion"], str):
#             return data["completion"]
#         # try simple nested shapes
#         if "results" in data and isinstance(data["results"], list) and data["results"]:
#             r0 = data["results"][0]
#             if isinstance(r0, dict):
#                 if "content" in r0 and isinstance(r0["content"], list) and r0["content"]:
#                     c0 = r0["content"][0]
#                     if isinstance(c0, dict):
#                         for k in ("text", "content", "output_text", "completion"):
#                             if k in c0 and isinstance(c0[k], str):
#                                 return c0[k]
#                 for k in ("text", "completion", "content"):
#                     if k in r0 and isinstance(r0[k], str):
#                         return r0[k]
#         if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
#             ch = data["choices"][0]
#             if isinstance(ch, dict) and "text" in ch:
#                 return ch["text"]

#     # fallback: return JSON string
#     return json.dumps(data)

# def extract_first_json(text: str):
#     """
#     Extract the first balanced JSON object from the text (simple robust method).
#     """
#     if not isinstance(text, str):
#         raise ValueError("Non-string input")

#     text = text.strip()
#     start = text.find("{")
#     if start == -1:
#         raise ValueError("No JSON object found")

#     level = 0
#     in_string = False
#     escape = False
#     for i in range(start, len(text)):
#         c = text[i]
#         if in_string:
#             if escape:
#                 escape = False
#             elif c == "\\":
#                 escape = True
#             elif c == '"':
#                 in_string = False
#         else:
#             if c == '"':
#                 in_string = True
#             elif c == "{":
#                 level += 1
#             elif c == "}":
#                 level -= 1
#                 if level == 0:
#                     candidate = text[start:i+1]
#                     return json.loads(candidate)
#     raise ValueError("Incomplete JSON object")

# def safe_map(field: str, value: str, default: int):
#     if not value:
#         return default
#     key = str(value).strip().lower()
#     fmap = reverse_maps.get(field, {})
#     if key in fmap:
#         return fmap[key]
#     key2 = key.replace(" ", "_").replace("-", "_")
#     if key2 in fmap:
#         return fmap[key2]
#     for name, code in fmap.items():
#         if name in key or key in name:
#             return code
#     return default

# def map_time(s: str):
#     lookup = {"afternoon":0,"early_morning":1,"evening":2,"late_night":3,"morning":4,"night":5}
#     if not s:
#         return lookup["morning"]
#     return lookup.get(s.strip().lower().replace(" ", "_"), lookup["morning"])

# def map_stops(s: str):
#     if s is None:
#         return reverse_maps.get("stops", {}).get("zero", 2)
#     s = str(s).lower()
#     if "non" in s or "zero" in s or s == "0":
#         return reverse_maps.get("stops", {}).get("zero", 2)
#     if "one" in s or "1" in s:
#         return reverse_maps.get("stops", {}).get("one", 0)
#     if "two" in s or "2" in s:
#         return reverse_maps.get("stops", {}).get("two_or_more", 1)
#     try:
#         return int(s)
#     except:
#         return reverse_maps.get("stops", {}).get("zero", 2)

# # ---------------------
# # Endpoints
# # ---------------------
# @app.post("/parse")
# def parse(query: Query):
#     prompt = f"""
# You are a flight booking assistant. Extract ONLY the following fields and return JSON exactly:
# airline, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left

# Return a single JSON object and nothing else. Example:
# {{"airline":"spicejet","source_city":"delhi","departure_time":"evening","stops":"non-stop","arrival_time":"night","destination_city":"mumbai","class":"economy","duration":2.5,"days_left":3}}

# User request: {query.text}
# """
#     raw = call_ollama(prompt)
#     if isinstance(raw, str) and raw.startswith("ERROR:"):
#         return {"parsed": {"error": raw}}
#     try:
#         parsed = extract_first_json(raw)
#     except Exception as e:
#         return {"parsed": {"error": "Could not parse response", "raw": raw, "err": str(e)}}
#     return {"parsed": parsed}

# @app.post("/chat_predict")
# def chat_predict(query: Query):
#     parsed_wrapper = parse(query).get("parsed", {})
#     if "error" in parsed_wrapper:
#         return {"error": "Could not extract flight details", "raw": parsed_wrapper}

#     parsed = parsed_wrapper

#     # defaults and normalization
#     airline_text = parsed.get("airline", "") or ""
#     source_text = parsed.get("source_city", "") or ""
#     departure_text = parsed.get("departure_time", "") or "morning"
#     stops_text = parsed.get("stops", "") or "zero"
#     arrival_text = parsed.get("arrival_time", "") or departure_text
#     destination_text = parsed.get("destination_city", "") or ""
#     class_text = parsed.get("class", "") or "economy"
#     try:
#         duration_val = float(parsed.get("duration", 1.0))
#     except:
#         duration_val = 1.0
#     try:
#         days_left_val = int(parsed.get("days_left", 1))
#     except:
#         days_left_val = 1

#     encoded = {
#         "airline": safe_map("airline", airline_text, 0),
#         "source_city": safe_map("source_city", source_text, 0),
#         "departure_time": map_time(departure_text),
#         "stops": map_stops(stops_text),
#         "arrival_time": map_time(arrival_text),
#         "destination_city": safe_map("destination_city", destination_text, 0),
#         "class": safe_map("class", class_text, 1),
#         "duration": duration_val,
#         "days_left": days_left_val
#     }

#     df = pd.DataFrame([encoded])
#     try:
#         price = model.predict(df)[0]
#     except Exception as e:
#         return {"error": f"Model predict failed: {e}", "encoded": encoded}

#     # human-readable decoded
#     decoded = {}
#     for k, v in encoded.items():
#         if k in mappings:
#             decoded[k] = mappings[k].get(str(v), v)
#         else:
#             decoded[k] = v

#     return {
#         "user_input": query.text,
#         "parsed_request": parsed,
#         "encoded_request": encoded,
#         "parsed_readable": decoded,
#         "predicted_price": round(float(price), 2)
#     }

# @app.post("/predict")
# def predict_price(flight: dict):
#     df = pd.DataFrame([flight])
#     try:
#         pred = model.predict(df)[0]
#     except Exception as e:
#         return {"error": f"Prediction failed: {e}"}
#     return {"predicted_price": float(pred)}



# llm/mistral_server.py
import os
import json
import logging
from typing import Any, Dict, Optional

import requests
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------------
# Logging & config
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mistral_server")

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # one level above llm/
ML_DIR = os.path.join(PROJECT_ROOT, "ml")

# Ollama (confirmed endpoint)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"
OLLAMA_TIMEOUT = 30

# Model artifact paths
BOOK_MODEL_PATH = os.path.join(ML_DIR, "booktime_model.pkl")
BOOK_LABEL_ENCODER_PATH = os.path.join(ML_DIR, "booktime_label_encoder.joblib")
ENCODERS_DIR = os.path.join(ML_DIR, "encoders")
PRICE_MODEL_PATH = os.path.join(ML_DIR, "model.joblib")  # optional older price model

# Expected feature columns (must match training)
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
# FastAPI App & Schemas
# -------------------------
app = FastAPI(title="SkyPredict - Ollama + Booking-Time-Classifier")

class QueryText(BaseModel):
    text: str

# -------------------------
# Utilities: LLM call + JSON extractor
# -------------------------
def call_ollama(prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
    """
    Simple call to Ollama /api/generate and return the generated text (or an ERROR string).
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

    # try parse JSON body returned by Ollama
    try:
        data = r.json()
    except Exception:
        # fall back to raw text
        return r.text

    # common shapes
    if isinstance(data, dict):
        if "response" in data and isinstance(data["response"], str):
            return data["response"]
        if "completion" in data and isinstance(data["completion"], str):
            return data["completion"]
        if "results" in data and isinstance(data["results"], list) and data["results"]:
            r0 = data["results"][0]
            if isinstance(r0, dict) and "content" in r0 and isinstance(r0["content"], list) and r0["content"]:
                c0 = r0["content"][0]
                if isinstance(c0, dict):
                    for key in ("text", "content", "output_text", "completion"):
                        if key in c0 and isinstance(c0[key], str):
                            return c0[key]
            for key in ("text", "completion", "content"):
                if isinstance(r0, dict) and key in r0 and isinstance(r0[key], str):
                    return r0[key]
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            ch0 = data["choices"][0]
            if isinstance(ch0, dict) and "text" in ch0:
                return ch0["text"]

    return json.dumps(data)


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
# Load models & encoders
# -------------------------
# booking-time classifier
if not os.path.exists(BOOK_MODEL_PATH):
    logger.error("Booking model not found at %s", BOOK_MODEL_PATH)
    book_model = None
else:
    try:
        book_model = joblib.load(BOOK_MODEL_PATH)
        logger.info("Loaded booking model from %s", BOOK_MODEL_PATH)
    except Exception as e:
        logger.exception("Failed loading booking model")
        book_model = None

# label encoder
if os.path.exists(BOOK_LABEL_ENCODER_PATH):
    try:
        label_le = joblib.load(BOOK_LABEL_ENCODER_PATH)
        logger.info("Loaded booking label encoder")
    except Exception as e:
        logger.exception("Failed loading booking label encoder")
        label_le = None
else:
    label_le = None
    logger.warning("Booking label encoder not found at %s", BOOK_LABEL_ENCODER_PATH)

# categorical encoders
encoders = {}
for name in ["airline", "origin", "destination", "stops", "departure_time", "class"]:
    p = os.path.join(ENCODERS_DIR, f"{name}_encoder.joblib")
    if os.path.exists(p):
        try:
            encoders[name] = joblib.load(p)
            logger.info("Loaded encoder for %s", name)
        except Exception:
            logger.exception("Failed to load encoder: %s", name)
    else:
        logger.warning("Encoder missing for %s (expected at %s)", name, p)

# optional old price prediction model
price_model = None
if os.path.exists(PRICE_MODEL_PATH):
    try:
        price_model = joblib.load(PRICE_MODEL_PATH)
        logger.info("Loaded price model from %s", PRICE_MODEL_PATH)
    except Exception:
        logger.exception("Failed to load price model; price endpoint will be disabled")

# -------------------------
# Featurization helper
# -------------------------
def safe_transform_encoder(enc, val) -> int:
    """
    Transform a text value to encoder code. Fallback to 0 on failure.
    """
    if enc is None:
        return 0
    try:
        # ensure string form
        return int(enc.transform([str(val)])[0])
    except Exception:
        try:
            # if encoder has classes_, try best-effort matching
            classes = [str(x).lower() for x in enc.classes_]
            key = str(val).lower()
            if key in classes:
                return int(classes.index(key))
        except Exception:
            pass
    return 0


def featurize_from_parsed(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build feature dict consistent with the training feature columns.
    Use default placeholders for history features if not available in parsed data.
    """
    # textual fields in parsed: airline, source_city, destination_city, departure_time, stops, class
    airline_val = parsed.get("airline", parsed.get("airline_name", ""))
    origin_val = parsed.get("source_city", parsed.get("origin", ""))
    dest_val = parsed.get("destination_city", parsed.get("destination", ""))
    dep_time_val = parsed.get("departure_time", "")
    stops_val = parsed.get("stops", "")
    class_val = parsed.get("class", "")

    airline_enc = safe_transform_encoder(encoders.get("airline"), airline_val)
    origin_enc = safe_transform_encoder(encoders.get("origin"), origin_val)
    dest_enc = safe_transform_encoder(encoders.get("destination"), dest_val)
    stops_enc = safe_transform_encoder(encoders.get("stops"), stops_val)
    dep_time_enc = safe_transform_encoder(encoders.get("departure_time"), dep_time_val)
    class_enc = safe_transform_encoder(encoders.get("class"), class_val)

    # numeric fields
    days_left = int(parsed.get("days_left", parsed.get("days_to_departure", 10)))
    # if 'price' field isn't present, allow current_price or 0
    price_val = parsed.get("price")
    if price_val is None:
        price_val = parsed.get("current_price", 0.0)
    try:
        price_val = float(price_val)
    except Exception:
        price_val = 0.0

    # history features - if you have a historical DB you can fill these; otherwise defaults used in training
    hist_min_7 = float(parsed.get("hist_min_7", -1.0))
    hist_mean_14 = float(parsed.get("hist_mean_14", -1.0))
    hist_std_30 = float(parsed.get("hist_std_30", -1.0))
    price_momentum_7 = float(parsed.get("price_momentum_7", 0.0))

    features = {
        "airline_enc": airline_enc,
        "origin_enc": origin_enc,
        "destination_enc": dest_enc,
        "days_left": days_left,
        "price": price_val,
        "hist_min_7": hist_min_7,
        "hist_mean_14": hist_mean_14,
        "hist_std_30": hist_std_30,
        "price_momentum_7": price_momentum_7,
        "stops_enc": stops_enc,
        "departure_time_enc": dep_time_enc,
        "class_enc": class_enc,
    }

    # Ensure order and keys match FEATURE_COLS (some names may differ: stops_enc vs stops_enc etc.)
    # Keep keys consistent with FEATURE_COLS
    standardized = {
        "airline_enc": features["airline_enc"],
        "origin_enc": features["origin_enc"],
        "destination_enc": features["destination_enc"],
        "days_left": features["days_left"],
        "price": features["price"],
        "hist_min_7": features["hist_min_7"],
        "hist_mean_14": features["hist_mean_14"],
        "hist_std_30": features["hist_std_30"],
        "price_momentum_7": features["price_momentum_7"],
        "stops_enc": features["stops_enc"],
        "departure_time_enc": features["departure_time_enc"],
        "class_enc": features["class_enc"],
    }

    return standardized

# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "booking_model_loaded": book_model is not None,
        "price_model_loaded": price_model is not None,
        "ollama_endpoint": OLLAMA_URL
    }


@app.post("/parse")
def parse_text(q: QueryText):
    """
    Parse free text via Ollama LLM and return the extracted JSON block (parsed fields).
    The LLM prompt asks for a single JSON object with keys:
      airline, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left
    """
    prompt = f"""
You are a flight booking assistant. Extract ONLY the following fields and return a JSON object with these exact keys:
airline, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left

Return only valid JSON and nothing else.

User request: {q.text}
"""
    raw = call_ollama(prompt)
    if isinstance(raw, str) and raw.startswith("ERROR:"):
        raise HTTPException(status_code=502, detail=f"Ollama error: {raw}")

    try:
        parsed = extract_first_json(raw)
    except Exception as e:
        # return raw snippet for debugging
        return {"parsed": {"error": "Could not parse response", "raw_snippet": raw, "parse_error": str(e)}}

    return {"parsed": parsed}


@app.post("/chat_predict")
def chat_predict(q: QueryText):
    """
    Full flow: natural language -> parse (LLM) -> featurize -> predict booking-time class
    Returns recommendation + probability map.
    """
    # 1) parse
    parse_result = parse_text(q)
    parsed = parse_result.get("parsed")
    if not isinstance(parsed, dict) or "error" in parsed:
        # pass the error forward
        return {"error": "Parsing failed", "parse_result": parsed}

    # 2) featurize
    features = featurize_from_parsed(parsed)
    try:
        X_df = pd.DataFrame([features], columns=FEATURE_COLS).astype(float).fillna(-1.0)
    except Exception as e:
        logger.exception("Featurize -> DataFrame failed")
        return {"error": f"Featurization failed: {e}", "features": features}

    # 3) predict using booking model
    if book_model is None or label_le is None:
        raise HTTPException(status_code=500, detail="Booking model or label encoder not available on server")

    try:
        # LightGBM Booster predict returns shape (n_samples, n_classes)
        probs = book_model.predict(X_df)
        # some saved sklearn wrappers may return 1D; normalize to 2D
        arr = np.array(probs)
        if arr.ndim == 1:
            # if vector returned, try reshape; otherwise wrap
            arr = arr.reshape(1, -1)
        probs_vec = arr[0]
        classes = list(label_le.classes_)
        if len(probs_vec) != len(classes):
            # sometimes sklearn pack returns raw logits; try softmax
            try:
                exps = np.exp(probs_vec - np.max(probs_vec))
                probs_vec = exps / exps.sum()
            except Exception:
                # fallback: uniform
                probs_vec = np.ones(len(classes)) / len(classes)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    proba_map = {classes[i]: float(probs_vec[i]) for i in range(len(classes))}
    best = classes[int(np.argmax(probs_vec))]

    # return parsed + encoded + recommendation
    return {
        "user_input": q.text,
        "parsed_request": parsed,
        "encoded_features": features,
        "recommendation": best,
        "probabilities": proba_map
    }


@app.post("/predict")
def predict_price(features: Dict[str, Any]):
    """
    Direct numeric prediction endpoint for the price model (if present).
    Expects keys matching the price model's training features (e.g., airline, source_city, ...)
    OR numeric encoded features. This endpoint is optional if you don't host a price model.
    """
    if price_model is None:
        raise HTTPException(status_code=404, detail="Price model not available on server")

    # Convert to DataFrame row
    try:
        df = pd.DataFrame([features])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    try:
        pred = price_model.predict(df)[0]
    except Exception as e:
        logger.exception("Price model prediction failed")
        raise HTTPException(status_code=500, detail=f"Price model prediction failed: {e}")

    return {"predicted_price": float(pred)}

# -------------------------
# If run directly, helpful message
# -------------------------
if __name__ == "__main__":
    print("This module is intended to be run by uvicorn, e.g.:")
    print("  uvicorn llm.mistral_server:app --reload --port 8001")
