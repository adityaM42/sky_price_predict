# llm/mistral_server.py
import os
import json
import re
import requests
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------
# Config
# ---------------------
ROOT = os.path.dirname(os.path.dirname(__file__))   # project root (one level above llm/)
MODEL_PATH = os.path.join(ROOT, "ml", "model.joblib")
MAPPINGS_PATH = os.path.join(ROOT, "ml", "mappings.json")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"   # your working endpoint
OLLAMA_MODEL = "mistral:latest"

# ---------------------
# Load model + mappings
# ---------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

if not os.path.exists(MAPPINGS_PATH):
    raise FileNotFoundError(f"Mappings not found at {MAPPINGS_PATH}")
with open(MAPPINGS_PATH, "r", encoding="utf-8") as f:
    mappings = json.load(f)

# Build reverse maps: text -> code
reverse_maps = {
    field: {v.lower(): int(k) for k, v in fmap.items()}
    for field, fmap in mappings.items()
}

# ---------------------
# FastAPI
# ---------------------
app = FastAPI(title="SkyPredict - Simple Mistral Server")

class Query(BaseModel):
    text: str

# ---------------------
# Helpers
# ---------------------
def call_ollama(prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
    """
    Simple call to Ollama /api/generate. Return textual completion or an error string.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=30)
        r.raise_for_status()
    except Exception as e:
        return f"ERROR: {e}"

    try:
        data = r.json()
    except Exception:
        return r.text

    # Prefer 'response' then 'completion'
    if isinstance(data, dict):
        if "response" in data and isinstance(data["response"], str):
            return data["response"]
        if "completion" in data and isinstance(data["completion"], str):
            return data["completion"]
        # try simple nested shapes
        if "results" in data and isinstance(data["results"], list) and data["results"]:
            r0 = data["results"][0]
            if isinstance(r0, dict):
                if "content" in r0 and isinstance(r0["content"], list) and r0["content"]:
                    c0 = r0["content"][0]
                    if isinstance(c0, dict):
                        for k in ("text", "content", "output_text", "completion"):
                            if k in c0 and isinstance(c0[k], str):
                                return c0[k]
                for k in ("text", "completion", "content"):
                    if k in r0 and isinstance(r0[k], str):
                        return r0[k]
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            ch = data["choices"][0]
            if isinstance(ch, dict) and "text" in ch:
                return ch["text"]

    # fallback: return JSON string
    return json.dumps(data)

def extract_first_json(text: str):
    """
    Extract the first balanced JSON object from the text (simple robust method).
    """
    if not isinstance(text, str):
        raise ValueError("Non-string input")

    text = text.strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    level = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
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
                    candidate = text[start:i+1]
                    return json.loads(candidate)
    raise ValueError("Incomplete JSON object")

def safe_map(field: str, value: str, default: int):
    if not value:
        return default
    key = str(value).strip().lower()
    fmap = reverse_maps.get(field, {})
    if key in fmap:
        return fmap[key]
    key2 = key.replace(" ", "_").replace("-", "_")
    if key2 in fmap:
        return fmap[key2]
    for name, code in fmap.items():
        if name in key or key in name:
            return code
    return default

def map_time(s: str):
    lookup = {"afternoon":0,"early_morning":1,"evening":2,"late_night":3,"morning":4,"night":5}
    if not s:
        return lookup["morning"]
    return lookup.get(s.strip().lower().replace(" ", "_"), lookup["morning"])

def map_stops(s: str):
    if s is None:
        return reverse_maps.get("stops", {}).get("zero", 2)
    s = str(s).lower()
    if "non" in s or "zero" in s or s == "0":
        return reverse_maps.get("stops", {}).get("zero", 2)
    if "one" in s or "1" in s:
        return reverse_maps.get("stops", {}).get("one", 0)
    if "two" in s or "2" in s:
        return reverse_maps.get("stops", {}).get("two_or_more", 1)
    try:
        return int(s)
    except:
        return reverse_maps.get("stops", {}).get("zero", 2)

# ---------------------
# Endpoints
# ---------------------
@app.post("/parse")
def parse(query: Query):
    prompt = f"""
You are a flight booking assistant. Extract ONLY the following fields and return JSON exactly:
airline, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left

Return a single JSON object and nothing else. Example:
{{"airline":"spicejet","source_city":"delhi","departure_time":"evening","stops":"non-stop","arrival_time":"night","destination_city":"mumbai","class":"economy","duration":2.5,"days_left":3}}

User request: {query.text}
"""
    raw = call_ollama(prompt)
    if isinstance(raw, str) and raw.startswith("ERROR:"):
        return {"parsed": {"error": raw}}
    try:
        parsed = extract_first_json(raw)
    except Exception as e:
        return {"parsed": {"error": "Could not parse response", "raw": raw, "err": str(e)}}
    return {"parsed": parsed}

@app.post("/chat_predict")
def chat_predict(query: Query):
    parsed_wrapper = parse(query).get("parsed", {})
    if "error" in parsed_wrapper:
        return {"error": "Could not extract flight details", "raw": parsed_wrapper}

    parsed = parsed_wrapper

    # defaults and normalization
    airline_text = parsed.get("airline", "") or ""
    source_text = parsed.get("source_city", "") or ""
    departure_text = parsed.get("departure_time", "") or "morning"
    stops_text = parsed.get("stops", "") or "zero"
    arrival_text = parsed.get("arrival_time", "") or departure_text
    destination_text = parsed.get("destination_city", "") or ""
    class_text = parsed.get("class", "") or "economy"
    try:
        duration_val = float(parsed.get("duration", 1.0))
    except:
        duration_val = 1.0
    try:
        days_left_val = int(parsed.get("days_left", 1))
    except:
        days_left_val = 1

    encoded = {
        "airline": safe_map("airline", airline_text, 0),
        "source_city": safe_map("source_city", source_text, 0),
        "departure_time": map_time(departure_text),
        "stops": map_stops(stops_text),
        "arrival_time": map_time(arrival_text),
        "destination_city": safe_map("destination_city", destination_text, 0),
        "class": safe_map("class", class_text, 1),
        "duration": duration_val,
        "days_left": days_left_val
    }

    df = pd.DataFrame([encoded])
    try:
        price = model.predict(df)[0]
    except Exception as e:
        return {"error": f"Model predict failed: {e}", "encoded": encoded}

    # human-readable decoded
    decoded = {}
    for k, v in encoded.items():
        if k in mappings:
            decoded[k] = mappings[k].get(str(v), v)
        else:
            decoded[k] = v

    return {
        "user_input": query.text,
        "parsed_request": parsed,
        "encoded_request": encoded,
        "parsed_readable": decoded,
        "predicted_price": round(float(price), 2)
    }

@app.post("/predict")
def predict_price(flight: dict):
    df = pd.DataFrame([flight])
    try:
        pred = model.predict(df)[0]
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}
    return {"predicted_price": float(pred)}
