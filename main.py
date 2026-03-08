"""
main.py  –  FastAPI Backend
────────────────────────────────────────────────────────────────
Coordinates between:
  • simulator.py  →  POST /ingest          (receives sensor readings)
  • ML models     →  runs RF + LSTM        (returns predictions)
  • React dashboard →  WebSocket /ws       (pushes every update live)

Endpoints
---------
  POST /ingest          Simulator posts a sensor reading here
  GET  /history         Last N readings + predictions (for chart init)
  GET  /health          Health check
  WS   /ws              WebSocket – dashboard subscribes here

Run
---
  pip install fastapi uvicorn joblib numpy scikit-learn tensorflow
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import collections
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────
BASE        = Path(__file__).parent / "backend"
ARTIFACTS   = BASE / "artifacts"        # LSTM (option A) + RF
ARTIFACTS_B = BASE / "artifacts_optionB"  # LSTM (option B – uses WQI_hat)

RF_MODEL_PATH  = ARTIFACTS / "rf_current_wqi.joblib"
LSTM_PATH      = ARTIFACTS / "lstm_wqi_multi_horizon.keras"
X_SCALER_PATH  = ARTIFACTS / "x_scaler.joblib"
Y_SCALER_PATH  = ARTIFACTS / "y_scaler.joblib"
LSTM_CFG_PATH  = ARTIFACTS / "config.json"

WQI_SAFE_THRESHOLD = 50
RF_FEATURES  = ["pH", "Turbidity_NTU", "TDS_ppm", "Temperature_C", "Flow_Rate_Lmin"]
MAX_HISTORY  = 120   # readings kept in memory for dashboard history


# ── Model Store ───────────────────────────────────────────────────
class Models:
    rf_model   = None
    lstm_model = None
    x_scaler   = None
    y_scaler   = None
    lookback   : int = 8
    feature_cols: list[str] = []
    window: collections.deque = collections.deque()

    def load(self):
        import tensorflow as tf  # deferred so startup logs are clean

        print("Loading ML models…")

        # Random Forest
        self.rf_model = joblib.load(RF_MODEL_PATH)
        print(f"  ✓ RF  ({RF_MODEL_PATH.name})")

        # LSTM
        self.lstm_model = tf.keras.models.load_model(str(LSTM_PATH))
        print(f"  ✓ LSTM ({LSTM_PATH.name})")

        # Scalers
        self.x_scaler = joblib.load(X_SCALER_PATH)
        self.y_scaler = joblib.load(Y_SCALER_PATH)
        print("  ✓ Scalers loaded")

        # Config
        with open(LSTM_CFG_PATH) as f:
            cfg = json.load(f)
        self.lookback     = cfg["lookback"]
        self.feature_cols = cfg["feature_cols"]   # includes "Water_Quality_Index"
        self.window       = collections.deque(maxlen=self.lookback)
        print(f"  ✓ Config  lookback={self.lookback}  features={self.feature_cols}")
        print("Models ready.\n")

    def predict_rf(self, reading: dict) -> dict:
        X   = np.array([[reading[c] for c in RF_FEATURES]])
        wqi = float(self.rf_model.predict(X)[0])
        return {
            "wqi":    round(wqi, 2),
            "status": "Safe" if wqi >= WQI_SAFE_THRESHOLD else "Unsafe",
        }

    def predict_lstm(self, reading: dict, rf_wqi: float) -> dict:
        # Feature set for LSTM: sensors + WQI (the RF-predicted one fills the slot)
        entry = {col: reading.get(col, 0.0) for col in RF_FEATURES}
        entry["Water_Quality_Index"] = rf_wqi
        self.window.append(entry)

        if len(self.window) < self.lookback:
            return {
                "ready":       False,
                "window_size": len(self.window),
                "required":    self.lookback,
            }

        arr        = np.array([[r[c] for c in self.feature_cols] for r in self.window], dtype=np.float32)
        arr_scaled = self.x_scaler.transform(arr)
        X_input    = arr_scaled[np.newaxis, ...]   # (1, lookback, n_features)

        y_scaled = self.lstm_model.predict(X_input, verbose=0)
        y_pred   = self.y_scaler.inverse_transform(y_scaled)[0]

        return {
            "ready":      True,
            "wqi_15min":  round(float(y_pred[0]), 2),
            "wqi_30min":  round(float(y_pred[1]), 2),
            "wqi_60min":  round(float(y_pred[2]), 2),
        }


store   = Models()
history: collections.deque = collections.deque(maxlen=MAX_HISTORY)


# ── WebSocket Connection Manager ──────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        print(f"  WS connected  (total={len(self.active)})")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        print(f"  WS disconnected  (total={len(self.active)})")

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ── App Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    yield


app = FastAPI(title="Water Quality Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schema ───────────────────────────────────────────────
class SensorReading(BaseModel):
    timestamp:        str
    node_id:          str = "NODE-001"
    pH:               float
    Turbidity_NTU:    float
    TDS_ppm:          float
    Temperature_C:    float
    Flow_Rate_Lmin:   float
    ground_truth_wqi: float | None = None


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": store.rf_model is not None}


@app.post("/ingest")
async def ingest(reading: SensorReading):
    data = reading.model_dump()

    rf_result   = store.predict_rf(data)
    lstm_result = store.predict_lstm(data, rf_result["wqi"])

    packet: dict[str, Any] = {
        "timestamp":        data["timestamp"],
        "node_id":          data["node_id"],
        "sensor": {k: data[k] for k in RF_FEATURES},
        "ground_truth_wqi": data.get("ground_truth_wqi"),
        "rf":               rf_result,
        "lstm":             lstm_result,
    }

    history.append(packet)
    await manager.broadcast(packet)
    return {"rf": rf_result, "lstm": lstm_result}


@app.get("/history")
def get_history(n: int = 60):
    items = list(history)[-n:]
    return {"count": len(items), "readings": items}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()   # keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Dev runner ────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)