"""
simulator.py  –  IoT Water-Quality Node Simulator
──────────────────────────────────────────────────
Reads water_quality_dataset.csv row-by-row and POSTs each reading
to the FastAPI backend at the same cadence as the original 15-min
timestamps (or compressed via --speed-factor).

Usage
-----
    python simulator.py                          # 1 s per reading (speed=900)
    python simulator.py --speed-factor 1         # real-time (15 min per reading)
    python simulator.py --speed-factor 0         # fire all rows instantly
    python simulator.py --dataset path/to/file.csv --url http://localhost:8000/ingest
"""

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────
DEFAULT_DATASET = Path(__file__).parent / "backend" / "water_quality_dataset.csv"
DEFAULT_URL     = "http://127.0.0.1:8000/ingest"
SENSOR_COLS     = ["pH", "Turbidity_NTU", "TDS_ppm", "Temperature_C", "Flow_Rate_Lmin"]
TIME_COL        = "Timestamp"
WQI_COL         = "Water_Quality_Index"   # ground-truth – included for reference


def parse_args():
    p = argparse.ArgumentParser(description="IoT Water-Quality Node Simulator")
    p.add_argument("--dataset",      default=str(DEFAULT_DATASET))
    p.add_argument("--url",          default=DEFAULT_URL, help="FastAPI /ingest endpoint")
    p.add_argument(
        "--speed-factor", type=float, default=900.0,
        help="Divide real gap (sec) by this. 900 → 15 min becomes 1 s. 0 = no sleep.",
    )
    p.add_argument("--loop", action="store_true", help="Restart from row 1 after finishing")
    return p.parse_args()


def load_csv(path: str):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def post(url: str, payload: dict) -> dict | None:
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        print(f"  ⚠  Could not reach server: {e.reason}  (is FastAPI running?)")
        return None


def main():
    args = parse_args()
    rows = load_csv(args.dataset)

    print("=" * 62)
    print("  💧 IoT Water-Quality Node Simulator")
    print("=" * 62)
    print(f"  Dataset : {args.dataset}  ({len(rows)} rows)")
    print(f"  Target  : {args.url}")
    print(f"  Speed   : {args.speed_factor}×  (0 = instant)")
    print(f"  Loop    : {args.loop}")
    print("=" * 62)

    run = 0
    while True:
        run += 1
        if run > 1:
            print(f"\n── Loop {run} ──────────────────────────────")

        prev_ts = None
        for idx, row in enumerate(rows):
            ts = datetime.fromisoformat(row[TIME_COL])

            # Throttle based on timestamp gap
            if prev_ts is not None and args.speed_factor > 0:
                gap   = (ts - prev_ts).total_seconds()
                sleep = gap / args.speed_factor
                time.sleep(max(sleep, 0))
            prev_ts = ts

            payload = {
                "timestamp": row[TIME_COL],
                "node_id":   "NODE-001",
                "ground_truth_wqi": float(row[WQI_COL]),   # for dashboard reference
            }
            for col in SENSOR_COLS:
                payload[col] = float(row[col])

            now = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{now}] #{idx+1:04d}  ts={row[TIME_COL]}")
            print(f"  pH={payload['pH']}  Turb={payload['Turbidity_NTU']} NTU  "
                  f"TDS={payload['TDS_ppm']} ppm  Temp={payload['Temperature_C']}°C  "
                  f"Flow={payload['Flow_Rate_Lmin']} L/min")

            result = post(args.url, payload)
            if result:
                rf   = result.get("rf", {})
                lstm = result.get("lstm", {})
                print(f"  → RF  : WQI={rf.get('wqi'):.1f}  [{rf.get('status')}]")
                if lstm.get("ready"):
                    print(f"  → LSTM: +15m={lstm.get('wqi_15min'):.1f}  "
                          f"+30m={lstm.get('wqi_30min'):.1f}  "
                          f"+60m={lstm.get('wqi_60min'):.1f}")
                else:
                    print(f"  → LSTM: warming up  ({lstm.get('window_size')}/{lstm.get('required')} readings)")

        if not args.loop:
            break

    print("\n✅  Simulation complete.")


if __name__ == "__main__":
    main()