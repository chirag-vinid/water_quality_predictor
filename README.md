# Water Quality Predictor

**Water Quality Monitoring and Leakage Detection System for Water Coolers**

An integrated IoT + Machine Learning system that combines real-time water-quality assessment, short-term forecasting, and unsupervised pipeline-leakage detection in a single ESP32-based hardware prototype and web dashboard — moving water monitoring from reactive alarms to proactive, forecast-aware intervention.

## Overview

Conventional water dispensing and monitoring setups rely on manual checks or static threshold alarms that only react after contamination or a leak has already happened. This project addresses that gap with three complementary layers running on a single architecture:

- **Instantaneous water-quality classification** — a Random Forest model computes a live Water Quality Index (WQI) from multi-sensor readings.
- **Multi-horizon forecasting** — an LSTM network predicts WQI trends 15, 30, and 60 minutes ahead.
- **Unsupervised leakage detection** — an Isolation Forest model flags abnormal flow patterns indicative of a pipeline leak, without needing labelled failure data.

All readings, predictions, and alerts are unified on a React dashboard for live monitoring.

## Features

- Real-time acquisition of pH, TDS, turbidity, temperature, and flow-rate data via an ESP32 microcontroller
- Instantaneous WQI computation using a Random Forest classifier
- Multi-horizon WQI forecasting (15 / 30 / 60 minutes) using an LSTM network with an 8-step sliding window
- Unsupervised flow-anomaly / leak detection using Isolation Forest
- Flask backend bridging the hardware sensing layer and the ML processing layer
- React dashboard for live sensor readings, WQI scores, forecasts, and leakage alerts
- Hardware and firmware logic validated in a Wokwi simulation prior to physical assembly

## Repository Structure

```
water_quality_predictor/
├── backend/                     # Flask backend and ML pipeline (Random Forest, LSTM, Isolation Forest)
├── dashboard/                   # React dashboard frontend
├── controller.ino               # ESP32 firmware: sensor acquisition and telemetry transmission
├── main.py                      # Entry point for the backend / ML pipeline
├── simulator.py                 # Generates simulated sensor telemetry (for testing without hardware)
├── water_quality_dataset.csv    # Dataset used to train and evaluate the ML models
├── wokwiLink.pdf                # Link to the Wokwi circuit simulation
├── requirements.txt             # Python dependencies
└── .gitignore
```

## System Architecture

```
Sensors → ESP32 → Telemetry (Flask backend) → Random Forest (WQI) + LSTM (forecast)
                                              → Isolation Forest (leak detection)
                                              → React Dashboard
```

The ESP32 forms the sensing layer and streams live pH, TDS, turbidity, and flow-rate telemetry to a Flask backend. The backend routes this data to the ML layer, which returns an instantaneous WQI, a multi-horizon WQI forecast, and a leak/anomaly flag. All outputs are surfaced together on the React dashboard.

## Hardware

- ESP32 microcontroller
- pH sensor
- TDS (Total Dissolved Solids) sensor
- Turbidity sensor
- Flow-rate sensor
- Circuit designed and validated in Wokwi before physical assembly (see `wokwiLink.pdf` and `controller.ino`)

## Machine Learning Models

| Model | Task | Input Features | Output |
|---|---|---|---|
| Random Forest | Instantaneous WQI assessment | pH, TDS, Turbidity, Flow Rate | Current WQI score / classification |
| LSTM | Multi-horizon WQI forecasting | pH, TDS, Turbidity, Flow Rate, current WQI (8-step sliding window) | WQI forecast at 15, 30, and 60 minutes |
| Isolation Forest | Leakage / anomaly detection | Flow Rate | Normal vs. anomalous flow flag |

## Dataset

The water-quality parameters (pH, turbidity, temperature, and related readings) used to train and evaluate the models were extracted from the publicly available [Water Quality and Pollution Monitoring Dataset](https://www.kaggle.com/datasets/ziya07/water-quality-and-pollution-monitoring-dataset) on Kaggle, and combined with TDS and flow-rate telemetry captured independently from the ESP32 prototype. The combined data used in this repository is provided in `water_quality_dataset.csv`.

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js and npm (for the dashboard)
- Arduino IDE or PlatformIO (to flash `controller.ino` to an ESP32)
- A [Wokwi](https://wokwi.com/) account if you want to run the simulated circuit instead of physical hardware (see `wokwiLink.pdf`)

### 1. Clone the repository

```bash
git clone https://github.com/chirag-vinid/water_quality_predictor.git
cd water_quality_predictor
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the backend / ML pipeline

```bash
python main.py
```

### 4. Simulate sensor data (optional, no hardware required)

```bash
python simulator.py
```

### 5. Flash the ESP32 firmware

1. Open `controller.ino` in the Arduino IDE.
2. Select your ESP32 board and the correct COM port.
3. Update the Wi-Fi credentials and backend endpoint in the sketch.
4. Upload the sketch to the board.

### 6. Run the dashboard

```bash
cd dashboard
npm install
npm start
```

## Performance (Provisional)

LSTM forecasting performance across horizons, using the currently available results:

| Forecast Horizon | MAE | RMSE | R² |
|---|---|---|---|
| 15 minutes | 1.82 | 2.31 | 0.95 |
| 30 minutes | 2.14 | 2.78 | 0.93 |
| 60 minutes | 2.67 | 3.42 | 0.89 |

> These values are provisional and will be updated as final results from the trained model and test dataset become available. Random Forest and Isolation Forest evaluation (R²/MAE/RMSE and Accuracy/F1-score, respectively) is still pending on their respective test sets.

## Limitations and Future Work

- **Edge computing** — perform basic anomaly detection and filtering on-device to reduce network load and response time.
- **Mobile application** — companion app for real-time readings, WQI values, and push notifications.
- **Additional sensors** — Dissolved Oxygen (DO), Electrical Conductivity (EC), ORP, and chlorine sensors for richer water-quality assessment.
- **Improved prediction models** — evaluate GRU, Bi-LSTM, and Transformer-based forecasters against the current LSTM.
- **Secure communication** — HTTPS/TLS and device/API authentication between the ESP32, backend, and dashboard.
- **Energy optimization** — ESP32 deep-sleep modes and event-based transmission for battery-powered deployments.
- **Multi-node monitoring** — extend to multiple ESP32 nodes reporting to a single backend and dashboard for larger distribution networks.

## Authors

Developed at the School of Computer Science and Engineering, Vellore Institute of Technology, Vellore.

- **Chirag Vinid** — Team Lead; system architecture, hardware setup, and simulation validation
- **Saket Thota** — Flask backend and sensor calibration
- **Srija Koppar** — Random Forest model and dataset preparation
- **Shaswatha S S** — LSTM forecasting model
- **Megh Chakravarty** — Isolation Forest model and React dashboard
- **Syamasudha Veeragandham** — Project Supervisor

## License

No license has been specified for this repository yet. Until a license is added, all rights are reserved by the authors.
