import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers

DATA_PATH = Path("water_quality_dataset.csv")
ARTIFACTS_DIR = Path("artifacts")

LOOKBACK = 8  # 2 hours (8 * 15min)
HORIZON_STEPS = [1, 2, 4]  # 15, 30, 60 minutes
TEST_RATIO = 0.15
VAL_RATIO = 0.15  # train = 0.70

SENSOR_COLS = ["pH", "Turbidity_NTU", "TDS_ppm", "Temperature_C", "Flow_Rate_Lmin"]
TARGET_COL = "Water_Quality_Index"
TIME_COL = "Timestamp"


def make_supervised(df_feat: pd.DataFrame, y: np.ndarray, lookback: int, horizon_steps: list[int]):
    X_list, Y_list = [], []
    max_h = max(horizon_steps)
    for i in range(lookback, len(df_feat) - max_h):
        X_window = df_feat.iloc[i - lookback : i].to_numpy(dtype=np.float32)
        Y = np.array([y[i + h] for h in horizon_steps], dtype=np.float32)
        X_list.append(X_window)
        Y_list.append(Y)
    return np.stack(X_list), np.stack(Y_list)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def eval_multi(y_true, y_pred, label):
    horizons = ["15min", "30min", "60min"]
    print(f"\n=== {label} ===")
    for k, hname in enumerate(horizons):
        mae = mean_absolute_error(y_true[:, k], y_pred[:, k])
        r = rmse(y_true[:, k], y_pred[:, k])
        print(f"{hname}: MAE={mae:.3f}  RMSE={r:.3f}")


def main():
    rf_path = ARTIFACTS_DIR / "rf_current_wqi.joblib"
    if not rf_path.exists():
        raise FileNotFoundError(
            f"Missing {rf_path}. Run: python train_rf_time_split.py"
        )
    rf_model = joblib.load(rf_path)

    df = pd.read_csv(DATA_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    if df[SENSOR_COLS + [TARGET_COL]].isna().any().any():
        raise ValueError("Dataset has missing values; please clean/impute before training.")

    # Time split on raw rows (same as your earlier LSTM script)
    n = len(df)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)
    n_train = n - n_val - n_test

    # Build WQI_hat (predicted current WQI) from sensors
    wqi_hat = rf_model.predict(df[SENSOR_COLS].to_numpy(dtype=np.float32)).astype(np.float32)
    df["WQI_hat"] = wqi_hat

    # Inputs to LSTM: sensors + predicted WQI (NOT the true WQI)
    feature_cols = SENSOR_COLS + ["WQI_hat"]
    X_full_df = df[feature_cols].copy()
    y_full = df[TARGET_COL].to_numpy(dtype=np.float32)  # targets remain TRUE WQI

    # Scale inputs on TRAIN ONLY (no leakage)
    x_scaler = StandardScaler()
    x_scaler.fit(X_full_df.iloc[:n_train].to_numpy(dtype=np.float32))

    X_scaled = x_scaler.transform(X_full_df.to_numpy(dtype=np.float32))
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    # Make sequences
    X_seq, Y_seq = make_supervised(X_scaled_df, y_full, lookback=LOOKBACK, horizon_steps=HORIZON_STEPS)

    # Sequence-aligned split (by raw end index)
    seq_end_raw_idx = np.arange(len(X_seq)) + (LOOKBACK - 1)
    train_mask = seq_end_raw_idx < (n_train - 1)
    val_mask = (seq_end_raw_idx >= (n_train - 1)) & (seq_end_raw_idx < (n_train + n_val - 1))
    test_mask = seq_end_raw_idx >= (n_train + n_val - 1)

    X_train, y_train = X_seq[train_mask], Y_seq[train_mask]
    X_val, y_val = X_seq[val_mask], Y_seq[val_mask]
    X_test, y_test = X_seq[test_mask], Y_seq[test_mask]

    # Scale multi-horizon targets
    y_scaler = StandardScaler()
    y_scaler.fit(y_train)
    y_train_s = y_scaler.transform(y_train)
    y_val_s = y_scaler.transform(y_val)

    print("Shapes:")
    print("X_train", X_train.shape, "y_train", y_train.shape)
    print("X_val  ", X_val.shape, "y_val  ", y_val.shape)
    print("X_test ", X_test.shape, "y_test ", y_test.shape)

    # Baseline (pipeline-consistent): persistence using last predicted current WQI (WQI_hat)
    end_idxs = seq_end_raw_idx[test_mask]
    persistence_hat = df["WQI_hat"].to_numpy(dtype=np.float32)[end_idxs].reshape(-1, 1)
    y_pred_persist = np.repeat(persistence_hat, repeats=3, axis=1)
    eval_multi(y_test, y_pred_persist, "Persistence baseline (predict last WQI_hat)")

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(LOOKBACK, len(feature_cols))),
            layers.LSTM(32, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(16),
            layers.Dense(16, activation="relu"),
            layers.Dense(3),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    model.fit(
        X_train,
        y_train_s,
        validation_data=(X_val, y_val_s),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_pred_s = model.predict(X_test, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_s)
    eval_multi(y_test, y_pred, "LSTM multi-horizon (inputs use WQI_hat)")

    out_dir = Path("artifacts_optionB")
    out_dir.mkdir(exist_ok=True)

    model.save(out_dir / "lstm_wqi_multi_horizon_wqihat.keras")
    joblib.dump(x_scaler, out_dir / "x_scaler_wqihat.joblib")
    joblib.dump(y_scaler, out_dir / "y_scaler.joblib")

    config = {
        "lookback": LOOKBACK,
        "horizon_steps": HORIZON_STEPS,
        "sensor_cols": SENSOR_COLS,
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "time_col": TIME_COL,
        "rf_model_path": str(rf_path),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\nSaved Option-B LSTM + scalers to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

