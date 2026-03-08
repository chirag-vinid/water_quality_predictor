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

LOOKBACK = 8                  # 2 hours (8 * 15min)
HORIZON_STEPS = [1, 2, 4]     # 15, 30, 60 minutes
TEST_RATIO = 0.15
VAL_RATIO = 0.15              # train = 0.70

FEATURE_COLS = [
    "pH",
    "Turbidity_NTU",
    "TDS_ppm",
    "Temperature_C",
    "Flow_Rate_Lmin",
    "Water_Quality_Index",     # include past WQI as an input feature
]
TARGET_COL = "Water_Quality_Index"
TIME_COL = "Timestamp"


def make_supervised(df_feat: pd.DataFrame, y: np.ndarray, lookback: int, horizon_steps: list[int]):
    X_list, Y_list = [], []
    max_h = max(horizon_steps)
    for i in range(lookback, len(df_feat) - max_h):
        X_window = df_feat.iloc[i - lookback:i].to_numpy(dtype=np.float32)
        Y = np.array([y[i + h] for h in horizon_steps], dtype=np.float32)
        X_list.append(X_window)
        Y_list.append(Y)
    return np.stack(X_list), np.stack(Y_list)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def eval_multi(y_true, y_pred, label):
    horizons = ["15min", "30min", "60min"]
    print(f"\n=== {label} ===")
    for k, hname in enumerate(horizons):
        mae = mean_absolute_error(y_true[:, k], y_pred[:, k])
        r = rmse(y_true[:, k], y_pred[:, k])
        print(f"{hname}: MAE={mae:.3f}  RMSE={r:.3f}")


def main():
    df = pd.read_csv(DATA_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # Basic sanity
    if df[FEATURE_COLS + [TARGET_COL]].isna().any().any():
        raise ValueError("Dataset has missing values; please clean/impute before training.")

    y_full = df[TARGET_COL].to_numpy(dtype=np.float32)
    X_full_df = df[FEATURE_COLS].copy()

    # Time-based split indices on the *raw rows* (before sequence building)
    n = len(df)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)
    n_train = n - n_val - n_test

    # Fit scalers on TRAIN ONLY (no leakage)
    x_scaler = StandardScaler()
    x_scaler.fit(X_full_df.iloc[:n_train].to_numpy(dtype=np.float32))

    # Transform all (train/val/test) with same scaler
    X_scaled = x_scaler.transform(X_full_df.to_numpy(dtype=np.float32))
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

    # Build supervised sequences from the *scaled* features
    X_seq, Y_seq = make_supervised(
        X_scaled_df, y_full, lookback=LOOKBACK, horizon_steps=HORIZON_STEPS
    )

    # Sequence-aligned split:
    # The first sequence ends at raw index LOOKBACK-1, so sequence i corresponds to raw end index (LOOKBACK+i-1).
    # To keep time integrity, split by the raw end index boundary.
    seq_end_raw_idx = np.arange(len(X_seq)) + (LOOKBACK - 1)

    train_mask = seq_end_raw_idx < (n_train - 1)
    val_mask = (seq_end_raw_idx >= (n_train - 1)) & (seq_end_raw_idx < (n_train + n_val - 1))
    test_mask = seq_end_raw_idx >= (n_train + n_val - 1)

    X_train, y_train = X_seq[train_mask], Y_seq[train_mask]
    X_val, y_val = X_seq[val_mask], Y_seq[val_mask]
    X_test, y_test = X_seq[test_mask], Y_seq[test_mask]

    # Scale targets (helps LSTM training); inverse-transform for metrics
    y_scaler = StandardScaler()
    y_scaler.fit(y_train)
    y_train_s = y_scaler.transform(y_train)
    y_val_s = y_scaler.transform(y_val)
    y_test_s = y_scaler.transform(y_test)

    print("Shapes:")
    print("X_train", X_train.shape, "y_train", y_train.shape)
    print("X_val  ", X_val.shape, "y_val  ", y_val.shape)
    print("X_test ", X_test.shape, "y_test ", y_test.shape)

    # Baseline: persistence using last observed WQI in the input window (unscaled target space)
    # last timestep in each window corresponds to raw time t-15min; we use its WQI value as forecast for all horizons
    wqi_idx_in_features = FEATURE_COLS.index("Water_Quality_Index")
    last_wqi_scaled = X_test[:, -1, wqi_idx_in_features]  # scaled
    # inverse-scale that single feature back? Easiest: compute persistence in original space from raw y:
    # For each test sequence, its end raw index is seq_end_raw_idx; last observed WQI is y_full[end_raw_idx]
    end_idxs = seq_end_raw_idx[test_mask]
    persistence = y_full[end_idxs].reshape(-1, 1)
    y_pred_persist = np.repeat(persistence, repeats=3, axis=1)
    eval_multi(y_test, y_pred_persist, "Persistence baseline (predict last WQI)")

    # LSTM model (simple + effective starter)
    tf.keras.utils.set_random_seed(42)

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(LOOKBACK, len(FEATURE_COLS))),
            layers.LSTM(32, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(16),
            layers.Dense(16, activation="relu"),
            layers.Dense(3),  # scaled target outputs
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train_s,
        validation_data=(X_val, y_val_s),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test (inverse-transform to original WQI units)
    y_pred_s = model.predict(X_test, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_s)

    eval_multi(y_test, y_pred, "LSTM multi-horizon")

    # Save artifacts
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    model.save(out_dir / "lstm_wqi_multi_horizon.keras")
    joblib.dump(x_scaler, out_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, out_dir / "y_scaler.joblib")

    config = {
        "lookback": LOOKBACK,
        "horizon_steps": HORIZON_STEPS,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "time_col": TIME_COL,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\nSaved model + scalers + config to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()