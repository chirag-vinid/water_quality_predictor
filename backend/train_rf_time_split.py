import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("water_quality_dataset.csv")
TIME_COL = "Timestamp"

FEATURE_COLS = ["pH", "Turbidity_NTU", "TDS_ppm", "Temperature_C", "Flow_Rate_Lmin"]
TARGET_COL = "Water_Quality_Index"

TEST_RATIO = 0.15
VAL_RATIO = 0.15  # train = 0.70


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    df = pd.read_csv(DATA_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    if df[FEATURE_COLS + [TARGET_COL]].isna().any().any():
        raise ValueError("Dataset has missing values; please clean/impute before training.")

    n = len(df)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)
    n_train = n - n_val - n_test

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.float32)

    X_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COL].to_numpy(dtype=np.float32)

    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.float32)

    # Train on train split, validate on val split
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
    )
    rf.fit(X_train, y_train)

    y_val_pred = rf.predict(X_val)
    print("=== Random Forest (val) ===")
    print(f"R2  : {r2_score(y_val, y_val_pred):.4f}")
    print(f"MAE : {mean_absolute_error(y_val, y_val_pred):.3f}")
    print(f"RMSE: {rmse(y_val, y_val_pred):.3f}")

    # Final model: train on train+val, evaluate on test
    X_train_full = df.iloc[: n_train + n_val][FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train_full = df.iloc[: n_train + n_val][TARGET_COL].to_numpy(dtype=np.float32)

    rf_final = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
    )
    rf_final.fit(X_train_full, y_train_full)

    y_test_pred = rf_final.predict(X_test)
    print("\n=== Random Forest (test) ===")
    print(f"R2  : {r2_score(y_test, y_test_pred):.4f}")
    print(f"MAE : {mean_absolute_error(y_test, y_test_pred):.3f}")
    print(f"RMSE: {rmse(y_test, y_test_pred):.3f}")

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    joblib.dump(rf_final, out_dir / "rf_current_wqi.joblib")
    config = {
        "time_col": TIME_COL,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "test_ratio": TEST_RATIO,
        "val_ratio": VAL_RATIO,
        "n_rows": int(n),
        "n_train_rows": int(n_train),
        "n_val_rows": int(n_val),
        "n_test_rows": int(n_test),
    }
    (out_dir / "rf_config.json").write_text(json.dumps(config, indent=2))
    print(f"\nSaved RF model to: {(out_dir / 'rf_current_wqi.joblib').resolve()}")


if __name__ == "__main__":
    main()

