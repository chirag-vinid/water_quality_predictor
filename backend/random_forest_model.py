import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import joblib

# ── Step 1: Load Dataset ──────────────────────────────────────────
df = pd.read_csv("water_quality_dataset.csv")
print("Dataset loaded!")
print(f"Total rows: {len(df)}")
print(df.head())

# ── Step 2: Define Features and Target ───────────────────────────
features = ['pH', 'Turbidity_NTU', 'TDS_ppm', 'Temperature_C', 'Flow_Rate_Lmin']
target = 'Water_Quality_Index'

X = df[features]
y = df[target]

# ── Step 3: Train-Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ── Step 4: Train Random Forest Model ────────────────────────────
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("\nModel training complete!")

# ── Step 5: Evaluate the Model ───────────────────────────────────
y_pred = rf_model.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"\n── Model Performance ──")
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")

# ── Step 6: Actual vs Predicted Plot ─────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual WQI")
plt.ylabel("Predicted WQI")
plt.title("Random Forest: Actual vs Predicted WQI")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()
print("Plot saved as actual_vs_predicted.png")

# ── Step 7: Feature Importance Plot ──────────────────────────────
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(7, 4))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("Feature importance plot saved!")

# ── Step 8: Water Safety Status ──────────────────────────────────
threshold = 50  # based on your WQI range (37 to 95)

results_df = pd.DataFrame({
    'Actual_WQI'   : y_test.values,
    'Predicted_WQI': y_pred
})
results_df['Status'] = results_df['Predicted_WQI'].apply(
    lambda x: 'Safe' if x >= threshold else 'Unsafe'
)
print(f"\nSample Predictions:\n{results_df.head(10)}")
print(f"\nStatus Counts:\n{results_df['Status'].value_counts()}")

# ── Step 9: Save the Model ────────────────────────────────────────
joblib.dump(rf_model, 'random_forest_wqi_model.pkl')
print("\nModel saved as random_forest_wqi_model.pkl ✅")