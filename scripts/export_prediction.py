import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

# === Paths ===
MODEL_DIR = "models"
DATA_PATH = "data_clean/application_final.csv"  # Adjust if needed
OUTPUT_PATH = "data_clean/prediction_results.csv"

# === Load model and artifacts ===
model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))

print("✅ Model and artifacts loaded successfully!")
print(f"Expected features: {len(features)}")

# === Load dataset ===
df = pd.read_csv(DATA_PATH)
print(f"📂 Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# === Align features safely ===
for f in features:
    if f not in df.columns:
        df[f] = 0
        print(f"⚠️ Missing feature added as 0: {f}")

# Keep only model features and reorder
X = df[features].copy()

# === Encode categorical columns ===
cat_cols = X.select_dtypes(include="object").columns.tolist()
if cat_cols:
    print(f"🔠 Encoding {len(cat_cols)} categorical columns...")
    for col in cat_cols:
        X[col] = pd.factorize(X[col])[0]

# === Handle scaling ===
# Only scale numeric columns that were part of the scaler's training
if hasattr(scaler, "feature_names_in_"):
    numeric_cols = [c for c in scaler.feature_names_in_ if c in X.columns]
else:
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.transform(X[numeric_cols])

# === Predict ===
y_pred_prob = model.predict_proba(X_scaled)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

# === Save results ===
df["PREDICTION"] = y_pred
df["PROBABILITY"] = y_pred_prob

df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Predictions exported to: {OUTPUT_PATH}")

# === Evaluate (if TARGET exists) ===
if "TARGET" in df.columns:
    y = df["TARGET"]
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_prob)
    print("\n📊 Model Evaluation on Full Data:")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   ROC-AUC  : {auc:.4f}")
