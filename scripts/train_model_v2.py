import os
import optuna
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ========== PATHS ==========
CLEAN_DATA_PATH = os.path.join("data_clean", "application_final.csv")
MODEL_DIR = "models"
PLOT_DIR = "docs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ========== LOAD DATA ==========
print("📂 Loading cleaned dataset...")
df = pd.read_csv(CLEAN_DATA_PATH)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# ========== FEATURE ENGINEERING ==========
print("🧩 Creating new ratio features (safe mode)...")

# Create features only if required columns exist
if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)

if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)

if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)

if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
    df["EMPLOYED_TO_AGE"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1)
else:
    print("⚠️ Skipping EMPLOYED_TO_AGE — missing DAYS_EMPLOYED or DAYS_BIRTH column")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

print("✅ Feature engineering completed.")


# ========== DEFINE FEATURES ==========
target_col = "TARGET"
y = df[target_col]
X = df.drop(columns=[target_col])

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# Encode categorical variables
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Scale numeric features
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# ========== TRAIN-TEST SPLIT ==========
print("🧮 Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ========== HANDLE IMBALANCE ==========
print("⚖️ Balancing data using SMOTE + undersampling...")
over = SMOTE(sampling_strategy=0.3, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.8)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)

print(f"✅ After balancing: {y_train_res.value_counts().to_dict()}")

# ========== OPTUNA TUNING ==========
print("🔍 Running Optuna hyperparameter optimization (XGBoost)...")

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5),
        "random_state": 42,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "auc"
    }

    model = XGBClassifier(**params)
    model.fit(X_train_res, y_train_res)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    return f1

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

print("✅ Best parameters found:")
print(study.best_params)

# ========== FINAL MODEL ==========
best_params = study.best_params
model = XGBClassifier(**best_params)
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print("\n🏁 Final Model Performance:")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ========== SAVE MODEL ==========
joblib.dump(model, os.path.join(MODEL_DIR, "xgb_optuna.pkl"))
joblib.dump(study.best_params, os.path.join(MODEL_DIR, "xgb_best_params.pkl"))
print("💾 Model and parameters saved successfully!")

# ========== FEATURE IMPORTANCE ==========
importance = model.feature_importances_
feat_imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)[:20]

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top 20 Feature Importances (Optimized XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "feature_importance_optuna.png"))
plt.close()

print("\n✅ Model training and optimization complete!")

# ======= SAVE ALL COMPONENTS FOR DEPLOYMENT =======
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(list(X_train.columns), "models/model_features.pkl")

print("✅ All model components saved successfully!")
