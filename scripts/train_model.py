import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, precision_recall_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# ================= CONFIGURATION =================
CLEAN_DATA_PATH = os.path.join("data_clean", "application_final.csv")
MODEL_DIR = "models"
PLOT_DIR = "docs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ================= LOAD DATA =================
print("📂 Loading cleaned dataset...")
df = pd.read_csv(CLEAN_DATA_PATH)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop any irrelevant ID columns
df.drop(columns=['SK_ID_CURR'], errors='ignore', inplace=True)

# Identify target and features
target_col = "TARGET"
X = df.drop(columns=[target_col])
y = df[target_col]

# ================= HANDLE CATEGORICALS =================
cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns

if len(cat_cols) > 0:
    print(f"🔤 Encoding {len(cat_cols)} categorical columns...")
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("Unknown")
        X[col] = le.fit_transform(X[col])
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
else:
    print("✅ No categorical columns found.")

# ================= TRAIN TEST SPLIT =================
print("📊 Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ================= APPLY SMOTE =================
print("⚖️ Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE: {y_train_res.value_counts().to_dict()}")

# ================= SCALE FEATURES =================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ================= FEATURE SELECTION (using XGBoost) =================
print("🌲 Running initial XGBoost for feature importance...")
xgb_temp = XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=200)
xgb_temp.fit(X_train_scaled, y_train_res)
feat_importance = pd.Series(xgb_temp.feature_importances_, index=X.columns)
important_feats = feat_importance[feat_importance > 0.01].index
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)[important_feats]
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)[important_feats]
print(f"✅ Using {len(important_feats)} important features for final training.")

# ================= LOGISTIC REGRESSION =================
print("\n🔹 Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train_res)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

lr_acc = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_roc = roc_auc_score(y_test, y_prob_lr)

print(f"✅ Logistic Regression Results -> Acc: {lr_acc:.4f}, F1: {lr_f1:.4f}, ROC-AUC: {lr_roc:.4f}")
joblib.dump(log_reg, os.path.join(MODEL_DIR, "logistic_model.pkl"))

# ================= XGBOOST HYPERPARAMETER TUNING =================
print("\n🚀 Tuning XGBoost parameters (this may take a few minutes)...")
params = {
    'n_estimators': [300],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 2, 5]
}

grid = GridSearchCV(
    estimator=XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid=params,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train_scaled, y_train_res)

best_xgb = grid.best_estimator_
print(f"✅ Best Parameters: {grid.best_params_}")

# ================= EVALUATE XGBOOST =================
y_prob_xgb = best_xgb.predict_proba(X_test_scaled)[:, 1]
y_pred_xgb = (y_prob_xgb > 0.3).astype(int)  # threshold tuning

xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_roc = roc_auc_score(y_test, y_prob_xgb)

print(f"✅ XGBoost Results -> Acc: {xgb_acc:.4f}, F1: {xgb_f1:.4f}, ROC-AUC: {xgb_roc:.4f}")
joblib.dump(best_xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

# ================= VISUALIZATIONS =================
# ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"XGBoost ROC-AUC: {xgb_roc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (XGBoost)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "roc_curve_xgb.png"))
plt.close()

# Feature importance plot
feat_imp_sorted = feat_importance.sort_values(ascending=False)[:20]
plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp_sorted.values, y=feat_imp_sorted.index)
plt.title("Top 20 Feature Importances (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "feature_importance_xgb.png"))
plt.close()

# ================= SUMMARY =================
print("\n📊 Final Model Summary:")
print(f"{'Model':<25}{'Accuracy':<12}{'F1 Score':<12}{'ROC-AUC':<12}")
print("-" * 60)
print(f"{'Logistic Regression':<25}{lr_acc:<12.4f}{lr_f1:<12.4f}{lr_roc:<12.4f}")
print(f"{'XGBoost (Tuned)':<25}{xgb_acc:<12.4f}{xgb_f1:<12.4f}{xgb_roc:<12.4f}")
print("\n✅ Model training complete! Results saved to /models and /docs/")
