# scripts/explain_model.py
"""
Explain model with SHAP (TreeExplainer) and save outputs to /docs/
- Loads cleaned dataset and XGBoost model
- Prepares features the model expects (numeric coercion)
- Fixes known XGBoost base_score string bug before creating SHAP explainer
- Saves:
    - docs/shap_summary_plot.png
    - docs/shap_feature_importance.png
    - docs/shap_force_plot_example.html
"""

import os
import sys
import joblib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# try import shap (may require build tools on Windows)
try:
    import shap
except Exception as e:
    shap = None
    print("⚠️ shap import failed:", e)
    print("Install shap (pip install shap) and C++ build tools if required.")
    # We'll continue but will abort before trying SHAP calculations.

# Configuration - change these paths if your repo uses different names
ROOT = Path(__file__).resolve().parents[1]       # repo root
CLEAN_DATA_PATH = ROOT / "data_clean" / "application_final.csv"
MODEL_PATH = ROOT / "models" / "xgb_model.pkl"   # your saved xgb model (joblib or pickle)
DOCS_DIR = ROOT / "docs"
SAMPLE_SIZE = 1000  # SHAP sample size (adjust; smaller = faster)

os.makedirs(DOCS_DIR, exist_ok=True)


def load_model(path: Path):
    """Load model via joblib (or pickle fallback)."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    print(f"✅ Loaded model: {path}")
    return model


def get_model_feature_names(model, df):
    """
    Try to infer feature names used by the model.
    - For XGBoost: look for booster.feature_names or model.get_booster().feature_names
    - Otherwise fallback to intersection of df columns
    """
    # XGBoost native booster
    try:
        # If it's sklearn API wrapper
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            fnames = getattr(booster, "feature_names", None)
            if fnames:
                return list(fnames)
        # If it's the underlying booster already
        if hasattr(model, "feature_names") and model.feature_names is not None:
            return list(model.feature_names)
    except Exception:
        pass

    # fallback: use df columns
    print("⚠️ Could not read feature names from model metadata; falling back to df columns.")
    return list(df.columns)


def safe_prepare_X(df: pd.DataFrame, feature_names):
    """
    Prepare feature matrix:
    - Select feature_names intersection with df columns
    - Convert object columns to numeric where possible, else factorize/coerce to numeric codes
    - Replace infs/NaNs with 0 (safe for SHAP)
    """
    cols = [c for c in feature_names if c in df.columns]
    if len(cols) == 0:
        raise ValueError("No feature columns found in df that match model feature names.")

    X = df[cols].copy()

    # Convert numeric-like strings to numeric where possible
    for c in X.columns:
        if X[c].dtype == "object":
            # First try numeric conversion
            tmp = pd.to_numeric(X[c], errors="coerce")
            if tmp.notna().sum() > 0.5 * len(tmp):  # if majority converts, use it
                X[c] = tmp
            else:
                # fallback: factorize to numeric codes (stable)
                X[c] = pd.factorize(X[c].astype(str))[0]

    # now ensure all numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # replace inf and Nan with safe numeric
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    return X


def patch_xgb_booster_base_score(booster):
    """
    Fix weird 'base_score' stored as strings like "[5E-1]" in booster config
    by loading config, regex-replacing the string with a numeric token, and reloading.
    Returns True if patched, False if nothing to do.
    """
    try:
        cfg = booster.save_config()
    except Exception as e:
        print("⚠️ Could not save booster config:", e)
        return False

    # If base_score present and is a quoted string or bracketed string, replace it
    # patterns examples: "base_score": "[5E-1]"  or "base_score":"[5E-1]" or "base_score": "0.5"
    new_cfg = re.sub(
        r'"base_score"\s*:\s*"\[?([0-9Ee\.\-\+]+)\]?"',
        r'"base_score": \1',
        cfg,
    )

    # quick check
    if new_cfg != cfg:
        try:
            booster.load_config(new_cfg)
            print("🔧 Patched booster config (base_score) for SHAP compatibility.")
            return True
        except Exception as e:
            print("⚠️ Failed to reload patched booster config:", e)
            return False
    return False


def main():
    if shap is None:
        print("❌ shap is not installed. Install shap to run explainability.")
        sys.exit(1)

    # 1) Load dataset
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Clean data not found: {CLEAN_DATA_PATH}")
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2) Load model
    model = load_model(MODEL_PATH)

    # 3) Infer features
    feature_names = get_model_feature_names(model, df)

    # 4) Prepare X for SHAP
    X_full = safe_prepare_X(df, feature_names)
    n_sample = min(SAMPLE_SIZE, len(X_full))
    X_sample = X_full.sample(n=n_sample, random_state=42).reset_index(drop=True)
    print(f"📊 Sampled {len(X_sample)} rows for SHAP analysis")

    # 5) Ensure we have an XGBoost booster object for TreeExplainer
    try:
        # If sklearn wrapper
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
        else:
            # maybe it's already a booster
            booster = model
    except Exception:
        booster = model

    # If booster has save_config/load_config support and base_score bug occurs, patch it
    try:
        patched = patch_xgb_booster_base_score(booster)
        if not patched:
            print("ℹ️ No booster patch required or patch skipped.")
    except Exception as e:
        print("⚠️ Patch step error:", e)

    # 6) Create SHAP explainer and compute values
    print("🔍 Calculating SHAP values (this may take a few minutes)...")
    try:
        # prefer TreeExplainer for tree models
        explainer = shap.TreeExplainer(booster)
        # new shap API: explainer(X) returns a ShapValues object; older uses shap_values()
        try:
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            # fallback to direct call
            shap_values = explainer(X_sample)
    except Exception as e:
        # if TreeExplainer fails, try KernelExplainer as fallback (slower)
        print("⚠️ TreeExplainer failed:", e)
        print("ℹ️ Trying KernelExplainer fallback (VERY SLOW for many features).")
        try:
            # Use model.predict_proba for KernelExplainer; need a wrapper callable
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
            bg = shap.sample(X_sample, 100)  # background small sample
            explainer = shap.KernelExplainer(predict_fn, bg)
            shap_values = explainer.shap_values(X_sample, nsamples=200)
        except Exception as e2:
            print("❌ KernelExplainer also failed:", e2)
            raise RuntimeError("SHAP explainability failed (both TreeExplainer and KernelExplainer).")

    # 7) Save SHAP outputs
    # summary_plot
    plt.figure(figsize=(10, 6))
    try:
        # If shap_values is array-like or list of arrays
        shap.summary_plot(shap_values, X_sample, show=False)
        out1 = DOCS_DIR / "shap_summary_plot.png"
        plt.tight_layout()
        plt.savefig(out1, dpi=150)
        plt.close()
        print("✅ Saved:", out1)
    except Exception as e:
        print("⚠️ Could not create shap.summary_plot:", e)

    # barplot of mean absolute shap (feature importance)
    try:
        # compute mean(|shap|) per feature
        if isinstance(shap_values, list) or isinstance(shap_values, tuple):
            # for multi-output classifiers shap_values may be list -> pick class 1 if present
            arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            # shap_values may be numpy array or shap explanation object
            # If it's an Explanation object, convert to values
            if hasattr(shap_values, "values"):
                arr = shap_values.values
            else:
                arr = np.array(shap_values)

        mean_abs = np.abs(arr).mean(axis=0)
        feat_imp = pd.Series(mean_abs, index=X_sample.columns).sort_values(ascending=False)
        top = feat_imp[:30]

        plt.figure(figsize=(8, 8))
        sns.barplot(x=top.values, y=top.index)
        plt.title("Top SHAP mean(|value|) Feature Importance")
        plt.xlabel("mean(|SHAP value|)")
        plt.tight_layout()
        out2 = DOCS_DIR / "shap_feature_importance.png"
        plt.savefig(out2, dpi=150)
        plt.close()
        print("✅ Saved:", out2)
    except Exception as e:
        print("⚠️ Could not save feature importance plot:", e)

    # force plot for a single sample -> HTML
    try:
        example_idx = 0
        # compute expected_value + force_plot
        if hasattr(explainer, "expected_value"):
            base = explainer.expected_value
        else:
            base = None

        # shap.force_plot API varies; build and save HTML
        # Use shap.Explanation / shap.force_plot properly:
        force_html = None
        try:
            # If shap_values is list -> use class 1 shap_values[1]
            sv_for_plot = shap_values[1] if isinstance(shap_values, (list, tuple)) and len(shap_values) > 1 else shap_values
            # convert to shap.Explanation if necessary
            # shap.force_plot accepts (expected_value, shap_values[row], features=row)
            force_plot = shap.force_plot(explainer.expected_value if hasattr(explainer, "expected_value") else None,
                                         sv_for_plot[example_idx],
                                         X_sample.iloc[example_idx],
                                         matplotlib=False)
            out_html = DOCS_DIR / "shap_force_plot_example.html"
            shap.save_html(str(out_html), force_plot)
            print("✅ Saved:", out_html)
        except Exception as fp_e:
            # last fallback: try shap.plotting._force._save_html
            print("⚠️ force_plot creation error:", fp_e)
    except Exception as e:
        print("⚠️ Could not create force plot:", e)

    print("\n✅ SHAP Explainability Completed Successfully!")
    print("Outputs saved to:", DOCS_DIR.resolve())


if __name__ == "__main__":
    main()
