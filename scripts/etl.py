import pandas as pd
import numpy as np
import os

# -------------------- CONFIG --------------------
RAW_PATH = "data_raw"
CLEAN_PATH = "data_clean"
os.makedirs(CLEAN_PATH, exist_ok=True)

# ======================================================
# 🧹 UNIVERSAL CLEANING UTILITIES
# ======================================================
def clean_dataframe(df, name="DataFrame"):
    print(f"\n🧹 Cleaning {name} ...")

    # Drop fully blank rows
    df.dropna(how="all", inplace=True)

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"   • Removed {before - after} duplicates")

    # Replace infinite with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fix negative values in numeric columns (except IDs)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if "ID" in col.upper():
            continue
        negatives = (df[col] < 0).sum()
        if negatives > 0:
            df.loc[df[col] < 0, col] = np.nan
            print(f"   • Fixed {negatives} negative values in {col}")

    print(f"✅ {name} cleaned → {df.shape}")
    return df


def clean_categoricals(df):
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.title()
    print("🔠 Standardized categorical formatting.")
    return df


def cap_outliers(df, numeric_cols):
    for col in numeric_cols:
        if col in ["SK_ID_CURR", "TARGET"]:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = df[col].copy()
        df[col] = np.clip(df[col], lower, upper)
        if (before != df[col]).sum() > 0:
            print(f"📉 Capped outliers in {col}")
    return df

# ======================================================
# 📦 TABLE-SPECIFIC CLEANING AND AGGREGATION
# ======================================================
def clean_application():
    print("📂 Loading application_train.csv ...")
    app = pd.read_csv(os.path.join(RAW_PATH, "application_train.csv"))
    app = clean_dataframe(app, "application_train")

    # Replace placeholder values
    app["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
    app["AMT_INCOME_TOTAL"].replace(0, 1e-6, inplace=True)

    # Derived features
    app["AGE_YEARS"] = (-app["DAYS_BIRTH"]) // 365
    app["YEARS_EMPLOYED"] = (-app["DAYS_EMPLOYED"]) // 365
    app["CREDIT_INCOME_RATIO"] = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"]
    app["ANNUITY_INCOME_RATIO"] = app["AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"]

    app = clean_categoricals(app)

    # Select relevant columns
    keep_cols = [
        "SK_ID_CURR", "TARGET", "CODE_GENDER", "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
        "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE", "AGE_YEARS", "YEARS_EMPLOYED",
        "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"
    ]
    app = app[keep_cols]
    return app


def aggregate_bureau():
    print("📂 Loading bureau.csv ...")
    bureau = pd.read_csv(os.path.join(RAW_PATH, "bureau.csv"))
    bureau = clean_dataframe(bureau, "bureau")
    bureau["CREDIT_ACTIVE"] = bureau["CREDIT_ACTIVE"].map({"Active": 1, "Closed": 0})

    agg = bureau.groupby("SK_ID_CURR").agg({
        "AMT_CREDIT_SUM": ["mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
        "CREDIT_ACTIVE": "mean"
    })
    agg.columns = [
        "BUREAU_CREDIT_MEAN", "BUREAU_CREDIT_MAX",
        "BUREAU_DEBT_MEAN", "BUREAU_DEBT_TOTAL",
        "BUREAU_ACTIVE_RATIO"
    ]
    agg.reset_index(inplace=True)
    return agg


def aggregate_prev_app():
    print("📂 Loading previous_application.csv ...")
    prev = pd.read_csv(os.path.join(RAW_PATH, "previous_application.csv"))
    prev = clean_dataframe(prev, "previous_application")

    agg = prev.groupby("SK_ID_CURR").agg({
        "AMT_CREDIT": ["mean", "max"],
        "AMT_ANNUITY": "mean",
        "DAYS_DECISION": "mean"
    })
    agg.columns = [
        "PREV_CREDIT_MEAN", "PREV_CREDIT_MAX",
        "PREV_ANNUITY_MEAN", "PREV_DAYS_DECISION_MEAN"
    ]
    agg.reset_index(inplace=True)
    return agg


def aggregate_installments():
    print("📂 Loading installments_payments.csv ...")
    inst = pd.read_csv(os.path.join(RAW_PATH, "installments_payments.csv"))
    inst = clean_dataframe(inst, "installments_payments")

    inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]
    agg = inst.groupby("SK_ID_CURR").agg({
        "PAYMENT_RATIO": "mean",
        "DAYS_ENTRY_PAYMENT": "mean"
    })
    agg.columns = ["INSTALL_PAYMENT_RATIO_MEAN", "INSTALL_PAYMENT_DELAY_MEAN"]
    agg.reset_index(inplace=True)
    return agg


def aggregate_credit_card():
    print("📂 Loading credit_card_balance.csv ...")
    cc = pd.read_csv(os.path.join(RAW_PATH, "credit_card_balance.csv"))
    cc = clean_dataframe(cc, "credit_card_balance")

    agg = cc.groupby("SK_ID_CURR").agg({
        "AMT_BALANCE": "mean",
        "AMT_CREDIT_LIMIT_ACTUAL": "mean",
        "SK_DPD": "mean"
    })
    agg.columns = ["CC_BALANCE_MEAN", "CC_LIMIT_MEAN", "CC_DPD_MEAN"]
    agg.reset_index(inplace=True)
    return agg

# ======================================================
# 🚀 MAIN ETL PIPELINE
# ======================================================
def main():
    print("🚀 Starting ETL pipeline...")

    app = clean_application()
    bureau = aggregate_bureau()
    prev = aggregate_prev_app()
    inst = aggregate_installments()
    cc = aggregate_credit_card()

    # Merge everything
    df = app.merge(bureau, on="SK_ID_CURR", how="left")
    df = df.merge(prev, on="SK_ID_CURR", how="left")
    df = df.merge(inst, on="SK_ID_CURR", how="left")
    df = df.merge(cc, on="SK_ID_CURR", how="left")

    print("\n🧠 Handling missing values, outliers, and formatting...")
    df = clean_categoricals(df)

    # Drop columns with >60% missing values
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > 0.6].index.tolist()
    if drop_cols:
        print(f"⚠️ Dropping columns with too many missing values: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)

    # Fill missing values
    cat_cols = df.select_dtypes(include="object").columns
    num_cols = df.select_dtypes(include=np.number).columns
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Cap outliers
    df = cap_outliers(df, num_cols)

    # Final clean pass
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    print("✅ Final dataset ready for ML training!")
    output_path = os.path.join(CLEAN_PATH, "application_final.csv")
    df.to_csv(output_path, index=False)
    print(f"💾 Saved cleaned dataset → {output_path}")


if __name__ == "__main__":
    main()
