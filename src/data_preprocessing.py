import pandas as pd
import numpy as np
from pathlib import Path
import sys


def standardize_text_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
            df_clean[col] = df_clean[col].replace(r"\s+", " ", regex=True)
    return df_clean


def multi_hot_encode_type(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    if "Type" in df_clean.columns:
        df_clean["Type"] = df_clean["Type"].str.replace(r"\s*,\s*", ", ", regex=True)
        type_dummies = df_clean["Type"].str.get_dummies(sep=", ").add_prefix("Type_")
        df_clean = pd.concat([df_clean.drop(columns=["Type"]), type_dummies], axis=1)
    return df_clean


def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[*] Initial dataset shape: {df.shape}")

    # 1. Drop rows missing the target variable
    df = df.dropna(subset=["Median_Price"]).copy()

    # 2. Clean Text
    text_cols = ["Area", "State", "Tenure", "Type"]
    df = standardize_text_columns(df, text_cols)

    if "Tenure" in df.columns:
        df["Tenure"] = df["Tenure"].replace({
            "Freehold, Leasehold": "Freehold and Leasehold",
            "Leasehold, Freehold": "Freehold and Leasehold",
        })

    # 3. Engineer Features (Safe row-by-row math)
    if "Median_PSF" in df.columns:
        df["Estimated_Size"] = df["Median_Price"] / df["Median_PSF"]
        df["Log_Estimated_Size"] = np.log1p(df["Estimated_Size"])
        df = df.drop(columns=["Median_PSF"])  # Drop to prevent leakage

    # 4. Encode Type
    df = multi_hot_encode_type(df)

    print(f"[*] Shape after preprocessing: {df.shape}")
    return df


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    input_path = project_root / "data" / "raw" / "malaysia_house_price_data_2025.csv"
    output_path = project_root / "data" / "processed" / "cleaned_malaysia_house_prices.csv"

    raw_df = pd.read_csv(input_path)
    clean_df = run_preprocessing_pipeline(raw_df)
    clean_df.to_csv(output_path, index=False)
    print("✅ Cleaned CSV generated successfully!")