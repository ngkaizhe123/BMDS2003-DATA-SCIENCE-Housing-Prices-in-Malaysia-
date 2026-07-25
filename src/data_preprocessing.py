import pandas as pd
import numpy as np
import sys
from pathlib import Path


def handle_missing_values(
    df: pd.DataFrame, target_col: str = "Median_Price"
) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=[target_col])
    if "Transactions" in df_clean.columns:
        df_clean["Transactions"] = df_clean["Transactions"].fillna(
            df_clean["Transactions"].median()
        )
    return df_clean


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def standardize_text_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
            df_clean[col] = df_clean[col].replace(r"\s+", " ", regex=True)
    return df_clean


def clean_tenure(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    if "Tenure" in df_clean.columns:
        df_clean["Tenure"] = df_clean["Tenure"].replace(
            {
                "Freehold, Leasehold": "Freehold and Leasehold",
                "Leasehold, Freehold": "Freehold and Leasehold",
            }
        )
    return df_clean


def multi_hot_encode_type(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    if "Type" in df_clean.columns:
        type_dummies = df_clean["Type"].str.get_dummies(sep=", ").add_prefix("Type_")

        df_clean = pd.concat([df_clean.drop(columns=["Type"]), type_dummies], axis=1)

    return df_clean


def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[*] Initial dataset shape: {df.shape}")

    df = handle_missing_values(df, target_col="Median_Price")

    text_cols_to_clean = ["Area", "State", "Tenure", "Type"]

    df = standardize_text_columns(df, text_cols_to_clean)
    print("Before cleaning:")
    print(df["Tenure"].value_counts())
    df = clean_tenure(df)
    print("\nAfter cleaning:")
    print(df["Tenure"].value_counts())

    # Multi-Hot Encoding for Property Type
    df = multi_hot_encode_type(df)

    df = remove_outliers_iqr(df, "Median_Price")
    print(f"[*] Shape after preprocessing: {df.shape}")
    return df


# NEW: This block only runs if you execute this file directly!
if __name__ == "__main__":
    # 1. Define paths
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    input_path = project_root / "data" / "raw" / "malaysia_house_price_data_2025.csv"
    output_dir = project_root / "data" / "processed"
    output_path = output_dir / "cleaned_malaysia_house_prices.csv"

    print(f"Loading raw data from {input_path}...")
    try:
        raw_df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {input_path}")
        sys.exit(1)

    # 2. Process the data
    print("Running preprocessing pipeline...")
    clean_df = run_preprocessing_pipeline(raw_df)

    # 3. Save the data
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving cleaned data to {output_path}...")
    clean_df.to_csv(output_path, index=False)
    print("✅ Data successfully preprocessed and saved!")
