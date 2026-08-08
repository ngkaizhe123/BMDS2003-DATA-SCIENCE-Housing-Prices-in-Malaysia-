import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json


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


def remove_township(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    if "Township" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Township"])
        print("[*] Township column removed.")
    return df_clean


def clean_state(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    state_counts = df["State"].value_counts()
    rare_states = state_counts[state_counts < 2].index
    strat_col = df["State"].replace(rare_states, "Other_State")  # Handle rare states
    df_clean["State"] = strat_col
    print("[*] Rare states replaced with 'Other_State'.")
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


def bucket_rare_areas(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    if "Area" in df.columns and "State" in df.columns:
        area_counts = df["Area"].value_counts()
        rare = area_counts[area_counts < min_count].index
        df = df.copy()
        df["Area"] = df["Area"].where(
            ~df["Area"].isin(rare), other="Other_" + df["State"]
        )
    print("[*] Rare areas bucketed into 'Other_State'.")
    print(f"[*] Dataset shape after bucketing rare areas: {df.shape}")
    return df


def multi_hot_encode_type(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    if "Type" in df_clean.columns:
        # Normalize separator: strip spaces around commas before splitting
        # Prevents wrong column names when raw data has no space after the comma
        df_clean["Type"] = df_clean["Type"].str.replace(r"\s*,\s*", ", ", regex=True)

        type_dummies = df_clean["Type"].str.get_dummies(sep=", ").add_prefix("Type_")

        df_clean = pd.concat([df_clean.drop(columns=["Type"]), type_dummies], axis=1)

    return df_clean


def add_area_density_features(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    if "Area" in df_clean.columns and "Transactions" in df_clean.columns:
        total_transactions = df_clean["Transactions"].sum()
        if total_transactions > 0:
            area_tx_sum = df_clean.groupby("Area")["Transactions"].transform("sum")
            df_clean["Area_Transaction_Density"] = area_tx_sum / total_transactions
            print("[*] Area Transaction Density feature successfully engineered.")

    return df_clean


def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[*] Initial dataset shape: {df.shape}")

    df = handle_missing_values(df, target_col="Median_Price")

    text_cols_to_clean = ["Area", "State", "Tenure", "Type"]

    df = remove_township(df)

    df = standardize_text_columns(df, text_cols_to_clean)
    print("Before cleaning:")
    print(df["Tenure"].value_counts())
    df = clean_tenure(df)
    print("\nAfter cleaning:")
    print(df["Tenure"].value_counts())

    # 1. ENGINEER THE SIZE FEATURE HERE
    if "Median_PSF" in df.columns:
        df["Estimated_Size"] = df["Median_Price"] / df["Median_PSF"]
        df["Log_Estimated_Size"] = np.log1p(df["Estimated_Size"])
        # Drop Median_PSF so the model doesn't cheat by knowing the price-per-sqft directly
        df = df.drop(columns=["Median_PSF"])

    # 2. ADD DENSITY AREA TRANSFORMATIONS (New Step)
    df = add_area_density_features(df)

    # 3. REMOVE OUTLIERS FIRST
    print("Outliers removal: ")
    print("Before removing outliers:", df.shape)
    df = remove_outliers_iqr(df, "Median_Price")
    print("[*] Removed outliers based on Median_Price.")
    df = remove_outliers_iqr(df, "Transactions")
    print("[*] Removed outliers based on Transactions.")
    print("After removing outliers:", df.shape)

    # 4. CLEAN RARE CATEGORIES AFTER OUTLIERS ARE REMOVED
    print("Before cleaning:")
    print(df["State"].value_counts())
    df = clean_state(df)
    print("\nAfter cleaning:")
    print(df["State"].value_counts())
    df = bucket_rare_areas(df, min_count=5)

    # 5. ENCODE TYPES
    df = multi_hot_encode_type(df)

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

    # =========================================================
    # 4. Generate LookUp table
    # =========================================================
    print("Generating State-Area lookup table from RAW data...")

    state_area_lookup = {}

    # Calculate total transactions from raw data for accurate density
    total_tx = raw_df["Transactions"].sum()

    # Iterate through all original states
    for state in raw_df["State"].dropna().unique():
        state_areas = raw_df[raw_df["State"] == state].dropna(subset=["Area"])

        area_dict = {}
        for area, group in state_areas.groupby("Area"):
            # Get median transactions, fallback to 16.0 if NaN
            med_tx = group["Transactions"].median()
            med_tx = med_tx if pd.notna(med_tx) else 16.0

            # Calculate original Area Transaction Density
            area_tx_sum = group["Transactions"].sum()
            density = area_tx_sum / total_tx if total_tx > 0 else 0.005

            area_dict[area] = {
                "Transactions": med_tx,
                "Area_Transaction_Density": density
            }

        state_area_lookup[state] = area_dict

    lookup_path = output_dir / "state_area_lookup_table.json"
    with open(lookup_path, "w") as f:
        json.dump(state_area_lookup, f, indent=4)

    print(f"✅ Lookup table successfully saved to {lookup_path}")