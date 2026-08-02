import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# --------------------------------------------------
# Project Path
# --------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

data_path = project_root / "data" / "raw" / "malaysia_house_price_data_2025.csv"
output_dir = project_root / "report_assets" / "plots" / "eda"
output_dir.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def save_histogram(df, column):
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"{column} Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / f"{column}_distribution.png", dpi=300)
    plt.close()


def save_boxplot(df, column):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[column])
    plt.title(f"{column} Boxplot")
    plt.tight_layout()
    plt.savefig(output_dir / f"{column}_boxplot.png", dpi=300)
    plt.close()


# Print top 15 values only, too many will cause the plot to be too tall or too crowded, making it difficult to read.
# Limited to 15 for better visualization.
def save_countplot(df, column):
    plt.figure(figsize=(8, 5))
    top = df[column].value_counts().head(15).index

    sns.countplot(data=df, y=column, order=top, palette="viridis")
    plt.title(f"{column} Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / f"{column}_countplot_top15.png", dpi=300)
    plt.close()


def count_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    df = pd.read_csv(data_path)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Shape : {df.shape}")

    print("\nData Types")
    print(df.dtypes)

    print("\nFirst Five Rows")
    print(df.head())

    print("\nDuplicate Rows")
    duplicate_value = df.duplicated().sum()
    print(duplicate_value)
    if duplicate_value == 0:
        print("No duplicate rows found.")

    print("\nUnique Values")
    print(df.nunique())

    # --------------------------------------------------
    # Missing Values
    # --------------------------------------------------

    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    missing = df.isnull().sum()
    print(missing)

    # If got missing value, then plot the missing values. Else, just display no missing value message.
    if missing.sum() > 0:
        missing = missing[missing > 0]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=missing.index, y=missing.values, palette="Reds")

        plt.ylabel("Missing Count")
        plt.xticks(rotation=45)
        plt.title("Missing Values")

        plt.tight_layout()
        plt.savefig(output_dir / "missing_values.png", dpi=300)
        plt.close()

    else:
        print("No missing values found.")

    # --------------------------------------------------
    # Descriptive Statistics
    # --------------------------------------------------

    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    print(df.describe(include="all"))

    # --------------------------------------------------
    # Numeric Features
    # --------------------------------------------------

    numeric_cols = df.select_dtypes(include=np.number).columns

    print("\nNumeric Columns")
    print(numeric_cols)

    # Histogram
    for col in numeric_cols:
        save_histogram(df, col)

    # Boxplot
    for col in numeric_cols:
        save_boxplot(df, col)

    # --------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
        plt.close()

    # --------------------------------------------------
    # Target Distribution
    # --------------------------------------------------

    if "Median_Price" in df.columns:
        plt.figure(figsize=(7, 5))
        sns.histplot(df["Median_Price"], bins=40, kde=True)
        plt.title("Target Distribution")
        plt.tight_layout()
        plt.savefig(output_dir / "target_distribution.png", dpi=300)
        plt.close()
        print("Original Skewness:", df["Median_Price"].skew())

        # Log Distribution

        plt.figure(figsize=(7, 5))
        sns.histplot(np.log1p(df["Median_Price"]), bins=40, kde=True)
        plt.title("Log Target Distribution")
        plt.tight_layout()
        plt.savefig(output_dir / "log_target_distribution.png", dpi=300)
        plt.close()
        print("Log Skewness:", np.log1p(df["Median_Price"]).skew())

    # --------------------------------------------------
    # Categorical Features
    # --------------------------------------------------

    categorical_cols = df.select_dtypes(include="object").columns

    print("\nCategorical Columns")
    print(categorical_cols)

    for col in categorical_cols:
        save_countplot(df, col)

    # --------------------------------------------------
    # Relationship Plot
    # --------------------------------------------------

    if "Transactions" in df.columns and "Median_Price" in df.columns:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df, x="Transactions", y="Median_Price", alpha=0.5)
        plt.title("Transactions vs Median Price")
        plt.tight_layout()
        plt.savefig(output_dir / "transactions_vs_price.png", dpi=300)
        plt.close()

    # --------------------------------------------------
    # Feature vs Target Analysis
    # --------------------------------------------------

    for col in ["State", "Tenure", "Type"]:
        if col in df.columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=col, y="Median_Price", data=df)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f"{col}_vs_price.png", dpi=300)
            plt.close()

    # --------------------------------------------------
    # Outliers Count
    # --------------------------------------------------
    print("\nOutliers Count: ")
    for col in numeric_cols:
        n = count_outliers(df[col])
        print(
            f"{col}: {n} ({n / len(df):.2%})"
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)

    print(f"Dataset Size : {df.shape}")
    print(f"Duplicate Rows : {df.duplicated().sum()}")
    print(f"Missing Values : {df.isnull().sum().sum()}")
    print("\nEDA Completed Successfully.")


if __name__ == "__main__":
    main()
