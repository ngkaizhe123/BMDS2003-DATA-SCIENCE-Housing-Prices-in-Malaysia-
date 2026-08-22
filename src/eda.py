import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from scipy.stats import chi2_contingency

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


def save_log_histogram(df, column):
    """Log1p-transformed view of a numeric column, so skew can be compared
    before/after the transform (mirrors what happens to the target during
    model training)."""
    plt.figure(figsize=(6, 4))
    sns.histplot(np.log1p(df[column]), kde=True, bins=30)
    plt.title(f"{column} Distribution (log1p)")
    plt.xlabel(f"log1p({column})")
    plt.tight_layout()
    plt.savefig(output_dir / f"{column}_log_distribution.png", dpi=300)
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
# Mixed-type association measures
#
# Pearson correlation only applies to numeric-numeric pairs. To build a
# heatmap that covers categorical features too (Area, State, Tenure, Type),
# we need two more measures:
#   - Cramer's V        : categorical vs categorical (0 to 1)
#   - Correlation Ratio  : numeric vs categorical (0 to 1)
# Unlike Pearson, both are unsigned (they measure strength, not direction).
# --------------------------------------------------
def cramers_v(x, y):
    """Bias-corrected Cramer's V between two categorical series."""
    try:
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape

        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min(kcorr - 1, rcorr - 1)

        if denom <= 0:
            return np.nan
        return np.sqrt(phi2corr / denom)
    except Exception:
        return np.nan


def correlation_ratio(categories, values):
    """Eta: how much of the numeric variable's variance is explained by
    grouping on the categorical variable. 0 = no association, 1 = perfect."""
    try:
        categories = np.asarray(categories)
        values = np.asarray(values, dtype=float)

        grand_mean = values.mean()
        ss_total = ((values - grand_mean) ** 2).sum()
        if ss_total == 0:
            return np.nan

        ss_between = 0.0
        for cat in np.unique(categories):
            group = values[categories == cat]
            ss_between += len(group) * (group.mean() - grand_mean) ** 2

        return np.sqrt(ss_between / ss_total)
    except Exception:
        return np.nan


def build_association_matrix(df, columns):
    """Symmetric matrix of pairwise associations across mixed column types."""
    n = len(columns)
    mat = pd.DataFrame(np.eye(n), index=columns, columns=columns)

    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(columns):
            if j <= i:
                continue

            is_num_i = pd.api.types.is_numeric_dtype(df[col_i])
            is_num_j = pd.api.types.is_numeric_dtype(df[col_j])

            if is_num_i and is_num_j:
                val = df[col_i].corr(df[col_j])
            elif is_num_i and not is_num_j:
                val = correlation_ratio(df[col_j], df[col_i])
            elif not is_num_i and is_num_j:
                val = correlation_ratio(df[col_i], df[col_j])
            else:
                val = cramers_v(df[col_i], df[col_j])

            mat.loc[col_i, col_j] = val
            mat.loc[col_j, col_i] = val

    return mat


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

    summary = df.describe(include="all")
    print(summary)
    summary.to_csv(output_dir / "descriptive_statistics.csv")

    # --------------------------------------------------
    # Numeric Features
    # --------------------------------------------------

    numeric_cols = df.select_dtypes(include=np.number).columns

    print("\nNumeric Columns")
    print(numeric_cols)

    # Histogram (raw)
    for col in numeric_cols:
        save_histogram(df, col)

    # Histogram (log1p) — skip Median_Price, it already gets a dedicated
    # target_distribution.png / log_target_distribution.png pair below.
    for col in numeric_cols:
        if col == "Median_Price":
            continue
        save_log_histogram(df, col)
        print(
            f"{col} skewness — raw: {df[col].skew():.2f}, "
            f"log1p: {np.log1p(df[col]).skew():.2f}"
        )

    # Boxplot
    for col in numeric_cols:
        save_boxplot(df, col)

    # --------------------------------------------------
    # Correlation Heatmap (numeric-only, Pearson)
    # --------------------------------------------------
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap (Numeric Features Only)")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
        plt.close()

    # --------------------------------------------------
    # Association Heatmap (all features: numeric + categorical)
    #
    # Township is excluded — same reasoning as the modeling pipeline: it's
    # a near-unique identifier per row, not a genuine categorical predictor.
    # --------------------------------------------------
    all_feature_cols = [c for c in df.columns if c != "Township"]
    print("\nBuilding all-feature association matrix for:", all_feature_cols)

    assoc_matrix = build_association_matrix(df, all_feature_cols)
    assoc_matrix.to_csv(output_dir / "all_features_association_matrix.csv")

    plt.figure(figsize=(9, 7))
    assoc_values = assoc_matrix.astype(float)
    mask = np.triu(np.ones_like(assoc_values, dtype=bool), k=1)
    sns.heatmap(
        assoc_values,
        mask=mask,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Association Strength"},
    )
    plt.title(
        "Association Strength — All Features\n(Pearson / Cramer's V / Correlation Ratio)"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "all_features_association_heatmap.png", dpi=300)
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
    # Area-State Relationship
    # --------------------------------------------------
    print("=" * 60)
    print("AREA-STATE RELATIONSHIP")
    print("=" * 60)

    mapping = df.groupby("Area")["State"].nunique()

    print("Maximum number of states per area:", mapping.max())

    if mapping.max() == 1:
        print("Each Area belongs to exactly one State.")
    else:
        print("Some Areas belong to multiple States.")

    area_counts = df["Area"].value_counts()

    print(area_counts.describe())

    print("\nAreas with fewer than 5 samples:", (area_counts < 5).sum())

    print("Percentage:", (area_counts < 5).mean() * 100)

    # --------------------------------------------------
    # Relationship Plot
    # --------------------------------------------------

    if "Transactions" in df.columns and "Median_Price" in df.columns:
        # -------------------------
        # Original Scatter Plot
        # -------------------------
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df, x="Transactions", y="Median_Price", alpha=0.5)
        plt.title("Transactions vs Median Price")
        plt.tight_layout()
        plt.savefig(output_dir / "transactions_vs_price.png", dpi=300)
        plt.close()

        # -------------------------
        # Log Scatter Plot
        # -------------------------
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=df, x="Transactions", y=np.log1p(df["Median_Price"]), alpha=0.5
        )
        plt.ylabel("Log(Median Price)")
        plt.title("Transactions vs Log(Median Price)")
        plt.tight_layout()
        plt.savefig(output_dir / "transactions_vs_price_log.png", dpi=300)
        plt.close()

    # --------------------------------------------------
    # Feature vs Target Analysis
    # --------------------------------------------------

    for col in ["State", "Tenure", "Type"]:

        if col not in df.columns:
            continue

        plot_df = df.copy()
        plot_col = col

        if col == "Type":
            counts = plot_df["Type"].value_counts()

            plot_df["Type_Plot"] = plot_df["Type"].where(
                plot_df["Type"].map(counts) >= 30, "Others"
            )

            plot_col = "Type_Plot"

        order = plot_df.groupby(plot_col)["Median_Price"].median().sort_values().index

        # -------------------------
        # Original Boxplot
        # -------------------------
        plt.figure(figsize=(12, 6))

        sns.boxplot(data=plot_df, x=plot_col, y="Median_Price", order=order)

        plt.xticks(rotation=45, ha="right")
        plt.title(f"{col} vs Median Price")
        plt.tight_layout()

        plt.savefig(output_dir / f"{col}_vs_price.png", dpi=300)

        plt.close()

        # -------------------------
        # Log Boxplot
        # -------------------------
        plt.figure(figsize=(12, 6))

        sns.boxplot(
            data=plot_df, x=plot_col, y=np.log1p(plot_df["Median_Price"]), order=order
        )

        plt.xticks(rotation=45, ha="right")
        plt.ylabel("log1p(Median Price)")
        plt.title(f"{col} vs Log(Median Price)")
        plt.tight_layout()

        plt.savefig(output_dir / f"{col}_vs_price_log.png", dpi=300)

        plt.close()

    # --------------------------------------------------
    # Outliers Count
    # --------------------------------------------------
    print("\nOutliers Count: ")
    for col in numeric_cols:
        n = count_outliers(df[col])
        print(f"{col}: {n} ({n / len(df):.2%})")

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
