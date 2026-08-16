import json
from pathlib import Path
import sys
import textwrap
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.data_preprocessing import run_preprocessing_pipeline
from src.utils import load_raw_dataset, split_dataset


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data_path = project_root / "data" / "raw" / "malaysia_house_price_data_2025.csv"
    output_dir = project_root / "report_assets" / "plots"
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    print(df.describe())
    df_visual = run_preprocessing_pipeline(df)
    sns.set_theme(style="whitegrid")

    # Chart 1: Comprehensive Trends
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    fig.suptitle(
        "Comprehensive Malaysia Housing Market Analysis",
        fontsize=16,
        fontweight="bold",
    )

    state_price = (
        df_visual.groupby("State")["Median_Price"]
        .median()
        .sort_values(ascending=False)
        .head(10)
    )
    sns.barplot(
        x=state_price.values,
        y=state_price.index,
        ax=axes[0, 0],
        hue=state_price.index,
        palette="viridis",
        legend=False,
    )
    axes[0, 0].set_title("Top 10 States by Median Price")

    # Type is multi-hot encoded into Type_* columns — reconstruct long-form for plotting
    type_cols = [c for c in df_visual.columns if c.startswith("Type_")]
    type_vol = (
        df_visual[type_cols]
        .multiply(df_visual["Transactions"], axis=0)
        .sum()
        .rename(lambda c: c.replace("Type_", ""))
        .sort_values(ascending=False)
        .head(8)
    )
    sns.barplot(
        x=type_vol.values,
        y=type_vol.index,
        ax=axes[0, 1],
        hue=type_vol.index,
        palette="magma",
        legend=False,
    )
    axes[0, 1].set_title("Most Traded Property Types")

    sns.histplot(
        df_visual["Median_Price"], bins=50, kde=True, ax=axes[1, 0], color="teal"
    )
    axes[1, 0].set_title("Distribution of Property Prices")

    sns.boxplot(
        data=df_visual,
        x="Tenure",
        y="Median_Price",
        ax=axes[1, 1],
        hue="Tenure",
        palette="Set2",
        legend=False,
    )
    axes[1, 1].set_title("Price Comparison: Freehold vs Leasehold")

    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_market_trend.png", dpi=300)
    print("✅ Saved comprehensive_market_trend.png")

    # Chart 2: Boxplot Features
    # Melt multi-hot Type_* columns into a long-form DataFrame for boxplot
    type_cols = [c for c in df_visual.columns if c.startswith("Type_")]
    df_type_long = (
        df_visual[type_cols + ["Median_Price"]]
        .melt(id_vars="Median_Price", var_name="Type", value_name="flag")
        .query("flag == 1")
        .drop(columns="flag")
    )
    df_type_long["Type"] = df_type_long["Type"].str.replace("Type_", "", regex=False)
    top_types = df_type_long["Type"].value_counts().head(8).index
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_type_long[df_type_long["Type"].isin(top_types)],
        x="Median_Price",
        y="Type",
        orient="h",
    )
    plt.title("Boxplot Analysis of Prices by Property Type")
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_features.png", dpi=300)
    print("✅ Saved boxplot_features.png")

    # MODEL PERFORMANCE COMPARISONS
    metrics_path = project_root / "report_assets" / "metrics.json"
    if metrics_path.exists():
        print("Generating Model Comparison Charts...")
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

        df_metrics = pd.DataFrame(metrics_data).T
        wrapped_labels = [
            textwrap.fill(str(label), width=12) for label in df_metrics.index
        ]

        # Chart: R² Comparison (Train vs Test side-by-side)
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_metrics))
        width = 0.35
        bars_train = ax.bar(
            x - width / 2,
            df_metrics["train_r2"],
            width,
            label="Train R²",
            color="#74b9ff",
        )
        bars_test = ax.bar(
            x + width / 2,
            df_metrics["test_r2"],
            width,
            label="Test R²",
            color="#00b894",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(wrapped_labels, rotation=0, ha="center")
        ax.set_xlabel("Models")
        ax.set_ylabel("R² Score")
        ax.set_title(
            "Model Comparison - Train vs Test R² Score (Higher is Better)",
            fontweight="bold",
        )
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right")
        ax.axhline(0, color="gray", linewidth=0.8)

        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{height:.4f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=9,
                )

        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison_r2.png", dpi=300)
        plt.close()
        print("✅ Saved model_comparison_r2.png")

        # Chart: Test MAE Comparison
        if "test_mae" in df_metrics.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=wrapped_labels,
                y=df_metrics["test_mae"],
                hue=wrapped_labels,
                palette="Set2",
                legend=False,
                ax=ax,
            )
            ax.set_xlabel("Models")
            ax.set_ylabel("Test MAE (Log Scale)")
            ax.set_title(
                "Model Comparison - Test MAE (Log Scale, Lower is Better)",
                fontweight="bold",
            )
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.4f}"))
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(
                    f"{height:.4f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    xytext=(0, 5),
                    textcoords="offset points",
                )
            ax.tick_params(axis="x", rotation=0)
            ax.set_ylim(0, df_metrics["test_mae"].max() * 1.25)
            plt.tight_layout()
            plt.savefig(output_dir / "model_comparison_test_mae.png", dpi=300)
            plt.close()
            print("✅ Saved model_comparison_test_mae.png")

        # Chart: Test RMSE Comparison
        if "test_rmse" in df_metrics.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=wrapped_labels,
                y=df_metrics["test_rmse"],
                hue=wrapped_labels,
                palette="muted",
                legend=False,
                ax=ax,
            )
            ax.set_xlabel("Models")
            ax.set_ylabel("Test RMSE (Log Scale)")
            ax.set_title(
                "Model Comparison - Test RMSE (Log Scale, Lower is Better)",
                fontweight="bold",
            )
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.4f}"))
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(
                    f"{height:.4f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    xytext=(0, 5),
                    textcoords="offset points",
                )
            ax.tick_params(axis="x", rotation=0)
            ax.set_ylim(0, df_metrics["test_rmse"].max() * 1.25)
            plt.tight_layout()
            plt.savefig(output_dir / "model_comparison_test_rmse.png", dpi=300)
            plt.close()
            print("✅ Saved model_comparison_test_rmse.png")

        # Chart: Test MAPE Comparison
        if "test_mape" in df_metrics.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            mape_pct = df_metrics["test_mape"] * 100
            sns.barplot(
                x=wrapped_labels,
                y=mape_pct,
                hue=wrapped_labels,
                palette="coolwarm",
                legend=False,
                ax=ax,
            )
            ax.set_xlabel("Models")
            ax.set_ylabel("Test MAPE (%)")
            ax.set_title(
                "Model Comparison - Test MAPE (Lower is Better)",
                fontweight="bold",
            )
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.2f}%",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="bottom",
                    xytext=(0, 5),
                    textcoords="offset points",
                )
            ax.tick_params(axis="x", rotation=0)
            ax.set_ylim(0, (df_metrics["test_mape"].max() * 100) * 1.15)
            plt.tight_layout()
            plt.savefig(output_dir / "model_comparison_test_mape.png", dpi=300)
            plt.close()
            print("✅ Saved model_comparison_test_mape.png")

    # MODEL SELF-DIAGNOSTICS (TRAIN VS TEST CHARTS)
    print(
        "Generating individual model diagnostics (Actual vs Predicted & Residuals)..."
    )
    categorical_features = ["Area", "State", "Tenure"]
    numerical_features = [
        "Transactions",
        "Log_Estimated_Size",
        "Area_Transaction_Density",
    ]

    # Retrieve both training and testing datasets
    X_train, X_test, y_train, y_test = split_dataset(
        df_visual, categorical_features, numerical_features, type_cols
    )

    prototype_dir = project_root / "prototype"
    model_files = {
        "Multiple Linear Regression": "linear_regression.pkl",
        "Support Vector Regression": "svr_regression.pkl",
        "Random Forest": "random_forest_regression.pkl",
        "XGBoost": "xgboost_regression.pkl",
    }

    for model_name, filename in model_files.items():
        model_path = prototype_dir / filename
        if not model_path.exists():
            continue

        try:
            model = joblib.load(model_path)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        except Exception as e:
            print(f"Skipping {model_name} due to load error: {e}")
            continue

        # 1. Actual vs Predicted Chart
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(
            x=y_train,
            y=y_train_pred,
            alpha=0.5,
            color="dodgerblue",
            label="Training Data",
            ax=ax,
        )
        sns.scatterplot(
            x=y_test,
            y=y_test_pred,
            alpha=0.6,
            color="red",
            label="Test Data",
            ax=ax,
        )

        min_val = min(
            y_train.min(), y_test.min(), y_train_pred.min(), y_test_pred.min()
        )
        max_val = max(
            y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max()
        )
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            lw=2,
            label="Ideal Fit Line",
        )

        ax.ticklabel_format(style="plain", axis="both")
        ax.set_title(f"Actual vs Predicted ({model_name})", fontweight="bold")
        ax.set_xlabel("Actual Prices (RM)")
        ax.set_ylabel("Predicted Prices (RM)")
        ax.legend(loc="upper left")
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(output_dir / f"actual_vs_predicted_{safe_name}.png", dpi=300)
        plt.close()

        # 2. Residuals vs Predicted Chart
        residuals_test = y_test - y_test_pred

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            x=y_test_pred,
            y=residuals_test,
            alpha=0.6,
            color="crimson",
            label="Test Data",
            ax=ax,
        )

        ax.axhline(0, color="black", linestyle="--", lw=2, label="Zero Error Line")
        ax.set_title(f"Residual Plot ({model_name})", fontweight="bold")
        ax.set_xlabel("Predicted Prices (RM)")
        ax.set_ylabel("Residuals (Actual - Predicted)(RM)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(output_dir / f"residuals_vs_predicted_{safe_name}.png", dpi=300)
        plt.close()

    print(
        "✅ Saved all updated Actual vs Predicted and Residual diagnostic charts for all models."
    )


if __name__ == "__main__":
    main()
