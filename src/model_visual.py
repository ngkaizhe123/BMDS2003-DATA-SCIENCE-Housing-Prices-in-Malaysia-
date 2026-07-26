import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.data_preprocessing import run_preprocessing_pipeline
from src.utils import load_dataset, split_dataset, load_model


def plot_residual_analysis(
    model_filename="xgboost_regression.pkl", model_title="XGBoost"
):
    prototype_dir = project_root / "prototype"
    model_path = prototype_dir / model_filename
    output_dir = project_root / "report_assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(
            f"Error: Model file {model_path} does not exist. Please train the model first."
        )
        return

    # Load dataset & model
    df = load_dataset(project_root)
    df = run_preprocessing_pipeline(df)

    categorical_features = ["Area", "State", "Tenure"]
    if "Type" in df.columns:
        categorical_features.append("Type")
    numerical_features = ["Transactions"]
    type_features = [col for col in df.columns if col.startswith("Type_")]

    X_train, X_test, y_train, y_test = split_dataset(
        df, categorical_features, numerical_features, type_features
    )

    model = load_model(model_path)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred

    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------------
    # Figure 1: Actual vs Predicted (Train & Test Comparison)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(
        y_train,
        y_train_pred,
        color="dodgerblue",
        alpha=0.5,
        edgecolors="none",
        label=f"Train Data (R² = {model.score(X_train, y_train):.4f})",
    )
    ax.scatter(
        y_test,
        y_test_pred,
        color="crimson",
        alpha=0.7,
        edgecolors="k",
        linewidth=0.5,
        label=f"Test Data",
    )

    # Ideal 1:1 Reference Line (Perfect Prediction Line)
    max_val = max(y_train.max(), y_test.max())
    min_val = min(y_train.min(), y_test.min())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        lw=2,
        label="Ideal Prediction (y = x)",
    )

    ax.set_title(
        f"{model_title}: Actual vs. Predicted Prices",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Actual Price (RM)", fontsize=11)
    ax.set_ylabel("Predicted Price (RM)", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"RM {int(x):,}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"RM {int(x):,}"))
    ax.legend(loc="upper left")

    plt.tight_layout()
    act_vs_pred_path = (
        output_dir / f"actual_vs_predicted_{model_title.lower().replace(' ', '_')}.png"
    )
    plt.savefig(act_vs_pred_path, dpi=300)
    plt.close()
    print(f"✅ Saved {act_vs_pred_path}")

    # -------------------------------------------------------------
    # Figure 2: Residuals vs Predicted (Error Analysis & Heteroscedasticity)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        y_test_pred,
        residuals_test,
        color="crimson",
        alpha=0.6,
        edgecolors="k",
        linewidth=0.5,
        label="Test Residuals",
    )
    ax.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Zero Residual Line (e = 0)",
    )

    ax.set_title(
        f"{model_title}: Residuals vs. Predicted Values",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Predicted Price (RM)", fontsize=11)
    ax.set_ylabel("Residuals (Actual - Predicted) (RM)", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"RM {int(x):,}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"RM {int(x):,}"))
    ax.legend(loc="upper left")

    plt.tight_layout()
    residual_path = (
        output_dir
        / f"residuals_vs_predicted_{model_title.lower().replace(' ', '_')}.png"
    )
    plt.savefig(residual_path, dpi=300)
    plt.close()
    print(f"✅ Saved {residual_path}")


def main():
    print("Generating Residual Analysis for XGBoost Regression...")
    plot_residual_analysis("xgboost_regression.pkl", "XGBoost")


if __name__ == "__main__":
    main()
