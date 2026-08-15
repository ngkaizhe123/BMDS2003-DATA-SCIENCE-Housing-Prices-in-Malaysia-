import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_preprocessing import run_preprocessing_pipeline
from src.utils import load_raw_dataset, split_dataset

# --------------------------------------------------
# Config
# --------------------------------------------------
CATEGORICAL_FEATURES = ["Area", "State", "Tenure"]
# Superset of numerical features across all 4 trained pipelines.
# Each model's own ColumnTransformer selects only the columns it was
# trained on by name, so passing this superset is safe for every model.
NUMERICAL_FEATURES = ["Transactions", "Log_Estimated_Size", "Area_Transaction_Density"]

MODEL_FILES = {
    "Multiple Linear Regression": "linear_regression.pkl",
    "Support Vector Regression": "svr_regression.pkl",
    "Random Forest": "random_forest_regression.pkl",
    "XGBoost": "xgboost_regression.pkl",
}


# --------------------------------------------------
# Core plotting function
# --------------------------------------------------
def plot_price_trend(model, model_name, X_test, y_test):
    """
    Build the Actual vs Predicted Price Trend line chart for a single
    fitted model and return the matplotlib Figure (caller decides whether
    to save it, show it, or hand it to Streamlit's st.pyplot()).
    """
    y_pred = model.predict(X_test)

    # Sort test samples by actual price so the "Actual Price" line forms
    # a smooth ascending trend; predictions are reordered to match.
    order = np.argsort(y_test.values)
    actual_sorted = y_test.values[order]
    predicted_sorted = np.asarray(y_pred)[order]
    sample_index = np.arange(len(actual_sorted))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(sample_index, actual_sorted, color="#E88B8B", linewidth=1.5, label="Actual Price")
    ax.plot(sample_index, predicted_sorted, color="#5B7FE0", linewidth=1, label="Predicted Price")

    ax.set_title(f"{model_name}: Actual vs Predicted Price Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Test Samples (Sorted by Price)")
    ax.set_ylabel("Price (RM)")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    return fig


# --------------------------------------------------
# Generate for all 4 models
# --------------------------------------------------
def generate_all_price_trends(prototype_dir: Path = None, df_visual: pd.DataFrame = None):
    """
    Load each trained model from the prototype/ directory, rebuild the same
    train/test split used during training, and return a dict mapping
    model_name -> matplotlib Figure, one entry per model that has a saved
    .pkl available.
    """
    prototype_dir = prototype_dir or (project_root / "prototype")

    if df_visual is None:
        df_raw = load_raw_dataset(project_root)
        df_visual = run_preprocessing_pipeline(df_raw)

    type_features = [col for col in df_visual.columns if col.startswith("Type_")]

    X_train, X_test, y_train, y_test = split_dataset(
        df_visual, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, type_features
    )

    figures = {}
    for model_name, filename in MODEL_FILES.items():
        model_path = prototype_dir / filename
        if not model_path.exists():
            print(f"Skipping {model_name}: {model_path} not found.")
            continue
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Skipping {model_name} due to load error: {e}")
            continue

        figures[model_name] = plot_price_trend(model, model_name, X_test, y_test)

    return figures


# --------------------------------------------------
# Standalone entry point: save PNGs to report_assets/plots
# --------------------------------------------------
def main():
    # Saved flat alongside the other model_visual.py outputs (not a
    # subfolder) so app.py's existing plots_dir.glob("*.png") discovery
    # in the Model Analytics tab picks these up automatically.
    output_dir = project_root / "report_assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = generate_all_price_trends()

    for model_name, fig in figures.items():
        safe_name = model_name.lower().replace(" ", "_")
        out_path = output_dir / f"price_trend_{safe_name}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")

    if not figures:
        print("No trained models found in prototype/. Run the training scripts first.")


if __name__ == "__main__":
    main()