import json
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_raw_dataset(project_root):
    data_path = project_root / "data" / "raw" / "malaysia_house_price_data_2025.csv"

    print("Loading raw dataset...")
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Raw dataset not found at {data_path}")


def load_dataset(project_root):
    data_path = (
        project_root / "data" / "processed" / "cleaned_malaysia_house_prices.csv"
    )

    print("Loading processed dataset...")
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {data_path}")


def split_dataset(
    df, categorical_features, numerical_features, type_features, target="Median_Price"
):
    X = df[categorical_features + numerical_features + type_features]
    y = df[target]

    # Only stratify by State if every state has at least 2 rows
    state_counts = df["State"].value_counts()
    rare_states = state_counts[state_counts < 2].index

    if len(rare_states) == 0:
        # All states have enough rows — safe to stratify
        return train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=df["State"]
        )
    else:
        # Rare states found — skip stratify to avoid crash
        print(
            f"Warning: Skipping stratify — rare states found with <2 rows: {list(rare_states)}"
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)


def build_preprocessor(numerical_features, categorical_features, type_features):
    numeric_transformer = Pipeline([("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        [
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            ("type", "passthrough", type_features),
        ]
    )


def print_metrics(model_name, y_true, y_pred):
    # 1. Calculate Real-World Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 2. Calculate Log-Transformed Metrics (The "0.something" illusion)
    # We apply the natural logarithm to both actual and predicted prices
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)

    log_mae = mean_absolute_error(y_true_log, y_pred_log)
    log_rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    log_mape = mean_absolute_percentage_error(y_true_log, y_pred_log)

    # Note: R-squared usually remains identical or very similar because it measures
    # variance explained, which scales proportionally.
    log_r2 = r2_score(y_true_log, y_pred_log)

    # 3. Display the comparison cleanly
    print("-" * 45)
    print(f"{model_name} Performance")
    print("-" * 45)
    print("REAL-WORLD METRICS (Actual Ringgit):")
    print(f"MAE: RM {mae:,.2f}")
    print(f"RMSE: RM {rmse:,.2f}")
    print(f"MAPE: {mape * 100:.2f}%")
    print(f"R² : {r2:.4f}")
    print("-" * 45)
    print("LOG-TRANSFORMED METRICS:")
    print(f"Log MAE: {log_mae:.4f}")
    print(f"Log RMSE: {log_rmse:.4f}")
    print(f"Log MAPE: {log_mape * 100:.2f}%")
    print(f"Log R²: {log_r2:.4f}")
    print("-" * 45)

    return mae, rmse, r2, mape


def save_model(model, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    print(f"Model saved to {output_path}")


def prepare_input_features(input_dict: dict) -> pd.DataFrame:
    """Formats single-property user inputs into the exact DataFrame structure expected by trained models."""
    df = pd.DataFrame([input_dict])

    all_type_cols = [
        "Type_Apartment",
        "Type_Bungalow",
        "Type_Cluster House",
        "Type_Condominium",
        "Type_Flat",
        "Type_Semi D",
        "Type_Service Residence",
        "Type_Terrace House",
        "Type_Town House",
    ]

    if "Type" in df.columns:
        selected_type = str(df["Type"].iloc[0])
        selected_types = [t.strip() for t in selected_type.split(",")]

        for col in all_type_cols:
            raw_type_name = col.replace("Type_", "")
            df[col] = 1 if raw_type_name in selected_types else 0

        df = df.drop(columns=["Type"])
    else:
        for col in all_type_cols:
            if col not in df.columns:
                df[col] = 0

    return df


# for models analysis and comparison
def evaluate_train_test_performance(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Compute Train vs Test metrics, print the summary block, and return a
    metrics dict where MAE and RMSE are stored in log1p scale and MAPE is
    stored as raw percentage."""
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_pred)

    # Calculate log-transformed MAE and RMSE
    train_mae_log = mean_absolute_error(np.log1p(y_train), np.log1p(y_train_pred))
    test_mae_log = mean_absolute_error(np.log1p(y_test), np.log1p(y_pred))
    train_rmse_log = np.sqrt(
        mean_squared_error(np.log1p(y_train), np.log1p(y_train_pred))
    )
    test_rmse_log = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred)))

    print("\nTrain vs Test Performance")
    print("-" * 40)
    print(f"Train Log MAE : {train_mae_log:.4f}")
    print(f"Test Log MAE  : {test_mae_log:.4f}")
    print(f"Train Log RMSE: {train_rmse_log:.4f}")
    print(f"Test Log RMSE : {test_rmse_log:.4f}")
    print(f"Train MAPE: {train_mape * 100:.2f}%")
    print(f"Test MAPE : {test_mape * 100:.2f}%")
    print(f"Train R\u00b2  : {train_r2:.4f}")
    print(f"Test R\u00b2   : {test_r2:.4f}")
    print(f"Final Gap : {train_r2 - test_r2:.4f}")

    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": float(train_mae_log),
        "test_mae": float(test_mae_log),
        "train_rmse": float(train_rmse_log),
        "test_rmse": float(test_rmse_log),
        "train_mape": float(train_mape),
        "test_mape": float(test_mape),
    }, y_pred


def save_metrics(model_name: str, metrics: dict, output_path: Path):
    """Persist a model's metrics dict to the shared metrics.json file.

    MAE and RMSE values are in log scale; MAPE is a plain ratio; R² is as-is.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[model_name] = metrics

    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Metrics for {model_name} saved to {output_path}")
