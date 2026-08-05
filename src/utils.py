import json
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    # Rare states that slipped through preprocessing (e.g. Putrajaya with 1 row)
    # will crash stratify — so we fall back to a normal split in that case
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
    r2 = r2_score(y_true, y_pred)

    # 2. Calculate Log-Transformed Metrics (The "0.something" illusion)
    # We apply the natural logarithm to both actual and predicted prices
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)

    log_mae = mean_absolute_error(y_true_log, y_pred_log)
    log_rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))

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
    print(f"R² : {r2:.4f}")
    print("-" * 45)
    print("LOG-TRANSFORMED METRICS:")
    print(f"Log MAE: {log_mae:.4f}")
    print(f"Log RMSE: {log_rmse:.4f}")
    print(f"Log R²: {log_r2:.4f}")
    print("-" * 45)

    return mae, rmse, r2


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
def save_metrics(model_name: str, metrics: dict, output_path: Path):
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