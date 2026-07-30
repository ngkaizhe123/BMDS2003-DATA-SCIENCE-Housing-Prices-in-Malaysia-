from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_dataset(project_root):
    data_path = (
        project_root / "data" / "processed" / "cleaned_malaysia_house_prices.csv"
    )

    print("Loading dataset...")
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {data_path}")


def split_dataset(
    df, categorical_features, numerical_features, type_features, target="Median_Price"
):
    X = df[categorical_features + numerical_features + type_features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=df["State"])


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
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("-" * 30)
    print(f"{model_name} Performance")
    print(f"MAE : RM {mae:,.2f}")
    print(f"RMSE: RM {rmse:,.2f}")
    print(f"R²  : {r2:.4f}")
    print("-" * 30)

    return mae, rmse, r2


def save_model(model, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    print(f"Model saved to {output_path}")


def load_model(path):
    return joblib.load(path)


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
