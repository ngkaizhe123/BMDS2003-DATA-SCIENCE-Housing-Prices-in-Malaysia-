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


def split_dataset(df, categorical_features, numerical_features, target="Median_Price"):
    X = df[categorical_features + numerical_features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_preprocessor(numerical_features, categorical_features):
    numeric_transformer = Pipeline([("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        [
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
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
