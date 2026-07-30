import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, MinMaxScaler

# Point python path cleanly to src
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root.parent / "src"))
sys.path.append(str(project_root))

from src.data_preprocessing import run_preprocessing_pipeline
from src.utils import (
    load_dataset,
    split_dataset,
    print_metrics,
    save_model,
)


def main():
    model_output_path = project_root / "prototype" / "linear_regression.pkl"

    categorical_features = ["Area", "State", "Tenure"]
    numerical_features = ["Transactions"]

    # Load, preprocess and split dataset using utils
    df = load_dataset(project_root)
    print("Running data preprocessing pipeline...")
    df = run_preprocessing_pipeline(df)
    type_features = [col for col in df.columns if col.startswith("Type_")]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_dataset(
        df, categorical_features, numerical_features, type_features
    )

    # Build preprocessor using utils
    print("Building preprocessing pipelines...")
    # Custom numeric_transformer for Linear Regression only:
    # PolynomialFeatures adds Transactions^2 so the model can learn
    # non-linear relationships, then StandardScaler normalizes the result
    numeric_transformer = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
        ("type", "passthrough", type_features),
    ])

    # Scale target (Median_Price) to 0-1 range so RMSE and MAE are
    # reported as 0.something instead of raw RM values
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    print("Training Advanced Multiple Linear Regression model...")
    # Wrap the linear regression inside a log-transformer to handle skewed house prices
    model_pipeline = TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model_pipeline.fit(X_train, y_train_scaled)

    print("Evaluating model...")
    y_pred_scaled = model_pipeline.predict(X_test)

    # Print metrics on scaled target → shows 0.something RMSE and MAE
    print_metrics("Linear Regression (Scaled 0-1)", y_test_scaled, y_pred_scaled)

    # Convert predictions back to real RM for actual use in Streamlit app
    y_pred_real = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Print metrics in real RM for reference
    print_metrics("Linear Regression (Real RM)", y_test, y_pred_real)

    # Save model using utils
    save_model(model_pipeline, model_output_path)


if __name__ == "__main__":
    main()