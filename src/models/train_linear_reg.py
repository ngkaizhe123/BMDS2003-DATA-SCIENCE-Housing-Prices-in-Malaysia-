import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

# Point python path cleanly to src
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root.parent / "src"))
sys.path.append(str(project_root))

from src.data_preprocessing import run_preprocessing_pipeline
from src.utils import (
    load_raw_dataset,
    split_dataset,
    build_preprocessor,
    print_metrics,
    save_model,
    save_metrics,
)


def main():
    model_output_path = project_root / "prototype" / "linear_regression.pkl"

    categorical_features = ["Area", "State", "Tenure"]
    numerical_features = [
        "Transactions",
        "Log_Estimated_Size",
    ]

    # Load raw dataset and run preprocessing pipeline ONCE
    df = load_raw_dataset(project_root)
    print("Running data preprocessing pipeline...")
    df = run_preprocessing_pipeline(df)
    type_features = [col for col in df.columns if col.startswith("Type_")]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_dataset(
        df, categorical_features, numerical_features, type_features
    )

    # Build preprocessor using utils
    print("Building preprocessing pipelines...")
    preprocessor = build_preprocessor(
        numerical_features, categorical_features, type_features
    )

    print("Training Advanced Multiple Linear Regression model...")
    # Wrap the linear regression inside a log-transformer to handle skewed house prices
    model_pipeline = TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model_pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)

    # =============================
    # Train vs Test Performance
    # =============================

    y_train_pred = model_pipeline.predict(X_train)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\nTrain vs Test Performance")
    print("-" * 40)

    print(f"Train MAE : RM {train_mae:,.2f}")
    print(f"Test MAE  : RM {test_mae:,.2f}")

    print(f"Train RMSE: RM {train_rmse:,.2f}")
    print(f"Test RMSE : RM {test_rmse:,.2f}")

    print(f"Train R\u00b2  : {train_r2:.4f}")
    print(f"Test R\u00b2   : {test_r2:.4f}")

    # Print metrics using utils
    print_metrics("Linear Regression", y_test, y_pred)
    metrics_output_path = project_root / "report_assets" / "metrics.json"
    save_metrics(
        "Linear Regression",
        {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
        },
        metrics_output_path,
    )

    # Save model using utils
    save_model(model_pipeline, model_output_path)


if __name__ == "__main__":
    main()
