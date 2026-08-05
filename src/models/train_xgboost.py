import numpy as np
import sys
from pathlib import Path

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
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
    save_model, save_metrics,
)


def main():
    model_output_path = project_root / "prototype" / "xgboost_regression.pkl"

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
    # This effectively drops Township and Median_PSF, preventing data leakage
    X_train, X_test, y_train, y_test = split_dataset(
        df, categorical_features, numerical_features, type_features
    )

    # Build preprocessor using utils
    print("Building preprocessing pipelines...")
    preprocessor = build_preprocessor(
        numerical_features, categorical_features, type_features
    )

    print("Training XGBoost Regression model...")
    # Wrap XGBRegressor in TransformedTargetRegressor to handle the right-skewed target variable
    model_pipeline = TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    XGBRegressor(
                        random_state=42, n_jobs=-1, objective="reg:squarederror"
                    ),
                ),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    param_dist = {
        "regressor__regressor__n_estimators": [200, 300, 400, 500],
        "regressor__regressor__max_depth": [4, 5, 6],
        "regressor__regressor__learning_rate": [0.03, 0.05, 0.08],
        "regressor__regressor__subsample": [0.7, 0.8, 0.9],
        "regressor__regressor__colsample_bytree": [0.7, 0.8, 1.0],
        "regressor__regressor__min_child_weight": [1, 2, 3, 5],
        "regressor__regressor__reg_alpha": [0, 0.1, 0.5, 1.0],
        "regressor__regressor__reg_lambda": [1.0, 3.0, 5.0, 10.0],
        "regressor__regressor__gamma": [0, 0.1, 0.2],
    }

    print("Implementing RandomizedSearchCV for hyperparameter tuning...")
    search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
    )
    print("Fitting the model with RandomizedSearchCV...")
    search.fit(X_train, y_train)
    print("Evaluating model...")

    # y_pred = model_pipeline.predict(X_test)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    # =============================
    # Train vs Test Performance
    # =============================

    y_train_pred = best_model.predict(X_train)

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

    print(f"Train R²  : {train_r2:.4f}")
    print(f"Test R²   : {test_r2:.4f}")

    # Print metrics using utils
    print_metrics("XGBoost", y_test, y_pred)
    metrics_output_path = project_root / "report_assets" / "metrics.json"
    save_metrics(
        "XGBoost",
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

    print(f"Best parameters found: {search.best_params_}")
    print(f"Best Model R2 Score: {search.best_score_}")
    print("-" * 30)

    # Save model using utils
    save_model(best_model, model_output_path)


if __name__ == "__main__":
    main()
