import numpy as np
import sys
from pathlib import Path
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_regression

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
    model_output_path = project_root / "prototype" / "svr_regression.pkl"

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

    print("Training Support Vector Regression model...")
    # Wrap SVR in TransformedTargetRegressor to handle the right-skewed target variable
    # SelectKBest placed after the preprocessor to perform dimensionality
    model_pipeline = TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("feature_selection", SelectKBest(score_func=f_regression)),
                ("regressor", SVR()),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    # Hyperparameter search space
    param_dist = {
        "regressor__regressor__kernel": ["rbf", "linear"],
        "regressor__regressor__C": [0.5, 1, 2,5,10,20],
        "regressor__regressor__epsilon": [0.01, 0.05, 0.1, 0.15, 0.2],
        "regressor__regressor__gamma": ["scale", 0.0005,0.001, 0.002, 0.005, 0.01, 0.02],
        "regressor__feature_selection__k": [15, 18, 20, "all"],
    }

    # 5-fold cross-validation on the training split only
    print("Implementing RandomizedSearchCV for hyperparameter tuning (5-fold CV)...")
    search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
    )

    print("Fitting SVR with RandomizedSearchCV...")
    search.fit(X_train, y_train)
    print("Evaluating model...")
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

    print(f"Train R\u00b2  : {train_r2:.4f}")
    print(f"Test R\u00b2   : {test_r2:.4f}")

    # Print regression metrics (R2, MAE, RMSE)
    print_metrics("Support Vector Regression", y_test, y_pred)
    metrics_output_path = project_root / "report_assets" / "metrics.json"
    save_metrics(
        "Support Vector Regression",
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
    print(f"Best Model R2 Score: {search.best_score_:.4f}")
    print("-" * 30)

    save_model(best_model, model_output_path)


if __name__ == "__main__":
    main()
