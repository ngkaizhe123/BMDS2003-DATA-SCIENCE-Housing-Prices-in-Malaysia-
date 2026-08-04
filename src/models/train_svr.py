import numpy as np
import sys
from pathlib import Path
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
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
    # NOTE: SVR is distance-based, so scaling numeric features (handled inside
    # build_preprocessor via StandardScaler)
    print("Building preprocessing pipelines...")
    preprocessor = build_preprocessor(
        numerical_features, categorical_features, type_features
    )

    print("Training Support Vector Regression model...")
    # Wrap SVR in TransformedTargetRegressor to handle the right-skewed target
    # variable, consistent with the other models in this project.
    model_pipeline = TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", SVR()),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    # Hyperparameter search space for SVR
    param_dist = {
        "regressor__regressor__kernel": ["rbf", "linear", "poly"],
        "regressor__regressor__C": [0.1, 1, 10, 50, 100],
        "regressor__regressor__epsilon": [0.01, 0.05, 0.1, 0.2],
        "regressor__regressor__gamma": ["scale", "auto"],
    }

    print("Implementing RandomizedSearchCV for hyperparameter tuning...")
    search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_dist,
        n_iter=20,
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

    # Print regression metrics (R2, MAE, RMSE)
    print_metrics("Support Vector Regression", y_test, y_pred)

    print(f"Best parameters found: {search.best_params_}")
    print(f"Best Model R2 Score: {search.best_score_:.4f}")
    print("-" * 30)

    # Export trained model for the Streamlit deployment prototype
    save_model(best_model, model_output_path)


if __name__ == "__main__":
    main()
