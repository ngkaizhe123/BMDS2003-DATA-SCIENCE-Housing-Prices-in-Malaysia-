import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
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
    load_processed_dataset,
    remove_outliers_train_only,
    split_dataset,
    build_preprocessor,
    print_metrics,
    save_model,
)


def main():
    model_output_path = project_root / "prototype" / "random_forest_regression.pkl"

    categorical_features = ["Area", "State", "Tenure"]
    numerical_features = [
        "Log_Estimated_Size",
        "Transactions",
    ]

    # 1. Load & clean row-by-row
    df = load_processed_dataset(project_root)
    type_features = [col for col in df.columns if col.startswith("Type_")]

    # 2. Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_dataset(
        df, categorical_features, numerical_features, type_features
    )

    # 3. Remove outliers ONLY from the training set
    print("Removing outliers from training data...")
    X_train, y_train = remove_outliers_train_only(X_train, y_train, ["Log_Transactions"])

    # 4. Build preprocessor (now handles medians and rare categories)
    print("Building preprocessing pipelines...")
    preprocessor = build_preprocessor(
        numerical_features, categorical_features, type_features
    )

    # 5. Wrap RandomForestRegressor in TransformedTargetRegressor to handle target skewness
    model_pipeline = TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(random_state=42)),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    # 6. Define hyperparameter distribution for tuning (CLO1 & CLO3 Rubric Requirement)
    param_dist = {
        "regressor__regressor__n_estimators": [400, 700, 600],
        "regressor__regressor__max_depth": [10, 20, 30, None],
        "regressor__regressor__min_samples_split": [2, 5, 10],
        "regressor__regressor__min_samples_leaf": [1, 2],
        "regressor__regressor__max_features": ["sqrt", "log2", 1, 0.3, 0.5],
    }

    print("Implementing RandomizedSearchCV for parameter tuning...")
    search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
    )

    print("Fitting Random Forest with RandomizedSearchCV...")
    search.fit(X_train, y_train)

    print("Evaluating model...")
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    # 7. Print regression metrics ($R^2$, MAE, RMSE)
    print_metrics("Random Forest", y_test, y_pred)

    print(f"Best parameters found: {search.best_params_}")
    print(f"Best Model R2 Score: {search.best_score_:.4f}")
    print("-" * 30)

    # 8. Export trained model for your Streamlit deployment prototype
    save_model(best_model, model_output_path)


if __name__ == "__main__":
    main()
