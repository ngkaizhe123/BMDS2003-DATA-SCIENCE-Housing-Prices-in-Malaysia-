import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import optuna
from optuna.samplers import TPESampler

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
    model_output_path = (
        project_root / "prototype" / "random_forest_regression(optuna).pkl"
    )

    categorical_features = ["Area", "State", "Tenure"]
    numerical_features = [
        "Transactions",
        "Estimated_Size",
        "Log_Estimated_Size",
        "Log_Transactions",
    ]

    # 1. Load raw dataset and run preprocessing pipeline ONCE
    df = load_raw_dataset(project_root)
    print("Running data preprocessing pipeline...")
    df = run_preprocessing_pipeline(df)

    type_features = [col for col in df.columns if col.startswith("Type_")]
    print("-" * 45)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_dataset(
        df, categorical_features, numerical_features, type_features
    )

    # 2. Build preprocessor matching the existing pipeline setup
    print("Building preprocessing pipelines...")
    preprocessor = build_preprocessor(
        numerical_features, categorical_features, type_features
    )

    # 3. Define Optuna objective function for hyperparameter tuning
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 25, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
            "max_features": trial.suggest_float("max_features", 0.25, 0.55),
        }

        # Build pipeline with trial parameters
        model_pipeline = TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", RandomForestRegressor(random_state=42, **params)),
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        )

        # Use cross-validation score as Optuna objective to maximize
        score = cross_val_score(
            model_pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
        ).mean()

        return score

    print("Implementing Optuna for parameter tuning (Fine Search)...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        objective,
        n_trials=100,
        show_progress_bar=True,
    )

    print(f"\nBest parameters found: {study.best_params}")
    print(f"Best CV R2 Score     : {study.best_value:.4f}")
    print("-" * 30)

    # 4. Retrain best model on full training set using best params from Optuna
    print("Training Random Forest Regression model with best parameters...")
    best_params = study.best_params
    best_model = TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(random_state=42, n_jobs=-1, **best_params),
                ),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    print("Fitting best model on full training set...")
    best_model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = best_model.predict(X_test)

    # 5. Print regression metrics (R², MAE, RMSE)
    print_metrics("Random Forest", y_test, y_pred)
    print("-" * 30)

    # 6. Export trained model for your Streamlit deployment prototype
    save_model(best_model, model_output_path)


if __name__ == "__main__":
    main()
