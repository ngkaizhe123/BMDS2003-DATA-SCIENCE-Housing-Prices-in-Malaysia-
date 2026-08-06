import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler, TargetEncoder

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


# Early Stopping Callback Class
class EarlyStoppingCallback:
    def __init__(self, early_stopping_rounds: int):
        self.early_stopping_rounds = early_stopping_rounds
        self.best_score = -np.inf
        self.stagnant_trials = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        # Check if the study's best value improved
        if study.best_value > self.best_score:
            self.best_score = study.best_value
            self.stagnant_trials = 0  # Reset counter if a new high score is found
        else:
            self.stagnant_trials += 1  # Increase counter if no improvement

        # Halt the study if the limit is reached
        if self.stagnant_trials >= self.early_stopping_rounds:
            print(
                f"\n[Early Stopping] Halting search! No improvement over the last {self.early_stopping_rounds} trials."
            )
            study.stop()


def main():
    model_output_path = (
        project_root / "prototype" / "random_forest_regression(optuna).pkl"
    )

    categorical_features = ["Area", "State", "Tenure"]
    numerical_features = ["Transactions", "Log_Estimated_Size"]

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
            "n_estimators": trial.suggest_int("n_estimators", 600, 1100, step=50),
            "max_depth": trial.suggest_int("max_depth", 35, 45),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 7),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
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

    early_stop = EarlyStoppingCallback(early_stopping_rounds=20)

    study.optimize(
        objective, n_trials=200, show_progress_bar=True, callbacks=[early_stop]
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

    # 5. Print regression metrics (R², MAE, RMSE)
    print_metrics("Random Forest", y_test, y_pred)
    metrics_output_path = project_root / "report_assets" / "metrics.json"
    save_metrics(
        "Random Forest (Optuna)",
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
    print("-" * 30)

    # 6. Export trained model for your Streamlit deployment prototype
    save_model(best_model, model_output_path)


if __name__ == "__main__":
    main()
