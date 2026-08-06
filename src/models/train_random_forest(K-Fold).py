import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
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
    print_metrics,
    save_model,
    save_metrics,
)

def build_preprocessor(numerical_features, categorical_features, type_features):
    numeric_transformer = Pipeline([("scaler", StandardScaler())])

    # This replaces One-Hot Encoding! It converts "Area" into a single, smooth numerical value.
    categorical_transformer = Pipeline(
        [("target_enc", TargetEncoder(smooth="auto", cv=5))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            ("type", "passthrough", type_features),
        ]
    )


def main():
    model_output_path = project_root / "prototype" / "random_forest_regression.pkl"

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

    print("Building preprocessing pipelines...")
    preprocessor = build_preprocessor(
        numerical_features, categorical_features, type_features
    )

    # 2. Hardcode the Optimal Parameters
    best_params = {
        "n_estimators": 850,
        "max_depth": 20,
        "min_samples_split": 10,
        "min_samples_leaf": 2,
        "max_features": 0.60,
    }

    print("\nImplementing Strategy 3: 5-Fold Out-of-Fold Averaging...")
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Arrays to store the blended predictions
    test_predictions_sum = np.zeros(len(X_test))
    train_predictions_sum = np.zeros(len(X_train))

    models_list = []

    # 3. K-Fold Training Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"   -> Training Fold {fold + 1}/{k}...")

        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]

        # Build fresh model for this specific fold
        fold_pipeline = TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", RandomForestRegressor(
                        random_state=42,
                        n_jobs=-1,
                        **best_params
                    )),
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        )

        fold_pipeline.fit(X_fold_train, y_fold_train)
        models_list.append(fold_pipeline)

        # Let this fold model vote on the ENTIRE training set and Test set
        train_predictions_sum += fold_pipeline.predict(X_train)
        test_predictions_sum += fold_pipeline.predict(X_test)

    # 4. Average the final predictions across all 5 models
    final_train_predictions = train_predictions_sum / k
    final_test_predictions = test_predictions_sum / k

    # =============================
    # Evaluate Blended Performance
    # =============================
    print("\nEvaluating K-Fold Averaged Model...")

    train_r2 = r2_score(y_train, final_train_predictions)
    test_r2 = r2_score(y_test, final_test_predictions)

    train_mae = mean_absolute_error(y_train, final_train_predictions)
    test_mae = mean_absolute_error(y_test, final_test_predictions)

    train_rmse = np.sqrt(mean_squared_error(y_train, final_train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, final_test_predictions))

    print("\nTrain vs Test Performance (5-Fold Blended)")
    print("-" * 40)
    print(f"Train MAE : RM {train_mae:,.2f}")
    print(f"Test MAE  : RM {test_mae:,.2f}")
    print(f"Train RMSE: RM {train_rmse:,.2f}")
    print(f"Test RMSE : RM {test_rmse:,.2f}")
    print(f"Train R²  : {train_r2:.4f}")
    print(f"Test R²   : {test_r2:.4f}")
    print(f"Final Gap : {train_r2 - test_r2:.4f}")

    # 5. Save metrics and export model
    print_metrics("Random Forest (K-Fold)", y_test, final_test_predictions)

    metrics_output_path = project_root / "report_assets" / "metrics.json"
    save_metrics(
        "Random Forest",
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
    save_model(models_list[0], model_output_path)


if __name__ == "__main__":
    main()