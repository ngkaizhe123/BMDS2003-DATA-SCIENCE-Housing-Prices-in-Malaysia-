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
    build_preprocessor,
    print_metrics,
    save_model,
    save_metrics,
)




def main():
    model_output_path = project_root / "prototype" / "random_forest_regression.pkl"

    categorical_features = ["Area", "State", "Tenure"]
    # Removed Log_Estimated_Size to prevent KeyError, relying safely on Estimated_Size
    numerical_features = ["Transactions", "Estimated_Size"]

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
        "n_estimators": 750,
        "max_depth": 41,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "max_features": 0.4,
    }

    print("\nImplementing Strategy 3: 5-Fold Out-of-Fold Averaging...")
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Arrays to store the blended predictions
    test_predictions_sum = np.zeros(len(X_test))
    oof_train_predictions = np.zeros(len(X_train))

    # To save the best overall model for Streamlit
    models_list = []

    # 3. K-Fold Training Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"   -> Training Fold {fold + 1}/{k}...")

        # Split training data into fold-train and fold-validation
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Build fresh model for this specific fold
        fold_pipeline = TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", RandomForestRegressor(
                        random_state=42,
                        n_jobs=-1,  # Safe to use -1 here inside the fold
                        **best_params
                    )),
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        )

        # Fit the fold model
        fold_pipeline.fit(X_fold_train, y_fold_train)
        models_list.append(fold_pipeline)

        # Generate Out-Of-Fold predictions for the training set evaluation
        oof_train_predictions[val_idx] = fold_pipeline.predict(X_fold_val)

        # Predict on the unseen global Test Set and accumulate the votes
        test_predictions_sum += fold_pipeline.predict(X_test)

    # 4. Average the final test predictions across all 5 models
    final_test_predictions = test_predictions_sum / k

    # =============================
    # Evaluate Blended Performance
    # =============================
    print("\nEvaluating K-Fold Averaged Model...")

    train_r2 = r2_score(y_train, oof_train_predictions)
    test_r2 = r2_score(y_test, final_test_predictions)

    train_mae = mean_absolute_error(y_train, oof_train_predictions)
    test_mae = mean_absolute_error(y_test, final_test_predictions)

    train_rmse = np.sqrt(mean_squared_error(y_train, oof_train_predictions))
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
    # We save Fold 1's model as the prototype representative for the Streamlit app
    # (In high-end production, you would export all 5 and average them live,
    # but saving one is standard for a deployment prototype assignment).
    save_model(models_list[0], model_output_path)


if __name__ == "__main__":
    main()