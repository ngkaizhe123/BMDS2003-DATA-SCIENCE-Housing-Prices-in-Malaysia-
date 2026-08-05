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


def _remove_correlated_type_features(df, type_features, threshold=0.90):
    """Remove redundant Type_ columns that are highly correlated with each other.

    One-hot / multi-hot encoded type columns can be linearly dependent or
    near-duplicate, which introduces noise for distance-based models like SVR.
    This helper drops one column from every pair whose absolute Pearson
    correlation exceeds *threshold*.
    """
    if len(type_features) < 2:
        return df, type_features

    corr_matrix = df[type_features].corr().abs()

    # Walk the upper triangle and collect columns to drop
    to_drop = set()
    for i in range(len(type_features)):
        for j in range(i + 1, len(type_features)):
            if corr_matrix.iloc[i, j] > threshold:
                # Drop the column that appears later
                to_drop.add(type_features[j])

    if to_drop:
        print(f"[SVR] Dropping {len(to_drop)} highly-correlated type feature(s): "
              f"{sorted(to_drop)}")
        df = df.drop(columns=list(to_drop))

    remaining = [f for f in type_features if f not in to_drop]
    return df, remaining


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

    # =========================================================
    # Feature Selection — remove redundant / highly-correlated
    # type features to reduce noise and dimensionality for SVR.
    # =========================================================
    print("Removing highly-correlated type features...")
    df, type_features = _remove_correlated_type_features(
        df, type_features, threshold=0.90
    )

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
    # SelectKBest is placed after the preprocessor to perform dimensionality
    # reduction on the full encoded feature set — its k is tuned via CV.
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

    # =========================================================
    # Hyperparameter search space — tuned to combat overfitting
    # =========================================================
    # • C lowered (0.1–1.0): stronger regularisation penalises
    #   complexity and prevents fitting noise.
    # • epsilon widened (0.2–0.5): a broader insensitive tube
    #   makes the model less sensitive to individual training
    #   points, reducing overfitting.
    # • gamma: explicit small floats replace 'scale'/'auto' so
    #   the RBF kernel uses a less flexible decision boundary.
    # • feature_selection__k: lets CV pick how many features to
    #   keep, further controlling effective dimensionality.
    # =========================================================
    param_dist = {
        "regressor__regressor__kernel": ["rbf", "linear"],
        "regressor__regressor__C": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        "regressor__regressor__epsilon": [0.1, 0.15, 0.2, 0.25, 0.3],
        "regressor__regressor__gamma": [0.01, 0.05, 0.1, "scale"],
        "regressor__feature_selection__k": [5, 10, 15, 20, "all"],
    }

    # 5-fold cross-validation on the training split only — never on test data.
    # n_iter raised to 50 to better explore the regularised search space.
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
    #remove later
    gap=train_r2-test_r2
    print(f"Gap Test R\u00b2   : {gap:4f}")

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

    # Export trained model for the Streamlit deployment prototype
    save_model(best_model, model_output_path)


if __name__ == "__main__":
    main()
