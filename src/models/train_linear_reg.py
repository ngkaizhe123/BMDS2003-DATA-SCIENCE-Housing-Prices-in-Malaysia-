import os
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Point python path cleanly to src
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root.parent / 'src'))
sys.path.append(str(project_root))

from src.data_preprocessing import run_preprocessing_pipeline

def main():
    data_path = project_root / 'data' / 'processed' / 'cleaned_malaysia_house_prices.csv'
    model_output_dir = project_root / 'prototype'
    model_output_path = model_output_dir / 'linear_regression.pkl'

    print("Loading dataset...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        return

    print("Running data preprocessing pipeline...")
    df = run_preprocessing_pipeline(df)

    target = 'Median_Price'
    categorical_features = ['Area', 'State', 'Tenure', 'Type']
    numerical_features = ['Transactions']

    X = df[categorical_features + numerical_features]
    y = df[target]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building preprocessing pipelines...")
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    print("Training Advanced Multiple Linear Regression model...")
    # Wrap the linear regression inside a log-transformer to handle skewed house prices
    model_pipeline = TransformedTargetRegressor(
        regressor=Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
        func=np.log1p,
        inverse_func=np.expm1
    )

    model_pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("-" * 30)
    print("Model Performance Metrics:")
    print(f"MAE:  RM {mae:,.2f}")
    print(f"RMSE: RM {rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)

    model_output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, model_output_path)
    print(f"Model saved successfully to {model_output_path}. Ready for deployment!")

if __name__ == "__main__":
    main()