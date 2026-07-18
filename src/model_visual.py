import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.data_preprocessing import run_preprocessing_pipeline


def main():
    data_path = project_root / 'data' / 'raw' / 'malaysia_house_price_data_2025.csv'
    model_path = project_root / 'prototype' / 'linear_regression.pkl'
    output_dir = project_root / 'report_assets' / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = run_preprocessing_pipeline(df)

    X = df[['Area', 'State', 'Tenure', 'Type', 'Transactions']]
    y = df['Median_Price']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Sort values to create a continuous "trend" line chart rather than scattered dots
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results_df = results_df.sort_values(by='Actual').reset_index(drop=True)

    plt.figure(figsize=(12, 6))

    # Plot Actual vs Predicted as overlapping lines mimicking a trading chart
    plt.plot(results_df.index, results_df['Actual'], color='indianred', alpha=0.7, label='Actual Price', linewidth=1.5)
    plt.plot(results_df.index, results_df['Predicted'], color='royalblue', alpha=0.8, label='Predicted Price',
             linewidth=1.5)

    # Add a vertical marker and badge mimicking the PDF's "Prediction Start Date"
    # We will use this to mark where luxury properties (top 20%) begin
    luxury_threshold_idx = int(len(results_df) * 0.8)
    luxury_price = results_df['Actual'].iloc[luxury_threshold_idx]

    plt.axvline(x=luxury_threshold_idx, color='#00b894', linestyle='-', linewidth=1.5, alpha=0.5)
    plt.text(luxury_threshold_idx, plt.ylim()[1] * 0.9, " Luxury Property Threshold ",
             color='white', backgroundcolor='#00b894', weight='bold', fontsize=10, ha='center')

    # Formatting to look highly professional
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xlabel('Test Samples (Sorted by Price)', fontsize=10)
    plt.ylabel('Price (RM)', fontsize=10)
    plt.title('Baseline Linear Regression: Actual vs Predicted Price Trend', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper left')

    # Light grid mimicking the trading chart background
    plt.grid(True, axis='y', linestyle='-', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'actual_vs_predicted_advanced.png', dpi=300)
    print("✅ Saved report_assets/plots/actual_vs_predicted_advanced.png")


if __name__ == "__main__":
    main()