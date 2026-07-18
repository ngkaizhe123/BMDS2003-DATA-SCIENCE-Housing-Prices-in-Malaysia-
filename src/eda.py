import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.data_preprocessing import run_preprocessing_pipeline


def main():
    data_path = project_root / 'data' / 'raw' / 'malaysia_house_price_data_2025.csv'
    output_dir = project_root / 'report_assets' / 'plots'
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    df_visual = run_preprocessing_pipeline(df)
    sns.set_theme(style="whitegrid")

    # Chart 1: Comprehensive Trends
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    fig.suptitle('Comprehensive Malaysia Housing Market Analysis', fontsize=16, fontweight='bold')

    state_price = df_visual.groupby('State')['Median_Price'].median().sort_values(ascending=False).head(10)
    sns.barplot(x=state_price.values, y=state_price.index, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Top 10 States by Median Price')

    type_vol = df_visual.groupby('Type')['Transactions'].sum().sort_values(ascending=False).head(8)
    sns.barplot(x=type_vol.values, y=type_vol.index, ax=axes[0, 1], palette='magma')
    axes[0, 1].set_title('Most Traded Property Types')

    sns.histplot(df_visual['Median_Price'], bins=50, kde=True, ax=axes[1, 0], color='teal')
    axes[1, 0].set_title('Distribution of Property Prices')

    sns.boxplot(data=df_visual, x='Tenure', y='Median_Price', ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_title('Price Comparison: Freehold vs Leasehold')

    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_market_trend.png', dpi=300)
    print("✅ Saved comprehensive_market_trend.png")

    # Chart 2: Boxplot Features
    plt.figure(figsize=(12, 6))
    top_types = df_visual['Type'].value_counts().head(8).index
    sns.boxplot(data=df_visual[df_visual['Type'].isin(top_types)], x='Median_Price', y='Type', orient='h')
    plt.title('Boxplot Analysis of Prices by Property Type')
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot_features.png', dpi=300)
    print("✅ Saved boxplot_features.png")


if __name__ == "__main__":
    main()