# Malaysia Housing Prices Prediction

An Data Science project designed to analyze and predict housing prices in Malaysia using real estate datasets. This
project explores data preprocessing, exploratory data analysis (EDA), feature engineering, and compares multiple
regression models to deliver an accurate price forecasting system.

Built as a collaborative data science assignment by a team of 4 members.

---

## 📂 Project Structure

This project adopts a modular production-ready Python architecture to ensure smooth collaboration and version control.

```text
Housing-Prices-Malaysia/
│
├── data/                      # Data storage (Git ignored large files)
│   ├── raw/                   # Raw CSV dataset from source
│   └── processed/             # Cleaned and engineered dataset ready for modeling
│
├── src/                       # Core source code
│   ├── __init__.py            # Treats src as a Python package
│   ├── eda.py                 # Exploratory Data Analysis script
│   ├── data_preprocessing.py  # Script for handling missing values, outliers, & encoding
│   ├── utils.py               # Shared utility functions (e.g., standardized evaluation metrics)
│   ├── model_visual.py       # Data visualization script
│   │
│   └── models/                     # Model training scripts (one per member)
│       ├── __init__.py
│       ├── train_linear_reg.py     # Baseline: Multiple Linear Regression
│       ├── train_xgboost.py        # Model 2: XGBoost Regressor
│       ├── train_random_forest.py  # Model 3: Random Forest Regressor
│       └── train_svr.py            # Model 4: Support Vector Regression (SVR)
│
├── prototype/                 # Compulsory Deployment Prototype
│   ├── app.py                 # Streamlit web application
│   └── best_model.pkl         # Serialized weights of the top-performing model
│
├── report_assets/             # Visualizations for the written report
│   └── plots/                 # Auto-generated charts (EDA, Feature Importance, etc.)
│
├── .gitignore                 # Specifies intentionally untracked files to ignore
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

```

---

## 🚀 How to Run the Streamlit Prototype

Follow these steps to set up a clean Python 3.12 virtual environment, install dependencies, and launch the Streamlit web application.

### Prerequisites
* **Python 3.12** installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/ngkaizhe123/BMDS2003-DATA-SCIENCE-Housing-Prices-in-Malaysia-.git
```

### 2. Set Up a Python 3.12 Virtual Environment
```bash
py -3.12 -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run prototype/app.py
```
