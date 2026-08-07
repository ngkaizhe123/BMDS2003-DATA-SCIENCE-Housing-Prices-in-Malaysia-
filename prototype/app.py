import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. CONFIGURATION & CORE LAYOUT
st.set_page_config(page_title="Housing Model Predictor", page_icon="🏠", layout="wide")


# =========================================================
# 2. BACKEND SERVICE LAYER
# =========================================================


@st.cache_resource
def load_models():
    """Loads all available models from the prototype directory."""
    prototype_dir = Path(__file__).resolve().parent
    models = {}

    lr_path = prototype_dir / "linear_regression.pkl"
    if lr_path.exists():
        models["Multiple Linear Regression"] = joblib.load(lr_path)

    svr_path = prototype_dir / "svr_regression.pkl"
    if svr_path.exists():
        models["Support Vector Regression"] = joblib.load(svr_path)

    xgb_path = prototype_dir / "xgboost_regression.pkl"
    if xgb_path.exists():
        models["XGBoost Regression"] = joblib.load(xgb_path)

    rf_path = prototype_dir / "random_forest_regression.pkl"
    if rf_path.exists():
        models["Random Forest Regression"] = joblib.load(rf_path)

    return models


@st.cache_data
def load_area_frequencies():
    """Area counts from the cleaned training data, used to warn users when a
    typed-in Area is rare or unseen in training."""
    project_root = Path(__file__).resolve().parent.parent
    data_path = (
        project_root / "data" / "processed" / "cleaned_malaysia_house_prices.csv"
    )
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path)
    if "Area" not in df.columns:
        return None
    return df["Area"].value_counts()


@st.cache_data
def load_raw_data():
    """Raw (pre-preprocessing) dataset, used by the EDA tab."""
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "malaysia_house_price_data_2025.csv"
    if not data_path.exists():
        return None
    return pd.read_csv(data_path)


def show_image_if_exists(path: Path, caption: str = None):
    """Display a saved plot if it exists, otherwise show a clear message
    instead of silently failing or crashing the tab."""
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.caption(
            f"⚠️ Plot not found: `{path.name}`. Run `python src/eda.py` to generate it."
        )


def _count_iqr_outliers(series: pd.Series) -> int:
    """Mirrors the IQR outlier-counting logic used in src/eda.py."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


# ---------------------------------------------------------
# Pipeline introspection helpers
#
# These walk into the fitted model (which may be wrapped in a
# TransformedTargetRegressor, then a Pipeline, then a ColumnTransformer)
# to recover exactly what the model expects, instead of hardcoding
# feature lists that can silently drift out of sync with training code.
# ---------------------------------------------------------


def _unwrap_column_transformer(model):
    """Return the fitted ColumnTransformer inside a model pipeline, if any."""
    candidate = model
    if hasattr(candidate, "regressor_"):  # fitted TransformedTargetRegressor
        candidate = candidate.regressor_
    elif hasattr(candidate, "regressor"):
        candidate = candidate.regressor

    if hasattr(candidate, "named_steps"):
        for step in candidate.named_steps.values():
            if hasattr(step, "transformers_") and hasattr(step, "feature_names_in_"):
                return step

    if hasattr(candidate, "transformers_") and hasattr(candidate, "feature_names_in_"):
        return candidate

    return None


def _unwrap_regressor(model):
    """Return the fitted final estimator (e.g. XGBRegressor, RandomForestRegressor)."""
    candidate = model
    if hasattr(candidate, "regressor_"):
        candidate = candidate.regressor_
    if hasattr(candidate, "named_steps"):
        return candidate.named_steps.get("regressor")
    return candidate


def get_expected_columns(model):
    """The exact raw input columns the model's preprocessor was fit on."""
    ct = _unwrap_column_transformer(model)
    if ct is not None and hasattr(ct, "feature_names_in_"):
        return list(ct.feature_names_in_)
    return None


def _get_onehot_categories(model, raw_col_name):
    """Known categories for a one-hot-encoded raw column (e.g. 'Area'), so we
    can detect when a user's typed-in value was never seen during training."""
    ct = _unwrap_column_transformer(model)
    if ct is None:
        return None
    try:
        for _, transformer, cols in ct.transformers_:
            cols = list(cols)
            if raw_col_name in cols:
                ohe = (
                    transformer.named_steps.get("onehot")
                    if hasattr(transformer, "named_steps")
                    else transformer
                )
                if hasattr(ohe, "categories_"):
                    idx = cols.index(raw_col_name)
                    return set(ohe.categories_[idx])
    except Exception:
        return None
    return None


def resolve_area_category(model, area_name, state_name):
    """
    Training buckets rare Areas into 'Other_<State>'. Mirror that here so a
    typed-in Area the model never saw doesn't silently get zeroed out by
    OneHotEncoder(handle_unknown='ignore') without the user knowing.

    Returns (resolved_area, was_remapped).
    """
    known_areas = _get_onehot_categories(model, "Area")
    if known_areas is None:
        return area_name, False  # can't verify — pass through as-is

    if area_name in known_areas:
        return area_name, False

    fallback = f"Other_{state_name}"
    if fallback in known_areas:
        return fallback, True

    return area_name, True  # still unresolved, encoder will zero it out


# ---------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------


def prepare_input_features(raw_input: dict, expected_columns: list):
    """
    Build a single-row DataFrame matching exactly what the trained
    preprocessor expects (pulled live from the model via `feature_names_in_`),
    so this never drifts out of sync with the training script again.

    Returns (df, notes) — notes describes any columns that had to be
    auto-derived or defaulted.
    """
    notes = []
    row = {}

    # Direct raw fields (Area, State, Tenure, Transactions, ...)
    for col in expected_columns:
        if col in raw_input:
            row[col] = raw_input[col]

    # Type_* multi-hot columns, derived from the raw "Type" selection
    type_cols = [c for c in expected_columns if c.startswith("Type_")]
    if type_cols and "Type" in raw_input:
        selected_types = [t.strip() for t in str(raw_input["Type"]).split(",")]
        for col in type_cols:
            raw_type_name = col.replace("Type_", "")
            row[col] = 1 if raw_type_name in selected_types else 0

    # Any remaining expected column: try to auto-derive Log_X from X,
    # otherwise fall back to a safe default and flag it.
    for col in expected_columns:
        if col in row:
            continue
        if col.startswith("Log_"):
            base_col = col.replace("Log_", "")
            if base_col in raw_input:
                row[col] = float(np.log1p(raw_input[base_col]))
                continue
        row[col] = 0
        notes.append(
            f"Missing features '{col}', filled with 0 by default, but prediction may be less accurate."
        )

    df = pd.DataFrame([row])[expected_columns]
    return df, notes


def predict_price(model, raw_input: dict):
    """Predict price and return (prediction, notes) where notes are
    user-facing caveats about the prediction's reliability."""
    expected_columns = get_expected_columns(model)
    if expected_columns is None:
        raise ValueError(
            "Unable to read expected feature columns from the model (missing feature_names_in_)."
            "Please confirm the model is trained and saved using a more recent version of the pipeline."
        )

    notes = []
    working_input = dict(raw_input)

    if "Area" in working_input and "State" in working_input:
        resolved_area, remapped = resolve_area_category(
            model, working_input["Area"], working_input["State"]
        )
        if remapped:
            notes.append(
                f"'{working_input['Area']}' not in training data's known areas, "
                f"reverted to using '{resolved_area}' (state-level estimate), prediction accuracy may be compromised."
            )
        working_input["Area"] = resolved_area

    df, fill_notes = prepare_input_features(working_input, expected_columns)
    notes.extend(fill_notes)

    pred = float(model.predict(df)[0])
    return pred, notes


def area_reliability_note(area_freq, area_name, threshold=5):
    """Extra warning based on how many training rows actually back this Area."""
    if area_freq is None:
        return None
    count = int(area_freq.get(area_name, 0))
    if count == 0:
        return None  # already covered by resolve_area_category's note
    if count < threshold:
        return f"Note: Training data only has {count} records for '{area_name}', prediction stability may be compromised."
    return None


# ---------------------------------------------------------
# SHAP explanation (tree models only: XGBoost, Random Forest)
# ---------------------------------------------------------


def explain_tree_prediction(model, raw_input: dict):
    """Best-effort SHAP waterfall for tree-based models. Returns a matplotlib
    figure, or None if unavailable (missing shap, or non-tree model)."""
    try:
        import shap
    except ImportError:
        return None

    regressor = _unwrap_regressor(model)
    is_tree_model = hasattr(regressor, "get_booster") or hasattr(
        regressor, "estimators_"
    )
    if regressor is None or not is_tree_model:
        return None

    ct = _unwrap_column_transformer(model)
    expected_columns = get_expected_columns(model)
    if ct is None or expected_columns is None:
        return None

    working_input = dict(raw_input)
    if "Area" in working_input and "State" in working_input:
        resolved_area, _ = resolve_area_category(
            model, working_input["Area"], working_input["State"]
        )
        working_input["Area"] = resolved_area

    df, _ = prepare_input_features(working_input, expected_columns)

    try:
        X_transformed = ct.transform(df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        feature_names = ct.get_feature_names_out()

        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_transformed)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_transformed[0],
                feature_names=feature_names,
            ),
            show=False,
        )
        plt.tight_layout()
        return fig
    except Exception:
        return None


def plot_market_comparison(state, prop_type, predicted_price, title="Market Benchmark"):
    prototype_dir = Path(__file__).resolve().parent
    project_root = prototype_dir.parent
    data_path = project_root / "data" / "raw" / "malaysia_house_price_data_2025.csv"

    if not data_path.exists():
        st.warning("Raw dataset file missing. Unable to render market comparison plot.")
        return

    df = pd.read_csv(data_path)
    state_data = df[df["State"] == state]
    if state_data.empty:
        return

    medians = (
        state_data.groupby("Type")["Median_Price"].median().sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x=medians.index, y=medians.values, ax=ax, color="lightgray")

    if prop_type in medians.index:
        x_pos = list(medians.index).index(prop_type)

        ax.scatter(x_pos, predicted_price, color="#00b894", s=250, marker="*", zorder=5)
        ax.axhline(y=predicted_price, color="#00b894", linestyle="--", alpha=0.8)

        ax.text(
            x_pos + 0.2,
            predicted_price,
            f" Predicted: RM {predicted_price:,.0f} ",
            color="white",
            backgroundcolor="#00b894",
            weight="bold",
            fontsize=10,
            va="center",
        )

    ax.set_title(
        f"{title}: Where your property stands in {state}",
        pad=20,
        fontsize=14,
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


def render_input_form(form_key):
    """Renders the common property feature input form."""
    st.info(
        "💡 **Tip:** Use a preset below to auto-fill the form, or enter your own custom details."
    )
    preset_choice = st.selectbox(
        "Auto Input Data (Optional):",
        [
            "Custom Input (Manual)",
            "Sample: Luxury Condo in KL",
            "Sample: Standard Terrace in Johor",
            "Sample: Affordable Flat in Penang",
        ],
        key=f"preset_{form_key}",
    )

    (
        default_state,
        default_area,
        default_tenure,
        default_type,
        default_tx,
        default_size,
    ) = (
        "Selangor",
        "",
        "Freehold",
        "Terrace House",
        10,
        1200,
    )

    if preset_choice == "Sample: Luxury Condo in KL":
        (
            default_state,
            default_area,
            default_tenure,
            default_type,
            default_tx,
            default_size,
        ) = (
            "Kuala Lumpur",
            "Mont Kiara",
            "Freehold",
            "Condominium",
            45,
            1450,
        )
    elif preset_choice == "Sample: Standard Terrace in Johor":
        (
            default_state,
            default_area,
            default_tenure,
            default_type,
            default_tx,
            default_size,
        ) = (
            "Johor",
            "Skudai",
            "Freehold",
            "Terrace House",
            120,
            1800,
        )
    elif preset_choice == "Sample: Affordable Flat in Penang":
        (
            default_state,
            default_area,
            default_tenure,
            default_type,
            default_tx,
            default_size,
        ) = (
            "Penang",
            "Ayer Itam",
            "Leasehold",
            "Flat",
            35,
            750,
        )

    user_features = None
    with st.form(form_key):
        st.subheader("Property Features")
        col1, col2 = st.columns(2)

        with col1:
            state_options = [
                "Kuala Lumpur",
                "Selangor",
                "Johor",
                "Penang",
                "Perak",
                "Negeri Sembilan",
                "Melaka",
                "Kedah",
                "Pahang",
                "Terengganu",
                "Kelantan",
                "Perlis",
                "Sabah",
                "Sarawak",
            ]
            state = st.selectbox(
                "State", options=state_options, index=state_options.index(default_state)
            )
            area = st.text_input(
                "Area / Township",
                value=default_area,
                placeholder="e.g., Cheras, Skudai",
            )
            tenure = st.selectbox(
                "Tenure",
                options=["Freehold", "Leasehold", "Freehold and Leasehold"],
                index=(
                    0
                    if default_tenure == "Freehold"
                    else (1 if default_tenure == "Leasehold" else 2)
                ),
            )

        with col2:
            type_options = [
                "Terrace House",
                "Condominium",
                "Apartment",
                "Semi D",
                "Bungalow",
                "Service Residence",
                "Flat",
                "Cluster House",
                "Town House",
            ]
            prop_type = st.selectbox(
                "Property Type",
                options=type_options,
                index=type_options.index(default_type),
            )
            transactions = st.number_input(
                "Number of Transactions", min_value=1, value=default_tx
            )
            estimated_size = st.number_input(
                "Estimated Built-up Size (sqft)",
                min_value=200,
                max_value=20000,
                value=default_size,
                step=50,
                help="Estimated built-up size of the property (square feet). This is a manual estimate, not an exact measurement.",
            )
            st.caption(
                "⚠️ This value is a manual estimate, and the model was trained using area derived from prices. "
                "There may be differences between the manual estimate and the training distribution, and the prediction should be taken as a reference only. "
            )

        submitted = st.form_submit_button("Predict Price", use_container_width=True)

        if submitted:
            if not area.strip():
                st.warning("Please enter an Area or Township name before submitting.")
            else:
                user_features = {
                    "Area": area.strip().title(),
                    "State": state,
                    "Tenure": tenure,
                    "Type": prop_type,
                    "Transactions": transactions,
                    "Estimated_Size": estimated_size,
                }

    return user_features


# =========================================================
# 3. FRONTEND UI & CONTROLLER LAYOUT
# =========================================================


def main():
    st.title("🏠 Malaysia Housing Price Predictor")

    models = load_models()
    area_freq = load_area_frequencies()

    if not models:
        st.error(
            "No models detected in `/prototype`. Please run your training scripts first."
        )
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎯 Single Prediction",
            "⚖️ Model Comparison",
            "📊 Model Analytics",
            "📈 EDA",
        ]
    )

    # --- TAB 1: SINGLE PREDICTION ---
    with tab1:
        st.markdown("Select a model and enter details below to generate a prediction.")

        selected_model_name = st.selectbox("Select Model to Use:", list(models.keys()))
        selected_model = models[selected_model_name]

        user_features = render_input_form("single_prediction_form")

        if user_features:
            try:
                pred, notes = predict_price(selected_model, user_features)

                st.success("Analysis Complete!")
                st.metric(
                    label=f"Estimated Median Price ({selected_model_name})",
                    value=f"RM {pred:,.2f}",
                )

                area_note = area_reliability_note(area_freq, user_features["Area"])
                if area_note:
                    notes.append(area_note)

                for note in notes:
                    st.warning(note)

                st.markdown("---")
                plot_market_comparison(
                    user_features["State"],
                    user_features["Type"],
                    pred,
                    title=f"Market Benchmark ({selected_model_name})",
                )

                shap_fig = explain_tree_prediction(selected_model, user_features)
                if shap_fig is not None:
                    with st.expander("🔍 Why? (SHAP Explanation)"):
                        st.caption(
                            "Values represent the contribution of each feature in the log price space, positive values represent an increase in the predicted price, and negative values represent a decrease."
                        )
                        st.pyplot(shap_fig)

            except Exception as e:
                st.error(f"Inference pipeline issue: {e}")

    # --- TAB 2: MODEL COMPARISON ---
    with tab2:
        st.subheader("Model Comparison Tool")
        st.markdown(
            "Run the same property features through all available models to compare their predictions side-by-side."
        )

        if len(models) < 2:
            st.info(
                "You need at least 2 models trained to compare. Currently only one model is loaded."
            )

        comp_features = render_input_form("comparison_form")

        if comp_features:
            st.markdown("### Prediction Results")
            cols = st.columns(len(models))

            predictions = {}
            all_notes = {}
            for idx, (m_name, m_obj) in enumerate(models.items()):
                try:
                    pred, notes = predict_price(m_obj, comp_features)
                    predictions[m_name] = pred
                    all_notes[m_name] = notes
                    cols[idx].metric(label=m_name, value=f"RM {pred:,.2f}")
                except Exception as e:
                    cols[idx].error(f"Error with {m_name}: {e}")

            flat_notes = sorted({n for notes in all_notes.values() for n in notes})
            for note in flat_notes:
                st.warning(note)

            if predictions:
                st.markdown("---")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(
                    x=list(predictions.keys()),
                    y=list(predictions.values()),
                    ax=ax,
                    palette="Set2",
                    legend=False,
                    hue=list(predictions.keys()),
                )
                ax.set_title("Model Prediction Comparison", fontweight="bold")
                ax.set_ylabel("Predicted Price (RM)")
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: format(int(x), ","))
                )

                for p in ax.patches:
                    ax.annotate(
                        f"RM {p.get_height():,.0f}",
                        (p.get_x() + p.get_width() / 2.0, p.get_height()),
                        ha="center",
                        va="bottom",
                        xytext=(0, 5),
                        textcoords="offset points",
                    )

                st.pyplot(fig)

    # --- TAB 3: MODEL ANALYTICS ---
    with tab3:
        st.subheader("Model Evaluation")
        st.markdown("Compare training vs test performance across all trained models.")

        current_path = Path(__file__).resolve().parent
        project_root = current_path.parent
        metrics_path = project_root / "report_assets" / "metrics.json"
        plots_dir = project_root / "report_assets" / "plots"

        if not metrics_path.exists():
            st.info(
                "No `metrics.json` found. "
                "Please call `save_metrics(model_name, metrics_dict, prototype_dir / 'metrics.json')` to generate and display the comparison results."
                "Here it will automatically generate and display the comparison results."
            )
        else:
            with open(metrics_path, "r") as f:
                all_metrics = json.load(f)

            if not all_metrics:
                st.info(
                    "`metrics.json` is empty, no models have been written to metrics yet."
                )
            else:
                metrics_df = pd.DataFrame(all_metrics).T
                metrics_df.index.name = "Model"

                required_cols = [
                    "train_r2",
                    "test_r2",
                    "train_mae",
                    "test_mae",
                    "train_rmse",
                    "test_rmse",
                    "train_mape",
                    "test_mape",
                ]
                for col in required_cols:
                    if col not in metrics_df.columns:
                        metrics_df[col] = np.nan

                metrics_df["gap_r2"] = metrics_df["train_r2"] - metrics_df["test_r2"]

                st.markdown("### Metrics Comparison Table")
                display_df = metrics_df.copy()
                display_df["train_r2"] = display_df["train_r2"].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )
                display_df["test_r2"] = display_df["test_r2"].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )
                display_df["gap_r2"] = display_df["gap_r2"].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )

                for col in ["train_mae", "test_mae", "train_rmse", "test_rmse"]:
                    display_df[col] = display_df[col].map(
                        lambda x: f"RM {x:,.0f}" if pd.notna(x) else "—"
                    )

                for col in ["train_mape", "test_mape"]:
                    display_df[col] = display_df[col].map(
                        lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "—"
                    )

                display_df = display_df.rename(
                    columns={
                        "train_r2": "Train R²",
                        "test_r2": "Test R²",
                        "gap_r2": "Gap Test R²",
                        "train_mae": "Train MAE",
                        "test_mae": "Test MAE",
                        "train_rmse": "Train RMSE",
                        "test_rmse": "Test RMSE",
                        "train_mape": "Train MAPE",
                        "test_mape": "Test MAPE",
                    }
                )

                display_cols = [
                    "Train R²",
                    "Test R²",
                    "Gap Test R²",
                    "Train MAPE",
                    "Test MAPE",
                    "Train MAE",
                    "Test MAE",
                    "Train RMSE",
                    "Test RMSE",
                ]
                display_df = display_df[display_cols]

                st.dataframe(display_df, use_container_width=True)

                st.markdown("### R² Comparison (Train vs Test)")
                show_image_if_exists(
                    plots_dir / "model_comparison_r2.png",
                    "Train vs Test R² Comparison"
                )

                st.markdown("### Test Error Comparison")
                show_image_if_exists(
                    plots_dir / "model_comparison_errors.png",
                    "Test MAE & RMSE Comparison"
                )

                st.markdown("### Actual vs Predicted Plots")
                actual_vs_pred_files = sorted(list(plots_dir.glob("actual_vs_predicted_*.png")))
                if actual_vs_pred_files:
                    for plot_file in actual_vs_pred_files:
                        model_title = plot_file.stem.replace("actual_vs_predicted_", "").replace("_", " ").title()
                        st.markdown(f"#### Actual vs Predicted ({model_title})")
                        show_image_if_exists(plot_file, f"Actual vs Predicted - {model_title}")
                else:
                    st.caption("No Actual vs Predicted plots found.")

                st.markdown("### Residuals vs Predicted Plots")
                residuals_files = sorted(list(plots_dir.glob("residuals_vs_predicted_*.png")))
                if residuals_files:
                    for plot_file in residuals_files:
                        model_title = plot_file.stem.replace("residuals_vs_predicted_", "").replace("_", " ").title()
                        st.markdown(f"#### Residuals vs Predicted ({model_title})")
                        show_image_if_exists(plot_file, f"Residuals vs Predicted - {model_title}")
                else:
                    st.caption("No Residuals vs Predicted plots found.")

                st.markdown("### Other Model Assessment Plots")
                other_plots = [
                    f for f in sorted(list(plots_dir.glob("*.png")))
                    if not f.name.startswith("actual_vs_predicted_")
                    and not f.name.startswith("residuals_vs_predicted_")
                    and f.name not in ["model_comparison_r2.png", "model_comparison_errors.png"]
                ]
                if other_plots:
                    for plot_file in other_plots:
                        title = plot_file.stem.replace("_", " ").title()
                        st.markdown(f"#### {title}")
                        show_image_if_exists(plot_file, title)

    # --- TAB 4: EDA ---
    with tab4:
        st.subheader("Exploratory Data Analysis")
        st.markdown(
            "Visualizations of the **raw dataset**, before preprocessing or feature "
            "engineering. Plots are generated by `src/eda.py` — run that script first "
            "if any plot below shows as missing."
        )

        current_path = Path(__file__).resolve().parent
        project_root = current_path.parent
        eda_dir = project_root / "report_assets" / "plots" / "eda"
        raw_df = load_raw_data()

        if raw_df is None:
            st.warning(
                "Raw dataset not found at `data/raw/malaysia_house_price_data_2025.csv`."
            )
        else:
            eda_tabs = st.tabs(
                [
                    "Overview",
                    "Target Variable",
                    "Numeric Features",
                    "Categorical Features",
                    "Feature vs Price",
                    "Correlation",
                ]
            )

            # ---------------- Overview ----------------
            with eda_tabs[0]:
                st.markdown("#### Dataset Shape & Quality")
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", f"{raw_df.shape[0]:,}")
                c2.metric("Columns", raw_df.shape[1])
                c3.metric("Duplicate Rows", int(raw_df.duplicated().sum()))

                missing = raw_df.isnull().sum()
                missing = missing[missing > 0]
                if len(missing) > 0:
                    st.markdown("Missing values were found in the raw data:")
                    show_image_if_exists(
                        eda_dir / "missing_values.png", "Missing Values by Column"
                    )
                else:
                    st.success("No missing values in the raw dataset.")

                st.markdown("#### Descriptive Statistics")
                desc_path = eda_dir / "descriptive_statistics.csv"
                if desc_path.exists():
                    st.dataframe(
                        pd.read_csv(desc_path, index_col=0), use_container_width=True
                    )
                else:
                    st.dataframe(
                        raw_df.describe(include="all"), use_container_width=True
                    )

            # ---------------- Target Variable ----------------
            with eda_tabs[1]:
                st.markdown("#### Median Price Distribution")
                if "Median_Price" in raw_df.columns:
                    skew_raw = raw_df["Median_Price"].skew()
                    skew_log = np.log1p(raw_df["Median_Price"]).skew()

                    c1, c2 = st.columns(2)
                    c1.metric("Skewness (raw)", f"{skew_raw:.2f}")
                    c2.metric("Skewness (log1p)", f"{skew_log:.2f}")

                    st.caption(
                        "Skewness far from 0 indicates a long tail. Housing prices are typically "
                        "right-skewed (a few very expensive properties pull the mean up), which is "
                        "why the target is log-transformed (`log1p`) before training, and predictions "
                        "are converted back with `expm1`."
                    )

                    img_c1, img_c2 = st.columns(2)
                    with img_c1:
                        show_image_if_exists(
                            eda_dir / "target_distribution.png", "Median Price (raw)"
                        )
                    with img_c2:
                        show_image_if_exists(
                            eda_dir / "log_target_distribution.png",
                            "Median Price (log1p)",
                        )
                else:
                    st.warning("`Median_Price` column not found in raw data.")

            # ---------------- Numeric Features ----------------
            with eda_tabs[2]:
                st.markdown("#### Numeric Feature Distributions")
                numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()

                if not numeric_cols:
                    st.info("No numeric columns found.")
                else:
                    for col in numeric_cols:
                        st.markdown(f"**{col}**")
                        n_outliers = _count_iqr_outliers(raw_df[col])
                        pct = n_outliers / len(raw_df) * 100
                        st.caption(
                            f"{n_outliers} potential outliers detected via the IQR rule "
                            f"({pct:.1f}% of rows)."
                        )
                        img_c1, img_c2 = st.columns(2)
                        with img_c1:
                            show_image_if_exists(
                                eda_dir / f"{col}_distribution.png",
                                f"{col} Distribution",
                            )
                        with img_c2:
                            show_image_if_exists(
                                eda_dir / f"{col}_boxplot.png", f"{col} Boxplot"
                            )
                        st.markdown("---")

            # ---------------- Categorical Features ----------------
            with eda_tabs[3]:
                st.markdown("#### Categorical Feature Distributions")
                st.caption(
                    "Top 15 most frequent categories are shown for readability; rarer "
                    "categories are omitted from the plot but still counted below."
                )
                categorical_cols = raw_df.select_dtypes(
                    include="object"
                ).columns.tolist()

                for col in categorical_cols:
                    n_unique = raw_df[col].nunique()
                    st.markdown(f"**{col}** — {n_unique} unique categories")

                    if n_unique > 50:
                        rare_pct = (raw_df[col].value_counts() < 5).mean() * 100
                        st.caption(
                            f"⚠️ High-cardinality column: {rare_pct:.1f}% of categories have "
                            f"fewer than 5 samples each. This sparsity can hurt model "
                            f"generalization for unseen or rarely-seen categories."
                        )

                    show_image_if_exists(
                        eda_dir / f"{col}_countplot_top15.png",
                        f"{col} — Top 15 Categories",
                    )
                    st.markdown("---")

            # ---------------- Feature vs Price ----------------
            with eda_tabs[4]:
                st.markdown("#### How Each Feature Relates to Price")

                st.markdown("**Transactions vs Median Price**")
                img_c1, img_c2 = st.columns(2)
                with img_c1:
                    show_image_if_exists(
                        eda_dir / "transactions_vs_price.png", "Transactions vs Price"
                    )
                with img_c2:
                    show_image_if_exists(
                        eda_dir / "transactions_vs_price_log.png",
                        "Transactions vs Log(Price)",
                    )
                st.markdown("---")

                for col in ["State", "Tenure", "Type"]:
                    if col not in raw_df.columns:
                        continue
                    st.markdown(f"**{col} vs Median Price**")
                    img_c1, img_c2 = st.columns(2)
                    with img_c1:
                        show_image_if_exists(
                            eda_dir / f"{col}_vs_price.png", f"{col} vs Price"
                        )
                    with img_c2:
                        show_image_if_exists(
                            eda_dir / f"{col}_vs_price_log.png", f"{col} vs Log(Price)"
                        )
                    st.markdown("---")

            # ---------------- Correlation ----------------
            with eda_tabs[5]:
                st.markdown("#### Correlation Between Numeric Features")
                show_image_if_exists(
                    eda_dir / "correlation_heatmap.png", "Correlation Heatmap"
                )

                if "Median_Price" in raw_df.columns and "Median_PSF" in raw_df.columns:
                    corr_val = raw_df["Median_Price"].corr(raw_df["Median_PSF"])
                    st.metric(
                        "Correlation: Median_Price vs Median_PSF", f"{corr_val:.3f}"
                    )
                    st.warning(
                        "This high correlation is expected, since `Median_PSF` is derived "
                        "from the same transactions as `Median_Price` (price per square "
                        "foot). This is exactly why `Median_PSF` — and any feature "
                        "reverse-engineered from it — is excluded from model training as a "
                        "target-leakage risk (see report methodology section)."
                    )


if __name__ == "__main__":
    main()
