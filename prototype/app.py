import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
    data_path = project_root / "data" / "processed" / "cleaned_malaysia_house_prices.csv"
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path)
    if "Area" not in df.columns:
        return None
    return df["Area"].value_counts()


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
    if hasattr(candidate, "regressor_"):        # fitted TransformedTargetRegressor
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
                ohe = transformer.named_steps.get("onehot") if hasattr(transformer, "named_steps") else transformer
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
        notes.append(f"Missing features '{col}', filled with 0 by default, but prediction may be less accurate.")

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
    is_tree_model = hasattr(regressor, "get_booster") or hasattr(regressor, "estimators_")
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

    default_state, default_area, default_tenure, default_type, default_tx, default_size = (
        "Selangor",
        "",
        "Freehold",
        "Terrace House",
        10,
        1200,
    )

    if preset_choice == "Sample: Luxury Condo in KL":
        default_state, default_area, default_tenure, default_type, default_tx, default_size = (
            "Kuala Lumpur",
            "Mont Kiara",
            "Freehold",
            "Condominium",
            45,
            1450,
        )
    elif preset_choice == "Sample: Standard Terrace in Johor":
        default_state, default_area, default_tenure, default_type, default_tx, default_size = (
            "Johor",
            "Skudai",
            "Freehold",
            "Terrace House",
            120,
            1800,
        )
    elif preset_choice == "Sample: Affordable Flat in Penang":
        default_state, default_area, default_tenure, default_type, default_tx, default_size = (
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

    tab1, tab2, tab3 = st.tabs(
        ["🎯 Single Prediction", "⚖️ Model Comparison", "📊 Model Analytics"]
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
        st.markdown("View training metrics and evaluation plots.")

        prototype_dir = Path(__file__).resolve().parent
        plots_dir = prototype_dir.parent / "report_assets" / "plots"

        plot_files = {
            "Linear Regression": plots_dir / "actual_vs_predicted_advanced.png",
            "XGBoost": plots_dir / "actual_vs_predicted_xgboost.png",
            "Random Forest": plots_dir / "actual_vs_predicted_randomforest.png",
            "SVR": plots_dir / "actual_vs_predicted_svr.png",
        }

        found_plot = False
        for name, path in plot_files.items():
            if path.exists():
                st.image(str(path), caption=f"Actual vs Predicted Values ({name})")
                found_plot = True

        if not found_plot:
            st.info(
                "Evaluation plots not found. Run your visualization scripts to generate them."
            )


if __name__ == "__main__":
    main()