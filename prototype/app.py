import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. CONFIGURATION & CORE LAYOUT
st.set_page_config(page_title="Housing Model Predictor", page_icon="🏠", layout="wide")


# 2. BACKEND SERVICE LAYER (Internal Functions)
@st.cache_resource
def load_models():
    """Loads all available models from the prototype directory."""
    prototype_dir = Path(__file__).resolve().parent
    models = {}

    lr_path = prototype_dir / "linear_regression.pkl"
    if lr_path.exists():
        models["Multiple Linear Regression"] = joblib.load(lr_path)

    xgb_path = prototype_dir / "xgboost_regression.pkl"
    if xgb_path.exists():
        models["XGBoost Regression"] = joblib.load(xgb_path)

    rf_path = prototype_dir / "random_forest_regression.pkl"
    if rf_path.exists():
        models["Random Forest Regression"] = joblib.load(rf_path)

    return models


def predict_price(model, input_dict: dict) -> float:
    df = pd.DataFrame([input_dict])
    return float(model.predict(df)[0])


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

        # Enhanced Marker mimicking the PDF's green indicator badge
        ax.scatter(x_pos, predicted_price, color="#00b894", s=250, marker="*", zorder=5)
        ax.axhline(y=predicted_price, color="#00b894", linestyle="--", alpha=0.8)

        # Add the badge text box
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

    # Preset logic
    default_state, default_area, default_tenure, default_type, default_tx = (
        "Selangor",
        "",
        "Freehold",
        "Terrace House",
        10,
    )

    if preset_choice == "Sample: Luxury Condo in KL":
        default_state, default_area, default_tenure, default_type, default_tx = (
            "Kuala Lumpur",
            "Mont Kiara",
            "Freehold",
            "Condominium",
            45,
        )
    elif preset_choice == "Sample: Standard Terrace in Johor":
        default_state, default_area, default_tenure, default_type, default_tx = (
            "Johor",
            "Skudai",
            "Freehold",
            "Terrace House",
            120,
        )
    elif preset_choice == "Sample: Affordable Flat in Penang":
        default_state, default_area, default_tenure, default_type, default_tx = (
            "Penang",
            "Ayer Itam",
            "Leasehold",
            "Flat",
            35,
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
                }

    return user_features


# 3. FRONTEND UI & CONTROLLER LAYOUT
def main():
    st.title("🏠 Malaysia Housing Price Predictor")

    models = load_models()
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
                pred = predict_price(selected_model, user_features)
                st.success("Analysis Complete!")
                st.metric(
                    label=f"Estimated Median Price ({selected_model_name})",
                    value=f"RM {pred:,.2f}",
                )

                st.markdown("---")
                plot_market_comparison(
                    user_features["State"],
                    user_features["Type"],
                    pred,
                    title=f"Market Benchmark ({selected_model_name})",
                )
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
            for idx, (m_name, m_obj) in enumerate(models.items()):
                try:
                    pred = predict_price(m_obj, comp_features)
                    predictions[m_name] = pred
                    cols[idx].metric(label=m_name, value=f"RM {pred:,.2f}")
                except Exception as e:
                    cols[idx].error(f"Error with {m_name}: {e}")

            if predictions:
                st.markdown("---")
                # Plot side-by-side comparison
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

                # Add value labels on top of bars
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
