import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. CONFIGURATION & CORE LAYOUT
st.set_page_config(
    page_title="Housing Baseline Model", page_icon="🏠", layout="centered"
)


# 2. BACKEND SERVICE LAYER (Internal Functions)
@st.cache_resource
def load_baseline_model():
    prototype_dir = Path(__file__).resolve().parent
    model_path = prototype_dir / "linear_regression.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def predict_price(model, input_dict: dict) -> float:
    df = pd.DataFrame([input_dict])
    return float(model.predict(df)[0])


def plot_market_comparison(state, prop_type, predicted_price):
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
        f"Market Benchmark: Where your property stands in {state}",
        pad=20,
        fontsize=14,
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


# 3. FRONTEND UI & CONTROLLER LAYOUT
def main():
    # Mimic the PDF Top Navigation using Streamlit Tabs
    st.title("🏠 Malaysia Housing Price Predictor")
    tab1, tab2 = st.tabs(["🎯 Single Property Prediction", "📊 Model Analytics"])

    model = load_baseline_model()

    with tab1:
        st.markdown(
            "Enter details below to generate a prediction via the **Multiple Linear Regression Baseline**."
        )
        if model is None:
            st.error(
                "No baseline model structure detected in `/prototype`. Please run your training script first."
            )
            return

        # AUTO-INPUT FEATURE (Inspired by PDF prototype)
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

        # Render Prediction Form Input
        user_features = None
        with st.form("prediction_form"):
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
                    "State",
                    options=state_options,
                    index=state_options.index(default_state),
                )
                area = st.text_input(
                    "Area / Township",
                    value=default_area,
                    placeholder="e.g., Cheras, Skudai",
                )
                tenure = st.selectbox(
                    "Tenure",
                    options=["Freehold", "Leasehold"],
                    index=0 if default_tenure == "Freehold" else 1,
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
                    st.warning(
                        "Please enter an Area or Township name before submitting."
                    )
                else:
                    user_features = {
                        "Area": area.strip(),
                        "State": state,
                        "Tenure": tenure,
                        "Type": prop_type,
                        "Transactions": transactions,
                    }

        # Handle Form Submission Inference
        if user_features:
            try:
                pred = predict_price(model, user_features)
                st.success("Analysis Complete!")
                st.metric(
                    label="Estimated Baseline Median Price", value=f"RM {pred:,.2f}"
                )

                # Render Comparative Visualizations
                st.markdown("---")
                plot_market_comparison(
                    user_features["State"], user_features["Type"], pred
                )

            except Exception as e:
                st.error(f"Inference pipeline issue: {e}")

    with tab2:
        st.subheader("Model Evaluation")
        st.markdown(
            "Baseline Multiple Linear Regression performance metrics and evaluation plots."
        )

        # Load the saved evaluation plot from report_assets
        prototype_dir = Path(__file__).resolve().parent
        plot_path = (
            prototype_dir.parent
            / "report_assets"
            / "plots"
            / "actual_vs_predicted_advanced.png"
        )

        if plot_path.exists():
            st.image(
                str(plot_path), caption="Actual vs Predicted Values (Linear Regression)"
            )
        else:
            st.info(
                "Evaluation plot not found. Run `python src/model_visual.py` to generate it."
            )


if __name__ == "__main__":
    main()
