
import streamlit as st
import pandas as pd
import joblib

# Load the saved model
saved = joblib.load("flower_model.pkl")
model = saved["model"]
feature_names = saved["feature_names"]
species_names = saved["species_names"]

st.set_page_config(
    page_title="Flower Species Prediction",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Flower Species Prediction")
st.write(
    "Enter the flower measurements below "
    "to predict its species."
)

st.info("Enter all measurements in centimeters (cm).")

# Input form
with st.form("flower_form"):
    st.subheader("Flower Measurements")

    sepal_length = st.number_input(
        "Sepal length (cm)",
        min_value=0.1,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    sepal_width = st.number_input(
        "Sepal width (cm)",
        min_value=0.1,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

    petal_length = st.number_input(
        "Petal length (cm)",
        min_value=0.1,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal width (cm)",
        min_value=0.1,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

    submitted = st.form_submit_button(
        "Predict Flower Species"
    )

if submitted:
    sample = pd.DataFrame(
        [[sepal_length, sepal_width,
          petal_length, petal_width]],
        columns=feature_names
    )

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    st.success(
        f"Predicted species: {species_names[prediction].title()}"
    )

    st.subheader("Model confidence estimates")

    probability_data = pd.DataFrame({
        "Species": [
            name.title() for name in species_names
        ],
        "Estimated probability": probabilities
    })

    st.bar_chart(
        probability_data.set_index("Species")
    )

    st.caption(
        "These probabilities are model estimates, "
        "not guarantees. Accuracy may vary for new flowers."
    )
