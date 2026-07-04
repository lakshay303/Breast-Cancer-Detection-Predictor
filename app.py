import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Detection Predictor",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# Load Model and Preprocessor
# --------------------------------------------------

@st.cache_resource
def load_pipeline():
    return joblib.load("breast_cancer_pipeline.pkl")

pipeline = load_pipeline()

# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("🩺 Breast Cancer Detection Predictor")

st.markdown("""
This application predicts whether a breast tumor is **Benign** or **Malignant**
using a Machine Learning model trained on the Wisconsin Breast Cancer Dataset.

Please enter the medical measurements below.
""")

st.divider()
# --------------------------------------------------
# User Inputs
# --------------------------------------------------

st.header("📋 Enter Tumor Measurements")

col1, col2, col3 = st.columns(3)

# -----------------------------
# Mean Features
# -----------------------------

with col1:

    st.subheader("📊 Mean Measurements")

    radius_mean = st.number_input("Radius Mean", value=14.12)
    texture_mean = st.number_input("Texture Mean", value=19.29)
    perimeter_mean = st.number_input("Perimeter Mean", value=91.97)
    area_mean = st.number_input("Area Mean", value=654.89)
    smoothness_mean = st.number_input("Smoothness Mean", value=0.096)
    compactness_mean = st.number_input("Compactness Mean", value=0.104)
    concavity_mean = st.number_input("Concavity Mean", value=0.089)
    concave_points_mean = st.number_input("Concave Points Mean", value=0.049)
    symmetry_mean = st.number_input("Symmetry Mean", value=0.181)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.062)

# -----------------------------
# Standard Error Features
# -----------------------------

with col2:

    st.subheader("📈 Standard Error")

    radius_se = st.number_input("Radius SE", value=0.405)
    texture_se = st.number_input("Texture SE", value=1.217)
    perimeter_se = st.number_input("Perimeter SE", value=2.866)
    area_se = st.number_input("Area SE", value=40.34)
    smoothness_se = st.number_input("Smoothness SE", value=0.007)
    compactness_se = st.number_input("Compactness SE", value=0.025)
    concavity_se = st.number_input("Concavity SE", value=0.031)
    concave_points_se = st.number_input("Concave Points SE", value=0.011)
    symmetry_se = st.number_input("Symmetry SE", value=0.021)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.003)

# -----------------------------
# Worst Features
# -----------------------------

with col3:

    st.subheader("📉 Worst Measurements")

    radius_worst = st.number_input("Radius Worst", value=16.27)
    texture_worst = st.number_input("Texture Worst", value=25.68)
    perimeter_worst = st.number_input("Perimeter Worst", value=107.26)
    area_worst = st.number_input("Area Worst", value=880.58)
    smoothness_worst = st.number_input("Smoothness Worst", value=0.132)
    compactness_worst = st.number_input("Compactness Worst", value=0.254)
    concavity_worst = st.number_input("Concavity Worst", value=0.272)
    concave_points_worst = st.number_input("Concave Points Worst", value=0.115)
    symmetry_worst = st.number_input("Symmetry Worst", value=0.290)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.084)

st.divider()

predict = st.button("🔍 Predict Diagnosis", use_container_width=True)
# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict:

    input_df = pd.DataFrame({

        "radius_mean": [radius_mean],
        "texture_mean": [texture_mean],
        "perimeter_mean": [perimeter_mean],
        "area_mean": [area_mean],
        "smoothness_mean": [smoothness_mean],
        "compactness_mean": [compactness_mean],
        "concavity_mean": [concavity_mean],
        "concave points_mean": [concave_points_mean],
        "symmetry_mean": [symmetry_mean],
        "fractal_dimension_mean": [fractal_dimension_mean],

        "radius_se": [radius_se],
        "texture_se": [texture_se],
        "perimeter_se": [perimeter_se],
        "area_se": [area_se],
        "smoothness_se": [smoothness_se],
        "compactness_se": [compactness_se],
        "concavity_se": [concavity_se],
        "concave points_se": [concave_points_se],
        "symmetry_se": [symmetry_se],
        "fractal_dimension_se": [fractal_dimension_se],

        "radius_worst": [radius_worst],
        "texture_worst": [texture_worst],
        "perimeter_worst": [perimeter_worst],
        "area_worst": [area_worst],
        "smoothness_worst": [smoothness_worst],
        "compactness_worst": [compactness_worst],
        "concavity_worst": [concavity_worst],
        "concave points_worst": [concave_points_worst],
        "symmetry_worst": [symmetry_worst],
        "fractal_dimension_worst": [fractal_dimension_worst]

    })


    prediction = pipeline.predict(input_df)[0]

    probability = pipeline.predict_proba(input_df)[0]

    confidence = max(probability) * 100

    st.divider()

    st.header("🩺 Prediction Result")

    if prediction == 0:

        st.success(
            f"✅ Diagnosis: **Benign (Non-Cancerous)**"
        )

    else:

        st.error(
            f"⚠️ Diagnosis: **Malignant (Cancerous)**"
        )

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    st.divider()

    st.warning(
        """
        **Disclaimer**

        This application is developed for educational and research purposes only.

        It should **not** be used as a substitute for professional medical advice, diagnosis, or treatment.

        Please consult a qualified healthcare professional for any medical concerns.
        """
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;'>

Developed with ❤️ by <b>Lakshay Verma</b>

AI & Machine Learning Intern

</div>
""",
unsafe_allow_html=True
)