import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Load model and features
with open(r'data/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(r'data/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.title("Hospital Readmission Risk Predictor")
st.write("Predicts whether a diabetic patient will be readmitted within 30 days.")

st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 5, 95, 65, step=10)
time_in_hospital = st.sidebar.slider("Days in hospital", 1, 14, 4)
number_inpatient = st.sidebar.slider("Prior inpatient visits", 0, 10, 1)
number_emergency = st.sidebar.slider("Prior emergency visits", 0, 10, 0)
num_medications = st.sidebar.slider("Number of medications", 1, 40, 15)
number_diagnoses = st.sidebar.slider("Number of diagnoses", 1, 16, 5)
num_lab_procedures = st.sidebar.slider("Lab procedures", 1, 100, 40)

if st.sidebar.button("Predict Risk"):
    # Build input row with zeros for all features
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # Fill in the values we collected
    input_data['age'] = age
    input_data['time_in_hospital'] = time_in_hospital
    input_data['number_inpatient'] = number_inpatient
    input_data['number_emergency'] = number_emergency
    input_data['num_medications'] = num_medications
    input_data['number_diagnoses'] = number_diagnoses
    input_data['num_lab_procedures'] = num_lab_procedures

    # Predict
    prob = model.predict_proba(input_data)[0][1]
    risk_percent = round(prob * 100, 1)

    st.subheader("Prediction Result")

    if prob >= 0.3:
        st.error(f"HIGH RISK: {risk_percent}% chance of readmission within 30 days")
    else:
        st.success(f"LOW RISK: {risk_percent}% chance of readmission within 30 days")

    # SHAP explanation
    st.subheader("Why this prediction?")
    explainer = shap.LinearExplainer(model, 
                    pd.DataFrame(np.zeros((100, len(feature_names))), 
                    columns=feature_names))
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_data.iloc[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig)