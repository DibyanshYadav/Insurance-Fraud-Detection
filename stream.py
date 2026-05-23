import streamlit as st
import pandas as pd
import joblib
import os
from datetime import date

# --- Helper to handle paths and loading ---
def load_file(file_name):
    # Searches in 'capstone' folder first, then the root directory
    primary_path = os.path.join('capstone', file_name)
    if os.path.exists(primary_path):
        return joblib.load(primary_path)
    elif os.path.exists(file_name):
        return joblib.load(file_name)
    else:
        st.error(f"Missing file: {file_name}. Ensure it is in the 'capstone' folder.")
        st.stop()

# Load model components
model = load_file('xgb_fraud_model.pkl')
trf = load_file('transformer.pkl')
metrics = load_file('model_metrics.pkl')
model_acc = metrics['accuracy']

# --- Page UI Configuration ---
st.set_page_config(page_title="Insurance Fraud AI", layout="wide")
st.title("🛡️ Health Insurance Fraud Detection System")

# Sidebar for Presentation Metrics
st.sidebar.header("Model Performance")
st.sidebar.metric(label="System Accuracy", value=f"{model_acc:.2%}")
st.sidebar.info("Capstone Project - I | IIT Patna")

# --- User Input Section with Clean Labels ---
st.subheader("Analyze New Claim")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Patient & Provider Profile")
    patient_age = st.number_input("Patient Age", 0, 110, 30, help="Age of the patient in years")
    patient_gender = st.selectbox("Patient Gender", ['Female', 'Male', 'Other'], help="Select gender of the patient")
    prev_claims_p = st.number_input("Patient’s Previous Claims History", 0, 100, 2, help="Number of claims this patient has made before")
    prev_claims_v = st.number_input("Provider’s Previous Claims History", 0, 1000, 15, help="Number of claims handled by this hospital/doctor before")
    distance = st.number_input("Calculated Distance (Miles)", 0.0, 500.0, 12.5, help="Distance between patient and hospital in miles")

with col2:
    st.markdown("### 💰 Financials")
    claim_amount = st.number_input("Claim Amount ($)", 0.0, 100000.0, 2500.0, help="Total amount claimed for this treatment")
    deductible_amount = st.number_input("Deductible Amount ($)", 0.0, 5000.0, 500.0, help="Amount patient must pay before insurance covers")
    copay_amount = st.number_input("CoPay Amount ($)", 0.0, 1000.0, 50.0, help="Amount patient pays along with insurance")
    claim_submitted_late = st.selectbox("Claim Submitted Late?", [False, True], help="Select True if claim was submitted late, otherwise False")

with col3:
    st.markdown("### 📅 Dates & Services")
    # Service and Discharge types kept exactly as per model requirements
    service_type = st.selectbox("Service Type", ['Inpatient', 'Outpatient', 'Pharmacy', 'Emergency Room', 'Laboratory', 'Ambulance'], help="Type of service: inpatient, outpatient, pharmacy, etc.")
    discharge_type = st.selectbox("Discharge Type", ['Home', 'Rehab/Skilled Nursing', 'Deceased', 'Against Medical Advice', 'Transfer'], help="How patient was discharged: home, transfer, etc.")
    
    # Raw date inputs for internal computation
    service_date = st.date_input("Service Date", date(2026, 4, 1), help="Date when treatment/service was given")
    claim_date = st.date_input("Claim Filing Date", date(2026, 4, 10), help="Date when claim was submitted")
    policy_expiry = st.date_input("Policy Expiration Date", date(2026, 12, 31), help="Date when insurance policy ends")

# --- Internal Feature Computation ---
# These are calculated behind the scenes for the model
claim_delay = (claim_date - service_date).days
days_to_expiry = (policy_expiry - claim_date).days

# --- Prediction Logic ---
st.divider()

if st.button("🚀 Run Fraud Analysis", type="primary", use_container_width=True):
    # Construct DataFrame (Column order must match X_train)
    input_df = pd.DataFrame([[
        claim_delay, days_to_expiry, claim_amount, patient_age, 
        deductible_amount, copay_amount, prev_claims_p, prev_claims_v, 
        distance, patient_gender, discharge_type, service_type, claim_submitted_late
    ]], columns=[
        'Claim_Delay', 'Days_To_Expiry', 'Claim_Amount', 'Patient_Age',
        'Deductible_Amount', 'CoPay_Amount', 'Number_of_Previous_Claims_Patient',
        'Number_of_Previous_Claims_Provider', 'Provider_Patient_Distance_Miles',
        'Patient_Gender', 'Discharge_Type', 'Service_Type', 'Claim_Submitted_Late'
    ])

    # 1. Pipeline Transformation
    transformed_data = trf.transform(input_df)
    
    # 2. Prediction & Probability (float conversion for Streamlit stability)
    prediction = model.predict(transformed_data)
    prob = float(model.predict_proba(transformed_data)[0][1])

    # --- Final Results Display ---
    if prediction[0] == 1:
        st.error(f"### 🚩 HIGH RISK DETECTED")
        st.progress(prob)
        st.write(f"Fraud Probability: **{prob:.2%}**")
        st.warning(f"Internal Metrics: Delay = {claim_delay} days | Expiry Window = {days_to_expiry} days")
    else:
        st.success(f"### ✅ LOW RISK")
        st.progress(prob)
        st.write(f"Fraud Probability: **{prob:.2%}**")
        st.info(f"Internal Metrics: Delay = {claim_delay} days | Expiry Window = {days_to_expiry} days")