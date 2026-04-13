import streamlit as st
import pandas as pd
import joblib

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict whether they will churn.")

# -------------------------
# LOAD MODEL (SAFE)
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -------------------------
# USER INPUTS
# -------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# -------------------------
# PREDICTION BUTTON
# -------------------------
if st.button("Predict"):

    try:
        # Convert input into DataFrame (MUST MATCH TRAINING FEATURES)
        input_data = pd.DataFrame([{
            "gender": 1 if gender == "Male" else 0,
            "SeniorCitizen": senior,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        # Prediction
        prediction = model.predict(input_data)

        # Output
        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is not likely to churn")

    except Exception as e:
        st.error(f"Something went wrong: {e}")