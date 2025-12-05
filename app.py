import streamlit as st
import pandas as pd
import requests
from feature_engineer import FeatureEngineer

# ⭐ Update this with your EC2 public IP
API_URL = "http://3.111.71.221:8000/predict"

st.set_page_config(page_title="Term Deposit Prediction", layout="centered")
st.title("💰 Bank Term Deposit Subscription Prediction")
st.markdown("Predict whether a client will subscribe to a term deposit using a deployed ML model API.")


# ============================================
# 🧍 USER INPUT FORM
# ============================================
def get_user_input():
    data = {
        'age': st.number_input('Age', 18, 100, 30),
        'job': st.selectbox('Job', [
            'technician', 'blue-collar', 'student', 'admin.', 'management',
            'entrepreneur', 'self-employed', 'unknown', 'services',
            'retired', 'housemaid', 'unemployed'
        ]),
        'marital': st.selectbox('Marital Status', ['married', 'single', 'divorced']),
        'education': st.selectbox('Education', ['secondary', 'primary', 'tertiary', 'unknown']),
        'default': st.selectbox('Default Credit', ['no', 'yes']),
        'balance': st.number_input('Account Balance', -2000, 100000, 1000),
        'housing': st.selectbox('Housing Loan', ['no', 'yes']),
        'loan': st.selectbox('Personal Loan', ['no', 'yes']),
        'contact': st.selectbox('Contact Type', ['cellular', 'unknown', 'telephone']),
        'day': st.number_input('Last Contact Day', 1, 31, 15),
        'month': st.selectbox('Last Contact Month', [
            'aug', 'jun', 'may', 'feb', 'apr', 'nov', 'jul',
            'jan', 'oct', 'mar', 'sep', 'dec'
        ]),
        'duration': st.number_input('Call Duration (sec)', 1, 5000, 200),
        'campaign': st.number_input('Contacts During Campaign', 1, 50, 1),
        'pdays': st.number_input('Days Since Last Contact', -1, 500, -1),
        'previous': st.number_input('Previous Contacts', 0, 50, 0),
        'poutcome': st.selectbox('Previous Outcome', ['unknown', 'other', 'failure', 'success'])
    }
    return pd.DataFrame([data])

input_df = get_user_input()


# ============================================
# 🔮 PREDICTION THROUGH FASTAPI
# ============================================
if st.button("Predict"):
    try:
        payload = {"data": input_df.iloc[0].to_dict()}

        response = requests.post(API_URL, json=payload).json()
        prediction = response["prediction"]
        proba = response["probability"]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("The client is **likely to subscribe** to a term deposit.")
        else:
            st.warning("The client is **not likely to subscribe**.")

        if proba is not None:
            st.subheader("Prediction Probability")
            st.write(f"**{proba:.2%}** chance of subscription.")

    except Exception as e:
        st.error(f"❌ API request failed: {e}")
