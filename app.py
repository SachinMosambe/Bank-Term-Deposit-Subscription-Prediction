import streamlit as st
import pandas as pd
import joblib
import boto3
import os

# ==========================
# 🪣 S3 Configuration
# ==========================
S3_BUCKET = "bank-term-model-bucket"       # 🔹 Replace with your bucket name
MODEL_KEY = "models/best_model.pkl"        # 🔹 Path inside your S3 bucket
LOCAL_MODEL_PATH = "best_model.pkl"        # Local cache path

# ==========================
# ⚙️ Load Model from S3
# ==========================
@st.cache_resource
def load_model():
    """Download model from S3 if not already cached locally"""
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.info("📦 Downloading model from S3...")
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)
        st.success("✅ Model downloaded successfully!")
    return joblib.load(LOCAL_MODEL_PATH)

model = load_model()

# ==========================
# 🧱 App UI
# ==========================
st.set_page_config(page_title="Term Deposit Prediction", layout="centered")
st.title("💰 Bank Term Deposit Subscription Prediction")
st.markdown("Predict whether a client will subscribe to a term deposit.")

# ==========================
# 🧍 User Input
# ==========================
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
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# ==========================
# 🔮 Prediction
# ==========================
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("The client is **likely to subscribe** to a term deposit.")
        else:
            st.warning("The client is **not likely to subscribe**.")

        st.subheader("Prediction Probability")
        st.write(f"**{proba:.2%}** chance of subscription.")
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")

