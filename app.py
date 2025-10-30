import streamlit as st
import pandas as pd
import joblib
import boto3
import io

# ==========================
# 🪣 S3 Configuration
# ==========================
S3_BUCKET = "bank-term-model-bucket"       # 🔹 Replace with your bucket name
MODEL_KEY = "models/best_model.pkl"        # 🔹 Path inside your S3 bucket

# ==========================
# ⚙️ Load Model Directly from S3
# ==========================
@st.cache_resource
def load_model_from_s3():
    """Load model directly into memory from S3"""
    st.info("📦 Loading model directly from S3...")
    s3 = boto3.client('s3')
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
        bytestream = io.BytesIO(obj['Body'].read())
        model = joblib.load(bytestream)
        st.success("✅ Model loaded successfully from S3!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model from S3: {e}")
        raise e

model = load_model_from_s3()

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
