import streamlit as st
import pandas as pd
import joblib
import boto3
import io
import botocore
import threading

# ==========================
# 🪣 S3 Configuration
# ==========================
S3_BUCKET = "bank-term-deposit-1"      
MODEL_KEY = "models/best_model.pkl"        # Path inside S3


_model = None
_model_lock = threading.Lock()


def _download_model():
    """Background thread: download model without blocking Streamlit."""
    global _model
    try:
        session = boto3.Session()

        # Highly optimized S3 client
        s3 = session.client(
            "s3",
            config=botocore.client.Config(
                max_pool_connections=50,
                retries={"max_attempts": 10, "mode": "adaptive"},
                connect_timeout=2,
                read_timeout=2,
            ),
        )

        buffer = io.BytesIO()
        s3.download_fileobj(S3_BUCKET, MODEL_KEY, buffer)
        buffer.seek(0)

        loaded = joblib.load(buffer)

        # Save model safely
        with _model_lock:
            _model = loaded

    except Exception as e:
        st.error(f"❌ Background model load failed: {e}")


@st.cache_resource(show_spinner=False)
def load_model_async():
    """Returns model if loaded; otherwise starts async loading."""
    global _model

    if _model is None:
        threading.Thread(target=_download_model, daemon=True).start()
        return None

    return _model


# Load model (async)
model = load_model_async()

# Loader UI
if model is None:
    st.info("⏳ Loading model from S3 in background... please wait 2–3 seconds.")
else:
    st.success("✅ Model loaded successfully!")

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
    if model is None:
        st.error("⏳ Model still loading... Please wait 2–3 seconds and try again.")
    else:
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
