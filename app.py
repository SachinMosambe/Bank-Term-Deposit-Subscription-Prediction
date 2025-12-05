import streamlit as st
import pandas as pd
import joblib
import boto3
import botocore
import threading
import io
import os

# ==========================
# 🪣 S3 Configuration
# ==========================
S3_BUCKET = "bank-term-deposit-1"
MODEL_KEY = "models/best_model.pkl"  # S3 path
LOCAL_MODEL_PATH = "cached_model.pkl"  # Local fast cache

# ==========================
# ⚡ Global State
# ==========================
_model = None
_loading = False
_lock = threading.Lock()


# ==========================
# 🔥 Background Model Loader
# ==========================
def _download_and_cache_model():
    """Download large model from S3 only once and cache locally."""
    global _model, _loading

    try:
        # If already cached locally → use it (FAST)
        if os.path.exists(LOCAL_MODEL_PATH):
            _model = joblib.load(LOCAL_MODEL_PATH)
            _loading = False
            return

        session = boto3.Session()
        s3 = session.client(
            "s3",
            config=botocore.client.Config(
                retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=40,
                connect_timeout=3,
                read_timeout=3,
            ),
        )

        buffer = io.BytesIO()
        s3.download_fileobj(S3_BUCKET, MODEL_KEY, buffer)

        buffer.seek(0)
        model = joblib.load(buffer)

        # Save for instant future loads
        joblib.dump(model, LOCAL_MODEL_PATH)

        with _lock:
            _model = model

    except Exception as e:
        st.error(f"❌ Model load failed: {e}")

    finally:
        _loading = False


# ==========================
# ⚡ Async Model Controller
# ==========================
@st.cache_resource(show_spinner=False)
def load_model():
    """Returns model or triggers async load."""
    global _model, _loading

    # Already loaded → return
    if _model is not None:
        return _model

    # Cached locally → load instantly
    if os.path.exists(LOCAL_MODEL_PATH):
        _model = joblib.load(LOCAL_MODEL_PATH)
        return _model

    # Start async download
    if not _loading:
        _loading = True
        threading.Thread(target=_download_and_cache_model, daemon=True).start()

    return None


# Trigger model load
model = load_model()

# ==========================
# UI Loader Status
# ==========================
if model is None:
    st.info("⏳ Loading 2.5GB model from S3... (only first time)")
else:
    st.success("✅ Model loaded successfully!")


# ==========================
# 🧱 Streamlit UI
# ==========================
st.set_page_config(page_title="Term Deposit Prediction", layout="centered")
st.title("💰 Bank Term Deposit Subscription Prediction")
st.markdown("Predict whether a client will subscribe to a term deposit.")


# ==========================
# 🧍 User Input Form
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
        st.error("⏳ Model still loading... Try again in a few seconds.")
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
            st.error(f"❌ Prediction failed: {e}")
