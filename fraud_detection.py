import streamlit as st
import pandas as pd
import joblib
import datetime
import os
from rag_engine import process_statement_with_rag

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Fraud Guard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# stSidebarCollapseButton hide already present
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
        
        [data-testid="stSidebarCollapseButton"] {
            display: none;
        }
        .sidebar-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            color: white;
            font-size: 48px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Final Groq API key (Loaded from .env, no UI display)
groq_api_key = os.getenv("GROQ_API_KEY", "")

# -------------------------
# Load trained model
# -------------------------
model = joblib.load("fraud_detection_pipeline.pkl")

# -------------------------
# App UI
# -------------------------
# -------------------------
# Sidebar for Navigation & Configuration
# -------------------------
with st.sidebar:
    st.markdown('<p class="sidebar-title">🛡️ FraudGuard AI</p>', unsafe_allow_html=True)
    app_mode = st.radio("Analysis Mode", ["Single Transaction", "Bank Statement (RAG)"])
    st.divider()

if app_mode == "Single Transaction":
    st.title("💳 AI Fraud Detection System")
    st.markdown("Enter transaction details to detect potential fraud using the ML model.")

    st.divider()

    # -------------------------
    # Input Layout (2 columns)
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0
        )

        oldbalanceOrg = st.number_input(
            "Old Balance (Sender)",
            min_value=0.0,
            value=0.0
        )

    with col2:
        newbalanceOrig = st.number_input(
            "New Balance (Sender)",
            min_value=0.0,
            value=0.0
        )

        oldbalanceDest = st.number_input(
            "Old Balance (Receiver)",
            min_value=0.0,
            value=0.0
        )

        newbalanceDest = st.number_input(
            "New Balance (Receiver)",
            min_value=0.0,
            value=0.0
        )

    st.divider()

    # -------------------------
    # Transaction History
    # -------------------------
    if "history" not in st.session_state:
        st.session_state.history = []

    # -------------------------
    # Prediction
    # -------------------------
    if st.button("🔍 Predict Fraud Risk"):
        progress_bar = st.progress(0)

        input_data = pd.DataFrame([{
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }])

        progress_bar.progress(50)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        progress_bar.progress(75)

        st.subheader("Prediction Result")

        # Risk meter
        st.progress(float(probability))

        st.metric(
            label="Fraud Probability",
            value=f"{probability*100:.2f}%"
        )

        if prediction == 1:
            st.error("⚠️ High Risk: This transaction may be FRAUD")
        else:
            st.success("✅ Safe: This transaction looks legitimate")

        progress_bar.progress(100)

        # Save history
        record = {
            "Time": datetime.datetime.now(),
            "Type": transaction_type,
            "Amount": amount,
            "Fraud Probability": round(probability*100,2),
            "Prediction": "Fraud" if prediction == 1 else "Safe"
        }

        st.session_state.history.append(record)

    st.divider()

    # -------------------------
    # Dashboard Section
    # -------------------------
    st.subheader("📊 Transaction History")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.dataframe(df, use_container_width=True)

        st.subheader("Fraud Risk Distribution")

        chart = df["Prediction"].value_counts()

        st.bar_chart(chart)

    else:
        st.info("No transactions yet. Run a prediction to see history.")

else:
    # -------------------------
    # RAG Mode UI
    # -------------------------
    st.title("📄 Bank Statement Analysis (RAG)")
    st.markdown("Upload your bank statement to identify suspicious transactions using AI Reasoning.")
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Bank Statement (PDF)", type=["pdf"])
    
    if uploaded_file:
        st.info(f"File uploaded: {uploaded_file.name}")
        
        if st.button("🔍 Run AI Fraud Audit"):
            if not groq_api_key:
                st.warning("⚠️ Please provide a Groq API Key in the sidebar to proceed.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📄 Analyzing PDF structure & layout...")
                progress_bar.progress(30)
                
                temp_filename = f"temp_{uploaded_file.name}"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                status_text.text("🔍 Extracting transactions (using AI Layout Engine if needed)...")
                progress_bar.progress(60)
                
                try:
                    report = process_statement_with_rag(temp_filename, groq_api_key)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Audit Complete!")
                    
                    st.subheader("📋 AI Audit Report")
                    st.markdown(report)
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"fraud_audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)