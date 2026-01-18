import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

SAVE_PATH = r"C:\Users\KATHERINE\OneDrive\Desktop\chrun_app"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_artifacts():
    model = pickle.load(open(os.path.join(SAVE_PATH, "churn_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(SAVE_PATH, "scaler.pkl"), "rb"))
    feature_names = pickle.load(open(os.path.join(SAVE_PATH, "feature_names.pkl"), "rb"))
    return model, scaler, feature_names

model, scaler, feature_names = load_model_artifacts()

# ---------------- HEADER ----------------
st.title("📊 Customer Churn Prediction")
st.caption("Predict customer churn risk using machine learning")
st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("Customer Details")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("Age", 18, 100, 35)
    tenure = st.number_input("Tenure (months)", 0, 120, 12)
    usage = st.number_input("Usage Frequency", 0, 100, 10)

with c2:
    support_calls = st.number_input("Support Calls", 0, 50, 1)
    payment_delay = st.number_input("Payment Delay (days)", 0, 60, 0)
    last_interaction = st.number_input("Days Since Last Interaction", 0, 365, 15)

with c3:
    total_spend = st.number_input("Total Spend", 0.0, 10000.0, 500.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    subscription = st.selectbox("Subscription Type", ["Premium", "Standard", "Basic"])
    contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Yearly"])

# ---------------- PREPROCESS ----------------
def preprocess_input():
    X = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    X['Age'] = age
    X['Tenure'] = tenure
    X['Usage Frequency'] = usage
    X['Support Calls'] = support_calls
    X['Payment Delay'] = payment_delay
    X['Total Spend'] = total_spend
    X['Last Interaction'] = last_interaction

    X['Gender_Male'] = 1 if gender == "Male" else 0
    X['Subscription Type_Premium'] = 1 if subscription == "Premium" else 0
    X['Subscription Type_Standard'] = 1 if subscription == "Standard" else 0
    X['Contract Length_Monthly'] = 1 if contract == "Monthly" else 0
    X['Contract Length_Quarterly'] = 1 if contract == "Quarterly" else 0

    return X

# ---------------- PREDICTION ----------------
if st.button("Predict Churn Risk", use_container_width=True):

    input_df = preprocess_input()
    input_scaled = scaler.transform(input_df)
    churn_prob = model.predict_proba(input_scaled)[0][1] * 100

    st.divider()
    st.subheader("Prediction Summary")

    # ---------------- RISK LEVEL ----------------
    if churn_prob < 30:
        st.success(f"Low Risk Customer  •  Churn Probability: {churn_prob:.2f}%")
    elif churn_prob < 60:
        st.warning(f"Medium Risk Customer  •  Churn Probability: {churn_prob:.2f}%")
    else:
        st.error(f"High Risk Customer  •  Churn Probability: {churn_prob:.2f}%")

    # ---------------- COMPACT LAYOUT ----------------
    colA, colB = st.columns([1, 1])

    # --------- PROBABILITY BAR (SMALL) ---------
    with colA:
        st.markdown("**Churn Probability**")
        chart_df = pd.DataFrame({
            "Probability (%)": [100 - churn_prob, churn_prob]
        }, index=["Not Churn", "Churn"])
        st.bar_chart(chart_df, height=200)

    # --------- RISK PROGRESS ---------
    with colB:
        st.markdown("**Risk Level Indicator**")
        st.progress(int(churn_prob))

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("Key Factors Influencing Churn")

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": model.coef_[0]
    }).sort_values(by="Impact", ascending=False)

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("⬆️ **Increases Churn Risk**")
        st.dataframe(coef_df.head(5), use_container_width=True, height=220)

    with col_neg:
        st.markdown("⬇️ **Reduces Churn Risk**")
        st.dataframe(coef_df.tail(5), use_container_width=True, height=220)
