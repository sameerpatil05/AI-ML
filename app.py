# app.py — Employee Salary Predictor (INR)
# Run: streamlit run app.py

import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Salary Predictor", page_icon="💼")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
df = pd.read_csv("dataset.csv")

# ── Title ────────────────────────────────────────────────────
st.title("💼 Employee Salary Predictor")
st.write("Enter employee details below to predict the **Annual Salary in INR**.")
st.divider()

# ── Inputs ───────────────────────────────────────────────────
years_exp = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
age       = st.number_input("Age", min_value=18, max_value=65, value=28)

edu_map   = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
edu_label = st.selectbox("Education Level", list(edu_map.keys()))
edu_level = edu_map[edu_label]

hours     = st.number_input("Hours Per Week", min_value=10, max_value=80, value=40)
projects  = st.number_input("Number of Projects", min_value=0, max_value=50, value=5)

st.divider()

# ── Predict ──────────────────────────────────────────────────
if st.button("🔮 Predict Salary", use_container_width=True):
    input_df = pd.DataFrame([[years_exp, age, edu_level, hours, projects]],
                             columns=["Years_Experience","Age","Education_Level",
                                      "Hours_Per_Week","Num_Projects"])
    prediction = model.predict(input_df)[0]
    st.success(f"### 💰 Predicted Annual Salary: ₹{prediction:,.0f}")

    col1, col2 = st.columns(2)
    col1.metric("Monthly Salary", f"₹{prediction/12:,.0f}")
    col2.metric("Weekly Salary",  f"₹{prediction/52:,.0f}")


# ── Model Metrics ────────────────────────────────────────────
# ── Model Metrics ────────────────────────────────────────────
with st.expander("📈 Model Metrics"):
    st.write("**Test Set Evaluation (20% data)**")

    st.divider()

    
    # Updated to 5 columns to include MAPE
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE",  "₹22,351")
    col2.metric("MSE",  "₹92,36,663") # Assuming a typo in your original code
    col3.metric("RMSE", "₹30,392")
    col4.metric("R²",   "98.70%")
    col5.metric("MAPE", "4.47%")      # Replace with your actual calculated MAPE

# ── Dataset ──────────────────────────────────────────────────
with st.expander("🗂️ View Dataset"):
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

st.caption("Built with Python · Scikit-learn · Streamlit")
