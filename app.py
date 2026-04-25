# app.py — Employee Salary Predictor (INR)
# Run: streamlit run app.py

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Salary Predictor", page_icon="💼")

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

# ── Load dataset ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

# ── Compute metrics as percentages ───────────────────────────
@st.cache_data
def compute_metrics():
    df = load_data()
    X = df[["Years_Experience","Age","Education_Level","Hours_Per_Week","Num_Projects"]]
    y = df["Annual_Salary_INR"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = load_model()
    y_pred = model.predict(X_test)
    mean_actual = y_test.mean()
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return {
        "MAE%":  round(mae  / mean_actual * 100, 2),   # % of mean salary
        "MSE%":  round(mse  / mean_actual**2 * 100, 2),# normalized %
        "RMSE%": round(rmse / mean_actual * 100, 2),   # % of mean salary
        "R2":    round(r2   * 100, 2),
        "MAPE":  round(mape, 2),
    }

model   = load_model()
df      = load_data()
metrics = compute_metrics()

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

# ── Model Metrics ─────────────────────────────────────────────
with st.expander("📈 Model Metrics"):
    st.write("**Test Set Evaluation (20% data)**")
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE",   f"{metrics['MAE%']}%")
    col2.metric("MSE",   f"{metrics['MSE%']}%")
    col3.metric("RMSE",  f"{metrics['RMSE%']}%")
    col4.metric("R²",    f"{metrics['R2']}%")
    col5.metric("MAPE",  f"{metrics['MAPE']}%")

# ── Dataset ──────────────────────────────────────────────────
with st.expander("🗂️ View Dataset"):
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

st.caption("Built with Python · Scikit-learn · Streamlit")
