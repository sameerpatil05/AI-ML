# ============================================================
#  app.py  —  Employee Salary Predictor  |  Streamlit UI
#  Run:  streamlit run app.py
# ============================================================

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="centered",
)

# ── LOAD MODEL ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── LOAD DATASET FOR EDA ────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

df = load_data()

# ── CUSTOM CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.4rem; font-weight:700; color:#1f4e79; text-align:center; }
    .sub-title   { font-size:1rem;   color:#555;      text-align:center; margin-bottom:1.5rem; }
    .section-hdr { font-size:1.15rem; font-weight:600; color:#1f4e79; margin-top:1rem; }
    .metric-card { background:#f0f6ff; border-radius:10px; padding:12px 18px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────
st.markdown('<p class="main-title">💼 Employee Salary Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Estimate annual salary using a trained Linear Regression model</p>', unsafe_allow_html=True)
st.divider()

# ── SIDEBAR — Dataset Info ───────────────────────────────────
with st.sidebar:
    st.header("📊 Dataset Overview")
    st.metric("Total Records",  len(df))
    st.metric("Features Used",  5)
    st.metric("Model R² Score", "98.76 %")
    st.markdown("---")
    st.markdown("**Feature Descriptions**")
    st.markdown("""
- **Years Experience** — Total work experience  
- **Age** — Employee age in years  
- **Education Level** — 0 = Bachelor, 1 = Master, 2 = PhD  
- **Hours / Week** — Weekly working hours  
- **Projects Handled** — Total projects completed  
    """)
    st.markdown("---")
    st.caption("Model: Linear Regression | sklearn")

# ── TAB LAYOUT ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Salary", "📈 Model Metrics", "🗂️ Dataset"])

# ────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-hdr">Enter Employee Details</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        years_exp = st.number_input(
            "🏅 Years of Experience",
            min_value=0, max_value=40,
            value=5, step=1,
            help="Total years of professional work experience"
        )
        age = st.number_input(
            "🎂 Age",
            min_value=18, max_value=65,
            value=28, step=1,
            help="Employee age in years"
        )
        edu_map = {"Bachelor's (0)": 0, "Master's (1)": 1, "PhD (2)": 2}
        edu_label = st.selectbox(
            "🎓 Education Level",
            options=list(edu_map.keys()),
            help="Highest education qualification"
        )
        edu_level = edu_map[edu_label]

    with col2:
        hours_per_week = st.number_input(
            "⏱️ Hours Per Week",
            min_value=10, max_value=80,
            value=40, step=1,
            help="Average number of hours worked per week"
        )
        num_projects = st.number_input(
            "📁 Number of Projects",
            min_value=0, max_value=50,
            value=5, step=1,
            help="Total number of projects handled"
        )

    st.markdown("")
    predict_btn = st.button("🚀 Predict Salary", use_container_width=True, type="primary")

    if predict_btn:
        input_data = np.array([[years_exp, age, edu_level, hours_per_week, num_projects]])
        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.success(f"### 💰 Predicted Annual Salary:  **${prediction:,.2f}**")

        # Breakdown card
        st.markdown("**📋 Input Summary**")
        summary = {
            "Years of Experience": years_exp,
            "Age":                 age,
            "Education Level":     edu_label,
            "Hours / Week":        hours_per_week,
            "Projects Handled":    num_projects,
        }
        summary_df = pd.DataFrame(summary.items(), columns=["Feature", "Value"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Salary range indicator
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Monthly Estimate",  f"${prediction/12:,.0f}")
        col_b.metric("Weekly Estimate",   f"${prediction/52:,.0f}")
        col_c.metric("Daily Estimate",    f"${prediction/260:,.0f}")

# ────────────────────────────────────────────────────────────
# TAB 2 — MODEL METRICS
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-hdr">Model Evaluation Metrics (Test Set — 20%)</p>', unsafe_allow_html=True)

    m_col1, m_col2 = st.columns(2)
    m_col1.metric("MAE  — Mean Absolute Error",     "$2,235.11")
    m_col2.metric("MSE  — Mean Squared Error",      "$9,236,637.98")
    m_col1.metric("RMSE — Root Mean Squared Error", "$3,039.18")
    m_col2.metric("R²   — Coefficient of Determination", "0.9876")

    st.markdown("---")
    st.markdown('<p class="section-hdr">Feature Coefficients</p>', unsafe_allow_html=True)

    coef_df = pd.DataFrame({
        "Feature":     ["Years_Experience", "Age", "Education_Level",
                        "Hours_Per_Week", "Num_Projects"],
        "Coefficient": [2223.13, 259.26, 8050.42, 403.39, 531.84],
        "Impact":      ["🔴 High", "🟠 Medium", "🔴 High", "🟡 Low", "🟡 Low"],
    })
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

    st.info("💡 **Intercept:** $30,821.32  |  A **1-unit increase** in Years of Experience adds ~$2,223 to salary on average.")

    st.markdown("---")
    st.markdown('<p class="section-hdr">Model Pipeline</p>', unsafe_allow_html=True)
    st.code("""
Step 1 → Load dataset.csv            (500 records, 5 features)
Step 2 → Data Cleaning                (null check + deduplication)
Step 3 → Exploratory Data Analysis    (correlation, statistics)
Step 4 → Feature Selection            (5 input features)
Step 5 → Train / Test Split           (80% / 20%)
Step 6 → LinearRegression.fit()       (sklearn)
Step 7 → Evaluate (MAE, MSE, RMSE, R²)
Step 8 → pickle.dump → model.pkl
    """, language="text")

# ────────────────────────────────────────────────────────────
# TAB 3 — DATASET
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-hdr">Dataset Preview (First 20 Rows)</p>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<p class="section-hdr">Statistical Summary</p>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("---")
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Total Rows",    len(df))
    col_d2.metric("Total Columns", len(df.columns))
    col_d3.metric("Missing Values", int(df.isnull().sum().sum()))

# ── FOOTER ──────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Python · Scikit-learn · Streamlit  |  Linear Regression Model  |  R² = 98.76%")
