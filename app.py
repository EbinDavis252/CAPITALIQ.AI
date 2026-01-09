import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px
import shap

from scipy.optimize import linprog

# ----------------------------------------------------
# App Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="CAPITALIQ-AI™ | Capital Allocation Advisor",
    layout="wide",
    page_icon="📊"
)

st.title("📊 CAPITALIQ-AI™")
st.subheader("AI-Driven Capital Allocation Advisor")
st.markdown("""
**Live Project | Grant Thornton Bharat LLP**  
*MSc Finance & Analytics — Individual Project*
""")

st.divider()

# ----------------------------------------------------
# Upload Dataset
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload Capital Allocation Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload the dataset to begin analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ----------------------------------------------------
# Column Validation
# ----------------------------------------------------
required_columns = [
    "Project_ID",
    "Department",
    "Investment_Capital",
    "Duration_Months",
    "Risk_Score",
    "Strategic_Alignment",
    "Market_Trend_Index",
    "Actual_ROI_Pct",
    "Actual_NPV"
]

missing = [c for c in required_columns if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

st.success("Dataset validated successfully")

# ----------------------------------------------------
# Dataset Preview
# ----------------------------------------------------
st.subheader("🔍 Dataset Preview")
st.dataframe(df, use_container_width=True)

# ----------------------------------------------------
# Feature Engineering
# ----------------------------------------------------
features = [
    "Investment_Capital",
    "Duration_Months",
    "Risk_Score",
    "Strategic_Alignment",
    "Market_Trend_Index"
]

X = df[features]
y_roi = df["Actual_ROI_Pct"]
y_npv = df["Actual_NPV"]

# ----------------------------------------------------
# Machine Learning Models
# ----------------------------------------------------
roi_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

npv_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

roi_model.fit(X, y_roi)
npv_model.fit(X, y_npv)

df["Predicted_ROI"] = roi_model.predict(X)
df["Predicted_NPV"] = npv_model.predict(X)

# ----------------------------------------------------
# Allocation Score
# ----------------------------------------------------
scaler = MinMaxScaler()

df["ROI_Norm"] = scaler.fit_transform(df[["Predicted_ROI"]])
df["NPV_Norm"] = scaler.fit_transform(df[["Predicted_NPV"]])

df["Allocation_Score"] = (
    0.4 * df["ROI_Norm"]
    + 0.3 * df["NPV_Norm"]
    + 0.2 * df["Strategic_Alignment"]
    - 0.1 * df["Risk_Score"]
)

# ----------------------------------------------------
# KPIs
# ----------------------------------------------------
st.subheader("📌 Portfolio Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Capital", f"₹{df['Investment_Capital'].sum():,.0f}")
c2.metric("Avg Predicted ROI", f"{df['Predicted_ROI'].mean():.2f}%")
c3.metric("Avg Predicted NPV", f"₹{df['Predicted_NPV'].mean():,.0f}")

st.divider()

# ----------------------------------------------------
# Capital Budget Optimization
# ----------------------------------------------------
st.subheader("💰 Capital Budget Optimization")

budget = st.number_input(
    "Enter Total Capital Budget (₹)",
    min_value=0.0,
    value=float(df["Investment_Capital"].sum() * 0.6),
    step=100000.0
)

# Linear Programming (maximize Allocation Score)
c = -df["Allocation_Score"].values
A = [df["Investment_Capital"].values]
b = [budget]
bounds = [(0, 1) for _ in range(len(df))]

opt = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

df["Selected"] = opt.x.round(0)

optimized_df = df[df["Selected"] == 1]

st.success(
    f"Optimized Capital Used: ₹{optimized_df['Investment_Capital'].sum():,.0f}"
)

st.dataframe(
    optimized_df[
        ["Project_ID", "Department", "Investment_Capital", "Allocation_Score"]
    ],
    use_container_width=True
)

st.divider()

# ----------------------------------------------------
# SHAP Explainability
# ----------------------------------------------------
st.subheader("🔍 AI Explainability (SHAP)")

explainer = shap.TreeExplainer(roi_model)
shap_values = explainer.shap_values(X)

shap_df = pd.DataFrame(
    np.abs(shap_values),
    columns=features
)

st.markdown("**Feature Importance for ROI Prediction**")

fig_shap = px.bar(
    shap_df.mean().sort_values(ascending=False),
    orientation="h",
    title="SHAP Feature Impact on ROI"
)

st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# ----------------------------------------------------
# DIFFERENT 3D GRAPH (NEW)
# ----------------------------------------------------
st.subheader("🧊 Strategic Risk–Priority 3D View")

fig_3d_new = px.scatter_3d(
    df,
    x="Strategic_Alignment",
    y="Risk_Score",
    z="Allocation_Score",
    color="Department",
    size="Investment_Capital",
    title="Strategy vs Risk vs Allocation Priority"
)

fig_3d_new.update_traces(marker=dict(opacity=0.85))
fig_3d_new.update_layout(margin=dict(l=0, r=0, t=40, b=0))

st.plotly_chart(fig_3d_new, use_container_width=True)

# ----------------------------------------------------
# Business Explanation
# ----------------------------------------------------
st.subheader("🧠 Executive Interpretation")

st.markdown("""
**How this system supports strategic capital allocation:**

- AI models forecast **ROI & NPV** using historical patterns
- SHAP explains **why predictions are made**
- Optimization selects **maximum-value projects under budget**
- 3D visualization highlights **strategy-risk-priority trade-offs**
- Enables **data-driven, defensible investment decisions**
""")

st.success("✅ CAPITALIQ-AI Analysis Completed Successfully")
