import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px

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
# File Upload
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload Capital Allocation Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload the dataset to begin the analysis.")
    st.stop()

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------
df = pd.read_csv(uploaded_file)

# ----------------------------------------------------
# Column Validation (CRITICAL)
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

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

st.success("✅ Dataset uploaded and validated successfully")

# ----------------------------------------------------
# Dataset Preview
# ----------------------------------------------------
st.subheader("🔍 Dataset Preview")
st.dataframe(df, use_container_width=True)

st.divider()

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
# Capital Allocation Score
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

df_sorted = df.sort_values("Allocation_Score", ascending=False)

# ----------------------------------------------------
# KPIs
# ----------------------------------------------------
st.subheader("📌 Portfolio Overview")

kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(
    "Total Capital Invested",
    f"₹{df['Investment_Capital'].sum():,.0f}"
)

kpi2.metric(
    "Average Predicted ROI",
    f"{df['Predicted_ROI'].mean():.2f}%"
)

kpi3.metric(
    "Average Predicted NPV",
    f"₹{df['Predicted_NPV'].mean():,.0f}"
)

st.divider()

# ----------------------------------------------------
# Allocation Ranking
# ----------------------------------------------------
st.subheader("🏆 Capital Allocation Priority")

fig_rank = px.bar(
    df_sorted.head(15),
    x="Allocation_Score",
    y="Project_ID",
    orientation="h",
    color="Allocation_Score",
    title="Top 15 Projects by Allocation Score"
)

st.plotly_chart(fig_rank, use_container_width=True)

# ----------------------------------------------------
# SAFE SIZE SCALING FOR 3D VISUAL
# ----------------------------------------------------
df["NPV_For_Size"] = df["Predicted_NPV"].replace(
    [np.inf, -np.inf], np.nan
)

df["NPV_For_Size"] = df["NPV_For_Size"].fillna(
    df["NPV_For_Size"].median()
)

df["NPV_For_Size"] = df["NPV_For_Size"].clip(lower=1)

size_scaler = MinMaxScaler(feature_range=(5, 40))
df["NPV_Size_Scaled"] = size_scaler.fit_transform(
    df[["NPV_For_Size"]]
)

# ----------------------------------------------------
# 3D Visualization
# ----------------------------------------------------
st.subheader("🧊 3D Investment Landscape")

fig_3d = px.scatter_3d(
    df,
    x="Risk_Score",
    y="Predicted_ROI",
    z="Investment_Capital",
    color="Department",
    size="NPV_Size_Scaled",
    title="Risk vs Return vs Capital Allocation"
)

fig_3d.update_traces(marker=dict(opacity=0.85))
fig_3d.update_layout(margin=dict(l=0, r=0, t=40, b=0))

st.plotly_chart(fig_3d, use_container_width=True)

# ----------------------------------------------------
# Scenario Analysis
# ----------------------------------------------------
st.subheader("🔄 Scenario Analysis")

market_adj = st.slider(
    "Market Trend Adjustment",
    0.8, 1.2, 1.0, 0.05
)

risk_adj = st.slider(
    "Risk Escalation Factor",
    1.0, 1.5, 1.0, 0.05
)

scenario_X = X.copy()
scenario_X["Market_Trend_Index"] *= market_adj
scenario_X["Risk_Score"] *= risk_adj

df["Scenario_ROI"] = roi_model.predict(scenario_X)
df["Scenario_NPV"] = npv_model.predict(scenario_X)

fig_scenario = px.scatter(
    df,
    x="Scenario_ROI",
    y="Scenario_NPV",
    color="Department",
    size="Investment_Capital",
    title="Scenario Impact on ROI & NPV"
)

st.plotly_chart(fig_scenario, use_container_width=True)

# ----------------------------------------------------
# Business Explanation
# ----------------------------------------------------
st.subheader("🧠 Decision Support Explanation")

st.markdown("""
**How CAPITALIQ-AI supports finance leaders:**

- Uses historical ROI & NPV to **forecast future project performance**
- Applies a **transparent capital allocation score**
- Balances **return, value creation, strategy, and risk**
- Enables **scenario-based stress testing**
- Provides **clear, defensible, AI-backed recommendations**
""")

st.success("✅ AI-Driven Capital Allocation Analysis Completed Successfully")
