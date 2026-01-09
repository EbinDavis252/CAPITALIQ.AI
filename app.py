import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from scipy.optimize import linprog
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
# Machine Learning Models (Forecasting)
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

# ----------------------------------------------------
# Portfolio KPIs
# ----------------------------------------------------
st.subheader("📌 Portfolio Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Capital Invested", f"₹{df['Investment_Capital'].sum():,.0f}")
c2.metric("Average Predicted ROI", f"{df['Predicted_ROI'].mean():.2f}%")
c3.metric("Average Predicted NPV", f"₹{df['Predicted_NPV'].mean():,.0f}")

st.divider()

# ----------------------------------------------------
# AI-Recommended Capital Allocation Strategy
# ----------------------------------------------------
st.subheader("🏆 AI-Recommended Capital Allocation Strategy")

df_ranked = df.sort_values("Allocation_Score", ascending=False)

fig_rank = px.bar(
    df_ranked.head(15),
    x="Allocation_Score",
    y="Project_ID",
    orientation="h",
    color="Allocation_Score",
    title="Top Projects by AI Allocation Score"
)

st.plotly_chart(fig_rank, use_container_width=True)

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

c = -df["Allocation_Score"].values
A = [df["Investment_Capital"].values]
b = [budget]
bounds = [(0, 1) for _ in range(len(df))]

result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

df["Selected"] = result.x.round(0)
optimized_df = df[df["Selected"] == 1]

st.success(
    f"Optimized Capital Utilized: ₹{optimized_df['Investment_Capital'].sum():,.0f}"
)

st.dataframe(
    optimized_df[
        ["Project_ID", "Department", "Investment_Capital", "Allocation_Score"]
    ],
    use_container_width=True
)

# ----------------------------------------------------
# Executive Recommendation Summary
# ----------------------------------------------------
st.subheader("📌 Executive Recommendation Summary")

top_projects = optimized_df.head(5)
project_list = ", ".join(top_projects["Project_ID"].astype(str).tolist())

st.markdown(f"""
### 🔍 Recommended Projects
**{project_list}**

### 💡 Rationale for Selection
- Selected projects demonstrate **superior AI-predicted ROI and NPV**
- Strong **strategic alignment** with organizational priorities
- Favorable **risk-adjusted performance** compared to alternatives

### 💰 Capital Budget Constraint
- **Available Capital Budget:** ₹{budget:,.0f}  
- **Capital Utilized:** ₹{optimized_df['Investment_Capital'].sum():,.0f}  
- Portfolio is optimized to **maximize value while remaining within budget limits**

### 🧠 Decision Support Insight
This AI-driven recommendation provides finance leaders with a
**clear, data-backed, and defensible basis** for capital allocation decisions.
""")

st.divider()

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
    title="Scenario Impact on ROI and NPV"
)

st.plotly_chart(fig_scenario, use_container_width=True)

st.divider()

# ----------------------------------------------------
# AI Explainability (Feature Importance)
# ----------------------------------------------------
st.subheader("🔍 AI Explainability")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": roi_model.feature_importances_
}).sort_values("Importance", ascending=False)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance for ROI Prediction"
)

st.plotly_chart(fig_importance, use_container_width=True)

st.divider()

# ----------------------------------------------------
# 3D Strategic Visualization
# ----------------------------------------------------
st.subheader("🧊 Strategic Risk–Priority 3D View")

fig_3d = px.scatter_3d(
    df,
    x="Strategic_Alignment",
    y="Risk_Score",
    z="Allocation_Score",
    color="Department",
    size="Investment_Capital",
    title="Strategy vs Risk vs Allocation Priority"
)

fig_3d.update_traces(marker=dict(opacity=0.85))
fig_3d.update_layout(margin=dict(l=0, r=0, t=40, b=0))

st.plotly_chart(fig_3d, use_container_width=True)

# ----------------------------------------------------
# Final Business Interpretation
# ----------------------------------------------------
st.subheader("🧠 Business Interpretation")

st.markdown("""
This AI-driven capital allocation system integrates forecasting,
optimization, scenario analysis, and explainability to support
**strategic, transparent, and value-maximizing investment decisions**
for finance leadership.
""")

st.success("✅ CAPITALIQ-AI Analysis Completed Successfully")
