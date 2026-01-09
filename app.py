import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------
# App Config
# ----------------------------------------------------
st.set_page_config(
    page_title="CAPITALIQ-AI™ | Capital Allocation Advisor",
    layout="wide",
    page_icon="📊"
)

st.title("📊 CAPITALIQ-AI™")
st.subheader("AI-Driven Capital Allocation Advisor")
st.markdown(
    "**Grant Thornton Bharat LLP – Live Project**  \n"
    "MSc Finance & Analytics | Individual Project"
)

st.divider()

# ----------------------------------------------------
# Upload Section
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload Project Investment Dataset (CSV)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # ------------------------------------------------
    # Data Preview
    # ------------------------------------------------
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # ------------------------------------------------
    # Feature Selection
    # ------------------------------------------------
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

    # ------------------------------------------------
    # ML Models
    # ------------------------------------------------
    roi_model = RandomForestRegressor(n_estimators=200, random_state=42)
    npv_model = RandomForestRegressor(n_estimators=200, random_state=42)

    roi_model.fit(X, y_roi)
    npv_model.fit(X, y_npv)

    df["Predicted_ROI"] = roi_model.predict(X)
    df["Predicted_NPV"] = npv_model.predict(X)

    # ------------------------------------------------
    # Allocation Score
    # ------------------------------------------------
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

    # ------------------------------------------------
    # KPI Section
    # ------------------------------------------------
    st.subheader("📌 Portfolio-Level Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Investment", f"₹{df['Investment_Capital'].sum():,.0f}")
    col2.metric("Avg Predicted ROI", f"{df['Predicted_ROI'].mean():.2f}%")
    col3.metric("Avg Predicted NPV", f"₹{df['Predicted_NPV'].mean():,.0f}")

    st.divider()

    # ------------------------------------------------
    # Allocation Ranking
    # ------------------------------------------------
    st.subheader("🏆 Capital Allocation Priority Ranking")

    fig_rank = px.bar(
        df_sorted.head(15),
        x="Allocation_Score",
        y="Project_ID",
        orientation="h",
        color="Allocation_Score",
        title="Top 15 Projects by Allocation Score"
    )

    st.plotly_chart(fig_rank, use_container_width=True)

    # ------------------------------------------------
    # 3D Visualization
    # ------------------------------------------------
    st.subheader("🧊 3D Risk–Return–Capital Analysis")

    fig_3d = px.scatter_3d(
        df,
        x="Risk_Score",
        y="Predicted_ROI",
        z="Investment_Capital",
        color="Department",
        size="Predicted_NPV",
        title="3D Investment Landscape"
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    # ------------------------------------------------
    # Scenario Analysis
    # ------------------------------------------------
    st.subheader("🔄 Scenario Analysis")

    market_factor = st.slider("Market Trend Adjustment", 0.8, 1.2, 1.0)
    risk_factor = st.slider("Risk Escalation Factor", 1.0, 1.5, 1.0)

    scenario_X = X.copy()
    scenario_X["Market_Trend_Index"] *= market_factor
    scenario_X["Risk_Score"] *= risk_factor

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

    # ------------------------------------------------
    # Explanation
    # ------------------------------------------------
    st.subheader("🧠 Model & Business Explanation")

    st.markdown("""
    **How this system supports finance leaders:**

    - Uses historical ROI & NPV to **predict future performance**
    - Ranks projects using a **transparent allocation score**
    - Balances **return, value, strategy, and risk**
    - Allows leadership to **simulate market and risk shocks**
    - Provides **clear, defensible, data-backed recommendations**
    """)

    st.success("✅ AI-Driven Capital Allocation Analysis Completed")
