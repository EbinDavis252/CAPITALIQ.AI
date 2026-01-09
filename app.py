import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CAPITALIQ.AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CAPITALIQ.AI")
st.subheader("AI-Driven Capital Allocation Advisor")
st.markdown("""
A professional AI-based decision support system to help finance leaders
allocate capital efficiently using risk-return analytics, forecasting,
and scenario evaluation.
""")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Select Module",
    [
        "Upload Data",
        "Market Data (Yahoo Finance)",
        "AI Scoring Engine",
        "Capital Allocation",
        "Scenario Analysis"
    ]
)

# ---------------- UPLOAD DATA ----------------
if section == "Upload Data":
    st.header("📁 Upload Investment Dataset")
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.success("Dataset uploaded successfully")
        st.dataframe(df)

        st.info(
            "This dataset will be used across AI scoring, capital allocation, "
            "and scenario analysis modules."
        )

# ---------------- MARKET DATA ----------------
elif section == "Market Data (Yahoo Finance)":
    st.header("📈 Market Benchmarking (Yahoo Finance)")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)")
    period = st.selectbox("Select Time Period", ["1y", "3y", "5y", "10y"])

    if ticker:
        data = yf.download(ticker, period=period)
        st.dataframe(data.tail())

        fig = px.line(
            data,
            x=data.index,
            y="Close",
            title=f"{ticker} Closing Price Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Market data can be used as a proxy for expected returns, "
            "volatility benchmarking, and scenario calibration."
        )

# ---------------- AI SCORING ----------------
elif section == "AI Scoring Engine":
    st.header("🤖 AI-Based Project Scoring")

    file = st.file_uploader("Upload dataset", type=["csv"], key="ai")
    if file:
        df = pd.read_csv(file)

        required_cols = {"Expected_Return", "Risk_Score"}
        if not required_cols.issubset(df.columns):
            st.error("Dataset must contain Expected_Return and Risk_Score columns.")
        else:
            df["Risk_Adjusted_Return"] = df["Expected_Return"] / df["Risk_Score"]

            scaler = MinMaxScaler()
            df["AI_Score"] = scaler.fit_transform(
                df[["Risk_Adjusted_Return"]]
            )

            st.success("AI scoring completed successfully")
            st.dataframe(df)

            fig = px.scatter(
                df,
                x="Risk_Score",
                y="Expected_Return",
                size="AI_Score",
                color="AI_Score",
                title="Risk vs Return with AI-Based Scoring"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Business Interpretation:**  
            Projects with higher AI scores offer superior return potential
            per unit of risk, making them more attractive for capital deployment.
            """)

# ---------------- CAPITAL ALLOCATION ----------------
elif section == "Capital Allocation":
    st.header("💰 Optimal Capital Allocation")

    file = st.file_uploader("Upload dataset", type=["csv"], key="alloc")
    if file:
        df = pd.read_csv(file)

        required_cols = {"Project_Name", "Expected_Return", "Risk_Score"}
        if not required_cols.issubset(df.columns):
            st.error("Dataset must contain Project_Name, Expected_Return, Risk_Score.")
        else:
            # Recompute AI Score safely
            df["Risk_Adjusted_Return"] = df["Expected_Return"] / df["Risk_Score"]
            scaler = MinMaxScaler()
            df["AI_Score"] = scaler.fit_transform(
                df[["Risk_Adjusted_Return"]]
            )

            total_budget = st.number_input(
                "Enter Total Capital Budget",
                min_value=0.0,
                value=1_000_000.0,
                step=100_000.0
            )

            df["Allocation_Weight"] = df["AI_Score"] / df["AI_Score"].sum()
            df["Allocated_Capital"] = df["Allocation_Weight"] * total_budget

            st.dataframe(
                df[["Project_Name", "Allocated_Capital"]]
            )

            fig = px.pie(
                df,
                names="Project_Name",
                values="Allocated_Capital",
                title="AI-Optimized Capital Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Allocation Logic:**  
            Capital is distributed proportionally based on AI-derived project
            attractiveness, ensuring optimal deployment under budget constraints.
            """)

# ---------------- SCENARIO ANALYSIS ----------------
elif section == "Scenario Analysis":
    st.header("🔍 Scenario Analysis")

    file = st.file_uploader("Upload dataset", type=["csv"], key="scenario")
    if file:
        df = pd.read_csv(file)

        if "Expected_Return" not in df.columns:
            st.error("Dataset must contain Expected_Return.")
        else:
            scenarios = {
                "Bull Case": 1.2,
                "Base Case": 1.0,
                "Bear Case": 0.8
            }

            scenario_data = []
            for name, factor in scenarios.items():
                temp = df.copy()
                temp["Scenario"] = name
                temp["Adjusted_Return"] = temp["Expected_Return"] * factor
                scenario_data.append(temp)

            scenario_df = pd.concat(scenario_data)

            fig = px.bar(
                scenario_df,
                x="Project_Name",
                y="Adjusted_Return",
                color="Scenario",
                barmode="group",
                title="Scenario-Based Expected Returns"
            )
            st.plotly_chart(fig, use_container_width=True)

            fig3d = px.scatter_3d(
                scenario_df,
                x="Risk_Score",
                y="Adjusted_Return",
                z="Initial_Investment",
                color="Scenario",
                title="3D Risk–Return–Investment Landscape"
            )
            st.plotly_chart(fig3d, use_container_width=True)

            st.markdown("""
            **Scenario Insight:**  
            This analysis evaluates project robustness across optimistic,
            normal, and stressed market environments.
            """)
