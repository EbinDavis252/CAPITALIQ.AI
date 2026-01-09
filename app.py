import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="CAPITALIQ.AI",
    layout="wide",
    page_icon="📊"
)

st.title("📊 CAPITALIQ.AI")
st.subheader("AI-Driven Capital Allocation Advisor")
st.markdown("""
*A decision-support system designed to help finance leaders allocate capital optimally using AI, risk analytics, and scenario-based insights.*
""")

# ------------------- SIDEBAR -------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Upload Data", "Market Data (Yahoo Finance)", "AI Analysis", "Capital Allocation", "Scenario Analysis"]
)

# ------------------- UPLOAD DATA -------------------
if section == "Upload Data":
    st.header("📁 Upload Investment Dataset")
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.success("Dataset successfully uploaded")
        st.dataframe(df)

        st.markdown("""
        **Dataset Overview:**  
        The uploaded data will be used to evaluate investment attractiveness, forecast returns,
        assess risk, and generate capital allocation recommendations.
        """)

# ------------------- MARKET DATA -------------------
elif section == "Market Data (Yahoo Finance)":
    st.header("📈 Fetch Market Data (Yahoo Finance)")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)")
    period = st.selectbox("Select Time Period", ["1y", "3y", "5y", "10y"])

    if ticker:
        data = yf.download(ticker, period=period)
        st.dataframe(data.tail())

        fig = px.line(
            data,
            x=data.index,
            y="Close",
            title=f"{ticker} Stock Price Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("Market data can be used as a benchmark or proxy for expected returns and volatility.")

# ------------------- AI ANALYSIS -------------------
elif section == "AI Analysis":
    st.header("🤖 AI-Driven Investment Analysis")

    file = st.file_uploader("Upload dataset for AI analysis", type=["csv"], key="ai")
    if file:
        df = pd.read_csv(file)

        st.subheader("Data Preview")
        st.dataframe(df)

        if {"Initial_Investment", "Expected_Return", "Risk_Score"}.issubset(df.columns):
            scaler = MinMaxScaler()
            df["Risk_Adjusted_Return"] = (
                df["Expected_Return"] / df["Risk_Score"]
            )

            df["AI_Score"] = scaler.fit_transform(
                df[["Risk_Adjusted_Return"]]
            )

            st.success("AI-based scoring completed")

            fig = px.scatter(
                df,
                x="Risk_Score",
                y="Expected_Return",
                size="AI_Score",
                color="AI_Score",
                title="Risk vs Return with AI Scoring"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Explanation:**  
            The AI score normalizes risk-adjusted returns to rank projects objectively.
            Higher AI scores indicate superior return potential per unit of risk.
            """)

# ------------------- CAPITAL ALLOCATION -------------------
elif section == "Capital Allocation":
    st.header("💰 Optimal Capital Allocation")

    file = st.file_uploader("Upload dataset for allocation", type=["csv"], key="alloc")
    if file:
        df = pd.read_csv(file)

        total_budget = st.number_input("Enter Total Capital Budget", value=1_000_000)

        df["Weight"] = df["AI_Score"] / df["AI_Score"].sum()
        df["Allocated_Capital"] = df["Weight"] * total_budget

        st.dataframe(df[["Project_Name", "Allocated_Capital"]])

        fig = px.pie(
            df,
            names="Project_Name",
            values="Allocated_Capital",
            title="Capital Allocation Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Allocation Logic:**  
        Capital is allocated proportionally based on AI-generated project attractiveness,
        ensuring optimal utilization under budget constraints.
        """)

# ------------------- SCENARIO ANALYSIS -------------------
elif section == "Scenario Analysis":
    st.header("🔍 Scenario-Based Analysis")

    file = st.file_uploader("Upload dataset for scenarios", type=["csv"], key="sc")
    if file:
        df = pd.read_csv(file)

        scenarios = {
            "Bull Case": 1.2,
            "Base Case": 1.0,
            "Bear Case": 0.8
        }

        scenario_df = []

        for scenario, factor in scenarios.items():
            temp = df.copy()
            temp["Scenario"] = scenario
            temp["Adjusted_Return"] = temp["Expected_Return"] * factor
            scenario_df.append(temp)

        scenario_df = pd.concat(scenario_df)

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
            title="3D Scenario Risk-Return-Investment View"
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.markdown("""
        **Scenario Analysis Insight:**  
        Evaluates project resilience under optimistic, normal, and stressed market conditions,
        helping leadership make informed, forward-looking decisions.
        """)
