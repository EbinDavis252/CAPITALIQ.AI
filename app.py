import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linprog

# ----------------------------------------------------
# 1. Page Configuration & Professional Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="CAPITALIQ-AI™ | Strategic Allocation",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for Background & Enterprise Look
st.markdown("""
    <style>
    /* Background Image */
    .stApp {
        background-image: linear_gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.9)), 
                          url('https://images.unsplash.com/photo-1611974765270-ca1258634369?q=80&w=2070&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Typography & Colors */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stMarkdown p {
        color: #e0e0e0 !important;
    }
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        color: #00CC96 !important;
        font-size: 28px !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #374151;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Sidebar: Global Strategy Controls
# ----------------------------------------------------
with st.sidebar:
    st.title("⚡ CAPITALIQ-AI™")
    st.caption("Strategic Decision Support System")
    st.markdown("---")
    
    st.subheader("1. Data Intelligence")
    hist_file = st.file_uploader("📂 Historical Data (Train)", type=["csv"])
    prop_file = st.file_uploader("📂 New Proposals (Predict)", type=["csv"])
    
    st.markdown("---")
    st.subheader("2. Strategic Constraints")
    budget_input = st.number_input("💰 Capital Budget (₹)", value=15000000.0, step=1000000.0)
    max_risk = st.slider("⚠️ Max Portfolio Risk", 1.0, 10.0, 6.5)
    
    st.markdown("---")
    st.subheader("3. Scenario Manager")
    market_shock = st.slider("📉 Market Shock Factor", -0.20, 0.20, 0.0, 0.01, help="Simulate a market crash (-20%) or boom (+20%)")
    
    st.markdown("---")
    st.info("Live Project: Grant Thornton Bharat LLP")

# ----------------------------------------------------
# 3. Main Application Logic
# ----------------------------------------------------
st.title("📈 Executive Capital Command Center")
st.markdown("_AI-Powered Analytics for High-Stakes Investment Decisions_")

if hist_file is None or prop_file is None:
    st.warning("⚠️ **System Standby:** Please upload project data to initialize the Strategic Engine.")
    st.stop()

# Load Data
@st.cache_data
def load_data(h, p):
    return pd.read_csv(h), pd.read_csv(p)

df_hist, df_prop = load_data(hist_file, prop_file)

# ----------------------------------------------------
# 4. Advanced ML Pipeline
# ----------------------------------------------------
features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
targets = ["Actual_ROI_Pct", "Actual_NPV"]

# Validate Columns
if not all(c in df_hist.columns for c in features + targets):
    st.error("❌ Data Schema Mismatch. Please check column headers.")
    st.stop()

# Train Models
with st.spinner('⚙️ Calibrating AI Models...'):
    rf_roi = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_npv = RandomForestRegressor(n_estimators=200, random_state=42)
    
    rf_roi.fit(df_hist[features], df_hist["Actual_ROI_Pct"])
    rf_npv.fit(df_hist[features], df_hist["Actual_NPV"])

# Prediction with Scenario Adjustment
df_prop["Pred_ROI"] = rf_roi.predict(df_prop[features]) * (1 + market_shock) # Apply Market Shock
df_prop["Pred_NPV"] = rf_npv.predict(df_prop[features]) * (1 + market_shock) # Apply Market Shock

# ----------------------------------------------------
# 5. Advanced Optimization Engine
# ----------------------------------------------------
# Linear Programming (Knapsack Problem)
c = -df_prop["Pred_NPV"].values 
A = [df_prop["Investment_Capital"].values]
b = [budget_input]
bounds = [(0, 1) for _ in range(len(df_prop))]

res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
df_prop["Selected"] = res.x.round(0) if res.success else 0
portfolio = df_prop[df_prop["Selected"] == 1]

# ----------------------------------------------------
# 6. Enterprise Dashboard Tabs
# ----------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary", 
    "🧠 Advanced Analytics",
    "⚡ Efficient Frontier", 
    "💰 Optimization Details",
    "🧊 Strategic 3D View"
])

# --- TAB 1: EXECUTIVE SUMMARY ---
with tab1:
    st.subheader("📌 Portfolio Performance at a Glance")
    
    # Hero Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects Funded", f"{len(portfolio)}", f"out of {len(df_prop)}")
    c2.metric("Capital Deployed", f"₹{portfolio['Investment_Capital'].sum()/1e6:.1f}M", f"Budget: ₹{budget_input/1e6:.1f}M")
    c3.metric("Projected NPV", f"₹{portfolio['Pred_NPV'].sum()/1e6:.2f}M", delta=f"{market_shock*100:+.0f}% Market Adj")
    c4.metric("Risk Profile", f"{portfolio['Risk_Score'].mean():.2f}", delta_color="inverse", delta="vs Max 6.5")

    st.divider()
    
    # Dual Charts
    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        st.markdown("##### 📅 ROI Forecast by Department")
        fig_bar = px.bar(
            portfolio, x="Department", y="Pred_ROI", color="Risk_Score",
            title="Avg ROI per Dept (colored by Risk)", barmode='group',
            color_continuous_scale='Redor'
        )
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_r:
        st.markdown("##### 🥧 Capital Allocation")
        fig_pie = px.pie(portfolio, values='Investment_Capital', names='Department', hole=0.5)
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 2: ADVANCED ANALYTICS ---
with tab2:
    st.subheader("🔍 Deep Dive: Correlations & Drivers")
    
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.markdown("##### 🔥 Correlation Heatmap")
        st.caption("How do features interact? (e.g., Does high Risk actually yield high ROI?)")
        
        # Compute Correlation
        corr_matrix = df_prop[features + ["Pred_ROI", "Pred_NPV"]].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_corr, use_container_width=True)

    with c_right:
        st.markdown("##### 📊 Distribution of Predicted Returns")
        fig_dist = px.histogram(
            df_prop, x="Pred_ROI", nbins=20, color="Department",
            title="ROI Distribution across all Proposals",
            marginal="box" # Adds a boxplot on top
        )
        fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_dist, use_container_width=True)

# --- TAB 3: EFFICIENT FRONTIER ---
with tab3:
    st.subheader("⚡ Efficient Frontier Analysis")
    st.markdown("Simulating **2,000 Random Portfolios** to visualize the Risk-Return Trade-off.")
    
    if st.button("🔄 Run Efficient Frontier Simulation"):
        with st.spinner("Simulating market scenarios..."):
            results = []
            for _ in range(2000):
                # Randomly select projects
                mask = np.random.rand(len(df_prop)) > 0.5
                sample = df_prop[mask]
                
                if sample["Investment_Capital"].sum() <= budget_input:
                    results.append({
                        "Risk": sample["Risk_Score"].mean(),
                        "Return": sample["Pred_ROI"].mean(),
                        "NPV": sample["Pred_NPV"].sum(),
                        "Count": len(sample)
                    })
            
            sim_df = pd.DataFrame(results)
            
            # Plot
            fig_ef = px.scatter(
                sim_df, x="Risk", y="Return", color="NPV",
                title="Efficient Frontier: Risk vs Return",
                labels={"Risk": "Portfolio Risk (Avg)", "Return": "Portfolio ROI (%)"},
                color_continuous_scale="Viridis",
                hover_data=["Count"]
            )
            
            # Mark Current Optimized Portfolio
            current_risk = portfolio["Risk_Score"].mean()
            current_return = portfolio["Pred_ROI"].mean()
            
            fig_ef.add_trace(go.Scatter(
                x=[current_risk], y=[current_return],
                mode='markers+text', marker=dict(color='red', size=15, symbol='star'),
                text=["AI Optimized"], textposition="top center", name="Selected"
            ))
            
            fig_ef.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_ef, use_container_width=True)

# --- TAB 4: OPTIMIZATION DETAILS ---
with tab4:
    st.subheader("📋 Detailed Selection Report")
    
    st.dataframe(
        portfolio[["Project_ID", "Department", "Investment_Capital", "Pred_ROI", "Pred_NPV", "Risk_Score", "Strategic_Alignment"]]
        .style.background_gradient(subset=["Pred_NPV"], cmap="Greens")
        .format({"Investment_Capital": "₹{:.0f}", "Pred_ROI": "{:.2f}%", "Pred_NPV": "₹{:.0f}"}),
        use_container_width=True
    )
    
    # Download Button
    csv = portfolio.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Portfolio Report (CSV)",
        data=csv,
        file_name="GrantThornton_Optimized_Portfolio.csv",
        mime="text/csv",
    )

# --- TAB 5: 3D STRATEGIC VIEW ---
with tab5:
    st.subheader("🧊 Multi-Dimensional Strategy Map")
    
    df_prop["Status"] = df_prop["Selected"].apply(lambda x: "Selected" if x==1 else "Rejected")
    
    fig_3d = px.scatter_3d(
        df_prop,
        x="Risk_Score",
        y="Strategic_Alignment",
        z="Pred_ROI",
        color="Status",
        color_discrete_map={"Selected": "#00CC96", "Rejected": "#EF553B"},
        size="Investment_Capital",
        opacity=0.9,
        title="Risk (X) vs Strategy (Y) vs ROI (Z)"
    )
    fig_3d.update_layout(
        scene = dict(
            xaxis = dict(backgroundcolor="rgb(0, 0, 0)", gridcolor="gray", showbackground=True),
            yaxis = dict(backgroundcolor="rgb(0, 0, 0)", gridcolor="gray", showbackground=True),
            zaxis = dict(backgroundcolor="rgb(0, 0, 0)", gridcolor="gray", showbackground=True),
        ),
        paper_bgcolor='rgba(0,0,0,0)', 
        font_color='white',
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# Footer
st.markdown("---")
st.caption("© 2026 CAPITALIQ-AI™ | Developed by KD | Grant Thornton Bharat LLP Live Project")
