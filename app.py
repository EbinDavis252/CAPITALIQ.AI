import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linprog

# ----------------------------------------------------
# 1. Page Configuration & "Glassmorphism" CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="CAPITALIQ-AI™ | Enterprise Edition",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# --- THEME ENGINE: GLASSMORPHISM & HIGH CONTRAST ---
st.markdown("""
    <style>
    /* 1. Global Background (Dark Luxury) */
    .stApp {
        background-image: linear_gradient(rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.95)), 
                          url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop');
        background-size: cover;
        background-attachment: fixed;
    }

    /* 2. Text Visibility Fixes (Force White) */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, li, span, label, .stDataFrame, .stRadio label {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* 3. Glass Cards */
    div[data-testid="stMetric"], div[data-testid="stExpander"] {
        background-color: rgba(30, 41, 59, 0.7); 
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }

    /* 4. Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a; 
        border-right: 1px solid #334155;
    }
    
    /* 5. Custom Radio Button (Navigation) */
    div[role="radiogroup"] > label > div:first-of-type {
        background-color: #0f172a;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Sidebar: Navigation & Strategy Controls
# ----------------------------------------------------
with st.sidebar:
    st.title("💎 CAPITALIQ-AI™")
    st.caption("Advanced Decision Support System")
    st.markdown("---")
    
    # --- A. NAVIGATION (Replaces Tabs) ---
    st.subheader("📍 Navigation")
    selected_page = st.radio(
        "Go to:", 
        ["🚀 Executive Summary", "🧠 AI Insights", "⚡ Efficient Frontier", "💰 Optimization Report", "🧊 Strategic 3D Map"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # --- B. STRATEGIC CONSTRAINTS ---
    st.subheader("⚙️ Strategic Constraints")
    budget_input = st.number_input("💰 Capital Budget (₹)", value=15000000.0, step=1000000.0)
    max_risk = st.slider("⚠️ Max Portfolio Risk", 1.0, 10.0, 6.5)
    
    st.markdown("---")
    
    # --- C. MARKET SIMULATOR ---
    st.subheader("📉 Market Simulator")
    market_shock = st.slider("Market Shock Factor", -0.20, 0.20, 0.0, 0.01, format="%+.0f%%")
    
    st.info("Live Project: Grant Thornton Bharat LLP")

# ----------------------------------------------------
# 3. Main Page: Data Ingestion (Moved Here)
# ----------------------------------------------------
st.title("📊 Executive Capital Command Center")

# --- DATA UPLOAD SECTION (Expanded by default until data is loaded) ---
with st.expander("📂 Data Initialization & Configuration", expanded=True):
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        hist_file = st.file_uploader("1. Upload Historical Data (Train)", type=["csv"])
    with col_u2:
        prop_file = st.file_uploader("2. Upload New Proposals (Predict)", type=["csv"])

    # Quick check to stop execution if data is missing
    if hist_file is None or prop_file is None:
        st.warning("⚠️ **System Standby:** Awaiting Data Streams...")
        st.stop() # Stops the script here until files are present

# Load Data
@st.cache_data
def load_data(h, p):
    return pd.read_csv(h), pd.read_csv(p)

df_hist, df_prop = load_data(hist_file, prop_file)

# ----------------------------------------------------
# 4. Advanced ML Pipeline (Runs regardless of page selection)
# ----------------------------------------------------
features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]

# Train Models
with st.spinner('⚙️ Calibrating AI Models & Optimizing Portfolio...'):
    rf_roi = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_npv = RandomForestRegressor(n_estimators=200, random_state=42)
    
    try:
        rf_roi.fit(df_hist[features], df_hist["Actual_ROI_Pct"])
        rf_npv.fit(df_hist[features], df_hist["Actual_NPV"])
    except Exception as e:
        st.error(f"Data Mismatch Error: {e}")
        st.stop()

    # Prediction with Scenario Adjustment
    df_prop["Pred_ROI"] = rf_roi.predict(df_prop[features]) * (1 + market_shock)
    df_prop["Pred_NPV"] = rf_npv.predict(df_prop[features]) * (1 + market_shock)
    df_prop["Efficiency_Ratio"] = df_prop["Pred_ROI"] / df_prop["Risk_Score"]

    # Optimization Engine
    c = -df_prop["Pred_NPV"].values 
    A = [df_prop["Investment_Capital"].values]
    b = [budget_input]
    bounds = [(0, 1) for _ in range(len(df_prop))]

    # Using 'highs' method for robustness
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    df_prop["Selected"] = res.x.round(0) if res.success else 0
    portfolio = df_prop[df_prop["Selected"] == 1]

# Common Plotly Layout for Dark Theme
def dark_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0")
    )
    return fig

# ----------------------------------------------------
# 5. Dynamic Page Rendering (Based on Sidebar Selection)
# ----------------------------------------------------

# --- PAGE 1: EXECUTIVE SUMMARY ---
if selected_page == "🚀 Executive Summary":
    st.subheader("📌 Portfolio Performance Snapshot")
    
    # Hero Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects Funded", f"{len(portfolio)}", f"out of {len(df_prop)}")
    c2.metric("Capital Deployed", f"₹{portfolio['Investment_Capital'].sum()/1e6:.1f}M", f"Limit: ₹{budget_input/1e6:.1f}M")
    c3.metric("Projected NPV", f"₹{portfolio['Pred_NPV'].sum()/1e6:.2f}M", delta=f"{market_shock*100:+.0f}% Scenario")
    c4.metric("Avg Risk Score", f"{portfolio['Risk_Score'].mean():.2f}", delta="Target < 6.5", delta_color="off")

    st.markdown("---")
    
    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        st.markdown("##### 📅 Projected ROI by Department")
        if not portfolio.empty:
            fig_bar = px.bar(
                portfolio, x="Department", y="Pred_ROI", color="Risk_Score",
                barmode='group', color_continuous_scale='Tealgrn',
                text_auto='.1f'
            )
            st.plotly_chart(dark_chart(fig_bar), use_container_width=True)
        else:
            st.info("No projects selected. Increase budget.")
    
    with col_r:
        st.markdown("##### 🥧 Budget Allocation")
        if not portfolio.empty:
            fig_pie = px.pie(portfolio, values='Investment_Capital', names='Department', hole=0.6, color_discrete_sequence=px.colors.sequential.Tealgrn_r)
            st.plotly_chart(dark_chart(fig_pie), use_container_width=True)

# --- PAGE 2: AI INSIGHTS ---
elif selected_page == "🧠 AI Insights":
    st.subheader("🔍 Deep Dive: Why these projects?")
    
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.markdown("##### 🔥 Correlation Drivers")
        corr_matrix = df_prop[features + ["Pred_ROI", "Pred_NPV"]].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        st.plotly_chart(dark_chart(fig_corr), use_container_width=True)

    with c_right:
        st.markdown("##### 📊 ROI vs Risk Distribution")
        fig_scat = px.scatter(
            df_prop, x="Risk_Score", y="Pred_ROI", color="Department",
            size="Investment_Capital", hover_data=["Project_ID"],
            title="All Proposals: Risk vs Reward"
        )
        st.plotly_chart(dark_chart(fig_scat), use_container_width=True)

# --- PAGE 3: EFFICIENT FRONTIER ---
elif selected_page == "⚡ Efficient Frontier":
    st.subheader("⚡ Efficient Frontier Simulation")
    st.markdown("Simulating **2,000 Portfolios** to find the optimal Risk-Return trade-off.")
    
    if st.button("🔄 Run Simulation"):
        with st.spinner("Crunching numbers..."):
            results = []
            total_cap = df_prop["Investment_Capital"].sum()
            avg_p = min(0.5, budget_input / (total_cap + 1)) 
            
            for _ in range(2000):
                mask = np.random.rand(len(df_prop)) < avg_p
                sample = df_prop[mask]
                if not sample.empty and sample["Investment_Capital"].sum() <= budget_input:
                    results.append({
                        "Risk": sample["Risk_Score"].mean(),
                        "Return": sample["Pred_ROI"].mean(),
                        "NPV": sample["Pred_NPV"].sum()
                    })
            
            sim_df = pd.DataFrame(results)
            
            if not sim_df.empty:
                fig_ef = px.scatter(
                    sim_df, x="Risk", y="Return", color="NPV",
                    labels={"Risk": "Portfolio Risk", "Return": "Portfolio ROI (%)"},
                    color_continuous_scale="Viridis"
                )
                # Add Current Portfolio Marker
                if not portfolio.empty:
                    fig_ef.add_trace(go.Scatter(
                        x=[portfolio["Risk_Score"].mean()], 
                        y=[portfolio["Pred_ROI"].mean()],
                        mode='markers', marker=dict(color='red', size=20, symbol='star'),
                        name="AI Selection"
                    ))
                st.plotly_chart(dark_chart(fig_ef), use_container_width=True)
            else:
                st.error("Simulation constraint too tight. Check budget.")

# --- PAGE 4: OPTIMIZATION REPORT ---
elif selected_page == "💰 Optimization Report":
    st.subheader("📋 Detailed Selection Report")
    st.markdown("Projects are ranked by **AI Score** and **Efficiency Ratio**.")
    
    display_cols = ["Project_ID", "Department", "Investment_Capital", "Pred_ROI", "Risk_Score", "Efficiency_Ratio"]
    
    st.dataframe(
        portfolio[display_cols]
        .sort_values("Efficiency_Ratio", ascending=False)
        .style.background_gradient(subset=["Efficiency_Ratio"], cmap="Greens")
        .format({"Investment_Capital": "₹{:.0f}", "Pred_ROI": "{:.1f}%", "Efficiency_Ratio": "{:.2f}"}),
        use_container_width=True,
        height=500
    )
    
    # Export
    csv = portfolio.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Report (CSV)", csv, "Optimized_Portfolio.csv", "text/csv")

# --- PAGE 5: 3D MAP ---
elif selected_page == "🧊 Strategic 3D Map":
    st.subheader("🧊 Strategic 3D Landscape")
    
    df_prop["Status"] = df_prop["Selected"].apply(lambda x: "Selected" if x==1 else "Rejected")
    
    fig_3d = px.scatter_3d(
        df_prop,
        x="Risk_Score", y="Strategic_Alignment", z="Pred_ROI",
        color="Status", color_discrete_map={"Selected": "#00e676", "Rejected": "#ff1744"},
        size="Investment_Capital", opacity=0.9
    )
    fig_3d.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)")
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig_3d, use_container_width=True)
