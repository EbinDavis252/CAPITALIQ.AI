import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linprog

# ----------------------------------------------------
# 1. Page Configuration & Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="CAPITALIQ-AI™ | Grant Thornton Live Project",
    layout="wide",
    page_icon="📊"
)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {color: #2c3e50;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Sidebar: Configuration & Controls
# ----------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python_logo_notext.svg/121px-Python_logo_notext.svg.png", width=50)
    st.title("CAPITALIQ-AI™")
    st.caption("AI-Driven Capital Allocation System")
    st.markdown("---")
    
    st.subheader("1. Data Ingestion")
    hist_file = st.file_uploader("📂 Upload Historical Data (Training)", type=["csv"], help="Upload 'grant_thornton_project_data.csv'")
    prop_file = st.file_uploader("📂 Upload New Proposals (Forecasting)", type=["csv"], help="Upload 'new_project_proposals.csv'")
    
    st.markdown("---")
    st.subheader("2. Constraints")
    
    # We will update the default budget dynamically if data is loaded
    budget_input = st.number_input("💰 Total Capital Budget (₹)", value=15000000.0, step=500000.0)
    
    max_risk = st.slider("⚠️ Max Portfolio Risk Tolerance", 1.0, 10.0, 6.5, help="Maximum acceptable average risk score (1-10) for the portfolio.")
    
    st.markdown("---")
    st.info("**Project:** Capital Allocation Advisor\n**Client:** Live Project (GT)")

# ----------------------------------------------------
# 3. Main Dashboard Logic
# ----------------------------------------------------
st.title("📊 Executive Decision Support System")
st.markdown("### Intelligent Capital Allocation & Portfolio Optimization")
st.markdown("This system utilizes **Random Forest Regression** for forecasting and **Linear Programming** for budget optimization.")

if hist_file is None or prop_file is None:
    st.warning("⚠️ **Action Required:** Please upload both **Historical Data** and **New Proposals** in the sidebar to begin analysis.")
    st.info("💡 **Tip:** Use the datasets generated/provided in the chat.")
    st.stop()

# Load Data
@st.cache_data
def load_data(h_file, p_file):
    df_h = pd.read_csv(h_file)
    df_p = pd.read_csv(p_file)
    return df_h, df_p

df_hist, df_prop = load_data(hist_file, prop_file)

# ----------------------------------------------------
# 4. Machine Learning Engine
# ----------------------------------------------------
# Define Features and Targets
features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
target_roi = "Actual_ROI_Pct"
target_npv = "Actual_NPV"

# Check if columns exist
required_hist_cols = features + [target_roi, target_npv]
if not all(col in df_hist.columns for col in required_hist_cols):
    st.error(f"❌ Historical Data missing required columns. Expected: {required_hist_cols}")
    st.stop()

required_prop_cols = features
if not all(col in df_prop.columns for col in required_prop_cols):
    st.error(f"❌ Proposals Data missing required columns. Expected: {required_prop_cols}")
    st.stop()

# Train Models
with st.spinner('🧠 Training AI Models on Historical Data...'):
    rf_roi = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_npv = RandomForestRegressor(n_estimators=200, random_state=42)
    
    X_train = df_hist[features]
    y_roi = df_hist[target_roi]
    y_npv = df_hist[target_npv]
    
    rf_roi.fit(X_train, y_roi)
    rf_npv.fit(X_train, y_npv)

# Predict on New Proposals
df_prop["Pred_ROI"] = rf_roi.predict(df_prop[features])
df_prop["Pred_NPV"] = rf_npv.predict(df_prop[features])

st.success("✅ AI Training Complete. Forecasting generated for new proposals.")

# ----------------------------------------------------
# 5. Interface Tabs
# ----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast & Strategy", 
    "⚖️ Portfolio Optimization", 
    "🎲 Risk Simulation (Monte Carlo)",
    "🧊 3D Strategic View"
])

# --- TAB 1: FORECAST & STRATEGY ---
with tab1:
    st.subheader("🔍 Project Valuation & AI Forecasting")
    
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Proposals", len(df_prop))
    c2.metric("Total Capital Requested", f"₹{df_prop['Investment_Capital'].sum():,.0f}")
    c3.metric("Avg Predicted ROI", f"{df_prop['Pred_ROI'].mean():.2f}%")
    c4.metric("Avg Predicted NPV", f"₹{df_prop['Pred_NPV'].mean():,.0f}")
    
    st.markdown("#### AI Feature Importance (What drives success?)")
    st.caption("The chart below shows which factors the AI model found most important when predicting ROI based on historical data.")
    
    # Feature Importance Plot
    col_chart, col_data = st.columns([2, 1])
    
    with col_chart:
        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": rf_roi.feature_importances_
        }).sort_values("Importance", ascending=True)
        
        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature", orientation='h', 
            title="Feature Importance (ROI Model)",
            color="Importance", color_continuous_scale="Blues",
            text_auto='.2f'
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with col_data:
        st.markdown("**Top 5 High-Value Proposals**")
        st.dataframe(
            df_prop[["Project_ID", "Department", "Pred_NPV"]].sort_values("Pred_NPV", ascending=False).head(5).style.format({"Pred_NPV": "₹{:.0f}"}),
            use_container_width=True,
            hide_index=True
        )

# --- TAB 2: OPTIMIZATION ---
with tab2:
    st.subheader("💰 Constrained Capital Optimization (Linear Programming)")
    st.markdown(f"""
    **Optimization Goal:** Maximize Total Portfolio NPV  
    **Hard Constraint:** Total Cost ≤ ₹{budget_input:,.0f}
    """)
    
    # Optimization Logic
    # 1. Variables: x (binary vector, 1=select, 0=reject)
    # 2. Objective Function: Maximize sum(NPV * x) -> Minimize sum(-NPV * x)
    # 3. Constraint: sum(Cost * x) <= Budget
    
    c = -df_prop["Pred_NPV"].values # Negative because linprog minimizes
    A = [df_prop["Investment_Capital"].values]
    b = [budget_input]
    bounds = [(0, 1) for _ in range(len(df_prop))]
    
    # Solve
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        df_prop["Selected"] = res.x.round(0)
        portfolio = df_prop[df_prop["Selected"] == 1]
        
        # Portfolio Stats
        p_cost = portfolio["Investment_Capital"].sum()
        p_npv = portfolio["Pred_NPV"].sum()
        p_risk = portfolio["Risk_Score"].mean()
        p_roi = portfolio["Pred_ROI"].mean()
        
        # Display Results
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Projects Selected", f"{len(portfolio)} / {len(df_prop)}")
        k2.metric("Capital Utilized", f"₹{p_cost:,.0f}", f"{p_cost/budget_input*100:.1f}% Utilized")
        k3.metric("Optimized NPV", f"₹{p_npv:,.0f}")
        k4.metric("Portfolio Risk", f"{p_risk:.2f}", delta_color="inverse" if p_risk > max_risk else "normal")
        
        if p_risk > max_risk:
            st.warning(f"⚠️ **Risk Alert:** The optimized portfolio risk ({p_risk:.2f}) exceeds your tolerance ({max_risk}). Consider manually rejecting high-risk projects or adjusting the budget.")
        else:
            st.success("✅ Portfolio is within risk tolerance levels.")
        
        st.divider()
        
        # Visuals
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            fig_pie = px.pie(portfolio, values='Investment_Capital', names='Department', title="Optimized Budget Allocation by Department", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with row1_col2:
            # ROI vs NPV Scatter for selected projects
            fig_scatter = px.scatter(
                portfolio, x="Pred_ROI", y="Pred_NPV", size="Investment_Capital", color="Department",
                title="Selected Projects: ROI vs NPV", hover_data=["Project_ID"]
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.markdown("### 📋 Detailed Allocation Table")
        st.dataframe(
            portfolio[["Project_ID", "Department", "Investment_Capital", "Pred_ROI", "Pred_NPV", "Risk_Score", "Strategic_Alignment"]]
            .style.format({"Investment_Capital": "₹{:.0f}", "Pred_ROI": "{:.2f}%", "Pred_NPV": "₹{:.0f}"}),
            use_container_width=True
        )
            
    else:
        st.error("Optimization failed. Constraints might be too tight or data issue.")

# --- TAB 3: MONTE CARLO ---
with tab3:
    st.subheader("🎲 Monte Carlo Simulation: Value at Risk (VaR)")
    st.markdown("Simulating **1,000 market scenarios** to stress-test the Optimized Portfolio.")
    
    if 'portfolio' in locals() and not portfolio.empty:
        iterations = 1000
        sim_results = []
        
        # Assumption: NPV can fluctuate by +/- 20% due to market volatility
        volatility = 0.20
        base_total_npv = portfolio["Pred_NPV"].sum()
        
        for _ in range(iterations):
            shock = np.random.normal(0, volatility) # Normal distribution shock
            simulated_npv = base_total_npv * (1 + shock)
            sim_results.append(simulated_npv)
            
        # Plot Distribution
        fig_hist = px.histogram(
            x=sim_results, nbins=50, 
            title="Probability Distribution of Portfolio Returns",
            labels={'x': 'Total Portfolio NPV (₹)'},
            color_discrete_sequence=['#2ecc71']
        )
        
        # Calculate VaR (5th percentile)
        var_95 = np.percentile(sim_results, 5)
        
        fig_hist.add_vline(x=base_total_npv, line_dash="dash", line_color="black", annotation_text="Expected NPV")
        fig_hist.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="95% VaR")
        fig_hist.update_layout(showlegend=False)
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.error(f"""
        **Risk Insight (VaR 95%):**
        While the Expected NPV is **₹{base_total_npv:,.0f}**, there is a 5% chance that due to adverse market conditions, 
        the portfolio value could drop to **₹{var_95:,.0f}**.
        """)
        
    else:
        st.info("⚠️ Please run the Optimization in the 'Portfolio Optimization' tab first to generate a portfolio for simulation.")

# --- TAB 4: 3D VISUAL ---
with tab4:
    st.subheader("🧊 Strategic Risk-Return Landscape")
    
    # We plot ALL proposals, and highlight selected ones
    if "Selected" in df_prop.columns:
        df_prop["Status"] = df_prop["Selected"].apply(lambda x: "Selected" if x==1 else "Rejected") 
        color_map = {"Selected": "#00CC96", "Rejected": "#EF553B"}
    else:
        df_prop["Status"] = "Pending"
        color_map = {"Pending": "grey"}
    
    fig_3d = px.scatter_3d(
        df_prop,
        x="Risk_Score",
        y="Strategic_Alignment",
        z="Pred_ROI",
        color="Status",
        color_discrete_map=color_map,
        size="Investment_Capital",
        hover_data=["Project_ID", "Department"],
        title="Strategy (Y) vs Risk (X) vs ROI (Z)",
        opacity=0.8
    )
    
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")
st.caption("© 2026 CAPITALIQ-AI™ | Grant Thornton Bharat LLP | Master's Project | Developed by KD")
