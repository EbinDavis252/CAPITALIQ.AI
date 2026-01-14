import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linprog # Kept for legacy support if needed
import pulp # NEW: For Advanced Optimization

# ----------------------------------------------------
# 1. Page Configuration & "Glassmorphism" CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="CAPITALIQ-AI™ | Enterprise Edition",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# --- THEME ENGINE: GLASSMORPHISM & HIGH CONTRAST (YOUR ORIGINAL CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    /* Global Settings */
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background-image: linear_gradient(rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.95)), 
                          url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop');
        background-size: cover;
        background-attachment: fixed;
    }

    /* Text Visibility Fixes (Force White) */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, li, span, label, .stDataFrame, .stRadio label {
        color: #e2e8f0 !important;
    }
    
    /* Glass Cards */
    div[data-testid="stMetric"], div[data-testid="stExpander"], div.stDataFrame {
        background-color: rgba(30, 41, 59, 0.6); 
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b1120; 
        border-right: 1px solid #334155;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #0f766e 0%, #0d9488 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.4);
    }
    
    /* Custom Radio Button (Navigation) */
    div[role="radiogroup"] > label > div:first-of-type {
        background-color: #0f172a;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Helper Functions (Caching & Logic)
# ----------------------------------------------------

@st.cache_data
def get_templates():
    """Generates sample CSV templates for the user to download."""
    # Historical Data Template
    df_h = pd.DataFrame({
        "Investment_Capital": np.random.randint(500000, 5000000, 50),
        "Duration_Months": np.random.randint(6, 36, 50),
        "Risk_Score": np.random.uniform(1, 10, 50).round(1),
        "Strategic_Alignment": np.random.uniform(1, 10, 50).round(1),
        "Market_Trend_Index": np.random.uniform(0.5, 1.5, 50).round(2),
        "Actual_ROI_Pct": np.random.uniform(5, 25, 50).round(1),
        "Actual_NPV": np.random.randint(100000, 2000000, 50)
    })
    
    # Proposal Data Template
    df_p = pd.DataFrame({
        "Project_ID": [f"PROJ-{i:03d}" for i in range(1, 21)],
        "Department": np.random.choice(['IT', 'R&D', 'Marketing', 'Ops'], 20),
        "Investment_Capital": np.random.randint(500000, 5000000, 20),
        "Duration_Months": np.random.randint(6, 36, 20),
        "Risk_Score": np.random.uniform(1, 10, 20).round(1),
        "Strategic_Alignment": np.random.uniform(1, 10, 20).round(1),
        "Market_Trend_Index": np.random.uniform(0.5, 1.5, 20).round(2)
    })
    return df_h, df_p

@st.cache_resource
def train_models(df_hist):
    """Trains the models once and caches them."""
    features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
    rf_roi = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_npv = RandomForestRegressor(n_estimators=200, random_state=42)
    
    try:
        rf_roi.fit(df_hist[features], df_hist["Actual_ROI_Pct"])
        rf_npv.fit(df_hist[features], df_hist["Actual_NPV"])
    except Exception as e:
        st.error(f"Training Error: {e}")
        st.stop()
    
    # Return models and feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_roi.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return rf_roi, rf_npv, importance

def calculate_dynamic_npv(row, wacc_rate):
    """
    NEW: Financial Rigor - Calculates NPV based on WACC.
    Assumption: ROI implies total return distributed evenly over duration.
    """
    total_return_value = row['Investment_Capital'] * (1 + (row['Pred_ROI'] / 100))
    annual_cash_flow = total_return_value / (row['Duration_Months'] / 12)
    
    # Discounted Cash Flow Summation
    years = row['Duration_Months'] / 12
    dcf = 0
    for t in range(1, int(years) + 2):
        if t <= years:
            dcf += annual_cash_flow / ((1 + wacc_rate) ** t)
        else:
            # Partial year handling
            remaining_fraction = years - int(years)
            if remaining_fraction > 0:
                dcf += (annual_cash_flow * remaining_fraction) / ((1 + wacc_rate) ** t)
                
    return dcf - row['Investment_Capital']

def run_advanced_optimization(df, budget):
    """
    NEW: Advanced Optimization using PuLP.
    Replaces the basic linprog for better constraint handling.
    """
    # Initialize Model
    prob = pulp.LpProblem("Capital_Allocation", pulp.LpMaximize)
    
    # Decision Vars
    selection_vars = pulp.LpVariable.dicts("Select", df.index, cat='Binary')
    
    # Objective: Maximize Dynamic NPV
    prob += pulp.lpSum([df.loc[i, "Dynamic_NPV"] * selection_vars[i] for i in df.index])
    
    # Global Constraint: Budget
    prob += pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in df.index]) <= budget
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Map Results
    df["Selected"] = [int(selection_vars[i].varValue) for i in df.index]
    return df

def dark_chart(fig):
    """Applies a consistent dark theme to Plotly charts."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0")
    )
    return fig

# ----------------------------------------------------
# 3. Sidebar: Navigation & Controls
# ----------------------------------------------------
with st.sidebar:
    st.title("💎 CAPITALIQ-AI™")
    st.caption("Strategic Portfolio Optimizer")
    st.markdown("---")
    
    # Navigation (Updated with new pages)
    selected_page = st.radio(
        "Navigation", 
        ["🏠 Home & Data", "🚀 Executive Summary", "🧠 AI Insights", "⚡ Efficient Frontier", "💰 Optimization Report", "🧊 Strategic 3D Map", "⚖️ Scenario Manager", "📄 AI Deal Memos"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Strategy Controls (Global)
    st.subheader("⚙️ Constraints & Sandbox")
    budget_input = st.number_input("Budget (₹)", value=15000000.0, step=500000.0)
    
    # NEW: WACC Input
    wacc_input = st.slider("WACC (%)", 5.0, 20.0, 10.0, help="Weighted Average Cost of Capital") / 100
    
    max_risk = st.slider("Max Portfolio Risk", 1.0, 10.0, 6.5)
    market_shock = st.slider("Market Scenario", -0.20, 0.20, 0.0, 0.01, format="%+.0f%%")
    
    st.markdown("---")
    
    # Data Reset Control
    st.markdown("### 🛠️ Data Controls")
    if st.button("🔄 Reset / Clear All Data", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ----------------------------------------------------
# 4. Main App Logic
# ----------------------------------------------------

# --- PAGE: HOME & DATA (The Landing Page) ---
if selected_page == "🏠 Home & Data":
    st.title("👋 Welcome to CapitalIQ-AI")
    
    col_intro, col_setup = st.columns([1.5, 1])
    
    with col_intro:
        st.markdown("### The Enterprise Standard for AI-Driven Capital Allocation")
        st.info("""
        **How it works:**
        1. **Upload** your historical project data (for training) and new proposals.
        2. **Configure** your budget, WACC, and risk appetite in the sidebar.
        3. **Analyze** the AI-optimized portfolio in the Dashboard.
        """)
        
        # Template Downloads
        h_temp, p_temp = get_templates()
        c1, c2 = st.columns(2)
        c1.download_button("📥 Download Train Template", h_temp.to_csv(index=False), "train_template.csv")
        c2.download_button("📥 Download Predict Template", p_temp.to_csv(index=False), "predict_template.csv")

    with col_setup:
        st.markdown("#### 📂 Initialize System")
        
        # --- UPLOAD LOGIC ---
        def disable_demo():
            st.session_state['use_demo'] = False

        with st.container():
            hist_file = st.file_uploader("1. Training Data (History)", type=["csv"], key="u_hist", on_change=disable_demo)
            prop_file = st.file_uploader("2. Proposal Data (New)", type=["csv"], key="u_prop", on_change=disable_demo)
            
            # Show "Load Demo" ONLY if no files are uploaded
            if not hist_file and not prop_file:
                st.markdown("---")
                if st.button("🚀 Load Demo Data", type="primary"):
                    st.session_state['use_demo'] = True
                    st.rerun()

    # --- DATA PROCESSING LOGIC ---
    data_source = None
    
    # 1. Check if user wants Demo Data
    if st.session_state.get('use_demo', False):
        df_hist, df_prop = get_templates()
        st.success("✅ Demo Data Loaded! Navigate to 'Executive Summary' to see results.")
        data_source = "demo"

    # 2. Check if user uploaded files (Prioritized)
    elif hist_file and prop_file:
        df_hist = pd.read_csv(hist_file)
        df_prop = pd.read_csv(prop_file)
        st.success("✅ Custom Data Uploaded & Saved!")
        data_source = "upload"

    # 3. No data yet
    else:
        st.warning("⚠️ Waiting for data streams... Upload files or Click 'Load Demo Data'")
        st.stop()

    # --- PERSISTENCE: Save to Session State ---
    if 'df_prop' not in st.session_state or st.session_state.get('data_source') != data_source:
        
        with st.spinner("⚙️ Training Models & Processing Data..."):
            # Train
            rf_roi, rf_npv, feature_imp = train_models(df_hist)
            
            # Predict (Initial pass)
            features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
            df_prop["Pred_ROI"] = rf_roi.predict(df_prop[features])
            df_prop["Pred_NPV"] = rf_npv.predict(df_prop[features])
            
            # Save everything
            st.session_state['df_prop'] = df_prop
            st.session_state['feature_imp'] = feature_imp
            st.session_state['rf_roi'] = rf_roi 
            st.session_state['rf_npv'] = rf_npv
            st.session_state['data_source'] = data_source
            
            # Refresh
            st.rerun()

# --- SHARED PROCESSING FOR DASHBOARD PAGES ---
# Check if data exists
if 'df_prop' not in st.session_state:
    st.warning("⚠️ Please go to '🏠 Home & Data' and load data first.")
    st.stop()

# Retrieve from state
df_prop = st.session_state['df_prop'].copy()
feature_imp = st.session_state['feature_imp']

# --- DYNAMIC UPDATES (SCENARIO ANALYSIS) ---
features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]

# 1. Apply Market Shock
df_prop["Pred_ROI"] = df_prop["Pred_ROI"] * (1 + market_shock)
df_prop["Pred_NPV"] = df_prop["Pred_NPV"] * (1 + market_shock)

# 2. NEW: Calculate Dynamic NPV based on WACC
df_prop["Dynamic_NPV"] = df_prop.apply(lambda row: calculate_dynamic_npv(row, wacc_input), axis=1)

# 3. Efficiency Metric
df_prop["Efficiency"] = df_prop["Pred_ROI"] / df_prop["Risk_Score"]

# --- OPTIMIZATION ENGINE (UPDATED TO PULP) ---
# Replaced linprog with PuLP for better stability and "Enterprise" logic
df_prop = run_advanced_optimization(df_prop, budget_input)
portfolio = df_prop[df_prop["Selected"] == 1]

# ----------------------------------------------------
# 5. Dashboard Pages
# ----------------------------------------------------

if selected_page == "🚀 Executive Summary":
    st.title("📊 Executive Dashboard")
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Projects Funded", f"{len(portfolio)}", f"/{len(df_prop)} Proposals")
    kpi2.metric("Capital Deployed", f"₹{portfolio['Investment_Capital'].sum()/1e6:.2f}M", f"Utilization: {portfolio['Investment_Capital'].sum()/budget_input*100:.1f}%")
    
    # Updated KPI to show WACC-Adjusted NPV
    kpi3.metric("Projected NPV (WACC Adj.)", f"₹{portfolio['Dynamic_NPV'].sum()/1e6:.2f}M", delta=f"Shock: {market_shock*100:+.0f}%")
    
    kpi4.metric("Avg Risk Profile", f"{portfolio['Risk_Score'].mean():.2f}", f"Max Limit: {max_risk}")

    st.markdown("---")
    
    # Charts (Your Original Charts)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📈 Capital Allocation by Department")
        if not portfolio.empty:
            fig = px.bar(portfolio, x="Department", y="Investment_Capital", color="Pred_ROI", 
                          title="Budget Distribution & ROI Heatmap", text_auto='.2s')
            st.plotly_chart(dark_chart(fig), use_container_width=True)
        else:
            st.info("No projects selected. Try increasing the budget.")
    
    with c2:
        st.subheader("⚡ Top ROI Drivers")
        st.dataframe(
            feature_imp.head(5).style.background_gradient(cmap='Greens'),
            use_container_width=True, hide_index=True
        )

elif selected_page == "🧠 AI Insights":
    st.subheader("🧠 The Brain Behind the Budget")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🔍 Feature Importance Analysis")
        fig_imp = px.bar(feature_imp, x="Importance", y="Feature", orientation='h', color="Importance")
        st.plotly_chart(dark_chart(fig_imp), use_container_width=True)
        
    with col2:
        st.markdown("##### 🎯 Prediction Accuracy vs Risk")
        fig_scat = px.scatter(df_prop, x="Risk_Score", y="Pred_ROI", size="Investment_Capital", 
                              color="Department", hover_name="Project_ID")
        st.plotly_chart(dark_chart(fig_scat), use_container_width=True)

elif selected_page == "⚡ Efficient Frontier":
    st.subheader("⚡ Efficient Frontier Simulation")
    st.markdown("Running Monte Carlo Simulation (1,000 Iterations)...")
    
    # Progress Bar
    progress_bar = st.progress(0)
    
    results = []
    total_cap = df_prop["Investment_Capital"].sum()
    avg_p = min(0.5, budget_input / (total_cap + 1))
    
    # Simulation Loop
    for i in range(1000):
        mask = np.random.rand(len(df_prop)) < avg_p
        sample = df_prop[mask]
        if not sample.empty and sample["Investment_Capital"].sum() <= budget_input:
            results.append({
                "Risk": sample["Risk_Score"].mean(),
                "Return": sample["Pred_ROI"].mean(),
                "NPV": sample["Dynamic_NPV"].sum()
            })
        if i % 100 == 0: progress_bar.progress(i/1000)
    
    progress_bar.empty() 
    
    sim_df = pd.DataFrame(results)
    if not sim_df.empty:
        fig_ef = px.scatter(sim_df, x="Risk", y="Return", color="NPV", title="Optimal Risk/Return Profiles")
        
        # Add Current Portfolio Marker
        if not portfolio.empty:
            fig_ef.add_trace(go.Scatter(
                x=[portfolio["Risk_Score"].mean()], y=[portfolio["Pred_ROI"].mean()],
                mode='markers', marker=dict(color='white', size=15, symbol='star'), name="Selected Portfolio"
            ))
        
        st.plotly_chart(dark_chart(fig_ef), use_container_width=True)
    else:
        st.error("Simulation failed to find valid portfolios. Check constraints.")

elif selected_page == "💰 Optimization Report":
    st.subheader("📋 Final Investment Schedule")
    
    tab1, tab2 = st.tabs(["✅ Selected Projects", "❌ Rejected Projects"])
    
    with tab1:
        st.dataframe(
            portfolio[["Project_ID", "Department", "Investment_Capital", "Pred_ROI", "Efficiency"]]
            .style.format({"Investment_Capital": "₹{:,.0f}", "Pred_ROI": "{:.1f}%", "Efficiency": "{:.2f}"})
            .background_gradient(subset=["Efficiency"], cmap="Greens"),
            use_container_width=True
        )
        csv = portfolio.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Portfolio (CSV)", csv, "Strategic_Portfolio.csv", "text/csv")
        
    with tab2:
        rejected = df_prop[df_prop["Selected"] == 0]
        st.dataframe(rejected[["Project_ID", "Investment_Capital", "Pred_ROI"]], use_container_width=True)

elif selected_page == "🧊 Strategic 3D Map":
    st.subheader("🧊 Portfolio Topology")
    df_prop["Status"] = df_prop["Selected"].apply(lambda x: "Funded" if x==1 else "Not Funded")
    
    fig_3d = px.scatter_3d(
        df_prop, x="Risk_Score", y="Strategic_Alignment", z="Pred_ROI",
        color="Status", size="Investment_Capital", opacity=0.8,
        color_discrete_map={"Funded": "#00e676", "Not Funded": "#ff1744"}
    )
    st.plotly_chart(dark_chart(fig_3d), use_container_width=True)

# --- NEW PAGE: SCENARIO MANAGER ---
elif selected_page == "⚖️ Scenario Manager":
    st.title("⚖️ Scenario Simulation")
    
    col_save, col_view = st.columns([1, 3])
    with col_save:
        scenario_name = st.text_input("Scenario Name", value="Base Case")
        if st.button("💾 Save Current State"):
            s_data = {
                "Name": scenario_name,
                "WACC": wacc_input,
                "Budget": budget_input,
                "NPV": portfolio['Dynamic_NPV'].sum(),
                "ROI": portfolio['Pred_ROI'].mean(),
                "Projects": len(portfolio)
            }
            if 'scenarios' not in st.session_state: st.session_state['scenarios'] = []
            st.session_state['scenarios'].append(s_data)
            st.success(f"Saved {scenario_name}!")
            
    with col_view:
        if 'scenarios' in st.session_state and st.session_state['scenarios']:
            s_df = pd.DataFrame(st.session_state['scenarios'])
            st.dataframe(s_df.style.format({"NPV": "₹{:,.0f}", "ROI": "{:.1f}%", "WACC": "{:.1f}%"}), use_container_width=True)
            
            # Scenario Comparison Chart
            fig_comp = px.bar(s_df, x="Name", y="NPV", color="ROI", title="Scenario NPV Comparison")
            st.plotly_chart(dark_chart(fig_comp), use_container_width=True)
        else:
            st.info("Adjust WACC/Budget in the sidebar, then click 'Save Current State' to compare scenarios.")

# --- NEW PAGE: AI DEAL MEMOS ---
elif selected_page == "📄 AI Deal Memos":
    st.title("📄 AI Investment Memos")
    
    col_app, col_rej = st.columns(2)
    
    with col_app:
        st.subheader("✅ Top Approvals")
        for i, row in portfolio.sort_values(by="Dynamic_NPV", ascending=False).head(3).iterrows():
            with st.expander(f"APPROVED: {row['Project_ID']} ({row['Department']})", expanded=True):
                st.markdown(f"""
                **Rationale:**
                * **NPV Contribution:** ₹{row['Dynamic_NPV']/1e5:.1f} Lakhs
                * **Strategic Fit:** {row['Strategic_Alignment']}/10
                * **Decision:** Approved due to high WACC-adjusted return.
                """)
                
    with col_rej:
        st.subheader("❌ Top Rejections")
        rejected = df_prop[df_prop["Selected"] == 0]
        for i, row in rejected.sort_values(by="Pred_ROI", ascending=False).head(3).iterrows():
            with st.expander(f"REJECTED: {row['Project_ID']} ({row['Department']})"):
                st.markdown(f"""
                **Rationale:**
                * **Issue:** Failed to beat capital cost hurdle or budget constraint.
                * **Risk Score:** {row['Risk_Score']}
                * **Decision:** Deferred to next fiscal cycle.
                """)
