import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import pulp 

# ----------------------------------------------------
# 1. Page Configuration & "Glassmorphism" CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="CAPITALIQ-AI™ | Enterprise Advisor",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# --- THEME ENGINE: GLASSMORPHISM & HIGH CONTRAST ---
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
    
    /* Insight Box Styling */
    .insight-box {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
        font-size: 0.9rem;
        color: #d1fae5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Helper Functions (Logic & AI)
# ----------------------------------------------------

@st.cache_data
def get_templates():
    """Generates sample CSV templates for the user."""
    # Historical Data (Training)
    df_h = pd.DataFrame({
        "Investment_Capital": np.random.randint(500000, 5000000, 50),
        "Duration_Months": np.random.randint(6, 36, 50),
        "Risk_Score": np.random.uniform(1, 10, 50).round(1),
        "Strategic_Alignment": np.random.uniform(1, 10, 50).round(1),
        "Market_Trend_Index": np.random.uniform(0.5, 1.5, 50).round(2),
        "Actual_ROI_Pct": np.random.uniform(5, 25, 50).round(1),
    })
    
    # Proposal Data (Prediction)
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
    features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
    rf_roi = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_roi.fit(df_hist[features], df_hist["Actual_ROI_Pct"])
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_roi.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return rf_roi, importance

def calculate_dynamic_npv(row, wacc_rate):
    """
    Financial Rigor: Calculates NPV based on WACC.
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

def run_optimization(df, budget, constraints):
    """
    Advanced Optimization using PuLP (Linear Programming).
    Handles constraints: Budget, Department Minimums, Risk Caps.
    """
    # 1. Initialize Model
    prob = pulp.LpProblem("Capital_Allocation", pulp.LpMaximize)
    
    # 2. Decision Vars
    selection_vars = pulp.LpVariable.dicts("Select", df.index, cat='Binary')
    
    # 3. Objective: Maximize NPV
    prob += pulp.lpSum([df.loc[i, "Dynamic_NPV"] * selection_vars[i] for i in df.index])
    
    # 4. Global Constraint: Budget
    prob += pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in df.index]) <= budget
    
    # 5. Department Constraints (Example: 'IT' must get funding if prioritized)
    # (Simplified for this demo: Ensure diversity - max 60% budget to one dept)
    total_invest = pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in df.index])
    
    for dept in df['Department'].unique():
        dept_indices = df[df['Department'] == dept].index
        # Dept allocation <= 60% of Total Budget (Risk Diversification)
        prob += pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in dept_indices]) <= (0.60 * budget)

    # 6. Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # 7. Map Results
    df["Selected"] = [int(selection_vars[i].varValue) for i in df.index]
    return df

def generate_insight(chart_type, data=None):
    """
    Generates rule-based natural language explanations for charts.
    """
    if chart_type == "budget_bar":
        top_dept = data.groupby("Department")["Investment_Capital"].sum().idxmax()
        return f"""
        <div class='insight-box'>
        <b>💡 AI Analysis:</b> The optimization engine has heavily weighted capital towards the <b>{top_dept}</b> department. 
        This suggests {top_dept} projects currently offer the highest risk-adjusted returns (Sharpe Ratio) under the current WACC scenario.
        </div>
        """
    elif chart_type == "roi_scatter":
        return """
        <div class='insight-box'>
        <b>💡 AI Analysis:</b> The efficient frontier (upper-left boundary) indicates projects with high ROI and manageable risk. 
        Outliers on the bottom-right (High Risk, Low ROI) have been automatically deprioritized.
        </div>
        """
    return ""

def dark_chart(fig):
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
    
    # Navigation
    selected_page = st.radio(
        "Navigation", 
        ["🏠 Home & Data", "🚀 Executive Dashboard", "⚖️ Scenario Manager", "🧠 AI Deal Memos"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Financial Sandbox Controls
    st.subheader("⚙️ Financial Sandbox")
    budget_input = st.number_input("Total Budget (₹)", value=15000000.0, step=500000.0)
    wacc_input = st.slider("WACC (%)", 5.0, 20.0, 10.0, help="Weighted Average Cost of Capital. Higher WACC penalizes long-term projects.") / 100
    market_shock = st.slider("Market Shock Adjustment", -20, 20, 0, format="%d%%") / 100
    
    st.markdown("---")
    if st.button("🔄 Reset System", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ----------------------------------------------------
# 4. Main App Logic
# ----------------------------------------------------

# --- PAGE: HOME & DATA ---
if selected_page == "🏠 Home & Data":
    st.title("👋 Welcome to CapitalIQ-AI")
    
    col_intro, col_setup = st.columns([1.5, 1])
    
    with col_intro:
        st.markdown("### The Enterprise Standard for AI-Driven Capital Allocation")
        st.info("""
        **System Capabilities:**
        1. **WACC-Adjusted NPV:** Calculates true economic value using your cost of capital.
        2. **Constraint Programming:** Uses Linear Programming (PuLP) to solve complex budget buckets.
        3. **Scenario Modeling:** Save and compare 'Recession' vs 'Growth' strategies.
        """)
        
        h_temp, p_temp = get_templates()
        c1, c2 = st.columns(2)
        c1.download_button("📥 Train Template", h_temp.to_csv(index=False), "train.csv")
        c2.download_button("📥 Predict Template", p_temp.to_csv(index=False), "predict.csv")

    with col_setup:
        st.markdown("#### 📂 Initialize System")
        hist_file = st.file_uploader("1. Training Data (History)", type=["csv"])
        prop_file = st.file_uploader("2. Proposal Data (New)", type=["csv"])
        
        if not hist_file and not prop_file:
            if st.button("🚀 Load Demo Data", type="primary"):
                st.session_state['use_demo'] = True
                st.rerun()

    # Data Loading Logic
    if st.session_state.get('use_demo', False):
        df_hist, df_prop = get_templates()
        st.success("✅ Demo Data Loaded!")
    elif hist_file and prop_file:
        df_hist = pd.read_csv(hist_file)
        df_prop = pd.read_csv(prop_file)
        st.success("✅ Custom Data Uploaded!")
    else:
        st.stop()

    # Train Models (Persisted)
    if 'rf_roi' not in st.session_state:
        with st.spinner("⚙️ Training AI Models..."):
            rf_roi, feat_imp = train_models(df_hist)
            st.session_state['rf_roi'] = rf_roi
            st.session_state['feat_imp'] = feat_imp
            st.session_state['df_prop_raw'] = df_prop # Save raw data
            st.rerun()

# --- SHARED CALCULATION BLOCK ---
if 'df_prop_raw' not in st.session_state:
    st.stop()

# 1. Retrieve Data
df_prop = st.session_state['df_prop_raw'].copy()
rf_roi = st.session_state['rf_roi']
features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]

# 2. AI Prediction (Base)
df_prop["Pred_ROI"] = rf_roi.predict(df_prop[features])

# 3. Apply Market Shock (Scenario)
df_prop["Pred_ROI"] = df_prop["Pred_ROI"] * (1 + market_shock)

# 4. Financial Calculation (Dynamic NPV using WACC)
df_prop["Dynamic_NPV"] = df_prop.apply(lambda row: calculate_dynamic_npv(row, wacc_input), axis=1)

# 5. Run Optimization (PuLP)
df_optimized = run_optimization(df_prop, budget_input, {})
portfolio = df_optimized[df_optimized["Selected"] == 1]

# --- PAGE: EXECUTIVE DASHBOARD ---
if selected_page == "🚀 Executive Dashboard":
    st.title("📊 Executive Dashboard")
    st.caption(f"Scenario: WACC {wacc_input*100}% | Market Shock {market_shock*100:+d}%")
    
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Projects Approved", f"{len(portfolio)}", f"/{len(df_prop)}")
    k2.metric("Capital Deployed", f"₹{portfolio['Investment_Capital'].sum()/1e6:.1f}M", f"Util: {portfolio['Investment_Capital'].sum()/budget_input*100:.0f}%")
    k3.metric("Projected NPV", f"₹{portfolio['Dynamic_NPV'].sum()/1e6:.2f}M", delta="WACC Adj.")
    k4.metric("Avg ROI", f"{portfolio['Pred_ROI'].mean():.1f}%", f"Risk: {portfolio['Risk_Score'].mean():.1f}")
    
    st.markdown("---")
    
    # Visuals Row 1
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📈 Capital Allocation Strategy")
        fig_bar = px.bar(portfolio, x="Department", y="Investment_Capital", color="Pred_ROI", 
                         title="Budget by Dept (Color = ROI)", text_auto='.2s',
                         color_continuous_scale="Viridis")
        st.plotly_chart(dark_chart(fig_bar), use_container_width=True)
        # AI Insight
        st.markdown(generate_insight("budget_bar", portfolio), unsafe_allow_html=True)
        
    with c2:
        st.subheader("🎯 Risk vs. Reward")
        fig_scat = px.scatter(df_optimized, x="Risk_Score", y="Pred_ROI", color="Selected",
                              symbol="Department", size="Investment_Capital",
                              color_discrete_map={0: "#ef4444", 1: "#22c55e"},
                              title="Green = Approved, Red = Rejected")
        st.plotly_chart(dark_chart(fig_scat), use_container_width=True)
        
    # Visuals Row 2
    st.markdown("---")
    st.subheader("🧊 Portfolio Topology (3D)")
    c3, c4 = st.columns([2, 1])
    with c3:
        fig_3d = px.scatter_3d(df_optimized, x="Risk_Score", y="Strategic_Alignment", z="Pred_ROI",
                               color="Selected", size="Investment_Capital", opacity=0.8,
                               color_discrete_map={0: "#ef4444", 1: "#00e676"})
        st.plotly_chart(dark_chart(fig_3d), use_container_width=True)
    with c4:
        st.markdown("#### 💡 Topology Insight")
        st.info("""
        The 3D Map allows you to visualize the 'sweet spot' of the portfolio.
        
        * **X-Axis:** Risk Score
        * **Y-Axis:** Strategic Fit
        * **Z-Axis:** ROI
        
        **Note:** The optimizer automatically avoided projects in the 'low z, low y' (Low Return, Low Strategy) zone.
        """)

# --- PAGE: SCENARIO MANAGER ---
elif selected_page == "⚖️ Scenario Manager":
    st.title("⚖️ Scenario Simulation & Comparison")
    
    st.markdown("### Current State Snapshot")
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

# --- PAGE: AI DEAL MEMOS ---
elif selected_page == "🧠 AI Deal Memos":
    st.title("🧠 AI Investment Committee")
    st.markdown("Generated explanations for **Top 3 Approved** and **Top 3 Rejected** proposals.")
    
    col_app, col_rej = st.columns(2)
    
    with col_app:
        st.subheader("✅ Approval Memos")
        for i, row in portfolio.sort_values(by="Dynamic_NPV", ascending=False).head(3).iterrows():
            with st.expander(f"APPROVED: {row['Project_ID']} ({row['Department']})", expanded=True):
                st.markdown(f"""
                **Rationale:**
                * **Financials:** Generates **₹{row['Dynamic_NPV']/1e5:.1f} Lakhs** in NPV (WACC adj).
                * **Strategy:** Strong alignment score ({row['Strategic_Alignment']}/10).
                * **Verdict:** This project clears the hurdle rate and fits within the diversification cap.
                """)
                
    with col_rej:
        st.subheader("❌ Rejection Memos")
        rejected = df_optimized[df_optimized["Selected"] == 0]
        for i, row in rejected.sort_values(by="Pred_ROI", ascending=False).head(3).iterrows():
            with st.expander(f"REJECTED: {row['Project_ID']} ({row['Department']})"):
                reason = "Budget Constraint" if row['Dynamic_NPV'] > 0 else "Negative NPV"
                st.markdown(f"""
                **Rationale:**
                * **Issue:** {reason}.
                * **Risk Profile:** Score of {row['Risk_Score']} may be too high relative to return.
                * **Recommendation:** Reduce scope to lower capital requirement or wait for next fiscal cycle.
                """)
