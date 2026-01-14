import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import pulp

# ----------------------------------------------------
# 1. Page Configuration & CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="CAPITALIQ-AI | Enterprise Edition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME ENGINE: HIGH CONTRAST PROFESSIONAL ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    /* Global Settings */
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background-image: linear_gradient(rgba(15, 23, 42, 0.95), rgba(15, 23, 42, 0.98)), 
                          url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop');
        background-size: cover;
        background-attachment: fixed;
    }

    /* Text Visibility Fixes */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, li, span, label, .stDataFrame, .stRadio label {
        color: #e2e8f0 !important;
    }
    
    /* Glass Cards */
    div[data-testid="stMetric"], div[data-testid="stExpander"], div.stDataFrame {
        background-color: rgba(30, 41, 59, 0.6); 
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 4px; 
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
        background: #0f766e;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #0d9488;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.4);
    }
    
    /* Analysis Box Styling */
    .analysis-box {
        border-left: 3px solid #3b82f6;
        background-color: rgba(59, 130, 246, 0.1);
        padding: 15px;
        margin-top: 10px;
        border-radius: 0 4px 4px 0;
    }
    .analysis-title {
        font-weight: 600;
        color: #60a5fa !important;
        margin-bottom: 5px;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Helper Functions (Logic & AI)
# ----------------------------------------------------

def standardize_columns(df):
    """Maps various user inputs to system standard names."""
    column_map = {
        'Capex': 'Investment_Capital', 'Cost': 'Investment_Capital', 'Budget': 'Investment_Capital',
        'Return': 'Actual_ROI_Pct', 'ROI': 'Actual_ROI_Pct',
        'Strategy': 'Strategic_Alignment', 'Strat': 'Strategic_Alignment',
        'Risk': 'Risk_Score', 'Dept': 'Department'
    }
    df = df.rename(columns=column_map)
    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

@st.cache_data
def get_templates():
    """Generates sample CSV templates."""
    df_h = pd.DataFrame({
        "Investment_Capital": np.random.randint(500000, 5000000, 50),
        "Duration_Months": np.random.randint(6, 36, 50),
        "Risk_Score": np.random.uniform(1, 10, 50).round(1),
        "Strategic_Alignment": np.random.uniform(1, 10, 50).round(1),
        "Market_Trend_Index": np.random.uniform(0.5, 1.5, 50).round(2),
        "Actual_ROI_Pct": np.random.uniform(5, 25, 50).round(1),
        "Actual_NPV": np.random.randint(100000, 2000000, 50)
    })
    
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
    for f in features:
        if f not in df_hist.columns: df_hist[f] = 0
            
    rf_roi = RandomForestRegressor(n_estimators=200, random_state=42)
    try:
        rf_roi.fit(df_hist[features], df_hist["Actual_ROI_Pct"])
    except Exception as e:
        st.error(f"Training Error: {e}")
        st.stop()
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_roi.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return rf_roi, importance

def calculate_dynamic_npv(row, wacc_rate):
    """Calculates NPV based on WACC."""
    total_return_value = row['Investment_Capital'] * (1 + (row['Pred_ROI'] / 100))
    duration = max(row['Duration_Months'], 1)
    annual_cash_flow = total_return_value / (duration / 12)
    years = duration / 12
    dcf = 0
    for t in range(1, int(years) + 2):
        if t <= years:
            dcf += annual_cash_flow / ((1 + wacc_rate) ** t)
        else:
            remaining_fraction = years - int(years)
            if remaining_fraction > 0:
                dcf += (annual_cash_flow * remaining_fraction) / ((1 + wacc_rate) ** t)
    return dcf - row['Investment_Capital']

def calculate_payback(row):
    """Calculates Payback Period in Years."""
    total_return_value = row['Investment_Capital'] * (1 + (row['Pred_ROI'] / 100))
    annual_cash_flow = total_return_value / (max(row['Duration_Months'], 1) / 12)
    if annual_cash_flow <= 0: return 99.9 
    return round(row['Investment_Capital'] / annual_cash_flow, 2)

def run_advanced_optimization(df, budget, min_dept_alloc_pct=0.0):
    """Advanced Optimization using PuLP with Dept Constraints."""
    prob = pulp.LpProblem("Capital_Allocation", pulp.LpMaximize)
    selection_vars = pulp.LpVariable.dicts("Select", df.index, cat='Binary')
    
    prob += pulp.lpSum([df.loc[i, "Dynamic_NPV"] * selection_vars[i] for i in df.index])
    prob += pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in df.index]) <= budget
    
    if min_dept_alloc_pct > 0 and 'Department' in df.columns:
        departments = df['Department'].unique()
        for dept in departments:
            dept_indices = df[df['Department'] == dept].index
            dept_req = df.loc[dept_indices, "Investment_Capital"].sum()
            target_min = budget * min_dept_alloc_pct
            if dept_req >= target_min:
                prob += pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in dept_indices]) >= target_min

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df["Selected"] = [int(selection_vars[i].varValue) if selection_vars[i].varValue is not None else 0 for i in df.index]
    return df

# --- IMPROVED VISUALIZATIONS ---

def generate_waterfall_chart(portfolio, wacc):
    """
    Replaces the 'disturbing' Tornado chart with a cleaner Waterfall/Bar chart 
    showing cumulative impact of sensitivities.
    """
    base_npv = portfolio['Dynamic_NPV'].sum()
    
    # Calculate impacts (deltas)
    impact_wacc_pos = (base_npv * 0.92) - base_npv  # Negative impact
    impact_wacc_neg = (base_npv * 1.08) - base_npv  # Positive impact
    impact_capex = (base_npv - (portfolio['Investment_Capital'].sum() * 0.10)) - base_npv
    impact_roi = (base_npv * 0.85) - base_npv

    # A "Diverging Bar Chart" is much cleaner than a standard Tornado
    scenarios = [
        {"Factor": "WACC Increase (+1%)", "Change": impact_wacc_pos}, 
        {"Factor": "Capex Overrun (+10%)", "Change": impact_capex},
        {"Factor": "ROI Shortfall (-10%)", "Change": impact_roi},
        {"Factor": "WACC Decrease (-1%)", "Change": impact_wacc_neg},
        {"Factor": "ROI Outperform (+10%)", "Change": (base_npv * 1.15) - base_npv}
    ]
    df_sens = pd.DataFrame(scenarios)
    
    fig = px.bar(df_sens, y="Factor", x="Change", orientation='h', 
                 title="Sensitivity Impact Analysis (Net NPV Change)",
                 color="Change", 
                 color_continuous_scale=["#ef4444", "#e2e8f0", "#22c55e"], # Red to Green
                 text_auto='.2s')
    
    fig.update_layout(xaxis_title="Impact on NPV (Currency)", yaxis_title="Risk Scenario")
    return dark_chart(fig)

def generate_3d_radar(df_prop):
    """
    Simulates a 3D-like experience using a filled Radar chart with multiple layers (traces).
    """
    categories = ["Risk_Score", "Strategic_Alignment", "Pred_ROI"]
    
    # Normalized Data (0-1 Scale) for fair comparison
    df_norm = df_prop.copy()
    df_norm["Risk_Score"] = df_norm["Risk_Score"] / 10
    df_norm["Strategic_Alignment"] = df_norm["Strategic_Alignment"] / 10
    df_norm["Pred_ROI"] = df_norm["Pred_ROI"] / 30 # Assumes max ROI ~30%
    
    # Clip to 1.0 max
    df_norm[categories] = df_norm[categories].clip(0, 1)

    funded = df_norm[df_norm["Selected"] == 1][categories].mean()
    rejected = df_norm[df_norm["Selected"] == 0][categories].mean()
    
    fig = go.Figure()
    
    # Layer 1: Rejected (Background)
    fig.add_trace(go.Scatterpolar(
        r=rejected.values, theta=categories, fill='toself', name='Rejected Avg',
        line_color='rgba(255, 23, 68, 0.5)', fillcolor='rgba(255, 23, 68, 0.2)'
    ))
    
    # Layer 2: Funded (Foreground - Popping out)
    fig.add_trace(go.Scatterpolar(
        r=funded.values, theta=categories, fill='toself', name='Funded Avg',
        line_color='#00e676', fillcolor='rgba(0, 230, 118, 0.4)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True,
        title="Portfolio Profile: Funded vs Rejected (Normalized)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0")
    )
    return fig

def generate_board_report(portfolio, budget, wacc):
    txt = f"CAPITALIQ-AI REPORT\n-------------------\nBudget: INR {budget:,.2f}\nDeployed: INR {portfolio['Investment_Capital'].sum():,.2f}\nNPV: INR {portfolio['Dynamic_NPV'].sum():,.2f}\nWACC: {wacc*100:.1f}%\n\nAPPROVED PROJECTS:\n"
    for _, row in portfolio.iterrows():
        txt += f"[x] {row['Project_ID']} | ROI: {row['Pred_ROI']:.1f}%\n"
    return txt

def dark_chart(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"), margin=dict(l=20, r=20, t=40, b=20))
    return fig

def render_analysis(text):
    st.markdown(f"<div class='analysis-box'><span class='analysis-title'>💡 AI EXPLANATION</span>{text}</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 3. Callback Functions (Nav Logic)
# ----------------------------------------------------

def process_data_callback():
    """Callback to process data immediately upon upload."""
    if st.session_state.u_hist is not None and st.session_state.u_prop is not None:
        try:
            df_hist = standardize_columns(pd.read_csv(st.session_state.u_hist))
            df_prop = standardize_columns(pd.read_csv(st.session_state.u_prop))
            
            rf_roi, feature_imp = train_models(df_hist)
            features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
            for f in features:
                if f not in df_prop.columns: df_prop[f] = 0
            
            df_prop["Pred_ROI"] = rf_roi.predict(df_prop[features])
            
            st.session_state['df_prop'] = df_prop
            st.session_state['feature_imp'] = feature_imp
            st.session_state.page_selection = "Executive Summary"
        except Exception as e:
            st.error(f"Error processing files: {e}")

def load_demo_callback():
    """Callback for Demo Data button."""
    df_hist, df_prop = get_templates()
    rf_roi, feature_imp = train_models(df_hist)
    features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
    df_prop["Pred_ROI"] = rf_roi.predict(df_prop[features])
    
    st.session_state['df_prop'] = df_prop
    st.session_state['feature_imp'] = feature_imp
    st.session_state.u_hist = "Demo"
    st.session_state.u_prop = "Demo"
    st.session_state.page_selection = "Executive Summary"

def reset_data_callback():
    st.session_state.clear()
    st.session_state.page_selection = "Home & Data"

# ----------------------------------------------------
# 4. Sidebar & Layout
# ----------------------------------------------------
with st.sidebar:
    st.title("CAPITALIQ-AI")
    st.caption("Strategic Portfolio Optimizer")
    st.markdown("---")
    
    if "page_selection" not in st.session_state:
        st.session_state.page_selection = "Home & Data"

    pages = ["Home & Data", "Executive Summary", "AI Insights", "Efficient Frontier", "Optimization Report", "Strategic 3D Map", "Scenario Manager", "AI Deal Memos"]

    selected_page = st.radio("NAVIGATION", pages, key="page_selection", label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("Constraints & Sandbox")
    budget_input = st.number_input("Budget (INR)", value=15000000.0, step=500000.0)
    wacc_input = st.slider("WACC (%)", 5.0, 20.0, 10.0, help="Weighted Average Cost of Capital") / 100
    min_dept_spend = st.slider("Min Dept. Allocation (%)", 0, 30, 0) / 100
    
    max_risk = st.slider("Max Portfolio Risk", 1.0, 10.0, 6.5)
    market_shock = st.slider("Market Scenario", -0.20, 0.20, 0.0, 0.01, format="%+.0f%%")
    
    st.markdown("---")
    st.button("Reset / Clear All Data", use_container_width=True, on_click=reset_data_callback)

    st.markdown("---")
    st.caption("© 2026 CapitalIQ-AI. Enterprise Edition. All Rights Reserved.")

# ----------------------------------------------------
# 5. Main Content
# ----------------------------------------------------

# --- PAGE: HOME & DATA ---
if selected_page == "Home & Data":
    st.title("Welcome to CapitalIQ-AI")
    
    if 'df_prop' in st.session_state:
        st.success("✅ Data System Online: Predictive Models Trained & Ready.")
        st.info("Your dataset is currently loaded in memory. You do not need to re-upload unless you want to change datasets.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Dashboard", type="primary"):
                st.session_state.page_selection = "Executive Summary"
                st.rerun()
        with col2:
            if st.button("Unload Current Data"):
                reset_data_callback()
                st.rerun()
    
    else:
        col_intro, col_setup = st.columns([1.5, 1])
        with col_intro:
            st.markdown("### The Enterprise Standard for AI-Driven Capital Allocation")
            st.info("""
            **Workflow:**
            1. **Upload** historical project data to train the predictive models.
            2. **Configure** financial constraints (Budget, WACC) in the sidebar.
            3. **Analyze** the optimized portfolio across various strategic dimensions.
            """)
            h_temp, p_temp = get_templates()
            c1, c2 = st.columns(2)
            c1.download_button("Download Train Template", h_temp.to_csv(index=False), "train_template.csv")
            c2.download_button("Download Predict Template", p_temp.to_csv(index=False), "predict_template.csv")

        with col_setup:
            st.markdown("#### Initialize System")
            with st.container():
                st.file_uploader("1. Training Data (History)", type=["csv"], key="u_hist", on_change=process_data_callback)
                st.file_uploader("2. Proposal Data (New)", type=["csv"], key="u_prop", on_change=process_data_callback)
                
                st.markdown("---")
                st.button("Load Demo Data", type="primary", on_click=load_demo_callback)

# --- SHARED DATA LOGIC ---
if selected_page != "Home & Data" and 'df_prop' not in st.session_state:
    st.warning("Please load data first.")
    st.stop()

if 'df_prop' in st.session_state:
    df_prop = st.session_state['df_prop'].copy()
    feature_imp = st.session_state['feature_imp']
    
    df_prop["Pred_ROI"] = df_prop["Pred_ROI"] * (1 + market_shock)
    df_prop["Dynamic_NPV"] = df_prop.apply(lambda row: calculate_dynamic_npv(row, wacc_input), axis=1)
    df_prop["Payback_Years"] = df_prop.apply(calculate_payback, axis=1)
    df_prop["Efficiency"] = df_prop["Pred_ROI"] / df_prop["Risk_Score"]
    
    df_prop = run_advanced_optimization(df_prop, budget_input, min_dept_spend)
    portfolio = df_prop[df_prop["Selected"] == 1]

# --- PAGE: EXECUTIVE SUMMARY ---
if selected_page == "Executive Summary":
    st.title("Executive Dashboard")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Projects Funded", f"{len(portfolio)}", f"Total: {len(df_prop)}")
    kpi2.metric("Capital Deployed", f"₹{portfolio['Investment_Capital'].sum()/1e6:.2f}M", f"Util: {portfolio['Investment_Capital'].sum()/budget_input*100:.1f}%")
    kpi3.metric("Projected NPV", f"₹{portfolio['Dynamic_NPV'].sum()/1e6:.2f}M", delta=f"Payback: {portfolio['Payback_Years'].mean():.1f} Yrs")
    kpi4.metric("Avg Risk Score", f"{portfolio['Risk_Score'].mean():.2f}", f"Max: {max_risk}")
    
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Capital Allocation by Department")
        if not portfolio.empty:
            # ENHANCED BAR CHART
            fig = px.bar(
                portfolio, 
                x="Department", 
                y="Investment_Capital", 
                color="Pred_ROI", 
                title="Budget Distribution (Colored by ROI)", 
                text_auto='.2s',
                color_continuous_scale="viridis" # Higher contrast
            )
            fig.update_layout(xaxis_title="Department", yaxis_title="Allocated Budget (INR)")
            st.plotly_chart(dark_chart(fig), use_container_width=True)
            render_analysis(f"""
            **What this shows:** The total money given to each department.
            <br>**How to read it:** Taller bars = More Money. Brighter Yellow/Green colors = Higher predicted Return on Investment (ROI).
            <br>**Insight:** If you see short, dark purple bars, that department is getting little money and offers low returns.
            """)
        else:
            st.info("No projects selected. Try increasing the budget.")
    with c2:
        st.subheader("Actionable Reports")
        report_txt = generate_board_report(portfolio, budget_input, wacc_input)
        st.download_button("📄 Download Board Report", report_txt, "Board_Report.txt", use_container_width=True)
        
        st.markdown("##### Top ROI Drivers")
        st.dataframe(feature_imp.head(3).style.background_gradient(cmap='Greens'), use_container_width=True, hide_index=True)
        render_analysis("""**Drivers:** These 3 factors are what the AI looks for when predicting if a project will succeed.""")

# --- PAGE: AI INSIGHTS ---
elif selected_page == "AI Insights":
    st.title("AI & Model Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 1. Sensitivity Analysis (Impact Chart)")
        if not portfolio.empty:
            fig_torn = generate_waterfall_chart(portfolio, wacc_input)
            st.plotly_chart(fig_torn, use_container_width=True)
            render_analysis("""
            **What this shows:** How fragile your profit (NPV) is to external shocks.
            <br>**How to read it:** Red bars on the left mean these events will *hurt* your profit. Green bars mean they will *help*.
            <br>**Insight:** If 'WACC Increase' has a huge red bar, your portfolio is very sensitive to interest rate hikes.
            """)
        else:
            st.warning("No portfolio to analyze.")
        
    with col2:
        st.markdown("##### 2. Predictive Drivers")
        fig_imp = px.bar(feature_imp, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Teal")
        st.plotly_chart(dark_chart(fig_imp), use_container_width=True)
        render_analysis("""
        **What this shows:** The recipe the AI uses to predict ROI.
        <br>**How to read it:** Longer bars = More important ingredients.
        """)

    st.markdown("---")
    st.markdown("##### 3. Portfolio Radar Profile")
    
    fig_radar = generate_3d_radar(df_prop)
    st.plotly_chart(fig_radar, use_container_width=True)
    render_analysis("""
    **What this shows:** A visual comparison between 'Winning' projects (Green) and 'Rejected' projects (Red).
    <br>**How to read it:** The further the shape stretches outward, the higher the score. You want the Green shape to be big in 'Strategy' and 'ROI', but small in 'Risk'.
    """)

# --- PAGE: EFFICIENT FRONTIER ---
elif selected_page == "Efficient Frontier":
    st.title("Efficient Frontier Simulation")
    
    sim_runs = st.slider("Monte Carlo Iterations", min_value=100, max_value=5000, value=1000, step=100)
    
    if st.button(f"Run {sim_runs} Simulation Scenarios"):
        st.markdown(f"Simulating {sim_runs} portfolio combinations...")
        progress_bar = st.progress(0)
        results = []
        total_cap = df_prop["Investment_Capital"].sum()
        avg_p = min(0.5, budget_input / (total_cap + 1))
        
        for i in range(sim_runs):
            mask = np.random.rand(len(df_prop)) < avg_p
            sample = df_prop[mask]
            if not sample.empty and sample["Investment_Capital"].sum() <= budget_input:
                results.append({"Risk": sample["Risk_Score"].mean(), "Return": sample["Pred_ROI"].mean(), "NPV": sample["Dynamic_NPV"].sum()})
            if i % 50 == 0: progress_bar.progress(min(i/sim_runs, 1.0))
        
        progress_bar.empty()
        
        sim_df = pd.DataFrame(results)
        if not sim_df.empty:
            # ENHANCED SCATTER
            fig_ef = px.scatter(
                sim_df, x="Risk", y="Return", color="NPV", 
                title="Risk vs Return Landscape", 
                color_continuous_scale="turbo", # Very high visibility scale
                labels={"Risk": "Average Risk Score", "Return": "Average ROI (%)"}
            )
            if not portfolio.empty:
                fig_ef.add_trace(go.Scatter(
                    x=[portfolio["Risk_Score"].mean()], 
                    y=[portfolio["Pred_ROI"].mean()], 
                    mode='markers+text', 
                    marker=dict(color='white', size=20, symbol='star', line=dict(width=2, color='black')), 
                    name="Your Portfolio",
                    text=["YOU ARE HERE"],
                    textposition="top center"
                ))
            st.plotly_chart(dark_chart(fig_ef), use_container_width=True)
            render_analysis("

[Image of Efficient Frontier Curve]
 **The Efficient Frontier:** The colorful cloud represents thousands of possible portfolios. The **White Star** is your current selection. If the star is at the top-left edge of the cloud, you have successfully maximized Return for your level of Risk.")
        else:
            st.error("Simulation failed to find valid portfolios within constraints.")

# --- PAGE: OPTIMIZATION REPORT ---
elif selected_page == "Optimization Report":
    st.title("Final Investment Schedule")
    tab1, tab2 = st.tabs(["Selected Projects", "Rejected Projects"])
    with tab1:
        st.dataframe(portfolio[["Project_ID", "Department", "Investment_Capital", "Pred_ROI", "Payback_Years", "Efficiency"]].style.format({"Investment_Capital": "₹{:,.0f}", "Pred_ROI": "{:.1f}%", "Payback_Years": "{:.1f} yrs", "Efficiency": "{:.2f}"}).background_gradient(subset=["Efficiency"], cmap="Greens"), use_container_width=True)
        csv = portfolio.to_csv(index=False).encode('utf-8')
        st.download_button("Export Portfolio (CSV)", csv, "Strategic_Portfolio.csv", "text/csv")
        render_analysis("**Explanation:** This table lists every project approved for funding. 'Efficiency' is a custom score calculated as (ROI / Risk). High efficiency means you are getting good returns for low risk.")
    with tab2:
        rejected = df_prop[df_prop["Selected"] == 0]
        st.dataframe(rejected[["Project_ID", "Investment_Capital", "Pred_ROI"]], use_container_width=True)
        render_analysis("**Explanation:** These projects were cut. Usually, this is because they had a low ROI, a very high Risk score, or simply didn't fit into the budget after funding better projects.")

# --- PAGE: STRATEGIC 3D MAP ---
elif selected_page == "Strategic 3D Map":
    st.title("Portfolio Topology")
    df_prop["Status"] = df_prop["Selected"].apply(lambda x: "Funded" if x==1 else "Not Funded")
    
    # ENHANCED 3D SCATTER
    fig_3d = px.scatter_3d(
        df_prop, 
        x="Risk_Score", 
        y="Strategic_Alignment", 
        z="Pred_ROI", 
        color="Status", 
        size="Investment_Capital", 
        opacity=0.9, 
        color_discrete_map={"Funded": "#00e676", "Not Funded": "#ff1744"},
        symbol="Status", # Adds shape differentiation
        labels={"Risk_Score": "Risk (Lower is Better)", "Strategic_Alignment": "Strategy Fit (Higher is Better)", "Pred_ROI": "ROI %"}
    )
    fig_3d.update_layout(scene=dict(
        xaxis_backgroundcolor="rgba(0,0,0,0)",
        yaxis_backgroundcolor="rgba(0,0,0,0)",
        zaxis_backgroundcolor="rgba(0,0,0,0)",
    ))
    st.plotly_chart(dark_chart(fig_3d), use_container_width=True)
    render_analysis("**How to Navigate:** Click and drag to rotate the cube. <br>**Goal:** You want to see Green bubbles floating in the top-back-left corner (High Strategy, High ROI, Low Risk). Red bubbles should ideally be clustered in the 'bad' corners (Low Strategy/Low ROI).")

# --- PAGE: SCENARIO MANAGER ---
elif selected_page == "Scenario Manager":
    st.title("Scenario Simulation")
    col_save, col_view = st.columns([1, 3])
    with col_save:
        scenario_name = st.text_input("Scenario Name", value="Base Case")
        if st.button("Save Current State"):
            s_data = {"Name": scenario_name, "WACC": wacc_input, "Budget": budget_input, "NPV": portfolio['Dynamic_NPV'].sum(), "ROI": portfolio['Pred_ROI'].mean(), "Projects": len(portfolio)}
            if 'scenarios' not in st.session_state: st.session_state['scenarios'] = []
            st.session_state['scenarios'].append(s_data)
            st.success(f"Saved {scenario_name}!")
    with col_view:
        if 'scenarios' in st.session_state and st.session_state['scenarios']:
            s_df = pd.DataFrame(st.session_state['scenarios'])
            st.dataframe(s_df.style.format({"NPV": "₹{:,.0f}", "ROI": "{:.1f}%", "WACC": "{:.1f}%"}), use_container_width=True)
            fig_comp = px.bar(s_df, x="Name", y="NPV", color="ROI", title="Scenario NPV Comparison")
            st.plotly_chart(dark_chart(fig_comp), use_container_width=True)
            render_analysis("**Explanation:** This tool lets you answer 'What If?' questions. Create a 'High Inflation' scenario by raising WACC in the sidebar, then save it here to compare against your 'Base Case'.")
        else:
            st.info("Adjust WACC/Budget in the sidebar, then click 'Save Current State' to compare scenarios.")

# --- PAGE: AI DEAL MEMOS ---
elif selected_page == "AI Deal Memos":
    st.title("AI Investment Memos")
    col_app, col_rej = st.columns(2)
    with col_app:
        st.subheader("Top Approvals")
        for i, row in portfolio.sort_values(by="Dynamic_NPV", ascending=False).head(3).iterrows():
            with st.expander(f"APPROVED: {row['Project_ID']} ({row['Department']})", expanded=True):
                st.markdown(f"**Rationale:**\n* **NPV Contribution:** ₹{row['Dynamic_NPV']/1e5:.1f} Lakhs\n* **Strategic Fit:** {row['Strategic_Alignment']}/10\n* **Decision:** Approved due to high WACC-adjusted return.")
    with col_rej:
        st.subheader("Top Rejections")
        rejected = df_prop[df_prop["Selected"] == 0]
        for i, row in rejected.sort_values(by="Pred_ROI", ascending=False).head(3).iterrows():
            with st.expander(f"REJECTED: {row['Project_ID']} ({row['Department']})"):
                st.markdown(f"**Rationale:**\n* **Issue:** Failed to beat capital cost hurdle or budget constraint.\n* **Risk Score:** {row['Risk_Score']}\n* **Decision:** Deferred to next fiscal cycle.")
    render_analysis("**Explanation:** These memos are auto-generated plain-English explanations. They tell you *why* the AI made the decision, so you can explain it to your boss or stakeholders.")
