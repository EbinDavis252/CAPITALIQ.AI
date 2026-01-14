import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestRegressor
import pulp
import sqlite3
import google.generativeai as genai
import time
from datetime import datetime, timedelta

# ----------------------------------------------------
# 1. Page Configuration & CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="CAPITALIQ-AI | Enterprise Edition",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊" 
)

# --- THEME ENGINE: ULTRA-CLEAN CORPORATE DARK ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear_gradient(135deg, #0f172a 0%, #1e293b 100%); }
    h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; font-weight: 700; letter-spacing: -0.5px; }
    p, li, span, label, div[data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }
    div[data-testid="stMetric"], div[data-testid="stExpander"], div.stDataFrame {
        background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; 
        padding: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #334155; }
    .brand-text {
        font-size: 26px; font-weight: 800;
        background: -webkit-linear-gradient(0deg, #38bdf8, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 1px; text-align: center; margin-bottom: 5px;
    }
    .brand-sub { font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: #64748b; text-align: center; margin-bottom: 30px; }
    .stButton>button { background: #2563eb; color: white; border: none; border-radius: 4px; font-weight: 600; text-transform: uppercase; font-size: 12px; height: 45px; transition: all 0.2s ease; }
    .stButton>button:hover { background: #1d4ed8; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3); }
    .analysis-box { border-left: 4px solid #38bdf8; background-color: rgba(56, 189, 248, 0.05); padding: 20px; margin-top: 15px; border-radius: 0 6px 6px 0; }
    .analysis-title { font-weight: 700; color: #38bdf8 !important; margin-bottom: 8px; display: block; text-transform: uppercase; font-size: 12px; letter-spacing: 1.5px; }
    .analysis-text { font-size: 14px; line-height: 1.6; color: #e2e8f0; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Database Layer (Persistence)
# ----------------------------------------------------
DB_FILE = "capital_planning.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scenarios 
                 (id INTEGER PRIMARY KEY, name TEXT, date TEXT, budget REAL, 
                  npv REAL, roi REAL, projects_count INTEGER, wacc REAL)''')
    conn.commit()
    conn.close()

def save_scenario_db(name, budget, npv, roi, count, wacc):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO scenarios (name, date, budget, npv, roi, projects_count, wacc) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (name, datetime.now().strftime("%Y-%m-%d %H:%M"), budget, npv, roi, count, wacc))
    conn.commit()
    conn.close()

def get_scenarios_db():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM scenarios ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

# ----------------------------------------------------
# 3. Helper Functions (Logic & AI)
# ----------------------------------------------------

def check_password():
    """Simple Authentication Wrapper"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("""<div class="brand-text">CAPITALIQ-AI</div><div class="brand-sub">Secure Login</div>""", unsafe_allow_html=True)
            pwd = st.text_input("Enter Access Token (Use 'admin'):", type="password")
            if st.button("Authenticate System"):
                if pwd == "admin":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Access Denied.")
        return False
    return True

def standardize_columns(df):
    """Maps various user inputs to system standard names."""
    column_map = {
        'Capex': 'Investment_Capital', 'Cost': 'Investment_Capital', 'Budget': 'Investment_Capital',
        'Return': 'Actual_ROI_Pct', 'ROI': 'Actual_ROI_Pct',
        'Strategy': 'Strategic_Alignment', 'Strat': 'Strategic_Alignment',
        'Risk': 'Risk_Score', 'Dept': 'Department', 'Mandatory': 'Is_Mandatory',
        'Group': 'Exclusion_Group'
    }
    df = df.rename(columns=column_map)
    
    # Ensure Advanced Logic Columns exist
    if 'Is_Mandatory' not in df.columns: df['Is_Mandatory'] = 0
    if 'Exclusion_Group' not in df.columns: df['Exclusion_Group'] = 0
    if 'Project_Description' not in df.columns: df['Project_Description'] = "Standard capital expansion project."
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df

@st.cache_data
def get_templates():
    """Generates sample CSV templates with new advanced columns."""
    df_h = pd.DataFrame({
        "Investment_Capital": np.random.randint(500000, 5000000, 50),
        "Duration_Months": np.random.randint(6, 36, 50),
        "Risk_Score": np.random.uniform(1, 10, 50).round(1),
        "Strategic_Alignment": np.random.uniform(1, 10, 50).round(1),
        "Market_Trend_Index": np.random.uniform(0.5, 1.5, 50).round(2),
        "Actual_ROI_Pct": np.random.uniform(5, 25, 50).round(1)
    })
    
    df_p = pd.DataFrame({
        "Project_ID": [f"PROJ-{i:03d}" for i in range(1, 21)],
        "Department": np.random.choice(['IT', 'R&D', 'Marketing', 'Ops'], 20),
        "Investment_Capital": np.random.randint(500000, 5000000, 20),
        "Duration_Months": np.random.randint(6, 36, 20),
        "Risk_Score": np.random.uniform(1, 10, 20).round(1),
        "Strategic_Alignment": np.random.uniform(1, 10, 20).round(1),
        "Market_Trend_Index": np.random.uniform(0.5, 1.5, 20).round(2),
        "Is_Mandatory": np.random.choice([0, 1], 20, p=[0.9, 0.1]),
        "Exclusion_Group": np.random.choice([0, 1, 2], 20, p=[0.8, 0.1, 0.1]), # 0 is no group
        "Project_Description": "Infrastructure upgrade for high velocity data processing."
    })
    return df_h, df_p

@st.cache_resource
def train_models(df_hist):
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
    """Calculates NPV assuming linear cash flow if annual data missing."""
    # Note: In a real app, we would look for 'CF_Y1', 'CF_Y2' columns.
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

def run_advanced_optimization(df, budget, min_dept_alloc_pct=0.0):
    """Advanced Solver with Mandatory & Mutually Exclusive constraints."""
    prob = pulp.LpProblem("Capital_Allocation", pulp.LpMaximize)
    selection_vars = pulp.LpVariable.dicts("Select", df.index, cat='Binary')
    
    # Objective: Maximize NPV
    prob += pulp.lpSum([df.loc[i, "Dynamic_NPV"] * selection_vars[i] for i in df.index])
    
    # 1. Budget Constraint
    prob += pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in df.index]) <= budget
    
    # 2. Mandatory Projects (Must be selected)
    for i in df.index:
        if df.loc[i, 'Is_Mandatory'] == 1:
            prob += selection_vars[i] == 1

    # 3. Mutually Exclusive Groups (Only 1 per group ID > 0)
    groups = df[df['Exclusion_Group'] > 0]['Exclusion_Group'].unique()
    for g in groups:
        group_indices = df[df['Exclusion_Group'] == g].index
        prob += pulp.lpSum([selection_vars[i] for i in group_indices]) <= 1

    # 4. Department Min Allocation
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

def generate_ai_memo(row, api_key):
    """Real AI Integration via Google Gemini"""
    if not api_key:
        return f"AI SIMULATION: Based on a risk score of {row['Risk_Score']} and ROI of {row['Pred_ROI']}%, this project is APPROVED."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Act as a cynical Investment Committee member. Analyze this project:
        ID: {row['Project_ID']}
        Desc: {row['Project_Description']}
        Cost: {row['Investment_Capital']}
        ROI: {row['Pred_ROI']}%
        Risk: {row['Risk_Score']}/10
        
        Write a 2-bullet point critique of why this project might fail and 1 strategic advantage. Be extremely concise.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Connection Error: {str(e)}"

# --- VISUALIZATIONS ---

def generate_gantt_chart(portfolio):
    """Creates a Project Timeline."""
    df_gantt = []
    start_date = datetime.now()
    for _, row in portfolio.iterrows():
        end_date = start_date + timedelta(days=int(row['Duration_Months']*30))
        df_gantt.append(dict(Task=row['Project_ID'], Start=start_date.strftime("%Y-%m-%d"), 
                             Finish=end_date.strftime("%Y-%m-%d"), Resource=row['Department']))
    
    fig = ff.create_gantt(df_gantt, index_col='Resource', show_colorbar=True, 
                          group_tasks=True, title="Implementation Timeline")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
    return fig

def generate_sensitivity_heatmap(portfolio, wacc_base):
    """Heatmap for WACC vs Inflation sensitivity."""
    wacc_range = [wacc_base - 0.02, wacc_base - 0.01, wacc_base, wacc_base + 0.01, wacc_base + 0.02]
    inflation_range = [0.02, 0.03, 0.04, 0.05, 0.06]
    
    z = []
    base_cash_flow = portfolio['Dynamic_NPV'].sum() + portfolio['Investment_Capital'].sum()
    
    for w in wacc_range:
        row_z = []
        for i in inflation_range:
            # Simplified sensitivity logic
            adjusted_npv = base_cash_flow / (1 + w + i) - portfolio['Investment_Capital'].sum()
            row_z.append(adjusted_npv)
        z.append(row_z)
        
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[f"{i*100:.0f}%" for i in inflation_range], 
        y=[f"{w*100:.1f}%" for w in wacc_range],
        colorscale='Viridis'))
    
    fig.update_layout(title="NPV Sensitivity (WACC vs Inflation)", 
                      xaxis_title="Inflation Rate", yaxis_title="WACC",
                      template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
    return fig

def render_analysis(text):
    st.markdown(f"<div class='analysis-box'><span class='analysis-title'>💡 EXECUTIVE INSIGHT</span><p class='analysis-text'>{text}</p></div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 4. Main App Logic
# ----------------------------------------------------

if not check_password():
    st.stop()

# --- Sidebar & Setup ---
with st.sidebar:
    st.markdown("""<div style="padding-top: 20px;"><div class="brand-text">CAPITALIQ-AI</div><div class="brand-sub">Enterprise Edition</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    
    if "page_selection" not in st.session_state: st.session_state.page_selection = "Home & Data"
    pages = ["Home & Data", "Executive Summary", "AI Insights", "Optimization Report", "Timeline", "Scenario Manager", "AI Deal Memos"]
    selected_page = st.radio("NAVIGATION", pages, key="page_selection", label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("Financial Assumptions")
    
    # CAPM WACC Builder
    with st.expander("WACC Builder (CAPM)"):
        rf = st.number_input("Risk Free Rate (%)", 3.0, 10.0, 4.0) / 100
        beta = st.number_input("Beta", 0.5, 3.0, 1.2)
        mkt_prem = st.number_input("Market Premium (%)", 3.0, 15.0, 6.0) / 100
        cost_equity = rf + beta * mkt_prem
        st.caption(f"Calculated Ke: {cost_equity*100:.2f}%")
        
        cost_debt = st.number_input("Cost of Debt (%)", 2.0, 15.0, 5.0) / 100
        tax_rate = st.number_input("Tax Rate (%)", 0.0, 40.0, 25.0) / 100
        equity_weight = st.slider("Equity % in Cap Structure", 0, 100, 70) / 100
        
        wacc_calc = (equity_weight * cost_equity) + ((1-equity_weight) * cost_debt * (1-tax_rate))
        st.markdown(f"**WACC: {wacc_calc*100:.2f}%**")
        wacc_input = wacc_calc

    budget_input = st.number_input("Total Budget (INR)", value=15000000.0, step=500000.0)
    min_dept_spend = st.slider("Min Dept. Allocation (%)", 0, 30, 0) / 100
    
    st.markdown("---")
    gemini_key = st.text_input("Gemini API Key (Optional)", type="password", help="Paste your Google Gemini API Key here for real AI features.")
    
    st.markdown("""<div style="text-align: center; font-size: 10px; color: #64748b;">© 2026 KD Technologies.<br>All Rights Reserved.</div>""", unsafe_allow_html=True)

# --- Data Processing Callback ---
def process_data_callback():
    if st.session_state.u_hist and st.session_state.u_prop:
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
        except Exception as e: st.error(f"Error processing files: {e}")

def load_demo_callback():
    df_hist, df_prop = get_templates()
    rf_roi, feature_imp = train_models(df_hist)
    features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
    df_prop["Pred_ROI"] = rf_roi.predict(df_prop[features])
    st.session_state['df_prop'] = df_prop
    st.session_state['feature_imp'] = feature_imp
    st.session_state.u_hist = "Demo"; st.session_state.u_prop = "Demo"
    st.session_state.page_selection = "Executive Summary"

# --- Shared Data Calculation ---
portfolio, rejected = pd.DataFrame(), pd.DataFrame()
if 'df_prop' in st.session_state:
    df_prop = st.session_state['df_prop'].copy()
    feature_imp = st.session_state['feature_imp']
    df_prop["Dynamic_NPV"] = df_prop.apply(lambda row: calculate_dynamic_npv(row, wacc_input), axis=1)
    df_prop["Efficiency"] = df_prop["Pred_ROI"] / df_prop["Risk_Score"]
    
    # Run Advanced Solver
    df_prop = run_advanced_optimization(df_prop, budget_input, min_dept_spend)
    portfolio = df_prop[df_prop["Selected"] == 1]
    rejected = df_prop[df_prop["Selected"] == 0]

# --- PAGE: HOME & DATA ---
if selected_page == "Home & Data":
    st.title("Enterprise Capital Planner")
    col_intro, col_setup = st.columns([1.5, 1])
    with col_intro:
        st.info("System Status: Secure & Online")
        st.markdown("### Workflow")
        st.markdown("1. **Upload History:** AI learns from past ROI.")
        st.markdown("2. **Configure WACC:** Use the CAPM builder in the sidebar.")
        st.markdown("3. **Optimize:** Solver handles 'Mandatory' projects and 'Mutually Exclusive' groups.")
        st.markdown("4. **Analyze:** Use the Timeline and Scenario Manager.")
        h_temp, p_temp = get_templates()
        c1, c2 = st.columns(2)
        c1.download_button("Download Train Template", h_temp.to_csv(index=False), "train.csv")
        c2.download_button("Download Proposal Template", p_temp.to_csv(index=False), "proposals.csv")
    with col_setup:
        st.markdown("#### Data Ingestion")
        st.file_uploader("1. History (Train)", type=["csv"], key="u_hist", on_change=process_data_callback)
        st.file_uploader("2. Proposals (Test)", type=["csv"], key="u_prop", on_change=process_data_callback)
        st.button("Load Demo Data", type="primary", on_click=load_demo_callback)

# --- PAGE: EXECUTIVE SUMMARY ---
elif selected_page == "Executive Summary":
    if portfolio.empty: st.warning("Please load data."); st.stop()
    st.title("Executive Dashboard")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Approved Projects", f"{len(portfolio)}", f"Rejected: {len(rejected)}")
    kpi2.metric("Capital Deployed", f"₹{portfolio['Investment_Capital'].sum()/1e6:.2f}M", f"Budget: ₹{budget_input/1e6:.2f}M")
    kpi3.metric("Projected NPV", f"₹{portfolio['Dynamic_NPV'].sum()/1e6:.2f}M", f"WACC: {wacc_input*100:.2f}%")
    kpi4.metric("Avg ROI", f"{portfolio['Pred_ROI'].mean():.2f}%", delta="Model Prediction")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Budget Allocation")
        fig = px.bar(portfolio, x="Department", y="Investment_Capital", color="Pred_ROI", title="Allocated Budget by Dept", color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Top Drivers")
        st.dataframe(feature_imp.head(5).style.background_gradient(cmap='Greens'), use_container_width=True, hide_index=True)

# --- PAGE: AI INSIGHTS ---
elif selected_page == "AI Insights":
    if portfolio.empty: st.warning("Please load data."); st.stop()
    st.title("AI & Sensitivity Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### NPV Sensitivity (WACC vs Inflation)")
        st.plotly_chart(generate_sensitivity_heatmap(portfolio, wacc_input), use_container_width=True)
        render_analysis("Heatmap shows NPV resilience. Darker areas indicate value destruction zones if inflation spikes.")
    with col2:
        st.markdown("##### Portfolio Efficiency (Risk vs Return)")
        fig_scatter = px.scatter(portfolio, x="Risk_Score", y="Pred_ROI", size="Investment_Capital", color="Department", hover_name="Project_ID")
        fig_scatter.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- PAGE: OPTIMIZATION REPORT ---
elif selected_page == "Optimization Report":
    if portfolio.empty: st.warning("Please load data."); st.stop()
    st.title("Final Investment Schedule")
    tab1, tab2 = st.tabs(["Approved", "Rejected"])
    with tab1:
        st.dataframe(portfolio[["Project_ID", "Department", "Investment_Capital", "Pred_ROI", "Is_Mandatory", "Efficiency"]].style.format({"Investment_Capital": "₹{:,.0f}", "Pred_ROI": "{:.1f}%"}).background_gradient(subset=["Efficiency"], cmap="Greens"), use_container_width=True)
    with tab2:
        st.dataframe(rejected[["Project_ID", "Investment_Capital", "Pred_ROI", "Risk_Score"]], use_container_width=True)

# --- PAGE: TIMELINE ---
elif selected_page == "Timeline":
    if portfolio.empty: st.warning("Please load data."); st.stop()
    st.title("Implementation Gantt Chart")
    st.plotly_chart(generate_gantt_chart(portfolio), use_container_width=True)
    render_analysis("Visualizes project duration and resource overlap. Ensure Departments are not bottlenecked by simultaneous project starts.")

# --- PAGE: SCENARIO MANAGER ---
elif selected_page == "Scenario Manager":
    if portfolio.empty: st.warning("Please load data."); st.stop()
    st.title("Scenario Persistence (SQLite)")
    
    col_save, col_view = st.columns([1, 2])
    with col_save:
        s_name = st.text_input("Scenario Name", "Base Case 2026")
        if st.button("Save to Database"):
            save_scenario_db(s_name, budget_input, portfolio['Dynamic_NPV'].sum(), portfolio['Pred_ROI'].mean(), len(portfolio), wacc_input)
            st.success("Saved to SQL Database!")
    
    with col_view:
        st.markdown("##### Historical Runs")
        df_hist = get_scenarios_db()
        if not df_hist.empty:
            st.dataframe(df_hist.style.format({'budget': '₹{:,.0f}', 'npv': '₹{:,.0f}', 'roi': '{:.1f}%', 'wacc': '{:.1%}'}), use_container_width=True)
        else:
            st.info("No scenarios saved in database yet.")

# --- PAGE: AI DEAL MEMOS ---
elif selected_page == "AI Deal Memos":
    if portfolio.empty: st.warning("Please load data."); st.stop()
    st.title("Generative AI Deal Memos")
    st.info("Using Google Gemini Pro for qualitative analysis.")
    
    for i, row in portfolio.head(3).iterrows():
        with st.expander(f"MEMO: {row['Project_ID']} - {row['Department']}", expanded=True):
            if st.button(f"Generate Memo for {row['Project_ID']}", key=f"btn_{i}"):
                with st.spinner("Consulting AI Investment Committee..."):
                    memo = generate_ai_memo(row, gemini_key)
                    st.markdown(memo)
            else:
                st.markdown("*Click generate to run LLM analysis...*")
