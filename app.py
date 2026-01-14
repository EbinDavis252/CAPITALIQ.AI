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

# ----------------------------------------------------
# 1. Page Configuration & CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="CAPITALIQ-AI | Enterprise Edition",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊" 
)

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
# 2. Database & Auth Layer
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
    c.execute("INSERT INTO scenarios (name, date, budget, npv, roi, projects_count, wacc) VALUES (?, datetime('now', 'localtime'), ?, ?, ?, ?, ?)",
              (name, budget, npv, roi, count, wacc))
    conn.commit()
    conn.close()

def get_scenarios_db():
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql("SELECT * FROM scenarios ORDER BY id DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

init_db()

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        return False
    return True

# ----------------------------------------------------
# 3. Helper Functions
# ----------------------------------------------------

def standardize_columns(df):
    column_map = {
        'Capex': 'Investment_Capital', 'Cost': 'Investment_Capital', 'Budget': 'Investment_Capital',
        'Return': 'Actual_ROI_Pct', 'ROI': 'Actual_ROI_Pct',
        'Strategy': 'Strategic_Alignment', 'Strat': 'Strategic_Alignment',
        'Risk': 'Risk_Score', 'Dept': 'Department', 
        'Mandatory': 'Is_Mandatory', 'Group': 'Exclusion_Group'
    }
    df = df.rename(columns=column_map)
    if 'Is_Mandatory' not in df.columns: df['Is_Mandatory'] = 0
    if 'Exclusion_Group' not in df.columns: df['Exclusion_Group'] = 0
    if 'Project_Description' not in df.columns: df['Project_Description'] = "Standard Project"
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

@st.cache_data
def get_templates():
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
        "Exclusion_Group": 0
    })
    return df_h, df_p

@st.cache_resource
def train_models(df_hist):
    features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
    for f in features:
        if f not in df_hist.columns: df_hist[f] = 0
    rf_roi = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_roi.fit(df_hist[features], df_hist["Actual_ROI_Pct"])
    importance = pd.DataFrame({'Feature': features, 'Importance': rf_roi.feature_importances_}).sort_values(by='Importance', ascending=False)
    return rf_roi, importance

def calculate_dynamic_npv(row, wacc_rate):
    total_return_value = row['Investment_Capital'] * (1 + (row['Pred_ROI'] / 100))
    duration = max(row['Duration_Months'], 1)
    annual_cash_flow = total_return_value / (duration / 12)
    years = duration / 12
    dcf = 0
    for t in range(1, int(years) + 2):
        if t <= years: dcf += annual_cash_flow / ((1 + wacc_rate) ** t)
        else: dcf += (annual_cash_flow * (years - int(years))) / ((1 + wacc_rate) ** t)
    return dcf - row['Investment_Capital']

def calculate_payback(row):
    total_return = row['Investment_Capital'] * (1 + (row['Pred_ROI'] / 100))
    annual = total_return / (max(row['Duration_Months'], 1) / 12)
    return round(row['Investment_Capital'] / annual, 2) if annual > 0 else 99.9

def run_advanced_optimization(df, budget, min_dept_alloc_pct=0.0):
    prob = pulp.LpProblem("Capital_Allocation", pulp.LpMaximize)
    selection_vars = pulp.LpVariable.dicts("Select", df.index, cat='Binary')
    prob += pulp.lpSum([df.loc[i, "Dynamic_NPV"] * selection_vars[i] for i in df.index])
    prob += pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in df.index]) <= budget
    
    if 'Is_Mandatory' in df.columns:
        for i in df.index:
            if df.loc[i, 'Is_Mandatory'] == 1: prob += selection_vars[i] == 1
            
    if 'Exclusion_Group' in df.columns:
        for g in df[df['Exclusion_Group'] > 0]['Exclusion_Group'].unique():
            prob += pulp.lpSum([selection_vars[i] for i in df[df['Exclusion_Group'] == g].index]) <= 1
            
    if min_dept_alloc_pct > 0 and 'Department' in df.columns:
        for dept in df['Department'].unique():
            dept_indices = df[df['Department'] == dept].index
            if df.loc[dept_indices, "Investment_Capital"].sum() >= budget * min_dept_alloc_pct:
                prob += pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in dept_indices]) >= budget * min_dept_alloc_pct
                
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df["Selected"] = [int(selection_vars[i].varValue) if selection_vars[i].varValue is not None else 0 for i in df.index]
    return df

def generate_ai_memo_text(row, api_key):
    if not api_key: return f"Decision: APPROVED\n\nProject exceeds risk-adjusted threshold."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Analyze project: {row.get('Project_Description','N/A')}. ROI: {row['Pred_ROI']}%. Risk: {row['Risk_Score']}. Cost: {row['Investment_Capital']}. Brief pro/con."
        return model.generate_content(prompt).text
    except Exception as e: return f"AI Error: {str(e)}"

# --- VISUALIZATION FUNCTIONS ---

def generate_professional_waterfall(portfolio):
    base_npv = portfolio['Dynamic_NPV'].sum()
    wacc_impact, inflation_impact, synergy_gain = -(base_npv*0.08), -(base_npv*0.05), (base_npv*0.12)
    final_npv = base_npv + wacc_impact + inflation_impact + synergy_gain
    fig = go.Figure(go.Waterfall(
        name="20", orientation="v", measure=["relative", "relative", "relative", "relative", "total"],
        x=["Base", "Cost of Cap", "Inflation", "Synergy", "Final"],
        y=[base_npv, wacc_impact, inflation_impact, synergy_gain, 0],
        text=[f"{x/1e6:.1f}M" for x in [base_npv, wacc_impact, inflation_impact, synergy_gain, final_npv]],
        decreasing={"marker":{"color":"#f43f5e"}}, increasing={"marker":{"color":"#10b981"}}, totals={"marker":{"color":"#3b82f6"}}
    ))
    fig.update_layout(title="Value Bridge", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def generate_benchmark_radar(df_prop):
    categories = ["Risk Safety", "Strategy", "ROI"]
    df_norm = df_prop.copy()
    df_norm["Risk Safety"] = 11 - df_norm["Risk_Score"]
    df_norm["Strategy"] = df_norm["Strategic_Alignment"]
    df_norm["ROI"] = (df_norm["Pred_ROI"] / df_norm["Pred_ROI"].max()) * 10 if df_norm["Pred_ROI"].max() > 0 else 0
    funded = df_norm[df_norm["Selected"] == 1][categories].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[8,8,8], theta=categories, fill='toself', name='Benchmark', line_color='rgba(255,255,255,0.3)'))
    fig.add_trace(go.Scatterpolar(r=[funded["Risk Safety"], funded["Strategy"], funded["ROI"]], theta=categories, fill='toself', name='Portfolio', line_color='#00e676'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10]), bgcolor="rgba(255,255,255,0.05)"), title="Portfolio vs Benchmark", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    return fig

def generate_3d_strategic_triangle(df_prop):
    funded = df_prop[df_prop["Selected"] == 1].copy()
    fig = go.Figure(data=[go.Scatter3d(
        x=funded['Strategic_Alignment'], y=funded['Pred_ROI'], z=10-funded['Risk_Score'],
        mode='markers', marker=dict(size=12, color=funded['Dynamic_NPV'], colorscale='Viridis'),
        text=funded['Project_ID']
    )])
    fig.update_layout(title="3D Strategy Map", scene=dict(xaxis_title='Strategy', yaxis_title='ROI', zaxis_title='Safety'), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,b=0,t=40))
    return fig

def generate_combo_scenario_chart(scenarios):
    # FIXED: Normalize column names to lowercase to match SQLite output
    df = pd.DataFrame(scenarios)
    df.columns = [x.lower() for x in df.columns]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["name"], y=df["npv"], name="NPV", marker_color="#3b82f6", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df["name"], y=df["roi"], name="ROI %", mode='lines+markers', line=dict(color="#fbbf24", width=3), yaxis="y2"))
    fig.update_layout(title="Scenario Analysis", yaxis=dict(title="NPV", side="left", showgrid=False), yaxis2=dict(title="ROI %", side="right", overlaying="y", showgrid=False), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2))
    return fig

# --- PDF GENERATOR ---
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'CAPITALIQ-AI Report', 0, 1, 'C')
            self.ln(10)
    def create_pdf_report(portfolio, rejected, budget, wacc):
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, f"Budget: {budget:,.2f}\nDeployed: {portfolio['Investment_Capital'].sum():,.2f}\nNPV: {portfolio['Dynamic_NPV'].sum():,.2f}\nWACC: {wacc:.1%}")
        pdf.ln()
        pdf.set_font('Arial', 'B', 10); pdf.cell(0, 10, "Approved Projects:", 0, 1)
        pdf.set_font('Arial', '', 9)
        for _, row in portfolio.iterrows(): pdf.cell(0, 5, f"{row['Project_ID']} | ROI: {row['Pred_ROI']:.1f}%", 0, 1)
        return pdf.output(dest='S').encode('latin-1')
except ImportError:
    PDF_AVAILABLE = False
    def create_pdf_report(portfolio, rejected, budget, wacc): return None

def generate_text_report(portfolio, budget, wacc):
    return f"Budget: {budget}\nNPV: {portfolio['Dynamic_NPV'].sum()}\nApproved: {len(portfolio)}"

def render_analysis(text):
    st.markdown(f"<div class='analysis-box'><span class='analysis-title'>💡 EXECUTIVE INSIGHT</span><p class='analysis-text'>{text}</p></div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 4. Main App Flow
# ----------------------------------------------------

if not check_login():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("""<div class="brand-text">CAPITALIQ-AI</div><div class="brand-sub">Secure Login</div>""", unsafe_allow_html=True)
        pwd = st.text_input("Enter Access Token (Use 'admin'):", type="password")
        if st.button("Authenticate System"):
            if pwd == "admin": st.session_state.authenticated = True; st.rerun()
            else: st.error("Access Denied.")
    st.stop()

with st.sidebar:
    st.markdown("""<div style="padding-top: 20px;"><div class="brand-text">CAPITALIQ-AI</div><div class="brand-sub">Enterprise Edition</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    if "page_selection" not in st.session_state: st.session_state.page_selection = "Home & Data"
    pages = ["Home & Data", "Executive Summary", "AI Insights", "Efficient Frontier", "Optimization Report", "Strategic 3D Map", "Scenario Manager", "AI Deal Memos"]
    selected_page = st.radio("NAVIGATION", pages, key="page_selection", label_visibility="collapsed")
    
    st.markdown("---")
    with st.expander("WACC Builder (CAPM)"):
        rf = st.number_input("Risk Free (%)", 2.0, 10.0, 4.0)/100
        beta = st.number_input("Beta", 0.5, 3.0, 1.2)
        mkt = st.number_input("Market Prem (%)", 2.0, 15.0, 6.0)/100
        cost_eq = rf + beta * mkt
        cost_d = st.number_input("Cost of Debt (%)", 2.0, 15.0, 5.0)/100
        tax = st.number_input("Tax Rate (%)", 0.0, 40.0, 25.0)/100
        eq_w = st.slider("Equity %", 0, 100, 70)/100
        wacc_input = (eq_w * cost_eq) + ((1-eq_w) * cost_d * (1-tax))
        st.caption(f"WACC: {wacc_input*100:.2f}%")
    
    budget_input = st.number_input("Budget (INR)", value=15000000.0, step=500000.0)
    min_dept_spend = st.slider("Min Dept %", 0, 30, 0)/100
    market_shock = st.slider("Market Scenario", -0.2, 0.2, 0.0, 0.05)
    gemini_key = st.text_input("Gemini API Key", type="password")
    if st.button("Clear All Data"): st.session_state.clear(); st.rerun()

# Data Handling
if selected_page == "Home & Data":
    st.title("Welcome to CapitalIQ-AI")
    c1, c2 = st.columns(2)
    with c1:
        st.file_uploader("History (Train)", type=["csv"], key="u_hist")
        st.file_uploader("Proposals (Test)", type=["csv"], key="u_prop")
        if st.button("Load Demo Data"):
            h, p = get_templates()
            st.session_state.u_hist = "Demo"; st.session_state.u_prop = "Demo"
            rf, imp = train_models(h)
            p["Pred_ROI"] = rf.predict(p[["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]])
            st.session_state.df_prop = p; st.session_state.feature_imp = imp
            st.rerun()
    with c2:
        if 'df_prop' in st.session_state:
            st.success("System Online")
            if st.button("Go to Dashboard"): st.session_state.page_selection = "Executive Summary"; st.rerun()

if 'df_prop' in st.session_state and selected_page != "Home & Data":
    df_prop = st.session_state.df_prop.copy()
    feature_imp = st.session_state.feature_imp
    df_prop["Pred_ROI"] = df_prop["Pred_ROI"] * (1+market_shock)
    df_prop["Dynamic_NPV"] = df_prop.apply(lambda r: calculate_dynamic_npv(r, wacc_input), axis=1)
    df_prop["Payback_Years"] = df_prop.apply(calculate_payback, axis=1)
    df_prop["Efficiency"] = df_prop["Pred_ROI"] / df_prop["Risk_Score"]
    
    # Optimization
    df_prop = run_advanced_optimization(df_prop, budget_input, min_dept_spend)
    portfolio = df_prop[df_prop["Selected"]==1]
    rejected = df_prop[df_prop["Selected"]==0]

    if selected_page == "Executive Summary":
        st.title("Executive Dashboard")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Funded", len(portfolio)); k2.metric("Deployed", f"{portfolio['Investment_Capital'].sum()/1e6:.1f}M")
        k3.metric("NPV", f"{portfolio['Dynamic_NPV'].sum()/1e6:.1f}M"); k4.metric("Avg ROI", f"{portfolio['Pred_ROI'].mean():.1f}%")
        
        c1, c2 = st.columns([2,1])
        with c1: st.plotly_chart(px.bar(portfolio, x="Department", y="Investment_Capital", color="Pred_ROI", title="Budget by Dept", template="plotly_dark"), use_container_width=True)
        with c2: 
            if PDF_AVAILABLE: st.download_button("Download Report", create_pdf_report(portfolio, rejected, budget_input, wacc_input), "Report.pdf")
            st.dataframe(feature_imp.head())

    elif selected_page == "AI Insights":
        st.title("AI Analytics")
        st.plotly_chart(generate_professional_waterfall(portfolio), use_container_width=True)
        st.plotly_chart(generate_benchmark_radar(df_prop), use_container_width=True)

    elif selected_page == "Efficient Frontier":
        st.title("Efficient Frontier")
        if st.button("Run Simulation"):
            res = []
            for _ in range(500):
                sam = df_prop.sample(frac=0.5)
                if sam["Investment_Capital"].sum() <= budget_input:
                    res.append({"Risk": sam["Risk_Score"].mean(), "Return": sam["Pred_ROI"].mean()})
            if res: st.plotly_chart(px.scatter(pd.DataFrame(res), x="Risk", y="Return", title="Risk vs Return", template="plotly_dark"), use_container_width=True)

    elif selected_page == "Optimization Report":
        st.title("Investment Schedule")
        st.dataframe(portfolio[["Project_ID", "Department", "Investment_Capital", "Pred_ROI"]].style.background_gradient(subset=["Pred_ROI"]), use_container_width=True)

    elif selected_page == "Strategic 3D Map":
        st.title("3D Topology")
        st.plotly_chart(generate_3d_strategic_triangle(df_prop), use_container_width=True)

    elif selected_page == "Scenario Manager":
        st.title("Scenario Manager (SQL)")
        c1, c2 = st.columns([1,3])
        with c1:
            name = st.text_input("Name", "Scenario A")
            if st.button("Save"):
                save_scenario_db(name, budget_input, portfolio['Dynamic_NPV'].sum(), portfolio['Pred_ROI'].mean(), len(portfolio), wacc_input)
                st.success("Saved!")
        with c2:
            df_s = get_scenarios_db()
            if not df_s.empty:
                st.plotly_chart(generate_combo_scenario_chart(df_s.to_dict('records')), use_container_width=True)
                st.dataframe(df_s)

    elif selected_page == "AI Deal Memos":
        st.title("AI Memos")
        for _, row in portfolio.head(3).iterrows():
            with st.expander(f"Memo: {row['Project_ID']}"):
                if st.button(f"Generate {row['Project_ID']}", key=row['Project_ID']):
                    st.write(generate_ai_memo_text(row, gemini_key))
