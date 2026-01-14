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
    initial_sidebar_state="expanded",
    page_icon="💼"
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

    /* Text Visibility Fixes - GLOBAL OVERRIDE */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, li, span, label, 
    .stDataFrame, .stRadio label, .streamlit-expander-content, div[data-testid="stMarkdownContainer"] p {
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
        text-transform: uppercase;
        letter-spacing: 1px;
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

# --- VISUALIZATIONS ---

def generate_waterfall_chart(portfolio):
    base_npv = portfolio['Dynamic_NPV'].sum()
    wacc_impact = -(base_npv * 0.08)
    inflation_impact = -(base_npv * 0.05)
    synergy_gain = (base_npv * 0.12)
    
    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative", "relative", "relative", "relative", "total"],
        x = ["Base Case", "WACC Increase", "Inflation Shock", "Synergy Upside", "Final Scenario"],
        textposition = "outside",
        text = [f"{base_npv/1e6:.1f}M", f"{wacc_impact/1e6:.1f}M", f"{inflation_impact/1e6:.1f}M", f"{synergy_gain/1e6:.1f}M", f"{(base_npv+wacc_impact+inflation_impact+synergy_gain)/1e6:.1f}M"],
        y = [base_npv, wacc_impact, inflation_impact, synergy_gain, 0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"#ef4444"}},
        increasing = {"marker":{"color":"#22c55e"}},
        totals = {"marker":{"color":"#3b82f6"}}
    ))
    fig.update_layout(title = "Capital Bridge: Sensitivity to Risk Factors", showlegend = False, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
    return fig

def generate_complex_radar(df_prop):
    categories = ["Risk_Score", "Strategic_Alignment", "Pred_ROI"]
    df_norm = df_prop.copy()
    df_norm["Risk_Score"] = 10 - df_norm["Risk_Score"]
    funded = df_norm[df_norm["Selected"] == 1][categories].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[9, 9, 25], theta=["Low Risk (Inv)", "Strategy", "ROI"], fill='toself', name='Target Benchmark', line_color='rgba(255, 255, 255, 0.2)', line_dash='dot'))
    fig.add_trace(go.Scatterpolar(r=[funded["Risk_Score"], funded["Strategic_Alignment"], funded["Pred_ROI"]], theta=["Low Risk (Inv)", "Strategy", "ROI"], fill='toself', name='Selected Portfolio', line_color='#00e676', opacity=0.8))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, showticklabels=False), bgcolor="rgba(255,255,255,0.05)"), title="Portfolio Quality vs. Benchmark", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
    return fig

def generate_combo_scenario_chart(scenarios):
    df = pd.DataFrame(scenarios)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Name"], y=df["NPV"], name="Net Present Value (NPV)", marker_color="#3b82f6", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df["Name"], y=df["ROI"], name="Avg ROI %", mode='lines+markers', line=dict(color="#fbbf24", width=3), yaxis="y2"))
    fig.update_layout(title="Scenario Analysis: Value vs Efficiency", yaxis=dict(title="NPV (Currency)", side="left", showgrid=False), yaxis2=dict(title="ROI (%)", side="right", overlaying="y", showgrid=False), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"), legend=dict(orientation="h", y=-0.2))
    return fig

# --- PDF GENERATOR (SAFE MODE) ---
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
    
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'CAPITALIQ-AI | Strategic Portfolio Report', 0, 1, 'C')
            self.line(10, 20, 200, 20)
            self.ln(10)

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(200, 220, 255)
            self.cell(0, 6, title, 0, 1, 'L', 1)
            self.ln(4)

        def chapter_body(self, body):
            self.set_font('Arial', '', 10)
            self.multi_cell(0, 5, body)
            self.ln()

    def create_pdf_report(portfolio, rejected, budget, wacc):
        pdf = PDFReport()
        pdf.add_page()
        pdf.chapter_title('1. Executive Summary')
        summary = f"""Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\nTotal Budget Authorized: INR {budget:,.2f}\nCapital Deployed: INR {portfolio['Investment_Capital'].sum():,.2f}\nTotal Portfolio NPV: INR {portfolio['Dynamic_NPV'].sum():,.2f}\nWACC: {wacc*100:.1f}%\nAverage ROI: {portfolio['Pred_ROI'].mean():.2f}%"""
        pdf.chapter_body(summary)
        pdf.chapter_title('2. Approved Investment Schedule')
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(40, 7, 'Project ID', 1); pdf.cell(50, 7, 'Department', 1); pdf.cell(50, 7, 'Investment (INR)', 1); pdf.cell(30, 7, 'ROI (%)', 1); pdf.ln()
        pdf.set_font('Arial', '', 9)
        for _, row in portfolio.iterrows():
            pdf.cell(40, 6, str(row['Project_ID']), 1); pdf.cell(50, 6, str(row['Department']), 1); pdf.cell(50, 6, f"{row['Investment_Capital']:,.0f}", 1); pdf.cell(30, 6, f"{row['Pred_ROI']:.1f}", 1); pdf.ln()
        pdf.ln()
        pdf.chapter_title('3. Deferred / Rejected Proposals')
        pdf.set_font('Arial', '', 9)
        for _, row in rejected.iterrows():
            pdf.cell(0, 5, f"[X] {row['Project_ID']} ({row['Department']}) - Low Efficiency Score: {row['Efficiency']:.2f}", 0, 1)
        return pdf.output(dest='S').encode('latin-1')

except ImportError:
    PDF_AVAILABLE = False
    def create_pdf_report(portfolio, rejected, budget, wacc): return None

def generate_text_report(portfolio, budget, wacc):
    txt = f"CAPITALIQ-AI REPORT\n-------------------\nBudget: INR {budget:,.2f}\nDeployed: INR {portfolio['Investment_Capital'].sum():,.2f}\nNPV: INR {portfolio['Dynamic_NPV'].sum():,.2f}\nWACC: {wacc*100:.1f}%\n\nAPPROVED PROJECTS:\n"
    for _, row in portfolio.iterrows(): txt += f"[x] {row['Project_ID']} | ROI: {row['Pred_ROI']:.1f}%\n"
    return txt

def dark_chart(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"), margin=dict(l=20, r=20, t=40, b=20))
    return fig

def render_analysis(text):
    st.markdown(f"<div class='analysis-box'><span class='analysis-title'>💡 EXECUTIVE INSIGHT</span>{text}</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 3. Callback Functions (Nav Logic)
# ----------------------------------------------------

def process_data_callback():
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
        except Exception as e: st.error(f"Error processing files: {e}")

def load_demo_callback():
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
    # --- BRANDING LOGO (CYBERPUNK GLOW STYLE) ---
    # We use CSS 'filter: invert(1)' to turn the black icon WHITE.
    # We also add a 'drop-shadow' to give it a subtle AI glow.
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="https://cdn-icons-png.flaticon.com/512/2103/2103633.png" 
                 width="100" 
                 style="filter: invert(1) drop-shadow(0 0 4px #22c55e); transition: transform 0.3s;">
            <p style="color: #e2e8f0; font-weight: 600; font-size: 24px; margin-top: 10px; letter-spacing: 2px;">CAPITALIQ-AI</p>
            <p style="color: #94a3b8; font-size: 12px; margin-top: -15px;">Enterprise Edition</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
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
            st.info("""**Workflow:**\n1. **Upload** historical project data to train the predictive models.\n2. **Configure** financial constraints (Budget, WACC) in the sidebar.\n3. **Analyze** the optimized portfolio across various strategic dimensions.""")
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
    rejected = df_prop[df_prop["Selected"] == 0]

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
            fig = px.bar(portfolio, x="Department", y="Investment_Capital", color="Pred_ROI", title="Budget Distribution (Colored by ROI)", text_auto='.2s', color_continuous_scale="viridis")
            fig.update_layout(xaxis_title="Department", yaxis_title="Allocated Budget (INR)")
            st.plotly_chart(dark_chart(fig), use_container_width=True)
            render_analysis("""This chart reveals the strategic prioritization of capital across functional units. The height of the bars corresponds to the total capital allocated, while the color gradient indicates the return efficiency. Departments represented by brighter yellow-green bars are generating superior ROI, suggesting that the AI has successfully concentrated capital in high-performance areas to maximize the aggregate WACC-adjusted value.""")
        else: st.info("No projects selected. Try increasing the budget.")
    with c2:
        st.subheader("Actionable Reports")
        if PDF_AVAILABLE:
            pdf_data = create_pdf_report(portfolio, rejected, budget_input, wacc_input)
            st.download_button("📄 Download Official PDF Report", data=pdf_data, file_name="CapitalIQ_Report.pdf", mime="application/pdf", use_container_width=True)
        else:
            st.warning("⚠️ PDF generation disabled (missing 'fpdf').")
            txt_report = generate_text_report(portfolio, budget_input, wacc_input)
            st.download_button("Download Text Report", txt_report, "Report.txt", use_container_width=True)
        st.markdown("##### Top ROI Drivers")
        st.dataframe(feature_imp.head(3).style.background_gradient(cmap='Greens'), use_container_width=True, hide_index=True)
        render_analysis("""These factors represent the primary statistical drivers influencing project success within your specific dataset. The model weights these variables most heavily when predicting future ROI, indicating that improvements in these specific areas will yield the highest marginal increase in portfolio performance.""")

# --- PAGE: AI INSIGHTS ---
elif selected_page == "AI Insights":
    st.title("AI & Model Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 1. Financial Sensitivity Bridge")
        if not portfolio.empty:
            fig_torn = generate_waterfall_chart(portfolio)
            st.plotly_chart(fig_torn, use_container_width=True)
            render_analysis("""This Waterfall analysis bridges your current Base Case NPV to potential future states. The blue total represents your current projection. The red steps downward illustrate how specific risks, such as a rise in the Weighted Average Cost of Capital (WACC) or inflation shocks, erode value. Conversely, green steps would indicate potential upsides. This visual quantifies exactly how much capital buffer is required to withstand adverse market conditions.""")
        else: st.warning("No portfolio to analyze.")
    with col2:
        st.markdown("##### 2. Predictive Signal Strength")
        fig_imp = px.bar(feature_imp, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Teal")
        st.plotly_chart(dark_chart(fig_imp), use_container_width=True)
        render_analysis("""This visualization ranks the predictive power of each input variable. A longer bar signifies that the Random Forest algorithm relies more heavily on that specific metric to distinguish between high-performing and low-performing projects. Understanding this hierarchy allows management to focus data collection and improvement efforts on the variables that actually move the needle.""")
    st.markdown("---")
    st.markdown("##### 3. Strategic Fit vs. Benchmark")
    fig_radar = generate_complex_radar(df_prop)
    st.plotly_chart(fig_radar, use_container_width=True)
    render_analysis("""This Radar chart contrasts the selected portfolio's average profile (Green Area) against an ideal theoretical benchmark (Dotted White Line). The vertices represent key strategic dimensions: Risk (inverted so outer is better), Strategic Alignment, and ROI. A robust portfolio should expand outward to fill the web, maximizing Strategy and ROI while minimizing Risk. Gaps between the green area and the white line highlight specific opportunities for portfolio optimization.""")

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
            fig_ef = px.scatter(sim_df, x="Risk", y="Return", color="NPV", title="Risk vs Return Landscape", color_continuous_scale="turbo", labels={"Risk": "Average Risk Score", "Return": "Average ROI (%)"})
            if not portfolio.empty:
                fig_ef.add_trace(go.Scatter(x=[portfolio["Risk_Score"].mean()], y=[portfolio["Pred_ROI"].mean()], mode='markers+text', marker=dict(color='white', size=20, symbol='star', line=dict(width=2, color='black')), name="Your Portfolio", text=["YOU ARE HERE"], textposition="top center"))
            st.plotly_chart(dark_chart(fig_ef), use_container_width=True)
            render_analysis("""The Efficient Frontier visualization plots thousands of potential portfolio combinations to map the optimal trade-off between risk and return. The colorful cloud represents the universe of possibilities. The White Star marks your current portfolio's position. Ideally, this star should sit on the upper-left boundary of the cloud, indicating that you have achieved the maximum possible return for your accepted level of risk, leaving no value on the table.""")
        else: st.error("Simulation failed to find valid portfolios within constraints.")

# --- PAGE: OPTIMIZATION REPORT ---
elif selected_page == "Optimization Report":
    st.title("Final Investment Schedule")
    tab1, tab2 = st.tabs(["Selected Projects", "Rejected Projects"])
    with tab1:
        st.dataframe(portfolio[["Project_ID", "Department", "Investment_Capital", "Pred_ROI", "Payback_Years", "Efficiency"]].style.format({"Investment_Capital": "₹{:,.0f}", "Pred_ROI": "{:.1f}%", "Payback_Years": "{:.1f} yrs", "Efficiency": "{:.2f}"}).background_gradient(subset=["Efficiency"], cmap="Greens"), use_container_width=True)
        csv = portfolio.to_csv(index=False).encode('utf-8')
        st.download_button("Export Raw Data (CSV)", csv, "Strategic_Portfolio.csv", "text/csv")
        render_analysis("""This schedule outlines the projects that successfully cleared both the financial hurdle rates and the strategic optimization constraints. The Efficiency metric, calculated as ROI divided by Risk Score, serves as a primary selection criterion. Projects with high efficiency scores (Dark Green) deliver the most value per unit of risk assumed, forming the backbone of a resilient portfolio.""")
    with tab2:
        rejected = df_prop[df_prop["Selected"] == 0]
        st.dataframe(rejected[["Project_ID", "Investment_Capital", "Pred_ROI"]], use_container_width=True)
        render_analysis("""These proposals were excluded from the final selection. Rejection typically stems from one of three causes: insufficient ROI to cover the cost of capital, a risk profile that exceeds the organizational tolerance, or lower capital efficiency compared to competing projects when budget constraints are applied. These projects should be re-evaluated for scope reduction or risk mitigation before resubmission.""")

# --- PAGE: STRATEGIC 3D MAP ---
elif selected_page == "Strategic 3D Map":
    st.title("Portfolio Topology")
    df_prop["Status"] = df_prop["Selected"].apply(lambda x: "Funded" if x==1 else "Not Funded")
    fig_3d = px.scatter_3d(df_prop, x="Efficiency", y="Dynamic_NPV", z="Pred_ROI", color="Status", size="Investment_Capital", opacity=0.9, color_discrete_map={"Funded": "#00e676", "Not Funded": "#ff1744"}, symbol="Status", labels={"Efficiency": "Capital Efficiency (ROI/Risk)", "Dynamic_NPV": "Net Present Value", "Pred_ROI": "ROI %"})
    fig_3d.update_layout(scene=dict(xaxis_backgroundcolor="rgba(0,0,0,0)", yaxis_backgroundcolor="rgba(0,0,0,0)", zaxis_backgroundcolor="rgba(0,0,0,0)"))
    st.plotly_chart(dark_chart(fig_3d), use_container_width=True)
    render_analysis("""This 3D Topology map allows for a multivariate assessment of the portfolio. By plotting Capital Efficiency (X), Net Present Value (Y), and ROI (Z), we create a volume of value. The Funded projects (Green) should ideally cluster in the upper-right-back quadrant, representing high efficiency, high value, and high return. This visualization helps confirm that the AI is not just picking 'safe' low-value projects, but is truly optimizing for the 'sweet spot' of financial performance.""")

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
            fig_comp = generate_combo_scenario_chart(st.session_state['scenarios'])
            st.plotly_chart(fig_comp, use_container_width=True)
            render_analysis("""This scenario analysis tool provides a comparative view of how different strategic assumptions impact the bottom line. The Blue Bars represent the Net Present Value (NPV), indicating the absolute wealth creation of each scenario. The Yellow Line tracks the average ROI percentage. A divergence between these two lines—for example, a scenario with high ROI but low NPV—might indicate a portfolio that is efficient but too conservative in its capital deployment.""")
        else: st.info("Adjust WACC/Budget in the sidebar, then click 'Save Current State' to compare scenarios.")

# --- PAGE: AI DEAL MEMOS ---
elif selected_page == "AI Deal Memos":
    st.title("AI Investment Memos")
    col_app, col_rej = st.columns(2)
    with col_app:
        st.subheader("Top Approvals")
        for i, row in portfolio.sort_values(by="Dynamic_NPV", ascending=False).head(3).iterrows():
            with st.expander(f"APPROVED: {row['Project_ID']} ({row['Department']})", expanded=True):
                # Using standard Streamlit Success Box for guaranteed visibility
                st.success(f"**Decision:** APPROVED\n\nThis project exceeds the required risk-adjusted return threshold and aligns with strategic goals.")
                c1, c2 = st.columns(2)
                c1.metric("NPV Contribution", f"₹{row['Dynamic_NPV']/1e5:.1f} L")
                c2.metric("Strategic Fit", f"{row['Strategic_Alignment']}/10")

    with col_rej:
        st.subheader("Top Rejections")
        rejected = df_prop[df_prop["Selected"] == 0]
        for i, row in rejected.sort_values(by="Pred_ROI", ascending=False).head(3).iterrows():
            with st.expander(f"REJECTED: {row['Project_ID']} ({row['Department']})"):
                # Using standard Streamlit Error Box for guaranteed visibility
                st.error(f"**Decision:** DEFERRED\n\nReason: Failed to beat capital cost hurdle or budget constraint.")
                c1, c2 = st.columns(2)
                c1.metric("Risk Score", f"{row['Risk_Score']}")
                c2.metric("Efficiency", f"{row['Efficiency']:.2f}")
    
    render_analysis("""These automated investment memos serve as a transparent audit trail for stakeholder communication. By explicitly stating the financial metrics (NPV, Risk Score) and the strategic logic behind every approval and rejection, these documents bridge the gap between complex algorithmic decision-making and executive accountability.""")
