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
    page_title="CapitalIQ | Enterprise Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔷"
)

# --- THEME ENGINE: FRESHBOOKS "CLEAN & FRIENDLY" ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;600;800&display=swap');

    /* FRESHBOOKS COLOR TOKENS */
    :root {
        --fb-blue: #0075DD;
        --fb-dark-blue: #0056a3;
        --fb-green: #22C55E;
        --fb-bg: #F2F6FA;
        --fb-text: #2D3E50;
        --fb-card-bg: #FFFFFF;
        --fb-border: #E1E7EE;
    }

    /* Global Reset & Typography */
    * { font-family: 'Hanken Grotesk', sans-serif; color: var(--fb-text); }
    
    .stApp {
        background-color: var(--fb-bg);
        background-image: none; /* Removed Dark Mode Image */
    }

    /* Headings */
    h1, h2, h3 {
        color: var(--fb-text) !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
    h3 { color: var(--fb-blue) !important; font-weight: 600; }

    /* FRESHBOOKS CARD STYLE (Metrics & Containers) */
    div[data-testid="stMetric"], div.stDataFrame, div.analysis-box {
        background-color: var(--fb-card-bg) !important;
        border: 1px solid var(--fb-border);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03); /* Soft Shadow */
        color: var(--fb-text);
    }

    /* Metric Values */
    div[data-testid="stMetricValue"] {
        color: var(--fb-blue) !important;
        font-weight: 800;
        font-size: 1.8rem;
    }

    /* Sidebar Styling - Clean White */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid var(--fb-border);
    }
    
    /* Sidebar Links/Radio Buttons */
    .stRadio label {
        color: var(--fb-text) !important;
        font-weight: 600;
        padding: 10px;
        border-radius: 6px;
        transition: background 0.2s;
    }
    .stRadio label:hover {
        background-color: #F2F6FA;
        color: var(--fb-blue) !important;
    }

    /* BUTTONS: FreshBooks Green & Blue */
    .stButton>button {
        background-color: var(--fb-blue);
        color: white;
        border: none;
        border-radius: 50px; /* Pill Shape */
        font-weight: 700;
        padding: 12px 24px;
        box-shadow: 0 4px 6px rgba(0, 117, 221, 0.2);
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: var(--fb-dark-blue);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 117, 221, 0.3);
        color: white;
    }
    
    /* Secondary Action (Reset) */
    button[kind="secondary"] {
        background-color: transparent;
        border: 2px solid var(--fb-border);
        color: var(--fb-text);
    }

    /* EXPANDER: Clean White Accordion */
    .streamlit-expanderHeader {
        background-color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid var(--fb-border);
        color: var(--fb-text);
        font-weight: 600;
    }
    
    /* Analysis Box (Insight) */
    .analysis-box {
        border-left: 4px solid var(--fb-green);
        background-color: #F0FDF4 !important; /* Very Light Green */
    }
    .analysis-title {
        color: var(--fb-green) !important;
        font-weight: 800;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 1px;
        display: block;
        margin-bottom: 8px;
    }
    
    /* Remove default Streamlit Headers coloring */
    .stMarkdown p { color: var(--fb-text) !important; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Helper Functions
# ----------------------------------------------------

def standardize_columns(df):
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
    features = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
    for f in features:
        if f not in df_hist.columns: df_hist[f] = 0
    rf_roi = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_roi.fit(df_hist[features], df_hist["Actual_ROI_Pct"])
    importance = pd.DataFrame({'Feature': features, 'Importance': rf_roi.feature_importances_}).sort_values(by='Importance', ascending=False)
    return rf_roi, importance

def calculate_dynamic_npv(row, wacc_rate):
    total_return = row['Investment_Capital'] * (1 + (row['Pred_ROI'] / 100))
    duration = max(row['Duration_Months'], 1)
    annual_cf = total_return / (duration / 12)
    years = duration / 12
    dcf = 0
    for t in range(1, int(years) + 2):
        if t <= years: dcf += annual_cf / ((1 + wacc_rate) ** t)
        else: dcf += (annual_cf * (years - int(years))) / ((1 + wacc_rate) ** t)
    return dcf - row['Investment_Capital']

def calculate_payback(row):
    total_return = row['Investment_Capital'] * (1 + (row['Pred_ROI'] / 100))
    annual_cf = total_return / (max(row['Duration_Months'], 1) / 12)
    if annual_cf <= 0: return 99.9 
    return round(row['Investment_Capital'] / annual_cf, 2)

def run_advanced_optimization(df, budget, min_dept_alloc_pct=0.0):
    prob = pulp.LpProblem("Capital_Allocation", pulp.LpMaximize)
    selection_vars = pulp.LpVariable.dicts("Select", df.index, cat='Binary')
    prob += pulp.lpSum([df.loc[i, "Dynamic_NPV"] * selection_vars[i] for i in df.index])
    prob += pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in df.index]) <= budget
    
    if min_dept_alloc_pct > 0 and 'Department' in df.columns:
        for dept in df['Department'].unique():
            indices = df[df['Department'] == dept].index
            if df.loc[indices, "Investment_Capital"].sum() >= (budget * min_dept_alloc_pct):
                prob += pulp.lpSum([df.loc[i, "Investment_Capital"] * selection_vars[i] for i in indices]) >= (budget * min_dept_alloc_pct)

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    df["Selected"] = [int(selection_vars[i].varValue) if selection_vars[i].varValue is not None else 0 for i in df.index]
    return df

# --- VISUALIZATION HELPERS (FreshBooks Style) ---
def fresh_chart(fig):
    """Updates Plotly charts to match FreshBooks clean light theme"""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#2D3E50", family="Hanken Grotesk"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#E1E7EE", zeroline=False)
    )
    return fig

def render_analysis(text):
    st.markdown(f"<div class='analysis-box'><span class='analysis-title'>💡 Expert Insight</span>{text}</div>", unsafe_allow_html=True)

# --- PDF GENERATOR (Safe Mode) ---
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'CAPITALIQ | Strategic Report', 0, 1, 'C')
            self.line(10, 20, 200, 20)
            self.ln(10)
        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(0, 117, 221) # FreshBooks Blue
            self.set_text_color(255, 255, 255)
            self.cell(0, 6, title, 0, 1, 'L', 1)
            self.set_text_color(0, 0, 0)
            self.ln(4)
        def chapter_body(self, body):
            self.set_font('Arial', '', 10)
            self.multi_cell(0, 5, body)
            self.ln()

    def create_pdf_report(portfolio, rejected, budget, wacc):
        pdf = PDFReport()
        pdf.add_page()
        pdf.chapter_title('1. Executive Summary')
        pdf.chapter_body(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\nTotal Budget: INR {budget:,.2f}\nDeployed: INR {portfolio['Investment_Capital'].sum():,.2f}\nNPV: INR {portfolio['Dynamic_NPV'].sum():,.2f}\nWACC: {wacc*100:.1f}%")
        pdf.chapter_title('2. Approved Projects')
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(40, 7, 'Project ID', 1); pdf.cell(50, 7, 'Dept', 1); pdf.cell(50, 7, 'Investment', 1); pdf.cell(30, 7, 'ROI %', 1); pdf.ln()
        pdf.set_font('Arial', '', 9)
        for _, row in portfolio.iterrows():
            pdf.cell(40, 6, str(row['Project_ID']), 1); pdf.cell(50, 6, str(row['Department']), 1); pdf.cell(50, 6, f"{row['Investment_Capital']:,.0f}", 1); pdf.cell(30, 6, f"{row['Pred_ROI']:.1f}", 1); pdf.ln()
        return pdf.output(dest='S').encode('latin-1')
except ImportError:
    PDF_AVAILABLE = False
    def create_pdf_report(*args): return None

# ----------------------------------------------------
# 3. Sidebar
# ----------------------------------------------------
def reset_data_callback():
    st.session_state.clear()
    st.session_state.page_selection = "Home & Data"

def process_data_callback():
    if st.session_state.u_hist and st.session_state.u_prop:
        try:
            df_h = standardize_columns(pd.read_csv(st.session_state.u_hist))
            df_p = standardize_columns(pd.read_csv(st.session_state.u_prop))
            rf, imp = train_models(df_h)
            feats = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
            for f in feats: 
                if f not in df_p.columns: df_p[f] = 0
            df_p["Pred_ROI"] = rf.predict(df_p[feats])
            st.session_state.update({'df_prop': df_p, 'feature_imp': imp, 'page_selection': "Executive Summary"})
        except Exception as e: st.error(f"Error: {e}")

def load_demo_callback():
    df_h, df_p = get_templates()
    rf, imp = train_models(df_h)
    feats = ["Investment_Capital", "Duration_Months", "Risk_Score", "Strategic_Alignment", "Market_Trend_Index"]
    df_p["Pred_ROI"] = rf.predict(df_p[feats])
    st.session_state.update({'df_prop': df_p, 'feature_imp': imp, 'u_hist': "Demo", 'u_prop': "Demo", 'page_selection': "Executive Summary"})

with st.sidebar:
    # FRESHBOOKS BRAND HEADER
    st.markdown("""
        <div style="background-color: #0075DD; padding: 20px; border-radius: 8px; margin-bottom: 20px; text-align: center;">
            <h2 style="color: white !important; margin: 0; font-size: 24px;">CapitalIQ</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 12px;">Enterprise Edition</p>
        </div>
    """, unsafe_allow_html=True)
    
    if "page_selection" not in st.session_state: st.session_state.page_selection = "Home & Data"
    
    pages = ["Home & Data", "Executive Summary", "AI Insights", "Efficient Frontier", "Optimization Report", "Strategic 3D Map", "Scenario Manager", "AI Deal Memos"]
    selected_page = st.radio("MENU", pages, key="page_selection", label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ⚙️ Constraints")
    budget_input = st.number_input("Budget (INR)", value=15000000.0, step=500000.0)
    wacc_input = st.slider("WACC (%)", 5.0, 20.0, 10.0) / 100
    min_dept_spend = st.slider("Min Dept. Alloc (%)", 0, 30, 0) / 100
    
    st.markdown("### 📈 Market")
    max_risk = st.slider("Max Risk", 1.0, 10.0, 6.5)
    market_shock = st.slider("Market Scenario", -0.20, 0.20, 0.0, 0.01, format="%+.0f%%")
    
    st.markdown("---")
    st.button("Reset Data", on_click=reset_data_callback, use_container_width=True)

# ----------------------------------------------------
# 4. Main Content
# ----------------------------------------------------

if selected_page == "Home & Data":
    st.title("Welcome back, User")
    st.markdown("### Let's optimize your portfolio today.")
    
    if 'df_prop' in st.session_state:
        st.success("Dataset Loaded & Models Ready")
        c1, c2 = st.columns(2)
        c1.button("Go to Dashboard", type="primary", on_click=lambda: st.session_state.update({'page_selection': 'Executive Summary'}))
        c2.button("Unload Data", on_click=reset_data_callback)
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.info("Upload your historical and proposal data to begin.")
            st.file_uploader("1. Historical Data", type="csv", key="u_hist", on_change=process_data_callback)
            st.file_uploader("2. Proposal Data", type="csv", key="u_prop", on_change=process_data_callback)
            st.button("Load Demo Data", type="primary", on_click=load_demo_callback)
        with c2:
            st.markdown("#### Templates")
            h_t, p_t = get_templates()
            st.download_button("Train Template", h_t.to_csv(index=False), "train.csv")
            st.download_button("Predict Template", p_t.to_csv(index=False), "predict.csv")

if 'df_prop' in st.session_state and selected_page != "Home & Data":
    df = st.session_state['df_prop'].copy()
    df["Pred_ROI"] *= (1 + market_shock)
    df["Dynamic_NPV"] = df.apply(lambda r: calculate_dynamic_npv(r, wacc_input), axis=1)
    df["Payback"] = df.apply(calculate_payback, axis=1)
    df = run_advanced_optimization(df, budget_input, min_dept_spend)
    portfolio = df[df["Selected"] == 1]
    rejected = df[df["Selected"] == 0]

    if selected_page == "Executive Summary":
        st.title("Executive Dashboard")
        
        # FRESHBOOKS STYLE METRICS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Projects Funded", f"{len(portfolio)}")
        m2.metric("Capital Deployed", f"₹{portfolio['Investment_Capital'].sum()/1e6:.1f}M", f"{portfolio['Investment_Capital'].sum()/budget_input*100:.1f}% Util")
        m3.metric("Projected NPV", f"₹{portfolio['Dynamic_NPV'].sum()/1e6:.1f}M")
        m4.metric("Avg Payback", f"{portfolio['Payback'].mean():.1f} Yrs")
        
        st.markdown("---")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Budget Allocation")
            if not portfolio.empty:
                fig = px.bar(portfolio, x="Department", y="Investment_Capital", color="Pred_ROI", 
                             color_continuous_scale=["#bbf7d0", "#22c55e", "#15803d"]) # Fresh greens
                st.plotly_chart(fresh_chart(fig), use_container_width=True)
                render_analysis("Budget is allocated to high-ROI departments first. Green intensity indicates better returns.")
        with c2:
            st.subheader("Reports")
            if PDF_AVAILABLE:
                pdf = create_pdf_report(portfolio, rejected, budget_input, wacc_input)
                st.download_button("📄 Download PDF Report", pdf, "report.pdf", "application/pdf", use_container_width=True)
            else:
                st.warning("Install fpdf for PDF reports")

    elif selected_page == "AI Insights":
        st.title("AI & Model Analytics")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Sensitivity Bridge")
            if not portfolio.empty:
                base = portfolio['Dynamic_NPV'].sum()
                fig = go.Figure(go.Waterfall(
                    measure=["relative", "relative", "total"],
                    x=["Base", "Risk Shock", "Final"],
                    y=[base, -(base*0.1), base*0.9],
                    connector={"line":{"color":"#E1E7EE"}},
                    decreasing={"marker":{"color":"#ef4444"}},
                    totals={"marker":{"color":"#0075DD"}}
                ))
                st.plotly_chart(fresh_chart(fig), use_container_width=True)
                render_analysis("Visualizes how market risks impact your bottom line.")
        with c2:
            st.markdown("##### Key Drivers")
            imp = st.session_state['feature_imp']
            fig = px.bar(imp, x="Importance", y="Feature", orientation='h', color_discrete_sequence=["#0075DD"])
            st.plotly_chart(fresh_chart(fig), use_container_width=True)

    elif selected_page == "Efficient Frontier":
        st.title("Risk vs. Return")
        sim_df = pd.DataFrame([{"Risk": np.random.uniform(1,10), "Return": np.random.uniform(5,25)} for _ in range(200)])
        fig = px.scatter(sim_df, x="Risk", y="Return", color="Return", color_continuous_scale="Blues")
        if not portfolio.empty:
            fig.add_trace(go.Scatter(x=[portfolio["Risk_Score"].mean()], y=[portfolio["Pred_ROI"].mean()], 
                                     marker=dict(color='#22C55E', size=15), name="You"))
        st.plotly_chart(fresh_chart(fig), use_container_width=True)
        render_analysis("Your portfolio (Green Dot) should aim for the top-left quadrant.")

    elif selected_page == "AI Deal Memos":
        st.title("Deal Memos")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Approved")
            for _, r in portfolio.head(3).iterrows():
                with st.expander(f"{r['Project_ID']} - {r['Department']}", expanded=True):
                    st.success(f"**Approved**\n\nROI: {r['Pred_ROI']:.1f}% | NPV: ₹{r['Dynamic_NPV']/1e5:.1f}L")
        with c2:
            st.subheader("❌ Deferred")
            for _, r in rejected.head(3).iterrows():
                with st.expander(f"{r['Project_ID']} - {r['Department']}"):
                    st.error(f"**Deferred**\n\nRisk: {r['Risk_Score']} | Efficiency: {r['Pred_ROI']/r['Risk_Score']:.2f}")
