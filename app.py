import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog

# ----------------------------------------------------
# 1. Page Configuration & Theme
# ----------------------------------------------------
st.set_page_config(
    page_title="CAPITALIQ-AI™ | Universal Edition",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# Glassmorphism & Dark Theme
st.markdown("""
    <style>
    .stApp {
        background-image: linear_gradient(rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.95)), 
                          url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop');
        background-size: cover;
        background-attachment: fixed;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, li, span, label, .stDataFrame, .stRadio label {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
    div[data-testid="stMetric"], div[data-testid="stExpander"] {
        background-color: rgba(30, 41, 59, 0.7); 
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a; 
        border-right: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Sidebar Controls
# ----------------------------------------------------
with st.sidebar:
    st.title("💎 CAPITALIQ-AI™")
    st.caption("Universal Capital Allocator")
    st.markdown("---")
    
    st.subheader("📍 Navigation")
    selected_page = st.radio(
        "Go to:", 
        ["🚀 Dashboard", "📊 Data Analysis", "⚡ Optimization Engine", "📤 Export Report"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.subheader("⚙️ Constraints")
    budget_input = st.number_input("💰 Total Budget Available", value=10000000.0, step=500000.0)
    
    st.info("Client Mode: Active")

# ----------------------------------------------------
# 3. Main Page: Universal Data Adapter
# ----------------------------------------------------
st.title("📊 Executive Capital Command Center")

# --- STEP 1: UPLOAD ---
with st.expander("📂 Step 1: Upload Client Dataset", expanded=True):
    uploaded_file = st.file_uploader("Upload any Project Proposal CSV", type=["csv"])
    
    if uploaded_file is None:
        st.info("👋 Waiting for data... Upload a CSV to begin.")
        st.stop()
    
    # Load raw data
    df_raw = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df_raw.head(3), use_container_width=True)

# --- STEP 2: MAPPING (The Magic Part) ---
st.markdown("### 🔌 Step 2: Configure Data Adapter")
st.caption("Map the columns from your uploaded file to the system requirements.")

cols = df_raw.columns.tolist()

c1, c2, c3, c4 = st.columns(4)
with c1:
    # Required: Project Name
    col_name = st.selectbox("🏷️ Project Name Column", options=cols, index=0)
with c2:
    # Required: Cost
    col_cost = st.selectbox("💰 Investment/Cost Column", options=cols, index=1 if len(cols)>1 else 0)
with c3:
    # Required: Value (NPV, ROI, or Profit)
    col_value = st.selectbox("📈 Return/Value Column (NPV/ROI)", options=cols, index=2 if len(cols)>2 else 0)
with c4:
    # Optional: Category
    col_cat = st.selectbox("📂 Category/Dept (Optional)", options=["None"] + cols, index=0)

# ----------------------------------------------------
# 4. Standardization Engine
# ----------------------------------------------------
# Create a standardized dataframe for internal processing
df = df_raw.copy()
df = df.rename(columns={
    col_name: "Project_Name",
    col_cost: "Cost",
    col_value: "Value"
})

# Handle Category
if col_cat != "None":
    df["Category"] = df_raw[col_cat]
else:
    df["Category"] = "General"

# Handle Missing/Bad Data
df["Cost"] = pd.to_numeric(df["Cost"], errors='coerce').fillna(0)
df["Value"] = pd.to_numeric(df["Value"], errors='coerce').fillna(0)

# Calculate ROI internally for comparison
df["Calculated_ROI"] = (df["Value"] / df["Cost"]).replace([np.inf, -np.inf], 0).fillna(0) * 100

# ----------------------------------------------------
# 5. Optimization Logic (Linear Programming)
# ----------------------------------------------------
# Objective: Maximize Value (NPV/Return)
# Constraint: Sum of Cost <= Budget
c = -df["Value"].values  # Negative because linprog minimizes
A = [df["Cost"].values]
b = [budget_input]
bounds = [(0, 1) for _ in range(len(df))] # 0 or 1 (Binary selection)

# Run Optimization
res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
df["Selected"] = res.x.round(0) if res.success else 0
portfolio = df[df["Selected"] == 1]

# ----------------------------------------------------
# 6. Dynamic Page Views
# ----------------------------------------------------
def dark_chart(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
    return fig

# --- VIEW: DASHBOARD ---
if selected_page == "🚀 Dashboard":
    st.subheader("📌 Executive Summary")
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Projects Approved", f"{len(portfolio)}", f"out of {len(df)}")
    m2.metric("Total Investment", f"₹{portfolio['Cost'].sum():,.0f}", f"Budget: ₹{budget_input:,.0f}")
    m3.metric("Total Value Created", f"₹{portfolio['Value'].sum():,.0f}", "Optimized Return")
    
    st.markdown("---")
    
    # Charts
    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### 🍩 Investment by Category")
        fig_pie = px.pie(portfolio, values="Cost", names="Category", hole=0.5, color_discrete_sequence=px.colors.sequential.Tealgrn_r)
        st.plotly_chart(dark_chart(fig_pie), use_container_width=True)
    
    with cr:
        st.markdown("#### 🚀 Value vs Cost (Bubble Chart)")
        fig_bub = px.scatter(
            df, x="Cost", y="Value", size="Cost", color="Selected",
            color_continuous_scale=["#ef5350", "#66bb6a"], # Red for rejected, Green for selected
            hover_data=["Project_Name"],
            labels={"Selected": "Status (1=Approved)"}
        )
        st.plotly_chart(dark_chart(fig_bub), use_container_width=True)

# --- VIEW: DATA ANALYSIS ---
elif selected_page == "📊 Data Analysis":
    st.subheader("🔍 Dataset Health & distribution")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Cost Distribution")
        fig_hist = px.histogram(df, x="Cost", nbins=20, color_discrete_sequence=["#26a69a"])
        st.plotly_chart(dark_chart(fig_hist), use_container_width=True)
    
    with col2:
        st.markdown("#### 📉 ROI Analysis")
        fig_box = px.box(df, x="Category", y="Calculated_ROI", color_discrete_sequence=["#ffa726"])
        st.plotly_chart(dark_chart(fig_box), use_container_width=True)

# --- VIEW: OPTIMIZATION ENGINE ---
elif selected_page == "⚡ Optimization Engine":
    st.subheader("⚡ Efficient Frontier & Selection")
    
    # Efficient Frontier Simulation
    if st.button("🔄 Run Monte Carlo Simulation"):
        with st.spinner("Simulating 1,000 Portfolio Scenarios..."):
            results = []
            for _ in range(1000):
                # Random selection logic
                mask = np.random.rand(len(df)) < 0.5
                sample = df[mask]
                if sample["Cost"].sum() <= budget_input:
                    results.append({
                        "Cost": sample["Cost"].sum(),
                        "Value": sample["Value"].sum(),
                        "Count": len(sample)
                    })
            sim_df = pd.DataFrame(results)
            
            fig_ef = px.scatter(
                sim_df, x="Cost", y="Value", color="Count",
                title="Portfolio Efficiency Frontier",
                labels={"Cost": "Total Portfolio Cost", "Value": "Total Portfolio Value"}
            )
            # Add Optimized Point
            fig_ef.add_trace(go.Scatter(
                x=[portfolio["Cost"].sum()], y=[portfolio["Value"].sum()],
                mode='markers', marker=dict(color='red', size=15, symbol='star'),
                name="AI Optimized Portfolio"
            ))
            st.plotly_chart(dark_chart(fig_ef), use_container_width=True)

# --- VIEW: EXPORT ---
elif selected_page == "📤 Export Report":
    st.subheader("📋 Final Decision Report")
    
    # Clean output dataframe
    output_df = portfolio[["Project_Name", "Category", "Cost", "Value", "Calculated_ROI"]].copy()
    output_df = output_df.sort_values("Value", ascending=False)
    
    st.dataframe(
        output_df.style.format({"Cost": "₹{:,.0f}", "Value": "₹{:,.0f}", "Calculated_ROI": "{:.1f}%"}),
        use_container_width=True,
        height=500
    )
    
    csv = output_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Approved Portfolio (CSV)",
        data=csv,
        file_name="Optimized_Capital_Allocation.csv",
        mime="text/csv"
    )
