import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Friday: Capital Allocation Advisor", layout="wide", page_icon="📈")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 20px; text-align: center;}
    .stAlert {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# --- 1. DATA FETCHING & PROCESSING ---
@st.cache_data
def fetch_data(tickers, start_date, end_date):
    """Fetches historical stock data."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """Calculates SMA, Volatility, and Returns for the XAI model."""
    indicators = pd.DataFrame(index=df.index)
    indicators['Close'] = df
    indicators['SMA_50'] = df.rolling(window=50).mean()
    indicators['SMA_200'] = df.rolling(window=200).mean()
    indicators['Daily_Return'] = df.pct_change()
    indicators['Volatility'] = df.rolling(window=20).std()
    # Momentum: Price vs 10 days ago
    indicators['Momentum'] = df / df.shift(10) - 1
    indicators.dropna(inplace=True)
    return indicators

def analyze_sentiment_live(ticker):
    """
    Fetches REAL news from Yahoo Finance and scores sentiment.
    """
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        
        headlines = []
        sentiment_scores = []
        
        if not news_list:
            return 0, ["No recent news found via API."]

        # Analyze top 5 news items
        for item in news_list[:5]:
            title = item.get('title', '')
            headlines.append(title)
            blob = TextBlob(title)
            sentiment_scores.append(blob.sentiment.polarity)
        
        avg_score = np.mean(sentiment_scores) if sentiment_scores else 0
        return avg_score, headlines
        
    except Exception as e:
        return 0, [f"Could not fetch news: {str(e)}"]

# --- 2. OPTIMIZATION ENGINE (WITH SCENARIO LOGIC) ---
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def optimize_portfolio(mean_returns, cov_matrix, max_weight_per_asset):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, 0.04) # Assuming 4% risk-free rate
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, max_weight_per_asset) for asset in range(num_assets))
    
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# --- 3. EXPLAINABLE AI (XAI) ENGINE ---
def train_and_explain_ai(ticker_data):
    """Trains a Random Forest to explain what drives the stock's movement."""
    df = calculate_technical_indicators(ticker_data)
    
    # Target: Predict next day's return
    df['Target'] = df['Daily_Return'].shift(-1)
    df.dropna(inplace=True)
    
    features = ['SMA_50', 'SMA_200', 'Volatility', 'Momentum']
    X = df[features]
    y = df['Target']
    
    if len(X) < 10: # Not enough data
        return None, None, None
        
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importances = model.feature_importances_
    explanation = pd.DataFrame({'Feature': features, 'Importance': importances})
    explanation = explanation.sort_values(by='Importance', ascending=False)
    
    return model, explanation, df.iloc[-1][features]

# --- 4. MAIN APPLICATION UI ---
st.title("🚀 AI-Driven Capital Allocation Advisor")
st.markdown("**User:** KD | **Client:** Grant Thornton Bharat LLP")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Input
    default_tickers = "AAPL, MSFT, GOOGL, TSLA, JPM"
    tickers_input = st.text_input("Tickers (Comma Separated)", default_tickers)
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    # Constraint
    max_weight = st.slider("Max Allocation per Asset (%)", 0.1, 1.0, 0.4, 
                           help="Compliance Rule: No single asset can exceed this %")
    
    # Scenario
    st.markdown("---")
    st.subheader("⚡ Stress Testing (What-If)")
    scenario_shock = st.select_slider(
        "Simulate Market Condition:",
        options=[-20, -10, 0, 10, 20],
        value=0,
        format_func=lambda x: f"{x}% Returns (Crash)" if x < 0 else (f"+{x}% Returns (Boom)" if x > 0 else "Neutral")
    )
    
    run_btn = st.button("Run AI Analysis", type="primary")

if run_btn:
    start = datetime.date(2023, 1, 1)
    end = datetime.date.today()
    
    with st.spinner('Fetching market data, training AI, and reading news...'):
        raw_data = fetch_data(tickers, start, end)
        
        if raw_data.empty:
            st.error("No data found. Please check ticker symbols.")
        else:
            # TABS
            tab1, tab2, tab3 = st.tabs(["📊 Portfolio Optimization", "🧠 XAI Logic", "📰 Live Sentiment"])
            
            # --- TAB 1: OPTIMIZATION ---
            with tab1:
                st.subheader("Optimal Capital Allocation")
                
                # Calculate returns
                returns = raw_data.pct_change()
                mean_returns = returns.mean()
                cov_matrix = returns.cov()
                
                # APPLY SCENARIO SHOCK
                # If scenario is -20%, we reduce expected annual returns by 20%
                # Since mean_returns is daily, we adjust roughly:
                if scenario_shock != 0:
                    shock_factor = scenario_shock / 100.0
                    # Adjusting mean expectations, not historical data
                    mean_returns = mean_returns + (shock_factor / 252) 
                    st.warning(f"⚠️ Simulation Active: Expected returns adjusted by {scenario_shock}%")

                # Optimize
                opt_results = optimize_portfolio(mean_returns, cov_matrix, max_weight)
                weights = opt_results.x
                
                # Display Results
                allocation_df = pd.DataFrame({'Ticker': tickers, 'Weight': weights})
                allocation_df['Weight'] = allocation_df['Weight'].apply(lambda x: round(x, 4))
                
                # Filter out near-zero weights for cleaner chart
                active_allocation = allocation_df[allocation_df['Weight'] > 0.001]
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(allocation_df.style.format({"Weight": "{:.2%}"}), use_container_width=True)
                    st.success(f"Objective: Maximize Sharpe Ratio (Risk-Adjusted Return)")
                with col2:
                    fig = px.pie(active_allocation, values='Weight', names='Ticker', 
                                 title='Recommended Portfolio', hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)

            # --- TAB 2: XAI ---
            with tab2:
                st.subheader("Explainable AI (Black Box Revealed)")
                st.write("Why is the model behaving this way? We use Random Forest Feature Importance.")
                
                selected_ticker = st.selectbox("Select Asset to Analyze", tickers)
                
                # Train small model for explanation
                if selected_ticker in raw_data.columns:
                    model, explanation, latest = train_and_explain_ai(raw_data[selected_ticker])
                    
                    if model:
                        c1, c2 = st.columns(2)
                        with c1:
                            fig_feat = px.bar(explanation, x='Importance', y='Feature', orientation='h',
                                              title=f"Drivers for {selected_ticker}")
                            st.plotly_chart(fig_feat, use_container_width=True)
                        with c2:
                            st.write(f"**Latest Indicators for {selected_ticker}:**")
                            st.json(latest.to_dict())
                            st.caption("The AI looks at these values to determine stability vs growth potential.")
                    else:
                        st.warning("Not enough data to train XAI model.")

            # --- TAB 3: SENTIMENT ---
            with tab3:
                st.subheader("Real-Time News Analysis")
                st.write("Fetching latest headlines from Yahoo Finance...")
                
                news_cols = st.columns(len(tickers))
                
                for i, ticker in enumerate(tickers):
                    score, headlines = analyze_sentiment_live(ticker)
                    
                    with news_cols[i % len(news_cols)]: # Wrap around columns if many tickers
                        st.markdown(f"#### {ticker}")
                        
                        # Sentiment Badge
                        if score > 0.1:
                            st.markdown("🟢 **Bullish**")
                        elif score < -0.1:
                            st.markdown("🔴 **Bearish**")
                        else:
                            st.markdown("⚪ **Neutral**")
                            
                        st.metric("Sentiment Score", f"{score:.2f}")
                        
                        with st.expander("Show Headlines"):
                            for h in headlines:
                                st.markdown(f"- {h}")

else:
    st.info("👈 Please enter tickers and click 'Run AI Analysis' to start.")
