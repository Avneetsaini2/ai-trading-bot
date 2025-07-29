import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from data_engine import FastDataEngine
import time

# Configure page
st.set_page_config(
    page_title="Fast NSE Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize data engine
@st.cache_resource
def get_data_engine():
    return FastDataEngine()

engine = get_data_engine()

# Main dashboard
st.title("🚀 Fast NSE Trading Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Controls")
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
selected_symbol = st.sidebar.selectbox("Select Symbol", list(engine.symbols.keys()))

# Live data section
col1, col2, col3, col4 = st.columns(4)

# Get live data
live_data = engine.get_live_data()

if not live_data.empty:
    # Display metrics
    for i, (_, row) in enumerate(live_data.iterrows()):
        col = [col1, col2, col3, col4][i % 4]
        
        color = "normal"
        if row['change'] > 0:
            color = "normal"
            delta_color = "normal"
        else:
            color = "normal" 
            delta_color = "inverse"
        
        col.metric(
            label=row['symbol'],
            value=f"₹{row['price']:.2f}",
            delta=f"{row['change']:.2f}%"
        )

# Chart section
st.markdown("## 📊 Price Chart")

try:
    # Load historical data for selected symbol
    hist_file = f"historical_data/{selected_symbol}.csv"
    df = pd.read_csv(hist_file, index_col=0, parse_dates=True)
    
    # Create candlestick chart
    fig = go.Figure(data=go.Candlestick(
        x=df.index[-100:],  # Last 100 days
        open=df['Open'][-100:],
        high=df['High'][-100:],
        low=df['Low'][-100:],
        close=df['Close'][-100:],
        name=selected_symbol
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df.index[-100:],
        y=df['SMA_20'][-100:],
        mode='lines',
        name='SMA 20',
        line=dict(color='orange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index[-100:],
        y=df['SMA_50'][-100:],
        mode='lines',
        name='SMA 50',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=f"{selected_symbol} - Last 100 Days",
        yaxis_title="Price (₹)",
        xaxis_title="Date",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    
    with col2:
        st.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
        
    with col3:
        price = df['Close'].iloc[-1]
        bb_upper = df['BB_Upper'].iloc[-1]
        bb_lower = df['BB_Lower'].iloc[-1]
        bb_position = ((price - bb_lower) / (bb_upper - bb_lower)) * 100
        st.metric("BB Position", f"{bb_position:.1f}%")
        
except Exception as e:
    st.error(f"Error loading data for {selected_symbol}: {e}")

# Auto refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()
