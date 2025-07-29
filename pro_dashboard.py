import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os

# Handle imports gracefully for cloud deployment
try:
    from cloud_trader import CloudTradingBot
    CLOUD_TRADER_AVAILABLE = True
except ImportError:
    CLOUD_TRADER_AVAILABLE = False
    st.warning("⚠️ Cloud trader module not available - running in demo mode")

# If cloud trader isn't available, create a mock class
if not CLOUD_TRADER_AVAILABLE:
    class MockTradingBot:
        def __init__(self):
            self.portfolio = {
                'cash': 95000,
                'positions': {
                    'RELIANCE': {'quantity': 10, 'avg_price': 2800, 'current_price': 2847},
                    'TCS': {'quantity': 5, 'avg_price': 4100, 'current_price': 4125},
                },
                'trade_history': [
                    {
                        'timestamp': '2025-01-15T10:30:00',
                        'symbol': 'RELIANCE',
                        'action': 'BUY',
                        'quantity': 10,
                        'price': 2800,
                        'confidence': 72
                    }
                ],
                'performance': {
                    'total_trades': 8,
                    'winning_trades': 6,
                    'total_pnl': 1240
                }
            }
        
        def calculate_total_portfolio_value(self):
            total = self.portfolio['cash']
            for pos in self.portfolio['positions'].values():
                total += pos['quantity'] * pos['current_price']
            return total
        
        def run_trading_session(self):
            st.success("🎯 Mock trading session completed! (Cloud trader not available)")


st.set_page_config(
    page_title="Pro Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.profit { color: #00d4aa; }
.loss { color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# Initialize bot
@st.cache_resource
def get_trading_bot():
    return CloudTradingBot()

bot = get_trading_bot()

# Header
st.title("📊 Professional Trading Dashboard")
st.markdown("*Your AI-Powered Trading Command Center*")

# Load portfolio data
portfolio = bot.portfolio

# Sidebar
st.sidebar.header("🎛️ Control Panel")

# Portfolio overview
total_value = bot.calculate_total_portfolio_value()
initial_value = 100000  # Starting amount
total_pnl = total_value - initial_value
pnl_percent = (total_pnl / initial_value) * 100

st.sidebar.markdown("### 💰 Portfolio Summary")
st.sidebar.metric("Total Value", f"₹{total_value:,.0f}", f"{pnl_percent:+.2f}%")
st.sidebar.metric("Cash Available", f"₹{portfolio['cash']:,.0f}")
st.sidebar.metric("Active Positions", len(portfolio['positions']))

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Portfolio Value", f"₹{total_value:,.0f}", f"₹{total_pnl:+,.0f}")

with col2:
    total_trades = portfolio['performance']['total_trades']
    st.metric("Total Trades", total_trades)

with col3:
    if total_trades > 0:
        win_rate = (portfolio['performance']['winning_trades'] / total_trades) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
    else:
        st.metric("Win Rate", "0%")

with col4:
    st.metric("Total P&L", f"₹{portfolio['performance']['total_pnl']:+,.0f}")

# Portfolio composition chart
if portfolio['positions']:
    st.markdown("## 📈 Portfolio Composition")
    
    position_data = []
    for symbol, pos in portfolio['positions'].items():
        position_value = pos['quantity'] * pos['current_price']
        position_data.append({
            'Symbol': symbol,
            'Quantity': pos['quantity'],
            'Avg Price': pos['avg_price'],
            'Current Price': pos['current_price'],
            'Value': position_value,
            'P&L': (pos['current_price'] - pos['avg_price']) * pos['quantity'],
            'P&L%': ((pos['current_price'] - pos['avg_price']) / pos['avg_price']) * 100
        })
    
    df_positions = pd.DataFrame(position_data)
    
    # Portfolio pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(df_positions, values='Value', names='Symbol', 
                        title="Portfolio Allocation")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # P&L bar chart
        fig_bar = px.bar(df_positions, x='Symbol', y='P&L', 
                        color='P&L', title="Position P&L",
                        color_continuous_scale=['red', 'white', 'green'])
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Positions table
    st.markdown("### 📋 Current Positions")
    
    # Style the dataframe
    def color_pnl(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        return f'color: {color}'
    
    styled_df = df_positions.style.applymap(color_pnl, subset=['P&L', 'P&L%'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

# Trading history
if portfolio['trade_history']:
    st.markdown("## 📜 Trading History")
    
    # Convert trade history to DataFrame
    df_trades = pd.DataFrame(portfolio['trade_history'])
    df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
    
    # Recent trades
    recent_trades = df_trades.tail(10).copy()
    recent_trades['timestamp'] = recent_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(recent_trades[['timestamp', 'symbol', 'action', 'quantity', 'price', 'confidence']], 
                use_container_width=True, hide_index=True)
    
    # Performance over time
    if len(df_trades) > 1:
        st.markdown("### 📊 Performance Over Time")
        
        # Calculate cumulative P&L
        df_trades['cumulative_pnl'] = 0
        running_pnl = 0
        
        for i, row in df_trades.iterrows():
            if 'pnl' in row and pd.notna(row['pnl']):
                running_pnl += row['pnl']
            df_trades.at[i, 'cumulative_pnl'] = running_pnl
        
        # Plot cumulative P&L
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=df_trades['timestamp'],
            y=df_trades['cumulative_pnl'],
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='green' if running_pnl > 0 else 'red')
        ))
        
        fig_pnl.update_layout(
            title="Cumulative P&L Over Time",
            xaxis_title="Date",
            yaxis_title="P&L (₹)",
            height=400
        )
        
        st.plotly_chart(fig_pnl, use_container_width=True)

# Live trading controls
st.markdown("## ⚡ Live Trading Controls")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Run Trading Session"):
        with st.spinner("🤖 AI analyzing market..."):
            bot.run_trading_session()
        st.success("Trading session completed!")
        st.rerun()

with col2:
    if st.button("📊 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()

with col3:
    if st.button("💾 Export Portfolio"):
        # Create downloadable portfolio report
        report = {
            'portfolio_value': total_value,
            'cash': portfolio['cash'],
            'positions': portfolio['positions'],
            'performance': portfolio['performance'],
            'generated_at': datetime.now().isoformat()
        }
        
        st.download_button(
            label="Download Portfolio Report",
            data=json.dumps(report, indent=2),
            file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

# Risk metrics
st.markdown("## ⚠️ Risk Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    # Portfolio concentration
    if portfolio['positions']:
        max_position = max([pos['quantity'] * pos['current_price'] for pos in portfolio['positions'].values()])
        concentration = (max_position / total_value) * 100
        st.metric("Max Position %", f"{concentration:.1f}%")
    else:
        st.metric("Max Position %", "0%")

with col2:
    # Cash ratio
    cash_ratio = (portfolio['cash'] / total_value) * 100
    st.metric("Cash Ratio", f"{cash_ratio:.1f}%")

with col3:
    # Drawdown (simplified)
    if portfolio['performance']['total_pnl'] < 0:
        drawdown = abs(portfolio['performance']['total_pnl'] / initial_value) * 100
        st.metric("Drawdown", f"{drawdown:.1f}%")
    else:
        st.metric("Drawdown", "0%")

# Auto-refresh
st.markdown("---")
if st.checkbox("🔄 Auto-refresh (every 5 minutes)", value=False):
    import time
    time.sleep(300)  # 5 minutes
    st.rerun()
