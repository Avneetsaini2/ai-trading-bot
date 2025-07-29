import streamlit as st
import pandas as pd
from datetime import datetime
from live_signals import LiveSignalGenerator
import time

st.set_page_config(
    page_title="Smart Trading Signals",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Smart Trading Robot - Live Signals")

# Initialize signal generator
@st.cache_resource
def get_signal_generator():
    return LiveSignalGenerator()

generator = get_signal_generator()

# Controls
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📡 Live Trading Signals")

with col2:
    if st.button("🔄 Refresh Signals"):
        st.cache_resource.clear()

# Generate current signals
try:
    with st.spinner("🧠 AI Robot analyzing market..."):
        signals = generator.generate_live_signals()
    
    if signals:
        # Convert to DataFrame for display
        df = pd.DataFrame(signals)
        
        # Style the dataframe
        def style_decision(val):
            if val == 'BUY':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'SELL':
                return 'background-color: #f8d7da; color: #721c24'
            else:
                return 'background-color: #fff3cd; color: #856404'
        
        # Display signals table
        styled_df = df[['symbol', 'price', 'change', 'decision', 'confidence', 'reason']].style.applymap(
            style_decision, subset=['decision']
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Summary metrics
        st.markdown("### 📊 Signal Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        buy_count = len(df[df['decision'] == 'BUY'])
        sell_count = len(df[df['decision'] == 'SELL'])
        hold_count = len(df[df['decision'] == 'HOLD'])
        avg_confidence = df['confidence'].mean()
        
        col1.metric("🟢 BUY Signals", buy_count)
        col2.metric("🔴 SELL Signals", sell_count)
        col3.metric("🟡 HOLD Signals", hold_count)
        col4.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%")
        
        # High confidence alerts
        high_confidence = df[df['confidence'] >= 65]
        if not high_confidence.empty:
            st.markdown("### 🚨 High Confidence Alerts")
            for _, signal in high_confidence.iterrows():
                alert_type = "success" if signal['decision'] == 'BUY' else "error" if signal['decision'] == 'SELL' else "warning"
                st.alert(f"**{signal['decision']}** {signal['symbol']} at ₹{signal['price']:.2f} - {signal['confidence']:.1f}% confidence", icon="🎯")
    
    else:
        st.error("❌ No signals generated. Check your data connection.")
        
except Exception as e:
    st.error(f"❌ Error generating signals: {e}")

# Auto refresh
st.markdown("---")
st.markdown("🔄 **Auto-refresh every 2 minutes** - Dashboard updates automatically")

# Auto refresh every 2 minutes
time.sleep(120)
st.rerun()
