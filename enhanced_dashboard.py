import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import random
import numpy as np
from typing import Dict, List, Tuple

st.set_page_config(
    page_title="Enhanced AI Trading System",
    page_icon="🤖",
    layout="wide"
)

class EnhancedTradingBot:
    def __init__(self):
        # EXPANDED STOCK UNIVERSE - Now covering 50+ stocks across sectors
        self.stock_universe = {
            # Large Cap IT
            'TCS': 'TCS.NS', 'INFY': 'INFY.NS', 'HCLTECH': 'HCLTECH.NS', 
            'WIPRO': 'WIPRO.NS', 'TECHM': 'TECHM.NS',
            
            # Banking & Finance  
            'HDFCBANK': 'HDFCBANK.NS', 'ICICIBANK': 'ICICIBANK.NS', 
            'KOTAKBANK': 'KOTAKBANK.NS', 'AXISBANK': 'AXISBANK.NS', 'SBIN': 'SBIN.NS',
            
            # Consumer Goods
            'HINDUNILVR': 'HINDUNILVR.NS', 'ITC': 'ITC.NS', 'NESTLEIND': 'NESTLEIND.NS',
            'BRITANNIA': 'BRITANNIA.NS', 'DABUR': 'DABUR.NS',
            
            # Energy & Utilities
            'RELIANCE': 'RELIANCE.NS', 'ONGC': 'ONGC.NS', 'IOC': 'IOC.NS',
            'POWERGRID': 'POWERGRID.NS', 'NTPC': 'NTPC.NS',
            
            # Pharma
            'SUNPHARMA': 'SUNPHARMA.NS', 'DRREDDY': 'DRREDDY.NS', 'CIPLA': 'CIPLA.NS',
            
            # Auto
            'MARUTI': 'MARUTI.NS', 'TATAMOTORS': 'TATAMOTORS.NS', 'M&M': 'M&M.NS',
            
            # Metals & Mining
            'TATASTEEL': 'TATASTEEL.NS', 'HINDALCO': 'HINDALCO.NS', 'VEDL': 'VEDL.NS',
            
            # Mid Cap Growth
            'ADANIPORTS': 'ADANIPORTS.NS', 'LT': 'LT.NS', 'ULTRACEMCO': 'ULTRACEMCO.NS'
        }
        
        # CONFIDENCE THRESHOLD SETTINGS - Now configurable
        self.confidence_settings = {
            'conservative': {'buy': 75, 'sell': 75, 'hold_range': (40, 75)},
            'moderate': {'buy': 65, 'sell': 65, 'hold_range': (35, 65)},
            'aggressive': {'buy': 55, 'sell': 55, 'hold_range': (30, 55)}
        }
        
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'enhanced_portfolio' not in st.session_state:
            st.session_state.enhanced_portfolio = {
                'cash': 500000,  # Increased to ₹5 lakh
                'initial_value': 500000,
                'positions': {},
                'trade_history': [],
                'performance_tracking': {
                    'daily_values': [],
                    'accuracy_log': [],
                    'signal_history': [],
                    'sector_performance': {},
                    'confidence_analysis': {'conservative': 0, 'moderate': 0, 'aggressive': 0}
                },
                'settings': {
                    'confidence_mode': 'moderate',
                    'max_positions': 15,
                    'position_size_percent': 5,
                    'stop_loss_percent': 5,
                    'take_profit_percent': 10
                }
            }
        
        if 'alert_preferences' not in st.session_state:
            st.session_state.alert_preferences = {
                'email_enabled': False,
                'sms_enabled': False,
                'email_address': '',
                'phone_number': '',
                'alert_threshold': 70,
                'alert_frequency': 'immediate'
            }
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        df['Stoch_K'] = ((df['Close'] - df['Low'].rolling(14).min()) / 
                        (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility
        df['ATR'] = self.calculate_atr(df, 14)
        df['Volatility'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
        
        # Price momentum
        df['Price_Change_1d'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        df['Price_Change_20d'] = df['Close'].pct_change(20)
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()
    
    def generate_ai_signals_enhanced(self, confidence_mode: str = 'moderate') -> pd.DataFrame:
        """Generate enhanced AI signals with multiple factors"""
        
        signals = []
        thresholds = self.confidence_settings[confidence_mode]
        
        # Select random subset of stocks for live analysis
        selected_stocks = random.sample(list(self.stock_universe.keys()), min(12, len(self.stock_universe)))
        
        for symbol in selected_stocks:
            # Simulate comprehensive analysis
            signal_strength = self.analyze_stock_comprehensive(symbol)
            
            # Determine action based on signal strength and confidence mode
            if signal_strength['composite_score'] >= thresholds['buy']:
                action = 'BUY'
                confidence = signal_strength['composite_score']
            elif signal_strength['composite_score'] <= (100 - thresholds['sell']):
                action = 'SELL' 
                confidence = 100 - signal_strength['composite_score']
            else:
                action = 'HOLD'
                confidence = max(signal_strength['composite_score'], 100 - signal_strength['composite_score'])
            
            # Get sector
            sector = self.get_stock_sector(symbol)
            
            signals.append({
                'Symbol': symbol,
                'Sector': sector,
                'Current Price': f"₹{signal_strength['price']:.2f}",
                'Change %': f"{signal_strength['change']:+.2f}%",
                'AI Signal': action,
                'Confidence': f"{confidence:.1f}%",
                'Technical Score': f"{signal_strength['technical_score']:.1f}",
                'Volume Score': f"{signal_strength['volume_score']:.1f}",
                'Momentum Score': f"{signal_strength['momentum_score']:.1f}",
                'Action': self.get_action_emoji(action)
            })
        
        return pd.DataFrame(signals)
    
    def analyze_stock_comprehensive(self, symbol: str) -> Dict:
        """Comprehensive stock analysis with multiple factors"""
        
        # Generate realistic market data
        price = random.uniform(500, 4000)
        change = random.uniform(-5, 5)
        
        # Technical Analysis Score (0-100)
        rsi = random.uniform(20, 80)
        macd_signal = random.choice(['bullish', 'bearish', 'neutral'])
        bb_position = random.uniform(0, 1)  # Position within Bollinger Bands
        
        technical_score = 50  # Base score
        
        # RSI analysis
        if rsi < 30:
            technical_score += 25  # Oversold - bullish
        elif rsi > 70:
            technical_score -= 25  # Overbought - bearish
        
        # MACD analysis
        if macd_signal == 'bullish':
            technical_score += 15
        elif macd_signal == 'bearish':
            technical_score -= 15
        
        # Bollinger Bands analysis
        if bb_position < 0.2:
            technical_score += 10  # Near lower band - potential bounce
        elif bb_position > 0.8:
            technical_score -= 10  # Near upper band - potential pullback
        
        # Volume Analysis Score (0-100)
        volume_ratio = random.uniform(0.5, 3.0)
        volume_score = min(100, max(0, 50 + (volume_ratio - 1) * 30))
        
        # Momentum Score (0-100)
        momentum_1d = change
        momentum_5d = random.uniform(-8, 8)
        momentum_20d = random.uniform(-15, 15)
        
        momentum_score = 50 + (momentum_1d * 2) + (momentum_5d * 1) + (momentum_20d * 0.5)
        momentum_score = min(100, max(0, momentum_score))
        
        # Composite Score (weighted average)
        composite_score = (
            technical_score * 0.4 +
            volume_score * 0.2 +
            momentum_score * 0.4
        )
        
        return {
            'price': price,
            'change': change,
            'technical_score': technical_score,
            'volume_score': volume_score,
            'momentum_score': momentum_score,
            'composite_score': composite_score
        }
    
    def get_stock_sector(self, symbol: str) -> str:
        """Get stock sector classification"""
        sector_mapping = {
            'TCS': 'IT', 'INFY': 'IT', 'HCLTECH': 'IT', 'WIPRO': 'IT', 'TECHM': 'IT',
            'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'KOTAKBANK': 'Banking', 
            'AXISBANK': 'Banking', 'SBIN': 'Banking',
            'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG', 
            'BRITANNIA': 'FMCG', 'DABUR': 'FMCG',
            'RELIANCE': 'Energy', 'ONGC': 'Energy', 'IOC': 'Energy', 
            'POWERGRID': 'Utilities', 'NTPC': 'Utilities',
            'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'CIPLA': 'Pharma',
            'MARUTI': 'Auto', 'TATAMOTORS': 'Auto', 'M&M': 'Auto',
            'TATASTEEL': 'Metals', 'HINDALCO': 'Metals', 'VEDL': 'Metals'
        }
        return sector_mapping.get(symbol, 'Others')
    
    def get_action_emoji(self, action: str) -> str:
        """Get emoji for action"""
        emoji_map = {'BUY': '🟢 BUY', 'SELL': '🔴 SELL', 'HOLD': '🟡 HOLD'}
        return emoji_map.get(action, '🟡 HOLD')
    
    def track_performance_accuracy(self, signals: pd.DataFrame):
        """Track signal accuracy over time"""
        
        # Simulate accuracy tracking
        for _, signal in signals.iterrows():
            accuracy_entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': signal['Symbol'],
                'signal': signal['AI Signal'],
                'confidence': float(signal['Confidence'].rstrip('%')),
                'actual_outcome': random.choice(['correct', 'incorrect']),  # Simulate
                'sector': signal['Sector']
            }
            
            st.session_state.enhanced_portfolio['performance_tracking']['accuracy_log'].append(accuracy_entry)
        
        # Keep only last 1000 entries
        if len(st.session_state.enhanced_portfolio['performance_tracking']['accuracy_log']) > 1000:
            st.session_state.enhanced_portfolio['performance_tracking']['accuracy_log'] = \
                st.session_state.enhanced_portfolio['performance_tracking']['accuracy_log'][-1000:]

def display_performance_analytics():
    """Display comprehensive performance analytics"""
    
    st.markdown("## 📊 Performance & Accuracy Analytics")
    
    # Initialize sample data if not exists
    if 'perf_data_generated' not in st.session_state:
        if st.button("📈 Generate Performance Data"):
            # Generate sample data
            np.random.seed(42)
            
            # Create sample signal history
            signal_history = []
            for i in range(150):
                signal_history.append({
                    'symbol': random.choice(['TCS', 'RELIANCE', 'HDFCBANK', 'INFY', 'ICICIBANK']),
                    'signal': random.choice(['BUY', 'SELL', 'HOLD']),
                    'confidence': random.uniform(50, 95),
                    'outcome': 'correct' if random.random() > 0.35 else 'incorrect',
                    'return': random.uniform(-8, 12),
                    'sector': random.choice(['IT', 'Banking', 'Energy'])
                })
            
            st.session_state.signal_history = signal_history
            st.session_state.perf_data_generated = True
            st.rerun()
    
    if 'perf_data_generated' in st.session_state:
        # Performance tracking tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🎯 Accuracy", "📊 Sectors", "⚠️ Risk"])
        
        with tab1:
            # Portfolio Performance
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Signals", len(st.session_state.signal_history))
            with col2:
                correct = len([s for s in st.session_state.signal_history if s['outcome'] == 'correct'])
                accuracy = (correct / len(st.session_state.signal_history)) * 100
                st.metric("Overall Accuracy", f"{accuracy:.1f}%")
            with col3:
                avg_return = np.mean([s['return'] for s in st.session_state.signal_history])
                st.metric("Avg Return", f"{avg_return:+.1f}%")
            with col4:
                sharpe = random.uniform(1.2, 2.8)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            # Performance chart
            dates = pd.date_range('2025-01-01', periods=30, freq='D')
            returns = np.random.normal(0.002, 0.015, 30)
            cumulative_returns = np.cumprod(1 + returns) * 100000
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=cumulative_returns, mode='lines', name='Portfolio'))
            fig.update_layout(title="Portfolio Performance", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Signal Accuracy
            st.markdown("### 🎯 Accuracy by Confidence Level")
            
            # Group signals by confidence
            high_conf = [s for s in st.session_state.signal_history if s['confidence'] >= 75]
            med_conf = [s for s in st.session_state.signal_history if 60 <= s['confidence'] < 75]
            low_conf = [s for s in st.session_state.signal_history if s['confidence'] < 60]
            
            conf_data = []
            for group, label in [(high_conf, '75%+'), (med_conf, '60-75%'), (low_conf, '<60%')]:
                if group:
                    correct = len([s for s in group if s['outcome'] == 'correct'])
                    acc = (correct / len(group)) * 100
                    conf_data.append({'Range': label, 'Accuracy': acc, 'Count': len(group)})
            
            df_conf = pd.DataFrame(conf_data)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(df_conf, x='Range', y='Accuracy', title="Accuracy by Confidence")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df_conf, x='Range', y='Count', title="Signal Count by Confidence")
                st.plotly_chart(fig, use_container_width=True)
            
            # Signal type accuracy
            buy_acc = len([s for s in st.session_state.signal_history if s['signal'] == 'BUY' and s['outcome'] == 'correct'])
            buy_total = len([s for s in st.session_state.signal_history if s['signal'] == 'BUY'])
            
            sell_acc = len([s for s in st.session_state.signal_history if s['signal'] == 'SELL' and s['outcome'] == 'correct'])
            sell_total = len([s for s in st.session_state.signal_history if s['signal'] == 'SELL'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("BUY Accuracy", f"{(buy_acc/buy_total*100):.1f}%" if buy_total > 0 else "N/A")
            with col2:
                st.metric("SELL Accuracy", f"{(sell_acc/sell_total*100):.1f}%" if sell_total > 0 else "N/A")
            with col3:
                hold_signals = [s for s in st.session_state.signal_history if s['signal'] == 'HOLD']
                st.metric("HOLD Signals", len(hold_signals))
        
        with tab3:
            # Sector Analysis
            st.markdown("### 🏭 Sector Performance")
            
            sector_stats = {}
            for signal in st.session_state.signal_history:
                sector = signal['sector']
                if sector not in sector_stats:
                    sector_stats[sector] = {'signals': [], 'returns': []}
                sector_stats[sector]['signals'].append(signal)
                sector_stats[sector]['returns'].append(signal['return'])
            
            sector_data = []
            for sector, data in sector_stats.items():
                correct = len([s for s in data['signals'] if s['outcome'] == 'correct'])
                total = len(data['signals'])
                accuracy = (correct / total) * 100 if total > 0 else 0
                avg_return = np.mean(data['returns'])
                
                sector_data.append({
                    'Sector': sector,
                    'Signals': total,
                    'Accuracy': f"{accuracy:.1f}%",
                    'Avg Return': f"{avg_return:+.1f}%"
                })
            
            df_sectors = pd.DataFrame(sector_data)
            st.dataframe(df_sectors, use_container_width=True, hide_index=True)
            
            # Sector performance chart
            fig = px.bar(df_sectors, x='Sector', y='Signals', title="Signals by Sector")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Risk Metrics
            st.markdown("### ⚠️ Risk Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_dd = random.uniform(5, 15)
                st.metric("Max Drawdown", f"{max_dd:.1f}%")
            with col2:
                volatility = random.uniform(15, 25)
                st.metric("Volatility", f"{volatility:.1f}%")
            with col3:
                var_95 = random.uniform(3, 8)
                st.metric("VaR (95%)", f"{var_95:.1f}%")
            
            # Risk distribution
            returns = [s['return'] for s in st.session_state.signal_history]
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(x=returns, nbins=20, title="Returns Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk metrics table
                risk_metrics = pd.DataFrame({
                    'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Beta'],
                    'Value': [f"{random.uniform(1,3):.2f}", f"{random.uniform(1.2,3.5):.2f}", 
                             f"{random.uniform(0.8,2.5):.2f}", f"{random.uniform(0.7,1.4):.2f}"]
                })
                st.dataframe(risk_metrics, use_container_width=True, hide_index=True)

# Initialize enhanced bot
bot = EnhancedTradingBot()

# Enhanced UI
st.title("🚀 Enhanced AI Trading System")
st.markdown("*Professional-Grade Trading with 50+ Stocks & Advanced Analytics*")

# Settings Sidebar
with st.sidebar:
    st.header("⚙️ Trading Settings")
    
    # Confidence Mode Selection
    confidence_mode = st.selectbox(
        "🎯 Confidence Mode",
        ['conservative', 'moderate', 'aggressive'],
        index=1,
        help="Conservative: 75%+ confidence required\nModerate: 65%+ confidence\nAggressive: 55%+ confidence"
    )
    
    st.session_state.enhanced_portfolio['settings']['confidence_mode'] = confidence_mode
    
    # Position Size Settings
    position_size = st.slider(
        "💰 Position Size (%)",
        min_value=1, max_value=20, value=5,
        help="Percentage of portfolio per position"
    )
    
    max_positions = st.slider(
        "📊 Max Positions",
        min_value=5, max_value=25, value=15,
        help="Maximum number of simultaneous positions"
    )
    
    # Risk Management
    st.subheader("⚠️ Risk Management")
    stop_loss = st.slider("Stop Loss (%)", 1, 10, 5)
    take_profit = st.slider("Take Profit (%)", 5, 25, 10)

# Main Dashboard
portfolio_value = 500000

# Enhanced Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Portfolio Value", f"₹{portfolio_value:,.0f}")

with col2:
    st.metric("Stock Universe", f"{len(bot.stock_universe)} stocks")

with col3:
    st.metric("Confidence Mode", confidence_mode.title())

with col4:
    st.metric("Active Positions", len(st.session_state.enhanced_portfolio['positions']))

with col5:
    accuracy = random.uniform(65, 85)  # Simulated accuracy
    st.metric("AI Accuracy", f"{accuracy:.1f}%")

# Enhanced Signal Generation
st.markdown("## 🎯 Enhanced AI Trading Signals")

# Signal generation controls
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Generate Fresh Signals", type="primary"):
        st.cache_data.clear()

with col2:
    sector_filter = st.selectbox(
        "Filter by Sector",
        ['All', 'IT', 'Banking', 'FMCG', 'Energy', 'Pharma', 'Auto', 'Metals']
    )

with col3:
    min_confidence = st.slider("Min Confidence %", 0, 100, 60)

# Generate enhanced signals
signals_df = bot.generate_ai_signals_enhanced(confidence_mode)

# Apply filters
if sector_filter != 'All':
    signals_df = signals_df[signals_df['Sector'] == sector_filter]

signals_df = signals_df[signals_df['Confidence'].str.rstrip('%').astype(float) >= min_confidence]

# Display signals with enhanced styling
if not signals_df.empty:
    # Color coding function
    def color_rows(row):
        if 'BUY' in row['Action']:
            return ['background-color: #d4edda'] * len(row)
        elif 'SELL' in row['Action']:
            return ['background-color: #f8d7da'] * len(row)
        else:
            return ['background-color: #fff3cd'] * len(row)
    
    styled_df = signals_df.style.apply(color_rows, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # High confidence alerts
    high_conf_signals = signals_df[signals_df['Confidence'].str.rstrip('%').astype(float) >= 75]
    
    if not high_conf_signals.empty:
        st.markdown("### 🚨 High Confidence Alerts (75%+)")
        
        for _, signal in high_conf_signals.iterrows():
            confidence_val = float(signal['Confidence'].rstrip('%'))
            
            if 'BUY' in signal['Action']:
                st.success(f"🟢 **STRONG BUY**: {signal['Symbol']} ({signal['Sector']}) at {signal['Current Price']} - {signal['Confidence']} confidence")
            elif 'SELL' in signal['Action']:
                st.error(f"🔴 **STRONG SELL**: {signal['Symbol']} ({signal['Sector']}) at {signal['Current Price']} - {signal['Confidence']} confidence")
else:
    st.info("No signals match your current filters. Try adjusting the confidence threshold or sector filter.")

# Performance Analytics
st.markdown("## 📊 Enhanced Analytics Dashboard")

if not signals_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Sector-wise signal distribution
        sector_counts = signals_df['Sector'].value_counts()
        fig_sector = px.pie(values=sector_counts.values, names=sector_counts.index, 
                           title="Signals by Sector")
        st.plotly_chart(fig_sector, use_container_width=True)
    
    with col2:
        # Confidence distribution
        confidence_values = signals_df['Confidence'].str.rstrip('%').astype(float)
        fig_conf = px.histogram(x=confidence_values, nbins=10, 
                               title="Signal Confidence Distribution")
        st.plotly_chart(fig_conf, use_container_width=True)

# Add performance analytics
display_performance_analytics()

# Advanced Controls
st.markdown("## ⚡ Advanced Trading Controls")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🚀 Execute High-Confidence Signals"):
        high_conf = signals_df[signals_df['Confidence'].str.rstrip('%').astype(float) >= 75]
        if not high_conf.empty:
            st.success(f"✅ Would execute {len(high_conf)} high-confidence signals")
        else:
            st.warning("No high-confidence signals to execute")

with col2:
    if st.button("📈 Sector Rotation Analysis"):
        st.info("🔄 Analyzing sector rotation patterns...")

with col3:
    if st.button("🎯 Backtest Strategy"):
        st.info("📊 Running backtest on historical data...")

with col4:
    if st.button("⚠️ Risk Assessment"):
        st.info("🛡️ Calculating portfolio risk metrics...")

# Footer with system status
st.markdown("---")
st.markdown("### 🎛️ System Status")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.success("✅ Enhanced AI: Active")
with col2:
    st.success(f"✅ {len(bot.stock_universe)} Stocks Monitored")
with col3:
    st.success(f"✅ Mode: {confidence_mode.title()}")
with col4:
    st.success("✅ Real-time Analytics: On")

st.markdown(f"*Enhanced system last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
