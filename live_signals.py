import pandas as pd
from smart_trader import SmartTrader
from data_engine import AlphaVantageEngine  # or whatever data engine you're using
import time
from datetime import datetime

class LiveSignalGenerator:
    def __init__(self):
        print("📡 Starting Live Signal Generator...")
        self.trader = SmartTrader()
        self.data_engine = AlphaVantageEngine()  # Your data source
        
    def calculate_current_indicators(self, symbol_data):
        """Calculate current technical indicators for live data"""
        # Get recent historical data to calculate indicators
        try:
            # Load historical data
            hist_file = f'historical_data/{symbol_data["symbol"]}.csv'
            df = pd.read_csv(hist_file, index_col=0, parse_dates=True)
            
            # Get the most recent values
            latest = df.iloc[-1]
            
            # Calculate current indicators (same as brain_builder.py)
            current_indicators = {
                'MA_5': df['Close'].rolling(5).mean().iloc[-1],
                'MA_20': df['Close'].rolling(20).mean().iloc[-1],
                'Above_MA5': 1 if symbol_data['price'] > df['Close'].rolling(5).mean().iloc[-1] else 0,
                'Above_MA20': 1 if symbol_data['price'] > df['Close'].rolling(20).mean().iloc[-1] else 0,
                'Price_Change': ((symbol_data['price'] - latest['Close']) / latest['Close']) * 100,
                'High_Volume': 1 if symbol_data.get('volume', 0) > df['Volume'].rolling(20).mean().iloc[-1] else 0,
                'RSI': self.calculate_current_rsi(df['Close'], symbol_data['price']),
                'RSI_Overbought': 0,  # Will calculate based on RSI
                'RSI_Oversold': 0     # Will calculate based on RSI
            }
            
            # Set RSI flags
            rsi = current_indicators['RSI']
            current_indicators['RSI_Overbought'] = 1 if rsi > 70 else 0
            current_indicators['RSI_Oversold'] = 1 if rsi < 30 else 0
            
            return current_indicators
            
        except Exception as e:
            print(f"❌ Error calculating indicators for {symbol_data['symbol']}: {e}")
            return None
    
    def calculate_current_rsi(self, historical_prices, current_price, period=14):
        """Calculate current RSI including today's price"""
        # Add current price to historical series
        prices = historical_prices.copy()
        prices.loc[datetime.now()] = current_price
        
        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def generate_live_signals(self):
        """Generate live trading signals for all stocks"""
        print(f"\n📊 Generating signals at {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 50)
        
        # Get live market data
        live_data = self.data_engine.get_live_data()
        
        if live_data.empty:
            print("❌ No live data available")
            return []
        
        signals = []
        
        for _, stock_data in live_data.iterrows():
            symbol = stock_data['symbol']
            
            # Calculate current technical indicators
            indicators = self.calculate_current_indicators(stock_data)
            
            if indicators is None:
                continue
            
            # Get AI decision
            decision, confidence, reason = self.trader.make_prediction(symbol, indicators)
            
            # Create signal
            signal = {
                'symbol': symbol,
                'price': stock_data['price'],
                'change': stock_data.get('change', 0),
                'decision': decision,
                'confidence': confidence,
                'reason': reason,
                'timestamp': datetime.now()
            }
            
            signals.append(signal)
            
            # Print signal
            emoji = "🟢" if decision == "BUY" else "🔴" if decision == "SELL" else "🟡"
            print(f"{emoji} {symbol}: {decision} at ₹{stock_data['price']:.2f} ({confidence:.1f}% confidence)")
            print(f"   💭 {reason}")
        
        return signals
    
    def run_continuous_signals(self, interval_minutes=5):
        """Run signal generation continuously"""
        print(f"🚀 Starting continuous signal generation (every {interval_minutes} minutes)")
        print("Press Ctrl+C to stop")
        
        while True:
            try:
                signals = self.generate_live_signals()
                
                # Save signals to file
                if signals:
                    df = pd.DataFrame(signals)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                    df.to_csv(f'live_signals_{timestamp}.csv', index=False)
                
                print(f"\n⏳ Waiting {interval_minutes} minutes for next signal generation...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n🛑 Signal generation stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("⏳ Retrying in 1 minute...")
                time.sleep(60)

if __name__ == "__main__":
    generator = LiveSignalGenerator()
    
    # Generate signals once
    print("📡 LIVE TRADING SIGNALS")
    print("=" * 30)
    signals = generator.generate_live_signals()
    
    # Ask if user wants continuous signals
    if signals:
        continuous = input("\n🔄 Run continuous signals? (y/n): ")
        if continuous.lower() == 'y':
            generator.run_continuous_signals()
