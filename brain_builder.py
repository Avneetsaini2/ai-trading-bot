import pandas as pd
import numpy as np
from datetime import datetime
import os

class StockBrain:
    def __init__(self):
        print("🧠 Creating a new stock brain...")
        self.memory = {}  # This is where we store what we learn
        
    def add_memory(self, symbol_name, stock_data):
        """Teach the brain about one stock"""
        print(f"📚 Teaching brain about {symbol_name}...")
        
        # Make a copy so we don't mess up the original
        df = stock_data.copy()
        
        # Step 1: Add "temperature readings" for stocks
        df = self.add_simple_indicators(df)
        
        # Step 2: Create "future knowledge" (what happened tomorrow?)
        df = self.add_future_targets(df)
        
        # Step 3: Store in brain's memory
        self.memory[symbol_name] = df
        
        print(f"✅ Brain learned {len(df)} days of {symbol_name} patterns")
        
    def add_simple_indicators(self, df):
        """Add simple 'temperature readings' for stocks"""
        print("   🌡️ Adding price temperature readings...")
        
        # 1. Moving Averages (like average temperature over time)
        df['MA_5'] = df['Close'].rolling(window=5).mean()    # 5-day average
        df['MA_20'] = df['Close'].rolling(window=20).mean()  # 20-day average
        
        # 2. Is price above or below average? (like "is it hotter than usual?")
        df['Above_MA5'] = (df['Close'] > df['MA_5']).astype(int)
        df['Above_MA20'] = (df['Close'] > df['MA_20']).astype(int)
        
        # 3. How much did price change today? (like temperature difference)
        df['Price_Change'] = df['Close'].pct_change() * 100  # Percentage change
        
        # 4. Volume indicator (like "how many people are outside?")
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['High_Volume'] = (df['Volume'] > df['Volume_MA']).astype(int)
        
        # 5. RSI - "Is the stock too hot or too cold?"
        df['RSI'] = self.calculate_simple_rsi(df['Close'])
        df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)  # Too hot
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)    # Too cold
        
        return df
    
    def calculate_simple_rsi(self, prices, period=14):
        """Calculate RSI (like a thermometer for stocks)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def add_future_targets(self, df):
        """Add 'what happened tomorrow?' column"""
        print("   🔮 Adding future knowledge...")
        
        # Did price go up tomorrow? (1 = yes, 0 = no)
        df['Tomorrow_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # How much did it change tomorrow?
        df['Tomorrow_Change'] = df['Close'].shift(-1).pct_change() * 100
        
        return df
    
    def save_brain(self):
        """Save the brain's memory to files"""
        print("💾 Saving brain's memory...")
        
        os.makedirs('brain_memory', exist_ok=True)
        
        for symbol, data in self.memory.items():
            # Save to CSV file
            data.to_csv(f'brain_memory/{symbol}_brain_data.csv')
            print(f"   ✅ Saved {symbol} memory")
    
    def show_what_brain_learned(self, symbol):
        """Show what the brain learned about a stock"""
        if symbol in self.memory:
            df = self.memory[symbol]
            
            print(f"\n🧠 What brain learned about {symbol}:")
            print(f"   📊 Total days studied: {len(df)}")
            print(f"   📈 Days price went up next day: {df['Tomorrow_Up'].sum()}")
            print(f"   📉 Days price went down next day: {len(df) - df['Tomorrow_Up'].sum()}")
            print(f"   🎯 Success rate if we always predicted 'up': {df['Tomorrow_Up'].mean()*100:.1f}%")
            
            # Show some patterns
            print(f"\n🔍 Interesting patterns found:")
            
            # When RSI is oversold, what usually happens?
            oversold_days = df[df['RSI_Oversold'] == 1]
            if len(oversold_days) > 0:
                success_rate = oversold_days['Tomorrow_Up'].mean() * 100
                print(f"   💡 When stock was 'too cold' (oversold): {success_rate:.1f}% chance of going up next day")
            
            # When volume is high, what happens?
            high_vol_days = df[df['High_Volume'] == 1]
            if len(high_vol_days) > 0:
                success_rate = high_vol_days['Tomorrow_Up'].mean() * 100
                print(f"   💡 When lots of people were trading: {success_rate:.1f}% chance of going up next day")
