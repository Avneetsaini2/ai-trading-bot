import pandas as pd
import os

def show_brain_insights():
    """Show the smartest patterns our brain found"""
    print("🧠 BRAIN INSIGHTS REPORT")
    print("=" * 30)
    
    brain_folder = 'brain_memory'
    
    if not os.path.exists(brain_folder):
        print("❌ No brain memory found. Run train_brain.py first!")
        return
    
    # Check each stock's brain data
    for file in os.listdir(brain_folder):
        if file.endswith('_brain_data.csv'):
            stock_name = file.replace('_brain_data.csv', '')
            
            print(f"\n📊 {stock_name} INSIGHTS:")
            print("-" * 20)
            
            try:
                df = pd.read_csv(f'{brain_folder}/{file}', index_col=0, parse_dates=True)
                
                # Remove rows with missing data
                df = df.dropna()
                
                if len(df) < 50:
                    print("   ❌ Not enough clean data")
                    continue
                
                # Overall success rate
                total_up_days = df['Tomorrow_Up'].sum()
                total_days = len(df)
                success_rate = (total_up_days / total_days) * 100
                
                print(f"   📈 Overall: {success_rate:.1f}% of days went up")
                
                # Best patterns
                print(f"   🔍 Best patterns found:")
                
                # Pattern 1: RSI Oversold
                oversold = df[df['RSI_Oversold'] == 1]
                if len(oversold) > 10:
                    rate = oversold['Tomorrow_Up'].mean() * 100
                    print(f"      💡 When RSI < 30: {rate:.1f}% success rate ({len(oversold)} times)")
                
                # Pattern 2: Price above both moving averages
                bullish = df[(df['Above_MA5'] == 1) & (df['Above_MA20'] == 1)]
                if len(bullish) > 10:
                    rate = bullish['Tomorrow_Up'].mean() * 100
                    print(f"      💡 When above both MAs: {rate:.1f}% success rate ({len(bullish)} times)")
                
                # Pattern 3: High volume + price drop
                vol_drop = df[(df['High_Volume'] == 1) & (df['Price_Change'] < -2)]
                if len(vol_drop) > 5:
                    rate = vol_drop['Tomorrow_Up'].mean() * 100
                    print(f"      💡 High volume + big drop: {rate:.1f}% bounce rate ({len(vol_drop)} times)")
                
            except Exception as e:
                print(f"   ❌ Error reading {stock_name}: {e}")

if __name__ == "__main__":
    show_brain_insights()
