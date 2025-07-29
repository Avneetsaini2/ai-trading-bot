from brain_builder import StockBrain
import pandas as pd
import os

def teach_brain_everything():
    """Teach the brain about all our stocks"""
    print("🎓 STOCK BRAIN TRAINING SCHOOL")
    print("=" * 40)
    
    # Create a new brain
    brain = StockBrain()
    
    # Find all our historical data files
    data_folder = 'historical_data'
    
    if not os.path.exists(data_folder):
        print("❌ No historical data found!")
        print("   Make sure you ran the data download first")
        return
    
    # Teach brain about each stock
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found in historical_data folder")
        return
    
    for file in csv_files:
        try:
            # Get stock name (remove .csv)
            stock_name = file.replace('.csv', '')
            
            # Load the data
            print(f"\n📖 Loading {stock_name} data...")
            stock_data = pd.read_csv(f'{data_folder}/{file}', index_col=0, parse_dates=True)
            
            # Make sure we have the basic columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in stock_data.columns for col in required_columns):
                # Teach brain about this stock
                brain.add_memory(stock_name, stock_data)
                
                # Show what brain learned
                brain.show_what_brain_learned(stock_name)
            else:
                print(f"❌ {stock_name}: Missing required data columns")
                
        except Exception as e:
            print(f"❌ Error teaching brain about {file}: {e}")
    
    # Save everything brain learned
    brain.save_brain()
    
    print(f"\n🎉 BRAIN TRAINING COMPLETE!")
    print(f"🧠 Brain now knows patterns for {len(brain.memory)} stocks")
    print(f"💾 All knowledge saved to 'brain_memory' folder")

if __name__ == "__main__":
    teach_brain_everything()
