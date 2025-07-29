import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime

class SmartTrader:
    def __init__(self):
        print("🤖 Creating Smart Trading Robot...")
        self.models = {}  # This will store our trained models
        self.features = [
            'MA_5', 'MA_20', 'Above_MA5', 'Above_MA20', 
            'Price_Change', 'High_Volume', 'RSI', 
            'RSI_Overbought', 'RSI_Oversold'
        ]
        
    def train_single_stock(self, stock_name):
        """Train the decision maker for one stock"""
        print(f"\n🎓 Training decision maker for {stock_name}...")
        
        try:
            # Load the brain data
            brain_file = f'brain_memory/{stock_name}_brain_data.csv'
            df = pd.read_csv(brain_file, index_col=0, parse_dates=True)
            
            # Clean the data (remove any rows with missing values)
            df = df.dropna()
            
            if len(df) < 100:
                print(f"   ❌ Not enough data for {stock_name} (need at least 100 days)")
                return False
            
            # Prepare the data for the robot
            # X = what the robot sees (features)
            # y = what we want it to predict (Tomorrow_Up)
            X = df[self.features]
            y = df['Tomorrow_Up']
            
            print(f"   📊 Using {len(X)} days of data with {len(self.features)} indicators")
            
            # Split data: 80% for training, 20% for testing
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            print(f"   📚 Training on {len(X_train)} days, testing on {len(X_test)} days")
            
            # Create and train the robot brain
            model = RandomForestClassifier(
                n_estimators=100,      # Use 100 decision trees
                max_depth=10,          # Don't make trees too complex
                random_state=42,       # For consistent results
                class_weight='balanced' # Handle imbalanced data
            )
            
            print("   🧠 Training the robot brain...")
            model.fit(X_train, y_train)
            
            # Test how well it learned
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"   🎯 Robot accuracy: {accuracy*100:.1f}%")
            
            # Show detailed performance
            if accuracy > 0.5:  # Better than random guessing
                print(f"   ✅ Good! Robot is better than random guessing")
                
                # Show feature importance (what the robot pays attention to most)
                feature_importance = model.feature_importances_
                important_features = sorted(zip(self.features, feature_importance), 
                                          key=lambda x: x[1], reverse=True)
                
                print(f"   🔍 Robot pays most attention to:")
                for feature, importance in important_features[:3]:
                    print(f"      • {feature}: {importance*100:.1f}% importance")
            else:
                print(f"   ⚠️  Robot isn't learning well (accuracy too low)")
            
            # Save the trained model
            os.makedirs('trained_models', exist_ok=True)
            model_file = f'trained_models/{stock_name}_model.pkl'
            joblib.dump(model, model_file)
            
            # Store in memory too
            self.models[stock_name] = model
            
            print(f"   💾 Model saved to {model_file}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error training {stock_name}: {e}")
            return False
    
    def train_all_stocks(self):
        """Train decision makers for all stocks"""
        print("🏫 SMART TRADER TRAINING ACADEMY")
        print("=" * 40)
        
        brain_folder = 'brain_memory'
        
        if not os.path.exists(brain_folder):
            print("❌ No brain memory found!")
            print("   Please run train_brain.py first")
            return
        
        # Find all brain data files
        brain_files = [f for f in os.listdir(brain_folder) if f.endswith('_brain_data.csv')]
        
        if not brain_files:
            print("❌ No brain data files found!")
            return
        
        successful_models = 0
        
        for file in brain_files:
            stock_name = file.replace('_brain_data.csv', '')
            
            if self.train_single_stock(stock_name):
                successful_models += 1
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"✅ Successfully trained {successful_models} models")
        print(f"🤖 Your trading robot is ready!")
    
    def make_prediction(self, stock_name, current_data):
        """Make a BUY/SELL/HOLD decision for current market data"""
        try:
            # Load the model if not in memory
            if stock_name not in self.models:
                model_file = f'trained_models/{stock_name}_model.pkl'
                if os.path.exists(model_file):
                    self.models[stock_name] = joblib.load(model_file)
                else:
                    return "HOLD", 50.0, "No model found"
            
            model = self.models[stock_name]
            
            # Prepare the current data in the same format as training
            features_data = []
            for feature in self.features:
                if feature in current_data:
                    features_data.append(current_data[feature])
                else:
                    features_data.append(0)  # Default value if missing
            
            # Get prediction probabilities
            probabilities = model.predict_proba([features_data])[0]
            
            # probabilities[0] = chance of going down
            # probabilities[1] = chance of going up
            
            up_probability = probabilities[1] * 100
            
            # Make decision based on confidence
            if up_probability >= 60:
                decision = "BUY"
                confidence = up_probability
                reason = f"Model is {up_probability:.1f}% confident price will rise"
            elif up_probability <= 40:
                decision = "SELL"
                confidence = 100 - up_probability
                reason = f"Model is {100-up_probability:.1f}% confident price will fall"
            else:
                decision = "HOLD"
                confidence = max(up_probability, 100-up_probability)
                reason = f"Model is uncertain (only {up_probability:.1f}% confident)"
            
            return decision, confidence, reason
            
        except Exception as e:
            return "HOLD", 50.0, f"Error: {e}"

# Test function
def test_smart_trader():
    """Test the smart trader with some sample data"""
    trader = SmartTrader()
    
    # Test with fake current market data
    test_data = {
        'MA_5': 100.5,
        'MA_20': 98.2,
        'Above_MA5': 1,      # Price is above 5-day average
        'Above_MA20': 1,     # Price is above 20-day average  
        'Price_Change': 2.5, # Price went up 2.5% today
        'High_Volume': 1,    # Volume was high today
        'RSI': 45,           # RSI is neutral
        'RSI_Overbought': 0, # Not overbought
        'RSI_Oversold': 0    # Not oversold
    }
    
    print("\n🧪 TESTING SMART TRADER")
    print("=" * 25)
    print("Sample market conditions:")
    print("• Price above both moving averages ✅")
    print("• Price up 2.5% today ✅") 
    print("• High trading volume ✅")
    print("• RSI neutral (45) ✅")
    
    # Test on available models
    model_folder = 'trained_models'
    if os.path.exists(model_folder):
        model_files = [f for f in os.listdir(model_folder) if f.endswith('_model.pkl')]
        
        for file in model_files[:3]:  # Test first 3 models
            stock_name = file.replace('_model.pkl', '')
            decision, confidence, reason = trader.make_prediction(stock_name, test_data)
            
            print(f"\n📊 {stock_name}:")
            print(f"   🎯 Decision: {decision}")
            print(f"   📈 Confidence: {confidence:.1f}%")
            print(f"   💭 Reason: {reason}")

if __name__ == "__main__":
    trader = SmartTrader()
    trader.train_all_stocks()
    test_smart_trader()
