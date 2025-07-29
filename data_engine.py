import requests
import pandas as pd
import time
from datetime import datetime
import os

class AlphaVantageEngine:
    def __init__(self):
        self.api_key = "4BNIN6S9MT2LFA3D"
        self.symbols = {
            'RELIANCE': 'RELIANCE.BSE',
            'TCS': 'TCS.BSE', 
            'HDFCBANK': 'HDFCBANK.BSE',
            'ICICIBANK': 'ICICIBANK.BSE',
            'INFY': 'INFY.BSE',
            'ITC': 'ITC.BSE',
            'HINDUNILVR': 'HINDUNILVR.BSE',
            'KOTAKBANK': 'KOTAKBANK.BSE'
        }
        
    def get_live_data(self):
        """Get current stock prices"""
        live_data = []
        
        for name, symbol in self.symbols.items():
            try:
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                if 'Global Quote' in data:
                    quote = data['Global Quote']
                    live_data.append({
                        'symbol': name,
                        'price': float(quote['05. price']),
                        'change': float(quote['09. change']),
                        'change_percent': quote['10. change percent'].rstrip('%'),
                        'volume': int(quote['06. volume']),
                        'timestamp': datetime.now()
                    })
                    print(f"✅ {name}: ₹{quote['05. price']}")
                    
                time.sleep(12)  # Free tier: 5 calls/minute
                
            except Exception as e:
                print(f"❌ {name}: {e}")
                continue
        
        return pd.DataFrame(live_data)
    
    def download_historical_data(self, days=365):
        """Download historical data"""
        print("📈 Downloading historical data from Alpha Vantage...")
        
        os.makedirs('historical_data', exist_ok=True)
        
        for name, symbol in self.symbols.items():
            try:
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'TIME_SERIES_DAILY',
                    'symbol': symbol,
                    'outputsize': 'full',
                    'apikey': self.api_key
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                if 'Time Series (Daily)' in data:
                    df = pd.DataFrame(data['Time Series (Daily)']).T
                    df.index = pd.to_datetime(df.index)
                    df = df.astype(float)
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    
                    # Add technical indicators
                    df = self.add_technical_indicators(df)
                    df.to_csv(f'historical_data/{name}.csv')
                    
                    print(f"✅ {name}: {len(df)} days saved")
                else:
                    print(f"❌ {name}: No data received")
                
                time.sleep(12)  # Respect rate limits
                
            except Exception as e:
                print(f"❌ {name}: {e}")
    
    def add_technical_indicators(self, df):
        """Same technical indicators as before"""
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        return df

# Test it
if __name__ == "__main__":
    engine = AlphaVantageEngine()
    engine.download_historical_data()
