import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from smart_trader import SmartTrader
from data_engine import AlphaVantageEngine
import logging

class CloudTradingBot:
    def __init__(self):
        print("☁️ Initializing Cloud Trading Bot...")
        self.trader = SmartTrader()
        self.data_engine = AlphaVantageEngine()
        self.portfolio = self.load_portfolio()
        self.alert_settings = self.load_alert_settings()
        
        # Set up logging
        logging.basicConfig(
            filename='trading_bot.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def load_portfolio(self):
        """Load portfolio data or create new one"""
        portfolio_file = 'portfolio.json'
        
        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                return json.load(f)
        else:
            # Create new portfolio
            portfolio = {
                'cash': 100000,  # Starting with ₹1 lakh virtual money
                'positions': {},
                'trade_history': [],
                'performance': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0,
                    'max_drawdown': 0,
                    'start_date': datetime.now().isoformat()
                }
            }
            self.save_portfolio(portfolio)
            return portfolio
    
    def save_portfolio(self, portfolio=None):
        """Save portfolio to file"""
        if portfolio is None:
            portfolio = self.portfolio
            
        with open('portfolio.json', 'w') as f:
            json.dump(portfolio, f, indent=2)
    
    def load_alert_settings(self):
        """Load alert settings"""
        settings_file = 'alert_settings.json'
        
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                return json.load(f)
        else:
            # Default settings
            settings = {
                'email_alerts': True,
                'sms_alerts': False,
                'email': 'your_email@gmail.com',
                'phone': '+91XXXXXXXXXX',
                'min_confidence': 70,  # Only alert for 70%+ confidence
                'gmail_password': 'your_app_password'  # Gmail app password
            }
            
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            
            print("📧 Please update alert_settings.json with your email and phone")
            return settings
    
    def send_email_alert(self, subject, message):
        """Send email alert"""
        try:
            if not self.alert_settings['email_alerts']:
                return
            
            msg = MimeMultipart()
            msg['From'] = self.alert_settings['email']
            msg['To'] = self.alert_settings['email']
            msg['Subject'] = subject
            
            msg.attach(MimeText(message, 'plain'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.alert_settings['email'], self.alert_settings['gmail_password'])
            text = msg.as_string()
            server.sendmail(self.alert_settings['email'], self.alert_settings['email'], text)
            server.quit()
            
            print("📧 Email alert sent!")
            
        except Exception as e:
            print(f"❌ Email alert failed: {e}")
    
    def calculate_position_size(self, confidence, price):
        """Calculate how much to invest based on confidence and risk management"""
        # Risk management: never risk more than 2% of portfolio
        max_risk = self.portfolio['cash'] * 0.02
        
        # Position size based on confidence
        if confidence >= 80:
            position_percent = 0.10  # 10% of portfolio for very high confidence
        elif confidence >= 70:
            position_percent = 0.05  # 5% of portfolio for high confidence
        else:
            position_percent = 0.02  # 2% of portfolio for moderate confidence
        
        position_value = self.portfolio['cash'] * position_percent
        quantity = int(position_value / price)
        
        return max(1, quantity)  # At least 1 share
    
    def execute_virtual_trade(self, symbol, action, price, quantity, confidence, reason):
        """Execute a virtual trade and update portfolio"""
        trade_value = price * quantity
        
        if action == 'BUY':
            if self.portfolio['cash'] >= trade_value:
                # Execute buy order
                self.portfolio['cash'] -= trade_value
                
                if symbol in self.portfolio['positions']:
                    # Add to existing position
                    existing = self.portfolio['positions'][symbol]
                    total_quantity = existing['quantity'] + quantity
                    avg_price = ((existing['avg_price'] * existing['quantity']) + 
                               (price * quantity)) / total_quantity
                    
                    self.portfolio['positions'][symbol] = {
                        'quantity': total_quantity,
                        'avg_price': avg_price,
                        'current_price': price,
                        'last_updated': datetime.now().isoformat()
                    }
                else:
                    # New position
                    self.portfolio['positions'][symbol] = {
                        'quantity': quantity,
                        'avg_price': price,
                        'current_price': price,
                        'last_updated': datetime.now().isoformat()
                    }
                
                # Record trade
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'value': trade_value,
                    'confidence': confidence,
                    'reason': reason
                }
                
                self.portfolio['trade_history'].append(trade)
                self.portfolio['performance']['total_trades'] += 1
                
                print(f"✅ BOUGHT {quantity} shares of {symbol} at ₹{price:.2f}")
                logging.info(f"BUY executed: {symbol} {quantity}@{price:.2f}")
                
                return True
            else:
                print(f"❌ Insufficient funds for {symbol} trade")
                return False
                
        elif action == 'SELL':
            if symbol in self.portfolio['positions'] and self.portfolio['positions'][symbol]['quantity'] >= quantity:
                # Execute sell order
                position = self.portfolio['positions'][symbol]
                
                # Calculate P&L
                pnl = (price - position['avg_price']) * quantity
                self.portfolio['cash'] += trade_value
                self.portfolio['performance']['total_pnl'] += pnl
                
                # Update position
                remaining_quantity = position['quantity'] - quantity
                if remaining_quantity > 0:
                    self.portfolio['positions'][symbol]['quantity'] = remaining_quantity
                else:
                    del self.portfolio['positions'][symbol]
                
                # Record trade
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'value': trade_value,
                    'pnl': pnl,
                    'confidence': confidence,
                    'reason': reason
                }
                
                self.portfolio['trade_history'].append(trade)
                self.portfolio['performance']['total_trades'] += 1
                
                if pnl > 0:
                    self.portfolio['performance']['winning_trades'] += 1
                
                print(f"✅ SOLD {quantity} shares of {symbol} at ₹{price:.2f} (P&L: ₹{pnl:.2f})")
                logging.info(f"SELL executed: {symbol} {quantity}@{price:.2f} P&L:{pnl:.2f}")
                
                return True
            else:
                print(f"❌ No position to sell for {symbol}")
                return False
        
        return False
    
    def run_trading_session(self):
        """Run one complete trading session"""
        print(f"\n🤖 Running trading session at {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        try:
            # Get live market data
            live_data = self.data_engine.get_live_data()
            
            if live_data.empty:
                print("❌ No market data available")
                return
            
            signals_generated = 0
            trades_executed = 0
            high_confidence_alerts = []
            
            for _, stock_data in live_data.iterrows():
                symbol = stock_data['symbol']
                current_price = stock_data['price']
                
                # Calculate technical indicators (reuse from live_signals.py)
                try:
                    hist_file = f'historical_data/{symbol}.csv'
                    df = pd.read_csv(hist_file, index_col=0, parse_dates=True)
                    latest = df.iloc[-1]
                    
                    indicators = {
                        'MA_5': df['Close'].rolling(5).mean().iloc[-1],
                        'MA_20': df['Close'].rolling(20).mean().iloc[-1],
                        'Above_MA5': 1 if current_price > df['Close'].rolling(5).mean().iloc[-1] else 0,
                        'Above_MA20': 1 if current_price > df['Close'].rolling(20).mean().iloc[-1] else 0,
                        'Price_Change': ((current_price - latest['Close']) / latest['Close']) * 100,
                        'High_Volume': 1 if stock_data.get('volume', 0) > df['Volume'].rolling(20).mean().iloc[-1] else 0,
                        'RSI': 50,  # Simplified for now
                        'RSI_Overbought': 0,
                        'RSI_Oversold': 0
                    }
                    
                    # Get AI decision
                    decision, confidence, reason = self.trader.make_prediction(symbol, indicators)
                    signals_generated += 1
                    
                    print(f"📊 {symbol}: {decision} ({confidence:.1f}%) - ₹{current_price:.2f}")
                    
                    # Execute trades for high confidence signals
                    if confidence >= self.alert_settings['min_confidence']:
                        high_confidence_alerts.append({
                            'symbol': symbol,
                            'decision': decision,
                            'confidence': confidence,
                            'price': current_price,
                            'reason': reason
                        })
                        
                        if decision == 'BUY':
                            quantity = self.calculate_position_size(confidence, current_price)
                            if self.execute_virtual_trade(symbol, 'BUY', current_price, quantity, confidence, reason):
                                trades_executed += 1
                                
                        elif decision == 'SELL' and symbol in self.portfolio['positions']:
                            quantity = min(self.portfolio['positions'][symbol]['quantity'], 
                                         self.calculate_position_size(confidence, current_price))
                            if self.execute_virtual_trade(symbol, 'SELL', current_price, quantity, confidence, reason):
                                trades_executed += 1
                
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {e}")
                    continue
            
            # Save portfolio
            self.save_portfolio()
            
            # Send alerts for high confidence signals
            if high_confidence_alerts:
                alert_message = f"🚨 HIGH CONFIDENCE TRADING ALERTS\n\n"
                for alert in high_confidence_alerts:
                    alert_message += f"• {alert['decision']} {alert['symbol']} at ₹{alert['price']:.2f} ({alert['confidence']:.1f}%)\n"
                    alert_message += f"  Reason: {alert['reason']}\n\n"
                
                self.send_email_alert("🤖 Trading Bot Alert", alert_message)
            
            # Session summary
            total_portfolio_value = self.calculate_total_portfolio_value()
            print(f"\n📈 SESSION SUMMARY:")
            print(f"   🎯 Signals generated: {signals_generated}")
            print(f"   ⚡ Trades executed: {trades_executed}")
            print(f"   💰 Portfolio value: ₹{total_portfolio_value:,.2f}")
            print(f"   🚨 High confidence alerts: {len(high_confidence_alerts)}")
            
        except Exception as e:
            print(f"❌ Trading session error: {e}")
            logging.error(f"Trading session error: {e}")
    
    def calculate_total_portfolio_value(self):
        """Calculate total portfolio value"""
        total_value = self.portfolio['cash']
        
        # Add value of all positions at current prices
        for symbol, position in self.portfolio['positions'].items():
            total_value += position['quantity'] * position['current_price']
        
        return total_value
    
    def run_continuous_trading(self, interval_minutes=30):
        """Run continuous trading bot"""
        print("🚀 CLOUD TRADING BOT STARTED")
        print("=" * 40)
        print(f"⏰ Trading every {interval_minutes} minutes")
        print(f"💰 Starting portfolio value: ₹{self.calculate_total_portfolio_value():,.2f}")
        print("Press Ctrl+C to stop")
        
        while True:
            try:
                self.run_trading_session()
                
                print(f"\n⏳ Waiting {interval_minutes} minutes for next session...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n🛑 Trading bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logging.error(f"Continuous trading error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    bot = CloudTradingBot()
    
    # Ask user what they want to do
    print("🤖 CLOUD TRADING BOT")
    print("=" * 25)
    print("1. Run single trading session")
    print("2. Start continuous trading")
    print("3. View portfolio status")
    
    choice = input("\nChoose option (1-3): ").strip()
    
    if choice == "1":
        bot.run_trading_session()
    elif choice == "2":
        bot.run_continuous_trading()
    elif choice == "3":
        portfolio_value = bot.calculate_total_portfolio_value()
        print(f"\n💰 Portfolio Value: ₹{portfolio_value:,.2f}")
        print(f"💵 Cash: ₹{bot.portfolio['cash']:,.2f}")
        print(f"📊 Positions: {len(bot.portfolio['positions'])}")
        print(f"📈 Total Trades: {bot.portfolio['performance']['total_trades']}")
        print(f"🎯 Win Rate: {(bot.portfolio['performance']['winning_trades'] / max(1, bot.portfolio['performance']['total_trades']) * 100):.1f}%")
