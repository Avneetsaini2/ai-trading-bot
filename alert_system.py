import smtplib
import requests
import json
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from datetime import datetime
import streamlit as st

class TradingAlertSystem:
    def __init__(self):
        self.load_alert_config()
    
    def load_alert_config(self):
        """Load alert configuration"""
        if 'alert_config' not in st.session_state:
            st.session_state.alert_config = {
                'email': {
                    'enabled': False,
                    'address': '',
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'password': ''  # Use app password for Gmail
                },
                'sms': {
                    'enabled': False,
                    'phone': '',
                    'service': 'textbelt'  # Free SMS service
                },
                'telegram': {
                    'enabled': False,
                    'bot_token': '',
                    'chat_id': ''
                },
                'thresholds': {
                    'min_confidence': 75,
                    'max_alerts_per_hour': 5
                }
            }
    
    def setup_alerts_ui(self):
        """Create alerts setup UI"""
        st.markdown("## 📱 Alert System Setup")
        
        # Email Alerts
        st.markdown("### 📧 Email Alerts")
        col1, col2 = st.columns(2)
        
        with col1:
            email_enabled = st.checkbox("Enable Email Alerts", 
                                       value=st.session_state.alert_config['email']['enabled'])
            
            if email_enabled:
                email_address = st.text_input("Email Address", 
                                            value=st.session_state.alert_config['email']['address'],
                                            placeholder="your.email@gmail.com")
                
                email_password = st.text_input("Gmail App Password", 
                                             type="password",
                                             help="Generate this from Gmail Settings > Security > App Passwords")
        
        with col2:
            if email_enabled:
                st.markdown("**📋 Gmail Setup Steps:**")
                st.markdown("""
                1. Go to Gmail Settings → Security
                2. Enable 2-Factor Authentication
                3. Generate App Password for 'Mail'
                4. Use that password here (not your regular password)
                """)
        
        # SMS Alerts
        st.markdown("### 📱 SMS Alerts")
        col1, col2 = st.columns(2)
        
        with col1:
            sms_enabled = st.checkbox("Enable SMS Alerts", 
                                     value=st.session_state.alert_config['sms']['enabled'])
            
            if sms_enabled:
                phone_number = st.text_input("Phone Number", 
                                           value=st.session_state.alert_config['sms']['phone'],
                                           placeholder="+91XXXXXXXXXX")
        
        with col2:
            if sms_enabled:
                st.markdown("**📋 SMS Info:**")
                st.markdown("""
                - Uses free TextBelt service
                - Limited to 1 SMS per day (free tier)
                - For unlimited SMS, upgrade to paid service
                """)
        
        # Telegram Alerts (Bonus)
        st.markdown("### 📨 Telegram Alerts")
        col1, col2 = st.columns(2)
        
        with col1:
            telegram_enabled = st.checkbox("Enable Telegram Alerts", 
                                         value=st.session_state.alert_config['telegram']['enabled'])
            
            if telegram_enabled:
                bot_token = st.text_input("Telegram Bot Token", 
                                        value=st.session_state.alert_config['telegram']['bot_token'],
                                        help="Create bot via @BotFather on Telegram")
                
                chat_id = st.text_input("Chat ID", 
                                      value=st.session_state.alert_config['telegram']['chat_id'],
                                      help="Get from @userinfobot on Telegram")
        
        with col2:
            if telegram_enabled:
                st.markdown("**📋 Telegram Setup:**")
                st.markdown("""
                1. Message @BotFather on Telegram
                2. Create new bot with /newbot
                3. Get bot token
                4. Message @userinfobot to get your Chat ID
                """)
        
        # Alert Thresholds
        st.markdown("### ⚙️ Alert Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            min_confidence = st.slider("Minimum Confidence for Alerts", 50, 95, 75)
            max_alerts = st.slider("Max Alerts per Hour", 1, 20, 5)
        
        with col2:
            alert_types = st.multiselect(
                "Alert for:",
                ['BUY signals', 'SELL signals', 'Portfolio updates', 'Risk warnings'],
                default=['BUY signals', 'SELL signals']
            )
        
        # Save settings
        if st.button("💾 Save Alert Settings", type="primary"):
            st.session_state.alert_config.update({
                'email': {
                    'enabled': email_enabled,
                    'address': email_address if email_enabled else '',
                    'password': email_password if email_enabled else '',
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587
                },
                'sms': {
                    'enabled': sms_enabled,
                    'phone': phone_number if sms_enabled else '',
                    'service': 'textbelt'
                },
                'telegram': {
                    'enabled': telegram_enabled,
                    'bot_token': bot_token if telegram_enabled else '',
                    'chat_id': chat_id if telegram_enabled else ''
                },
                'thresholds': {
                    'min_confidence': min_confidence,
                    'max_alerts_per_hour': max_alerts
                },
                'alert_types': alert_types
            })
            
            st.success("✅ Alert settings saved!")
    
    def send_email_alert(self, subject: str, message: str):
        """Send email alert"""
        try:
            config = st.session_state.alert_config['email']
            
            if not config['enabled'] or not config['address']:
                return False
            
            msg = MimeMultipart()
            msg['From'] = config['address']
            msg['To'] = config['address']
            msg['Subject'] = subject
            
            msg.attach(MimeText(message, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['address'], config['password'])
            text = msg.as_string()
            server.sendmail(config['address'], config['address'], text)
            server.quit()
            
            return True
            
        except Exception as e:
            st.error(f"Email alert failed: {e}")
            return False
    
    def send_sms_alert(self, message: str):
        """Send SMS alert using TextBelt"""
        try:
            config = st.session_state.alert_config['sms']
            
            if not config['enabled'] or not config['phone']:
                return False
            
            url = 'https://textbelt.com/text'
            data = {
                'phone': config['phone'],
                'message': message,
                'key': 'textbelt'  # Free tier key
            }
            
            response = requests.post(url, data=data)
            result = response.json()
            
            return result.get('success', False)
            
        except Exception as e:
            st.error(f"SMS alert failed: {e}")
            return False
    
    def send_telegram_alert(self, message: str):
        """Send Telegram alert"""
        try:
            config = st.session_state.alert_config['telegram']
            
            if not config['enabled'] or not config['bot_token'] or not config['chat_id']:
                return False
            
            url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
            data = {
                'chat_id': config['chat_id'],
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data)
            return response.status_code == 200
            
        except Exception as e:
            st.error(f"Telegram alert failed: {e}")
            return False
    
    def send_trading_alert(self, signal_data: dict):
        """Send comprehensive trading alert"""
        
        # Check if alert meets threshold
        confidence = float(signal_data['confidence'])
        min_conf = st.session_state.alert_config['thresholds']['min_confidence']
        
        if confidence < min_conf:
            return
        
        # Create alert message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        subject = f"🤖 AI Trading Alert: {signal_data['action']} {signal_data['symbol']}"
        
        message = f"""
🤖 AI TRADING ALERT

Symbol: {signal_data['symbol']}
Action: {signal_data['action']}
Price: ₹{signal_data['price']:.2f}
Confidence: {confidence:.1f}%
Sector: {signal_data.get('sector', 'N/A')}

Technical Analysis:
- RSI: {signal_data.get('rsi', 'N/A')}
- MACD: {signal_data.get('macd', 'N/A')}
- Volume: {signal_data.get('volume_signal', 'N/A')}

Timestamp: {timestamp}

🎯 This is an automated alert from your AI Trading System
        """
        
        # Send via all enabled channels
        alert_sent = False
        
        if st.session_state.alert_config['email']['enabled']:
            if self.send_email_alert(subject, message):
                alert_sent = True
                st.success("📧 Email alert sent!")
        
        if st.session_state.alert_config['sms']['enabled']:
            sms_message = f"AI Alert: {signal_data['action']} {signal_data['symbol']} at ₹{signal_data['price']:.2f} ({confidence:.1f}% conf)"
            if self.send_sms_alert(sms_message):
                alert_sent = True
                st.success("📱 SMS alert sent!")
        
        if st.session_state.alert_config['telegram']['enabled']:
            if self.send_telegram_alert(message):
                alert_sent = True
                st.success("📨 Telegram alert sent!")
        
        return alert_sent

# Alert system instance
alert_system = TradingAlertSystem()
