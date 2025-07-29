import subprocess
import sys

# Install everything in one shot
packages = [
    'yfinance==0.2.28',
    'streamlit==1.28.1', 
    'plotly==5.17.0',
    'pandas==2.1.4',
    'scikit-learn==1.3.2',
    'ta-lib-binary==0.4.28'
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
print("✅ All packages installed! Ready to build!")
