# 📊 Nova Options Scanner

The **Nova Options Scanner** is a custom Streamlit application designed to help quickly identify high-probability, defined-risk option trades. It’s built to follow Nova’s trading rules:  
- Probability of Profit (POP) focus  
- Defined max loss ($200–$300 typical target)  
- Narrow vertical spreads and iron condors  
- Clear, easy-to-read tables  

---

## 🚀 Features
- **Ticker Input** – Enter any stock ticker and fetch live option chains via Yahoo Finance.  
- **Expiration Selector** – Choose the exact expiration date.  
- **Custom Filters** – Control spread width, max loss, and minimum POP.  
- **Strategy Modes** – Bull Put, Bear Call, or Iron Condor.  
- **Auto Analysis** – Nova’s AI automatically summarizes scan results and gives quick takeaways.  
- **Always-Visible Chat** – Talk directly with Nova for trade guidance.  

---

## 🛠️ Tech Stack
- **Python 3.11+**
- [Streamlit](https://streamlit.io/) – frontend framework
- [yfinance](https://pypi.org/project/yfinance/) – live option chain data
- [OpenAI](https://platform.openai.com/) – AI trade summaries
- **Custom Nova Modules** – `nova_utils.py`, `nova_math.py`, `nova_rules.py`

---

## 📦 Installation & Setup
Clone the repo:
```bash
git clone https://github.com/Novastradingcompany/nova-options-scanner.git
cd nova-options-scanner
