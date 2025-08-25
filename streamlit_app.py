import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Five-Factor Model Analyzer", layout="wide")
st.title("Fama-French Five-Factor Model Analyzer")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter stock ticker", "AAPL")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("2025-01-01"))

# Load stock data
@st.cache_data
def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)['Adj Close']
    returns = stock.pct_change().dropna()
    return returns

returns = get_stock_data(ticker, start_date, end_date)

st.subheader(f"{ticker} Daily Returns")
st.line_chart(returns)

# Load Fama-French 5 Factor data
@st.cache_data
def get_ff5_data():
    ff5 = pd.read_csv(
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        skiprows=3, skipfooter=3, engine='python'
    )
    ff5.index = pd.to_datetime(ff5['Date'], format='%Y%m%d')
    ff5 = ff5[['Mkt-RF','SMB','HML','RMW','CMA','RF']].astype(float)/100
    return ff5

ff5 = get_ff5_data()
ff5 = ff5.loc[start_date:end_date]

# Merge stock returns with factor data
data = pd.merge(returns, ff5, left_index=True, right_index=True)
data['Excess'] = data[ticker] - data['RF']

# Regression
X = data[['Mkt-RF','SMB','HML','RMW','CMA']]
X = sm.add_constant(X)
y = data['Excess']

model = sm.OLS(y, X).fit()

st.subheader("Five-Factor Regression Results")
st.text(model.summary())

# Plot factor sensitivities
st.subheader("Factor Loadings (Betas)")
fig, ax = plt.subplots()
sns.barplot(x=X.columns[1:], y=model.params[1:])
ax.set_ylabel("Beta")
st.pyplot(fig)

# Optional: Show alpha
st.subheader(f"Alpha: {model.params['const']:.4f} per day")
