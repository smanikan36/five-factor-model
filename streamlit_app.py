import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Five-Factor Model (Manual Input)", layout="wide")
st.title("Fama-French Five-Factor Model Analyzer (Manual Input)")

st.write("""
Enter your **stock returns** and the **five factor returns** (including risk-free rate) manually. 
The app will calculate excess returns, run the five-factor regression, and display factor loadings and alpha.
""")

# --- User Input for Returns ---
st.subheader("Step 1: Input Your Data")

num_periods = st.number_input("Number of periods (e.g., days or months):", min_value=2, value=10)

st.write("Enter your stock returns, risk-free rate, and the five factors (as decimals, e.g., 0.01 for 1%)")

# Initialize empty lists
stock_returns = []
rf = []
mkt_rf = []
smb = []
hml = []
rmw = []
cma = []

for i in range(int(num_periods)):
    st.markdown(f"**Period {i+1}**")
    stock_returns.append(st.number_input(f"Stock return {i+1}", key=f"r{i}"))
    rf.append(st.number_input(f"Risk-free rate {i+1}", key=f"rf{i}"))
    mkt_rf.append(st.number_input(f"Mkt-RF {i+1}", key=f"m{i}"))
    smb.append(st.number_input(f"SMB {i+1}", key=f"smb{i}"))
    hml.append(st.number_input(f"HML {i+1}", key=f"hml{i}"))
    rmw.append(st.number_input(f"RMW {i+1}", key=f"rmw{i}"))
    cma.append(st.number_input(f"CMA {i+1}", key=f"cma{i}"))

# --- Build DataFrame ---
data = pd.DataFrame({
    "Stock": stock_returns,
    "RF": rf,
    "Mkt-RF": mkt_rf,
    "SMB": smb,
    "HML": hml,
    "RMW": rmw,
    "CMA": cma
})

data["Excess"] = data["Stock"] - data["RF"]

# --- Run Regression ---
X = data[["Mkt-RF","SMB","HML","RMW","CMA"]]
X = sm.add_constant(X)
y = data["Excess"]

model = sm.OLS(y, X).fit()

st.subheader("Five-Factor Regression Results")
st.text(model.summary())

# --- Plot Factor Loadings ---
st.subheader("Factor Loadings (Betas)")
fig, ax = plt.subplots()
sns.barplot(x=X.columns[1:], y=model.params[1:])
ax.set_ylabel("Beta")
st.pyplot(fig)

st.subheader(f"Alpha (Intercept): {model.params['const']:.4f} per period")
