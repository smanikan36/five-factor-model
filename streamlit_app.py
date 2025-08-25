import streamlit
import pandas
import statsmodels.api
import matplotlib.pyplot
import seaborn

streamlit.set_page_config(page_title="Five-Factor Model (Manual Input)", layout="wide")
streamlit.title("Fama-French Five-Factor Model Analyzer (Manual Input)")

streamlit.write("""
Enter your **stock returns** and the **five factor returns** (including risk-free rate) manually. 
The app will calculate excess returns, run the five-factor regression, and display factor loadings and alpha.
""")

streamlit.subheader("Step 1: Input Your Data")

num_periods = streamlit.number_input("Number of periods (e.g., days or months):", min_value=2, value=10)

streamlit.write("Enter your stock returns, risk-free rate, and the five factors (as decimals, e.g., 0.01 for 1%)")

stock_returns = []
risk_free_rates = []
market_excess_returns = []
size_factors = []
value_factors = []
profitability_factors = []
investment_factors = []

for i in range(int(num_periods)):
    streamlit.markdown(f"**Period {i+1}**")
    stock_returns.append(streamlit.number_input(f"Stock return {i+1}", key=f"r{i}"))
    risk_free_rates.append(streamlit.number_input(f"Risk-free rate {i+1}", key=f"rf{i}"))
    market_excess_returns.append(streamlit.number_input(f"Mkt-RF {i+1}", key=f"m{i}"))
    size_factors.append(streamlit.number_input(f"SMB {i+1}", key=f"smb{i}"))
    value_factors.append(streamlit.number_input(f"HML {i+1}", key=f"hml{i}"))
    profitability_factors.append(streamlit.number_input(f"RMW {i+1}", key=f"rmw{i}"))
    investment_factors.append(streamlit.number_input(f"CMA {i+1}", key=f"cma{i}"))

data_frame = pandas.DataFrame({
    "Stock": stock_returns,
    "RF": risk_free_rates,
    "Mkt-RF": market_excess_returns,
    "SMB": size_factors,
    "HML": value_factors,
    "RMW": profitability_factors,
    "CMA": investment_factors
})

data_frame["Excess"] = data_frame["Stock"] - data_frame["RF"]

regression_X = data_frame[["Mkt-RF","SMB","HML","RMW","CMA"]]
regression_X = statsmodels.api.add_constant(regression_X)
regression_y = data_frame["Excess"]

regression_model = statsmodels.api.OLS(regression_y, regression_X).fit()

streamlit.subheader("Five-Factor Regression Results")
streamlit.text(regression_model.summary())

streamlit.subheader("Factor Loadings (Betas)")
figure, axis = matplotlib.pyplot.subplots()
seaborn.barplot(x=regression_X.columns[1:], y=regression_model.params[1:], ax=axis)
axis.set_ylabel("Beta")
streamlit.pyplot(figure)

streamlit.subheader(f"Alpha (Intercept): {regression_model.params['const']:.4f} per period")
