"""Now we will backtest a trading strategy. We will take a position Wednesday and sell on Thursday
right after the EIA report is released.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

#Model Data
df = pd.read_csv("merged_data.csv", parse_dates=["date"])
freeze = pd.read_csv("weekly_production_tmin.csv", parse_dates=["date"])

df = pd.merge(df, freeze[["date", "freeze_intensity"]], on="date", how="inner")

TOTAL_CAPACITY = 4280 #capacity of US natural gas storage 
df["pct_full_lag1"] = df["storage_bcf"].shift(1) / TOTAL_CAPACITY
df["storage_change_lag1"] = df["storage_change"].shift(1)
df = df.dropna().reset_index(drop=True)

X = df[["HDD", "CDD", "pct_full_lag1", "storage_change_lag1", "freeze_intensity"]]
y = df["storage_change"]

#Walk forward validation
train_window = 104 #2 years of weekly data for training
predictions = []
for i in range(train_window, len(df)):
    X_train = X.iloc[:i]
    y_train = y.iloc[:i]
    X_pred = X.iloc[i:i+1]

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_pred)[0]
    predictions.append({"date":df["date"].iloc[i], "predicted": pred, "actual": y.iloc[i]})

pred_df = pd.DataFrame(predictions)
print(pred_df.head())
print(f"Prediction generated for {len(pred_df)} weeks.")

# Trading strategy, naive benchmark of five year average for that week 
df_bench = df[["date", "storage_change"]].copy()
df_bench["week_of_year"] = df_bench["date"].dt.isocalendar().week.astype(int)
df_bench["year"] = df_bench["date"].dt.year.astype(int)


benchmarks = []
for index, row in pred_df.iterrows():
    week = int(row["date"].isocalendar().week)
    year = int(row["date"].year)
    #we need to only use prior 5 years of the same week
    same_week = df_bench[(df_bench["week_of_year"] == week) & (df_bench["year"] < year) & (df_bench["year"] >= year-5)]["storage_change"]
    benchmarks.append(same_week.mean())
    


pred_df["benchmark"] = benchmarks
pred_df = pred_df.dropna().reset_index(drop=True) 
pred_df["benchmark_error"] = pred_df["benchmark"] - pred_df["actual"]
pred_df["benchmark_std"] = pred_df["benchmark_error"].rolling(window=52).std()
pred_df = pred_df.dropna().reset_index(drop=True)
#for our signal
pred_df["signal"]= pred_df["predicted"] - pred_df["benchmark"]
pred_df["position"] = 0
pred_df.loc[pred_df["signal"] < -1.5 *pred_df["benchmark_std"], "position"] = 1   # long
pred_df.loc[pred_df["signal"] > 1.5 * pred_df["benchmark_std"], "position"] = -1   # short


#Henry hib futures prioce

ng = yf.download("NG=F", start=pred_df["date"].min(), end=pred_df["date"].max(), interval="1d")
ng = ng[["Close"]].reset_index()
ng.columns = ["date", "price"]
ng["date"] = pd.to_datetime(ng["date"]).dt.tz_localize(None)
ng["date"] = ng["date"].dt.normalize()

#Since we are only trading on Wednesday and Thursday, we need to filter only those
wed_price = ng[ng["date"].dt.weekday == 2].copy() # Wednesday
thu_price = ng[ng["date"].dt.weekday == 3].copy() # Thursday

wed_price = wed_price.rename(columns={"price": "wed_price", "date":"wed_date"})
thu_price = thu_price.rename(columns={"price": "thu_price", "date":"thu_date"})

#match wednesday to thursday
wed_price["thu_date"] = wed_price["wed_date"] + pd.Timedelta(days=1)
prices = pd.merge(wed_price, thu_price, on="thu_date", how="inner") 

print(prices.head())


#Time to merge prices with predictions
pred_df["thu_date"] = pred_df["date"] - pd.Timedelta(days=1)
backtest = pd.merge(pred_df, prices, on="thu_date", how="inner")
backtest = backtest[(backtest["position"] != 0) & backtest["position"].notna()].reset_index(drop=True)

#P and L calculation
trans_cost = 0.01
contract_size = 10000

backtest["price_change"] = backtest["thu_price"] - backtest["wed_price"]
backtest["pnl"] = ((backtest["position"] * backtest["price_change"]) - trans_cost)* contract_size
backtest["cumulative_pnl"] = backtest["pnl"].cumsum()

print(f"\nTotal P&L: ${backtest['pnl'].sum():,.0f}")
print(f"Win rate: {(backtest['pnl'] > 0).mean():.1%}")
print(f"Average P&L per trade: ${backtest['pnl'].mean():,.0f}")
print(f"Sharpe (annualized): {backtest['pnl'].mean() / backtest['pnl'].std() * (52**0.5):.2f}")

pred_df.to_csv("predictions.csv", index=False)

short_only = backtest[backtest["position"] == -1].copy()
print(f"Short trades: {len(short_only)}")
print(f"Short P&L: ${short_only['pnl'].sum():,.0f}")
print(f"Short win rate: {(short_only['pnl'] > 0).mean():.1%}")
print(f"Sharpe: {short_only['pnl'].mean() / short_only['pnl'].std() * (52**0.5):.2f}")
short_only["cumulative_pnl"] = short_only["pnl"].cumsum()
plt.figure(figsize=(12,5))
plt.plot(short_only["thu_date"], short_only["cumulative_pnl"])
plt.title("Short-Only Strategy: Cumulative P&L")
plt.xlabel("Date")
plt.ylabel("Cumulative P&L ($)")
plt.axhline(0, color="red", linestyle="--")
plt.tight_layout()

import scipy.stats as stats

# binomial test: is 65.8% win rate statistically different from 50% (random)?
n_trades = 38
n_wins = int(0.658 * 38)  # ~25 wins
p_value = stats.binomtest(n_wins, n_trades, p=0.5, alternative='greater').pvalue
print(f"Wins: {n_wins}/{n_trades}")
print(f"P-value: {p_value:.4f}")
print(f"Statistically significant at 5%: {p_value < 0.05}")
