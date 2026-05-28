import yfinance as yf
import pandas as pd
import numpy as np

# pull 1-hour NG futures data (last 2 years)
ng_intraday = yf.download("NG=F", period="2y", interval="1h")
ng_intraday = ng_intraday[["Close"]].reset_index()
ng_intraday.columns = ["datetime", "price"]
ng_intraday["datetime"] = pd.to_datetime(ng_intraday["datetime"]).dt.tz_localize(None)

ng_intraday["date"] = ng_intraday["datetime"].dt.date
ng_intraday["weekday"] = ng_intraday["datetime"].dt.weekday
ng_intraday["hour"] = ng_intraday["datetime"].dt.hour
# Wednesday 11PM entry (last price before EIA release day)
wed_entry = ng_intraday[
    (ng_intraday["weekday"] == 2) & 
    (ng_intraday["hour"] == 23)
][["datetime", "price"]].rename(columns={"datetime": "wed_dt", "price": "entry_price"})

# add thursday date for merging
wed_entry["thu_date"] = pd.to_datetime(wed_entry["wed_dt"].dt.date) + pd.Timedelta(days=1)

# Thursday 10AM exit (closest to 10:30 release)
thu_exit = ng_intraday[
    (ng_intraday["weekday"] == 3) & 
    (ng_intraday["hour"] == 10)
][["datetime", "price"]].rename(columns={"datetime": "thu_dt", "price": "exit_price"})

thu_exit["thu_date"] = pd.to_datetime(thu_exit["thu_dt"].dt.date)

# merge
prices_intraday = pd.merge(wed_entry, thu_exit, on="thu_date", how="inner")
print(f"Price pairs: {len(prices_intraday)}")
print(prices_intraday.head())

pred_df = pd.read_csv("predictions.csv", parse_dates=["date"])
# merge with pred_df — match on thu_date
pred_df["thu_date"] = pd.to_datetime(pred_df["date"]) - pd.Timedelta(days=1)

backtest_intraday = pd.merge(pred_df, prices_intraday, on="thu_date", how="inner")
backtest_intraday = backtest_intraday[
    backtest_intraday["position"].notna() & 
    (backtest_intraday["position"] != 0)
].reset_index(drop=True)

print(f"Tradeable weeks: {len(backtest_intraday)}")

# P&L
trans_cost = 0.01
contract_size = 10000

backtest_intraday["price_change"] = backtest_intraday["exit_price"] - backtest_intraday["entry_price"]
backtest_intraday["pnl"] = (backtest_intraday["position"] * backtest_intraday["price_change"] - trans_cost) * contract_size
backtest_intraday["cumulative_pnl"] = backtest_intraday["pnl"].cumsum()

print(f"Total P&L: ${backtest_intraday['pnl'].sum():,.0f}")
print(f"Win rate: {(backtest_intraday['pnl'] > 0).mean():.1%}")
print(f"Trades: {len(backtest_intraday)}")
print(f"Sharpe: {backtest_intraday['pnl'].mean() / backtest_intraday['pnl'].std() * (52**0.5):.2f}")

# direction check
backtest_intraday["actual_surprise"] = backtest_intraday["actual"] - backtest_intraday["benchmark"]
backtest_intraday["direction_correct"] = (
    ((backtest_intraday["position"] == 1) & (backtest_intraday["actual_surprise"] < 0)) |
    ((backtest_intraday["position"] == -1) & (backtest_intraday["actual_surprise"] > 0))
)

short_only = backtest_intraday[backtest_intraday["position"] == -1].copy()
print(f"Short trades: {len(short_only)}")
print(f"Short P&L: ${short_only['pnl'].sum():,.0f}")
print(f"Short win rate: {(short_only['pnl'] > 0).mean():.1%}")
print(f"Sharpe(short): {short_only['pnl'].mean() / short_only['pnl'].std() * (52**0.5):.2f}")