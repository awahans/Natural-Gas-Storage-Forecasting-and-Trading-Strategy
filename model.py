#Import relevant libraries and packages
import pandas as pd
import numpy as np
from data import get_storage, storage_change
from weather import get_weekly_degree_days 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

#Load the storage data
storage = get_storage()
storage = storage_change(storage)

weather = pd.read_csv("weekly_degree_days.csv", parse_dates=["date"])


"""EIA gathers storage data every Friday at 9 AM CST, and as a result of which, for weather data we are doing weekly
resampling ending on Thursday. So, to align weather data with EIA data, we shift weather data by a day
to Friday.
"""
weather["date"] = weather["date"] + pd.Timedelta(days=1)
df = pd.merge(storage, weather, on='date', how = 'inner')
df = df.dropna().reset_index(drop=True) # Drop any rows with missing values and reset the index


#Plot the raw data to see relationship between HDD and storage change
import matplotlib.pyplot as plt
plt.scatter(df["storage_change"], df["HDD"])
plt.xlabel("Weekly HDD")
plt.ylabel("Weekly Change in Storage (BCF)")
plt.title("Relationship between Weekly HDD and Change in Storage")
#Linear trend is visible

#Feature Engineering 
Total_capacity = 4280 #Total capacity of US natural gas storage in BCF
df["pct_full_lag1"] = df["storage_bcf"].shift(1)/Total_capacity
df["storage_change_lag1"] = df["storage_change"].shift(1)
df = df.dropna().reset_index(drop=True) # Drop any rows with missing values and reset the index
df.to_csv("model_data.csv", index=False)

#MODEL FUNCTION
def run_model(X,y, model, label, n_splits = 5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []
    r2_scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_scores.append(mse)
        r2_scores.append(r2)
    mse_mean = np.mean(mse_scores)
    r2_mean = np.mean(r2_scores)
    label_name = label
    return (label_name, mse_mean, r2_mean)
        
#MODEL Comparision
results = {}
y= df["storage_change"]
#Baseline: HDD and CDD only 
X1 = df[["HDD", "CDD"]]
model1 = LinearRegression()
results["HDD_CDD"] = run_model(X1,y, model1, "HDD_CDD")

#Model 2: Adding lagged storage remaining
X2 = df[["HDD", "CDD", "pct_full_lag1"]]
model2 = LinearRegression()
results["HDD_CDD_PCT_FULL"] = run_model(X2,y, model2, "HDD_CDD_PCT_FULL")

#Model 3: Adding lagged storage change
X3 = df[["HDD", "CDD", "pct_full_lag1", "storage_change_lag1"]]
model3 = LinearRegression()
results["HDD_CDD_PCT_FULL_LAG"] = run_model(X3,y, model3, "HDD_CDD_PCT_FULL_LAG")

"""Noting, our assumption of constant production might be false on extreme cold resulting in freezing of wellheads
, two features were tested- binary variable "freeze_event" and continuous variable "freeze_intensity" to capture the effect of freezing on storage change."""
#Freezing intensity is weighted based on production levels 
#For production freeze data
#Weekly_production_tmin.csv created from production_weather.py
freeze = pd.read_csv("weekly_production_tmin.csv", parse_dates=["date"])
df["date"] = pd.to_datetime(df["date"])
df = pd.merge(df, freeze[["date", "freeze_event", "freeze_intensity"]], on="date", how="inner")
df = df.dropna().reset_index(drop=True) # Drop any rows with missing values and reset the index

#freeze_severe: flags only the most extreme freeze weeks (top 5%, above the 95th percentile),
#distinct from freeze_event which fires on any freeze, mild or severe. Lets the model apply a
#fixed kink on top of the freeze_intensity slope just for catastrophic weeks, without the
#leverage blowup that squaring freeze_intensity caused
"""Severe freeeze weeks are weeks that are in the top 5 percentile"""
freeze_severe_threshold = df["freeze_intensity"].quantile(0.95)
df["freeze_severe"] = (df["freeze_intensity"] >= freeze_severe_threshold).astype(int)

# #Model 4: Adding freeze event
X4 = df[["HDD", "CDD", "pct_full_lag1","storage_change_lag1", "freeze_event"]]
model4 = LinearRegression()
results["HDD_CDD_PCT_FULL_LAG_FREEZE_event"] = run_model(X4,y, model4, "HDD_CDD_PCT_FULL_LAG_FREEZE_event")

#Model 5: Using freeze intensity instead of just event
X5 = df[["HDD", "CDD", "pct_full_lag1", "storage_change_lag1", "freeze_intensity"]]
model5 = LinearRegression()
results["HDD_CDD_PCT_FULL_LAG_FREEZE_intensity"] = run_model(X5,y, model5, "HDD_CDD_PCT_FULL_LAG_FREEZE_intensity")



"""New feature holiday_week was added to capture effect of US federal holidays.
Industrial demand drops and residential demand rises during holidays, which is not captured by weather features"""
from pandas.tseries.holiday import USFederalHolidayCalendar
holiday_dates = USFederalHolidayCalendar().holidays(start=df["date"].min() - pd.Timedelta(days=7),
                                                      end=df["date"].max())
#EIA week ends Thursday, reported Friday - flag if a holiday falls in the preceding Fri-Thu span
df["holiday_week"] = df["date"].apply(
    lambda d: int(any((d - pd.Timedelta(days=6) <= h <= d) for h in holiday_dates)))


#XGBoost with all features
X6 = df[["HDD", "CDD", "pct_full_lag1", "storage_change_lag1", "freeze_intensity"]]
model6 = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
results["XGB_ALL_FEATURES"] = run_model(X6,y, model6, "XGB_ALL_FEATURES")

#Model 7: Adding freeze_severe dummy (top 5% most extreme freeze weeks)
X7 = df[["HDD", "CDD", "pct_full_lag1", "storage_change_lag1", "freeze_intensity", "freeze_severe"]]
model7 = LinearRegression()
results["HDD_CDD_PCT_FULL_LAG_FREEZE_intensity_SEVERE"] = run_model(X7,y, model7, "HDD_CDD_PCT_FULL_LAG_FREEZE_intensity_SEVERE")

#Model 8: Residual boosting - linear model5 captures the seasonal signal, then XGBoost is trained
#on model5's residuals to pick up the nonlinear leftover (e.g. the under-predicted freeze weeks).
#Final prediction = linear prediction + XGB residual correction. Both stages are fit inside each
#CV fold on training data only, so the residual model never sees the test fold.
def run_residual_model(X, y, xgb_params, label, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores, r2_scores = [], []
    oof_pred = pd.Series(np.nan, index=y.index)  # out-of-fold predictions for honest plotting
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        lin = LinearRegression().fit(X_train, y_train)
        train_resid = y_train - lin.predict(X_train)
        xgb = XGBRegressor(objective="reg:squarederror", **xgb_params).fit(X_train, train_resid)
        y_pred = lin.predict(X_test) + xgb.predict(X_test)
        oof_pred.iloc[test_index] = y_pred
        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
    return (label, np.mean(mse_scores), np.mean(r2_scores)), oof_pred

xgb_resid_params = dict(n_estimators=200, learning_rate=0.05, max_depth=3,
                        subsample=0.8, min_child_weight=5, random_state=42)
results["LINEAR_plus_XGB_RESIDUALS"], df["oof_pred_hybrid"] = run_residual_model(
    X5, y, xgb_resid_params, "LINEAR_plus_XGB_RESIDUALS")

#Model 9: HDD/CDD hinge (threshold) features - hinge(x, tau) = max(x - tau, 0): zero below tau,
#then grows linearly above it. Tests whether extreme-cold or extreme-hot weeks push storage change
#beyond what a straight line through all of HDD/CDD would predict
def run_hinge_model(df, base_cols, y, hdd_q=0.75, cdd_q=0.75, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores, r2_scores = [], []
    oof_pred = pd.Series(np.nan, index=y.index)
    for train_index, test_index in tscv.split(df):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]
        tau_hdd = train_df["HDD"].quantile(hdd_q)
        tau_cdd = train_df["CDD"].quantile(cdd_q)

        X_train = train_df[base_cols].copy()
        X_train["hdd_hinge"] = np.maximum(train_df["HDD"] - tau_hdd, 0)
        X_train["cdd_hinge"] = np.maximum(train_df["CDD"] - tau_cdd, 0)

        X_test = test_df[base_cols].copy()
        X_test["hdd_hinge"] = np.maximum(test_df["HDD"] - tau_hdd, 0)
        X_test["cdd_hinge"] = np.maximum(test_df["CDD"] - tau_cdd, 0)

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        oof_pred.iloc[test_index] = y_pred
        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
    return (np.mean(mse_scores), np.mean(r2_scores)), oof_pred

base_cols_hinge = ["HDD", "CDD", "pct_full_lag1", "storage_change_lag1", "freeze_intensity"]
(hinge_mse, hinge_r2), df["oof_pred_hinge"] = run_hinge_model(df, base_cols_hinge, y)
results["HDD_CDD_PCT_FULL_LAG_FREEZE_intensity_HINGE"] = (
    "HDD_CDD_PCT_FULL_LAG_FREEZE_intensity_HINGE", hinge_mse, hinge_r2)

#Model 10: HDD/CDD hinge features + XGB residual boosting combined - the hinge features gave the
#linear stage a real (not spurious) improvement specifically on severe-freeze weeks, while the XGB
#residual stage tightened the bulk of ordinary weeks. Same per-fold tau discipline as Model 9, and
#the residual stage still only ever sees training-fold residuals, same as Model 8.
def run_hinge_residual_model(df, base_cols, y, xgb_params, hdd_q=0.75, cdd_q=0.75, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores, r2_scores = [], []
    oof_pred = pd.Series(np.nan, index=y.index)
    for train_index, test_index in tscv.split(df):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]
        tau_hdd = train_df["HDD"].quantile(hdd_q)
        tau_cdd = train_df["CDD"].quantile(cdd_q)

        X_train = train_df[base_cols].copy()
        X_train["hdd_hinge"] = np.maximum(train_df["HDD"] - tau_hdd, 0)
        X_train["cdd_hinge"] = np.maximum(train_df["CDD"] - tau_cdd, 0)

        X_test = test_df[base_cols].copy()
        X_test["hdd_hinge"] = np.maximum(test_df["HDD"] - tau_hdd, 0)
        X_test["cdd_hinge"] = np.maximum(test_df["CDD"] - tau_cdd, 0)

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        lin = LinearRegression().fit(X_train, y_train)
        train_resid = y_train - lin.predict(X_train)
        xgb = XGBRegressor(objective="reg:squarederror", **xgb_params).fit(X_train, train_resid)
        y_pred = lin.predict(X_test) + xgb.predict(X_test)
        oof_pred.iloc[test_index] = y_pred
        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
    return (np.mean(mse_scores), np.mean(r2_scores)), oof_pred

(hinge_resid_mse, hinge_resid_r2), df["oof_pred_hinge_hybrid"] = run_hinge_residual_model(
    df, base_cols_hinge, y, xgb_resid_params)
results["HINGE_plus_XGB_RESIDUALS"] = ("HINGE_plus_XGB_RESIDUALS", hinge_resid_mse, hinge_resid_r2)

#Model 11: champion (hinge + XGB residual) plus holiday_week - honest calendar feature, no
#leakage risk since holidays are fixed calendar facts, not something computed from the target
base_cols_holiday = base_cols_hinge + ["holiday_week"]
(holiday_mse, holiday_r2), df["oof_pred_hinge_hybrid_holiday"] = run_hinge_residual_model(
    df, base_cols_holiday, y, xgb_resid_params)
results["HINGE_plus_XGB_RESIDUALS_plus_HOLIDAY"] = (
    "HINGE_plus_XGB_RESIDUALS_plus_HOLIDAY", holiday_mse, holiday_r2)
print(f"\nHoliday_week feature: {int(df['holiday_week'].sum())} of {len(df)} weeks flagged")
print(f"Champion without holiday: {hinge_resid_mse:.1f} MSE | with holiday: {holiday_mse:.1f} MSE")
"""Stress test to see how sensitive is the hinge model to tau threshold choice"""
stress_rows = []
for q in [0.5, 0.6, 0.65, 0.75, 0.85, 0.95]:
    (mse_h, r2_h), oof_h = run_hinge_model(df, base_cols_hinge, y, hdd_q=q, cdd_q=q)
    (mse_hr, r2_hr), oof_hr = run_hinge_residual_model(df, base_cols_hinge, y, xgb_resid_params, hdd_q=q, cdd_q=q)

    tmp = df[["storage_change", "freeze_severe"]].copy()
    tmp["oof_h"], tmp["oof_hr"] = oof_h, oof_hr
    tmp = tmp.dropna()
    sev = tmp["freeze_severe"] == 1

    stress_rows.append({
        "tau_quantile": q,
        "hdd_tau_full_data": df["HDD"].quantile(q).round(1),   # reference only, not used in CV
        "cdd_tau_full_data": df["CDD"].quantile(q).round(1),
        "hinge_MSE": round(mse_h, 1), "hinge_R2": round(r2_h, 4),
        "hinge_MAE_severe": round((tmp.loc[sev,"storage_change"] - tmp.loc[sev,"oof_h"]).abs().mean(), 1),
        "hinge+resid_MSE": round(mse_hr, 1), "hinge+resid_R2": round(r2_hr, 4),
        "hinge+resid_MAE_severe": round((tmp.loc[sev,"storage_change"] - tmp.loc[sev,"oof_hr"]).abs().mean(), 1),
    })

print("\nHinge threshold stress test (tau quantile swept for hdd_hinge and cdd_hinge together):")
print(pd.DataFrame(stress_rows).to_string(index=False))

#Summarize results
results_df = pd.DataFrame(results).T.reset_index().rename(columns={0: "Model", 1: "Average MSE", 2: "Average R-squared"})
print(results_df.drop(columns=["index"]).to_string(index=False))

#Out-of-fold comparison on the same CV test folds: model5 alone vs linear + XGB residuals,
#split by severe-freeze weeks vs everything else to see where the residual model actually helps
df["oof_pred_linear"] = np.nan
for train_index, test_index in TimeSeriesSplit(n_splits=5).split(X5):
    lin = LinearRegression().fit(X5.iloc[train_index], y.iloc[train_index])
    df.loc[df.index[test_index], "oof_pred_linear"] = lin.predict(X5.iloc[test_index])
oof = df.dropna(subset=["oof_pred_linear", "oof_pred_hybrid", "oof_pred_hinge", "oof_pred_hinge_hybrid"])
severe = oof["freeze_severe"] == 1
print(f"\nOut-of-fold MAE (BCF) over {len(oof)} test weeks, {severe.sum()} of them severe-freeze:")
for name, col in [("Linear (model5)        ", "oof_pred_linear"),
                   ("Linear + XGB residual  ", "oof_pred_hybrid"),
                   ("Linear + HDD/CDD hinge ", "oof_pred_hinge"),
                   ("Hinge + XGB residual   ", "oof_pred_hinge_hybrid")]:
    err = (oof["storage_change"] - oof[col]).abs()
    print(f"  {name}: all={err.mean():.1f} | severe freeze weeks={err[severe].mean():.1f} | other weeks={err[~severe].mean():.1f}")

#Actual vs Predicted plot for the best model (Linear + XGB residual correction, Model 8)
import matplotlib.pyplot as plt
df["predicted"] = df["oof_pred_hinge_hybrid"]
df["residual"] = df["storage_change"] - df["predicted"]  # actual - predicted: negative = model under-predicted the draw
df = df.dropna(subset=["predicted"]).reset_index(drop=True)  # drop earliest weeks with no out-of-fold prediction (first CV fold's training set)

def plot_actual_vs_predicted(data, title_suffix, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})

    ax1.scatter(data["date"], data["storage_change"], label="Actual", alpha=0.7,
                s=55, color='#1f77b4', edgecolors='#0d3d61', linewidth=0.5)
    ax1.scatter(data["date"], data["predicted"], label="Predicted", alpha=0.7,
                s=55, color='#ff7f0e', edgecolors='#b35400', linewidth=0.5)
    ax1.set_ylabel("Storage Change (BCF)", fontsize=12, fontweight='bold')
    ax1.set_title(f"Actual vs Predicted Weekly Storage Change - Linear + XGB Residuals ({title_suffix})", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)

    colors = np.where(data["residual"] < 0, '#2ca02c', '#d62728')
    ax2.bar(data["date"], data["residual"], width=5, color=colors, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel("Actual - Predicted\n(BCF)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color='#2ca02c', label='Under-predicted the draw'),
                         Patch(color='#d62728', label='Over-predicted the draw')],
               fontsize=9, loc='best')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    mae = np.mean(np.abs(data["residual"]))
    rmse = np.sqrt(np.mean(data["residual"]**2))
    print(f"\n{title_suffix}: saved to {filename}")
    print(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"  MAE: {mae:.1f} BCF | RMSE: {rmse:.1f} BCF")

#Most recent 2 years (last 104 weeks)
recent_2yr = df.tail(104).reset_index(drop=True)
plot_actual_vs_predicted(recent_2yr, "Most Recent 2 Years", "actual_vs_predicted_recent_2yr.png")

#The 2 years before that (weeks 105-208 from the end)
prior_2yr = df.iloc[-208:-104].reset_index(drop=True)
plot_actual_vs_predicted(prior_2yr, "Prior 2 Years", "actual_vs_predicted_prior_2yr.png")

print("\nDone.")
print(df.tail(20))

# Split residuals into the two regimes
ordinary_residuals = df.loc[df["freeze_severe"] == 0, "residual"]
severe_residuals   = df.loc[df["freeze_severe"] == 1, "residual"]

# 10th/90th percentile of each group's past residuals -> an 80% band
ordinary_lo, ordinary_hi = ordinary_residuals.quantile([0.10, 0.90])
severe_lo,   severe_hi   = severe_residuals.quantile([0.10, 0.90])

print(f"Ordinary weeks:  point forecast {ordinary_lo:+.1f} to {ordinary_hi:+.1f} BCF")
print(f"Severe weeks:    point forecast {severe_lo:+.1f} to {severe_hi:+.1f} BCF")

# Attach a band to every row, picking the right width based on that row's regime
df["pi_lo"] = df["predicted"] + np.where(df["freeze_severe"] == 1, severe_lo, ordinary_lo)
df["pi_hi"] = df["predicted"] + np.where(df["freeze_severe"] == 1, severe_hi, ordinary_hi)
