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


"""EIA data is always gathered on Friday at 9 AM, and such for weather data we did weekly
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

#After working with these four features, I wanted to add a features that also traces production
#Freezing event is a binary variable that indicates if any of the natural gas basins are not producing gas due to low T
#Freezing intensity is rather a continuous variable that tracks the severity and scale of freezing
#Freezing intensity is weighted based on production levels 
#For production freeze data
#Weekly_production_tmin.csv created from production_weather.py
freeze = pd.read_csv("weekly_production_tmin.csv", parse_dates=["date"])
df["date"] = pd.to_datetime(df["date"])
df = pd.merge(df, freeze[["date", "freeze_event", "freeze_intensity"]], on="date", how="inner")
df = df.dropna().reset_index(drop=True) # Drop any rows with missing values and reset the index

# #Model 4: Adding freeze event
X4 = df[["HDD", "CDD", "pct_full_lag1","storage_change_lag1", "freeze_event"]]
model4 = LinearRegression()
results["HDD_CDD_PCT_FULL_LAG_FREEZE_event"] = run_model(X4,y, model4, "HDD_CDD_PCT_FULL_LAG_FREEZE_event")

#Model 5: Using freeze intensity instead of just event
X5 = df[["HDD", "CDD", "pct_full_lag1", "storage_change_lag1", "freeze_intensity"]]
model5 = LinearRegression()
results["HDD_CDD_PCT_FULL_LAG_FREEZE_intensity"] = run_model(X5,y, model5, "HDD_CDD_PCT_FULL_LAG_FREEZE_intensity")

#XGBoost with all features
X6 = df[["HDD", "CDD", "pct_full_lag1", "storage_change_lag1", "freeze_intensity"]]
model6 = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=3)
results["XGB_ALL_FEATURES"] = run_model(X6,y, model6, "XGB_ALL_FEATURES")

#Summarize results
results_df = pd.DataFrame(results).T.reset_index().rename(columns={0: "Model", 1: "Average MSE", 2: "Average R-squared"})
print(results_df.drop(columns=["index"]).to_string(index=False))

"""Although XGB Boost for all features has the best MSE, and R sqaured, it is not significant different
from linear regression model with HDD, CSS, lagged percentage full and lagged storage change.
"""
#Actual vs Predicted plt.plot for the best model
import matplotlib.pyplot as plt

best_model = model5
best_model.fit(X5, y)
df["predicted"] = best_model.predict(X5)
plt.figure(figsize=(12,6))
plt.plot(df["date"], df["storage_change"], label="Actual", alpha=0.7)
plt.plot(df["date"], df["predicted"], label="Predicted", alpha=0.7)
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.xlabel("Date")
plt.ylabel("Storage Change (Bcf)")
plt.title("Actual vs Predicted Weekly Natural Gas Storage Change")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

print("\nDone. Chart saved to actual_vs_predicted.png")
