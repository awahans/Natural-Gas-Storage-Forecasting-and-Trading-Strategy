#checking correlation for freeze and storage change
import pandas as pd
df_model = pd.read_csv("merged_data.csv", parse_dates=["date"])
df_freeze = pd.read_csv("weekly_production_tmin.csv", parse_dates=["date"])

df_model = pd.merge(df_model, df_freeze, on="date", how="inner")

for threshold in [24,15,10,0]:
    df_model[f"freeze_below_{threshold}"] = (df_model["prod_min"] < threshold).astype(int)
    correlation = df_model["storage_change"].corr(df_model[f"freeze_below_{threshold}"])
    count = df_model[f"freeze_below_{threshold}"].sum()
    print(f"Correlation between storage change and freeze below {threshold} F: {correlation:.3f}")


#Strong correlation found between the threshold of 15 F and storage change, will update in the production weather model