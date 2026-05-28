import requests
import pandas as pd

from dotenv import load_dotenv
import os
from features import compute_degree_days
load_dotenv()
import time
import numpy as np

TOKEN = os.getenv("NOAA_TOKEN")

PRODUCTION_STATIONS = {
    "Midland_TX":    "GHCND:USW00023023",  # Permian Basin
    "Pittsburgh_PA": "GHCND:USW00094823",  # Marcellus/Utica
    "Shreveport_LA": "GHCND:USW00013957",  # Haynesville
    "Casper_WY":     "GHCND:USW00024089",  # DJ Basin
}

def get_production_weather_data(station_id, start="2010-01-01", end="2026-05-01"):
    url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": TOKEN} #NOAA API requires a token in the header
    all_data = []

    # loop year by year to stay within NOAA's 1-year limit
    years = range(2010, 2027)
    for year in years:
        offset = 1 #NOAA API uses 1-based indexing for offset
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"
        
        while True:
            params = {
                "datasetid": "GHCND",
                "datatypeid": ["TMIN"],
                "stationid": station_id,
                "startdate": year_start,
                "enddate": year_end,
                "units": "standard",
                "limit": 1000,
                "offset": offset,
            }
            r = requests.get(url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json().get("results", []) #Dig into the JSON response to get the actual data else return empty list
            if not data:
                break
            all_data.extend(data)
            offset += 1000 #Since our limit is 1000, we need to increment the offset by 1000 to get the next batch of data
            time.sleep(0.2)  # be nice to the API
        
        print(f"  {year} done ({len(all_data)} rows so far)")

    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.pivot_table(index="date", columns="datatype", values="value").reset_index() # Pivot the data to have TMAX and TMIN as columns instead of rows
    df["Tmin"] = df["TMIN"]
    return df[["date", "Tmin"]]

#For freeze intensity, we will work on the production weights
weights = {
    "Midland_TX": 0.35,    
    "Pittsburgh_PA": 0.47,  
    "Shreveport_LA": 0.15,  
    "Casper_WY": 0.03,     
}

def get_all_production_weather_data():
    dfs = []
    for name, station_id in PRODUCTION_STATIONS.items():
        print(f"Fetching data for {name}...")
        df = get_production_weather_data(station_id)
        df = df.rename(columns={"Tmin": f"Tmin_{name}"}) # Rename the Tmin column to include the station name for clarity
        dfs.append(df.set_index("date"))

    combined_dfs = pd.concat(dfs, axis= 1)
    #weighted daily freezing intensity
    daily_intensity = pd.Series(0, index=combined_dfs.index)
    for name, weight in weights.items():
        col = f"Tmin_{name}"
        daily_intensity += weight * np.maximum(15 - combined_dfs[col], 0)
    combined_dfs["freeze_intensity"] = daily_intensity
    #For freezing event , binary variable!
    combined_dfs["prod_min"] = combined_dfs.filter(like="Tmin").min(axis=1) 

    # Get the minimum temperature across all stations for each date
    weekly = combined_dfs[["prod_min", "freeze_intensity"]].resample("W-FRI").agg({
    "prod_min": "min",
    "freeze_intensity": "sum"}).reset_index()


    weekly["freeze_event"] = (weekly["prod_min"] < 15).astype(int) # Create a binary variable indicating if there was a freeze event (min temp below 32 F)

    return weekly[["date", "prod_min", "freeze_event", "freeze_intensity"]]

if __name__ == "__main__":
    df = get_all_production_weather_data()
    print(df[df["freeze_event"] == 1].head(10))
    print(f"\nTotal freeze weeks: {df['freeze_event'].sum()}")
    df.to_csv("weekly_production_tmin.csv", index=False)

