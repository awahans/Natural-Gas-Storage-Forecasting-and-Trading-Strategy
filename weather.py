#Relevant packages for API calls
import requests
import pandas as pd
from dotenv import load_dotenv
import os
from features import compute_degree_days
load_dotenv()
import time

TOKEN = os.getenv("NOAA_TOKEN")

#Assuking major demand is in these 5 cities, we will get their Temperature data
STATIONS = {
    "NYC": "GHCND:USW00094728",
    "Chicago": "GHCND:USW00094846",
    "Houston": "GHCND:USW00012918",
    "Denver": "GHCND:USW00023062",
    "Atlanta": "GHCND:USW00013874"
}

def get_weather_data(station_id, start="2010-01-01", end="2026-05-01"):
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
                "datatypeid": ["TMAX", "TMIN"],
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
    df["temperature"] = (df["TMAX"] + df["TMIN"]) / 2
    return df[["date", "temperature"]]

#Function to get weather data for all cities and calculate average temperature across all cities
def get_all_weather_data():
    dfs = []
    for city, station in STATIONS.items():
        print(f"Fetching {city}...")
        df = get_weather_data(station) # Weather data for each
        df = df.rename(columns={"temperature": f"{city}_temp"}) #renaming the column to include city name 
        dfs.append(df.set_index("date")) #set the date as index 
    combined= pd.concat(dfs, axis=1).reset_index() # Combines all dataframes into one and resets the index as date
    combined["avg_temp"] = combined.filter(like="_temp").mean(axis=1) # selects columns that contain "_temp" and calculates the average temperature across all cities for each date
    return combined[["date","avg_temp"]]

#Using compute degree days get HDD and CDD for each day
#Resample that to weekly data by summing degree days from Friday to Thursday
#As the storage data is measured at Friday at 9 AM
def get_weekly_degree_days():
    df = pd.read_csv("temperatures.csv",parse_dates=["date"])
    degree_days = compute_degree_days(df["avg_temp"])
    df = pd.concat([df, degree_days], axis=1)
    #Now for weekly
    weekly = df.set_index("date")[["HDD", "CDD"]].resample("W-THU").sum().reset_index() # Resample the data to weekly frequency and sum the HDD and CDD for each week
    return weekly

if __name__ == "__main__":
    weekly = get_weekly_degree_days()
    print(weekly.head())
    weekly.to_csv("weekly_degree_days.csv", index=False)


