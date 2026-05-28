#Using EIA API to get storage data, then calculate weekly change in storage
#import relevant libraries for API calls and data manipulation
import requests
import pandas as pd
from dotenv import load_dotenv
import os
#load the EIA API key from .env file
load_dotenv()  
API_KEY = os.getenv("EIA_API_KEY")

def get_storage():
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    params = {
        "api_key": API_KEY,
        "frequency": "weekly",
        "data[0]": "value",\
        "facets[series][]": "NW2_EPG0_SWO_R48_BCF",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000
        }
    
    r = requests.get(url, params=params)
    r.raise_for_status() #Crash if we get an error response
    raw= r.json() # Parse the Json response to dictionaries for python 
    rows= raw['response']['data']
    df = pd.DataFrame(rows)
    df = df[["period", "value"]].rename(columns={"period": "date", "value": "storage_bcf"})
    df["date"] = pd.to_datetime(df['date'])
    return df

#Calculate weekly change which is something of our interest in this project as we want to predict this change
def storage_change(df):
    df["storage_bcf"] = pd.to_numeric(df["storage_bcf"])
    df["storage_change"] = df["storage_bcf"].diff()
    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = get_storage()
    df = storage_change(df)
    print(df.tail())
    