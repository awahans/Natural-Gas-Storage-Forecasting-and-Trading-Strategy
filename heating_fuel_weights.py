
#Uses Census ACS House heating fuel to calculate weights for HDD for different cities
import requests
import pandas as pd
from dotenv import load_dotenv
import os
import time
load_dotenv()

API_KEY = os.getenv("CENSUS_API_KEY")

def get_with_retry(url, params, tries=20, delay=3):
    for attempt in range(tries):
        r = requests.get(url, params=params)
        if r.text.strip().startswith("["):
            return r
        time.sleep(delay)
    raise RuntimeError(f"Census API still not returning valid JSON after {tries} tries - check the key/request")

from gas_weights import DIVISIONS

#Census state FIPS codes, needed because the ACS API returns states by FIPS, not postal abbreviation
STATE_FIPS = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL",
    "13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME",
    "24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH",
    "34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
    "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY",
}
#We chose 2022 as a baseline year to calculate weight. There is no significant change in weights calculated at 2012 and 2022
def get_heating_fuel_by_state(year=2022):
    r = get_with_retry(f"https://api.census.gov/data/{year}/acs/acs5", params={
        "get": "NAME,B25040_001E,B25040_002E",  #B25040_001E = total occupied units, _002E = utility gas
        "for": "state:*",
        "key": API_KEY,
    })
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["gas_heated_units"] = pd.to_numeric(df["B25040_002E"])
    df["state"] = df["state"].map(STATE_FIPS)
    return df.dropna(subset=["state"])[["state", "gas_heated_units"]]

#From the state calculate weights for the region
def get_heating_division_weights(year=2022):
    df = get_heating_fuel_by_state(year)
    state_division = {s: d for d, states in DIVISIONS.items() for s in states}
    df["division"] = df["state"].map(state_division)
    df = df.dropna(subset=["division"])
    by_division = df.groupby("division")["gas_heated_units"].sum()
    return (by_division / by_division.sum()).rename("hdd_weight_acs")

if __name__ == "__main__":
    weights = get_heating_division_weights()
    print(weights.round(3))
    weights.to_csv("division_weights_heating.csv")

#Calculates the weights for each division based on number of households using gas for heating. 