import pandas as pd
import numpy as np

#Simple function to compute the heating degree days and cooling degree days
#65 F is the standard base T for HDD and CDD calulations

def compute_degree_days(temp_series, base_temp = 65):
    hdd = np.maximum(0, base_temp - temp_series) #HDD is the number of degrees the T is below base T of 65 F
    cdd = np.maximum(0, temp_series - base_temp) #CDD is the number of degrees that T is above base T of 65 F
    return pd.DataFrame({"HDD": hdd, "CDD": cdd})


#Test case
test = pd.Series([30, 65, 95])
print(compute_degree_days(test))
print("Test passed!")