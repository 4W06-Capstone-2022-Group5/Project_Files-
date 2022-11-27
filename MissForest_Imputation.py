import math
import numpy as np
import sys
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from missforest.miss_forest import MissForest

from MissForestExtra import MissForestExtra

# Getting the steel data with 30% removed outputs
url = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Steel%20Fatigue%20database.csv'
steel_data = pd.read_csv(url)

# Print the steel data to make sure it got imported :p 
# print(steel_data.to_string)

# Instantiate miss forest and impute the dataset
mf = MissForest()
print(mf.fit_transform(steel_data))    # Prints the imputed data set 

# Export the imputed data set into a CSV file (each run will provide new imputations)
# mf.fit_transform(steel_data).to_csv('Imputed Data Set', sep='\t', encoding='utf-8')
