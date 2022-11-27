import pandas as pd
import numpy as np

# Getting the steel data with 30% removed outputs
url = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Steel%20Fatigue%20database.csv'
steel_data = pd.read_csv(url)

##---------------------------------------------------#
## USING MICE
##---------------------------------------------------#
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
#from sklearn.linear_model import LinearRegression

#lr = LinearRegression()
#imp = IterativeImputer(estimator=lr,missing_values=np.nan, max_iter=10, verbose=2, imputation_order='roman',random_state=0)
#X = imp.fit_transform(steel_data)

## convert to dataframe
#X = pd.DataFrame(X) 

## prints the imputed data set 
#print(X)    

## export the imputed data set into a csv file (each run will provide new imputations)
#X.to_csv('MICE_30P-Removed.csv', sep='\t', encoding='utf-8')

##---------------------------------------------------#
## USING KNN
##---------------------------------------------------#
#from sklearn.impute import KNNImputer

## We're using the number of nearest neighbours to be '2' here. We should try a selection of nearest neighbours and determine which gives the best results
#imputer = KNNImputer(n_neighbors=2)
#X = imputer.fit_transform(steel_data)

## Convert to Dataframe
#X = pd.DataFrame(X) 

## Prints the imputed data set 
#print(X)    

## Export the imputed data set into a CSV file (each run will provide new imputations)
#X.to_csv('KNN_2NN_30P-removed.csv', encoding='utf-8')



#---------------------------------------------------#
# USING GAIN
#---------------------------------------------------#
# separate folder for this

#---------------------------------------------------#
# USING MIDA
#---------------------------------------------------#