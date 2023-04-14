# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import numpy as np
import sys
import pandas as pd
import sklearn
import argparse
from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
from utils import rmse_loss
from tqdm import tqdm

from gain import gain
import tensorflow.compat.v1 as tf

import statistics
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from missforest.miss_forest import MissForest

from MissForestExtra import MissForestExtra

import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------------- #
# COMBINED FUNCTION 
# ------------------------------------------------------------------- #
def testing_imputation(data, data_full, num_properties, impute_method = 'empty'):


    '''Impute missing values in data_x
  
    Args:
        - data: Original data with missing values stored in a DATAFRAME
        - num_properties: Number of properties within dataset (must be placed at END of dataset)
        - impute_method: Desired imputation method
            - 'final': Chooses the imputation method based on lowest error achieved between MF and KNN on a subsample of the dataset
            - 'MF': Uses the MissForest method
            - 'GAIN': Uses the GAIN method
            - 'MICE': Uses the MICE method
            - 'KNN': Uses the KNN method
            - 'MI': Uses the Mean Imputation method
      
    Returns:
        - data_imputed: imputed data
        - mae, rmse: MAE and RMSE error metrics
        - N: Number of missing values which were filled
    '''

    # ------------------------------------------------------------------- #
    # Preprocessing
    # ------------------------------------------------------------------- #
    
    # If dataset columns have only 0's or NANs, remove that column
    for c in reversed(range(0, data.shape[1])):
        if data.iloc[:,c].sum() == 0 or data.iloc[:,c].sum() == np.nan:
            data = data.drop(data.columns[c], axis=1)
            data_full = data_full.drop(data_full.columns[c], axis=1)

    # Isolate features from data
    X = data.iloc[:,:-num_properties]
    y = data.iloc[:,-num_properties:]

    # Initialize imputed dataframe. Will add column by column
    data_imputed = X.copy()

    # Initialize error vectors
    rmse_vector = []
    mae_vector = []

    # ------------------------------------------------------------------- #
    # Imputation column-by-column
    # ------------------------------------------------------------------- #
    for i in range(0, y.shape[1]):
        # Add the 'current column to be imputed' to the end of X 
        sub_data = X.copy()
        sub_data[f"impute_xxx"] = y.iloc[:, i]

        # Check which values are missing & must be imputed
        NANcheck = sub_data.isnull()

        # Make a copy of the dataset, remove all rows containing a 'NAN' value in the physical property column.
        data_nonan = sub_data.copy()
        data_nonan = data_nonan.dropna()

        # Remove 10% of data in data_nonan. This will be used for error calculations
        row_choices = range(0, data_nonan.shape[0])
        num_remove = round(len(row_choices) * 0.10)
        rows = data_nonan.shape[0]
        cols = data_nonan.shape[1] 
        vector = np.ones((rows,cols))
        num_col_remove = 1          # Number of columns to remove data from
        for c in range(cols-num_col_remove, cols):
            vector[0:(num_remove), c] = np.nan
            random.shuffle((vector[:,c]))

        # Removing
        err_data_full = data_nonan.copy()
        err_data = np.multiply(err_data_full, vector)

        # Mark where the missing values are in 'err_data'
        NANcheck2 = err_data.isnull()

        # Calculate std deviation of entire err_data set
        std_sd = err_data_full.values.std(ddof=1)

        # ------------------------------------------------------------------- #
        # Imputation
        # ------------------------------------------------------------------- #
        if impute_method == 'final':
            #-------------------------------------------------------------------#
            # Error calculation
            #-------------------------------------------------------------------#
            # Create dummy datasets
            KNN_err_data = err_data.copy()
            MF_err_data = err_data.copy()

            # Impute the artificially produced missing values USING KNN, then compare to known values
            imputer = KNNImputer(n_neighbors=2)
            KNN_err_data_imputed = imputer.fit_transform(KNN_err_data)

            # Convert the imputed dataset from an array back into a dataframe
            KNN_err_data_imputed = pd.DataFrame(KNN_err_data_imputed)  

            # Compute total error between predictions and known vlaues USING KNN (NRMSE + NMAE) 
            rmse_summ = 0
            mae_summ = 0
            N = 0
            for k in range(NANcheck2.shape[0]):         # Iterate over rows
                for j in range(NANcheck2.shape[1]):     # Iterate over columns
                    val = NANcheck2.iat[k, j]           # If it had been a missing value here, then count twd error calculation
                    if val == True:
                        diff = err_data_full.iat[k, j] - KNN_err_data_imputed.iat[k, j]
                        abs_diff = abs(diff)
                        N = N + 1
                        rmse_summ = rmse_summ + diff**2
                        mae_summ = mae_summ + abs_diff

            KNN_rmse = math.sqrt(rmse_summ/N) / std_sd
            KNN_mae = (mae_summ / N) / std_sd
            KNN_tot_err = KNN_mae + KNN_rmse

            
            # Next, impute USING MISSFOREST
            mf = MissForest()
            MF_err_data_imputed = mf.fit_transform(MF_err_data)

            # Convert the imputed dataset from an array back into a dataframe
            MF_err_data_imputed = pd.DataFrame(MF_err_data_imputed)  

            # Compute total error between predictions and known vlaues USING MISSFOREST (NRMSE + NMAE)
            rmse_summ = 0
            mae_summ = 0
            N = 0
            for k in range(NANcheck2.shape[0]):         # Iterate over rows
                for j in range(NANcheck2.shape[1]):     # Iterate over columns
                    val = NANcheck2.iat[k, j]           # If it had been a missing value here, then count twd error calculation
                    if val == True:
                        diff = err_data_full.iat[k, j] - MF_err_data_imputed.iat[k, j]
                        abs_diff = abs(diff)
                        N = N + 1
                        rmse_summ = rmse_summ + diff**2
                        mae_summ = mae_summ + abs_diff

            MF_rmse = math.sqrt(rmse_summ/N) / std_sd
            MF_mae = (mae_summ / N) / std_sd
            MF_tot_err = MF_mae + MF_rmse

            # Whichever has lowest combined error, add that to the error vector 
            if KNN_tot_err < MF_tot_err:
                mae_vector.append(KNN_mae)
                rmse_vector.append(KNN_rmse)
            else: 
                mae_vector.append(MF_mae)
                rmse_vector.append(MF_rmse)

            #-------------------------------------------------------------------#
            # Full imputation (based on whichever gives lowest error)
            #-------------------------------------------------------------------#
            if KNN_tot_err < MF_tot_err: 
                # Perform the KNN algorithm to get an imputed dataset. We're using the number of nearest neighbours to be '2' here. 
                imputer = KNNImputer(n_neighbors=2)
                sub_data_imputed = imputer.fit_transform(sub_data)

                # Convert the imputed dataset from an array back into a dataframe
                sub_data_imputed = pd.DataFrame(sub_data_imputed)

                # Display which method was chosen
                chosen_method = 'KNN'
                print('\n',f'The chosen imputation method was {chosen_method}.','\n')

            else: 
                # Perform the MissForest algorithm to get an imputed dataset.
                mf = MissForest()
                sub_data_imputed = mf.fit_transform(sub_data)

                # Convert the imputed dataset from an array back into a dataframe
                sub_data_imputed = pd.DataFrame(sub_data_imputed)

                # Display which method was chosen
                chosen_method = 'MissForest'
                print('\n',f'The chosen imputation method was {chosen_method}.','\n')

        # ------------------------------------------------------------------- #
        # Imputation based on MISSFOREST
        # ------------------------------------------------------------------- #
        elif impute_method == 'MF':
            # Perform the MissForest algorithm to get an imputed dataset.
            mf = MissForest()
            sub_data_imputed = mf.fit_transform(sub_data)

            # Convert the imputed dataset from an array back into a dataframe
            sub_data_imputed = pd.DataFrame(sub_data_imputed)

        # ------------------------------------------------------------------- #
        # Imputation based on GAIN
        # ------------------------------------------------------------------- #
        elif impute_method == 'GAIN':
            # Perform the GAIN algorithm to get an imputed dataset. Change it from a df to an array for GAIN to work
            sub_data = sub_data.to_numpy()

            # Perform the GAIN algorithm to get an imputed dataset. Using default values atm 
            gain_parameters = {'batch_size': 128,
                               'hint_rate': 0.9,
                               'alpha': 100,
                               'iterations': 10000}
            sub_data_imputed = gain(sub_data, gain_parameters)

            # Convert the imputed dataset from an array back into a dataframe
            sub_data_imputed = pd.DataFrame(sub_data_imputed)

        # ------------------------------------------------------------------- #
        # Imputation based on MICE
        # ------------------------------------------------------------------- #        
        elif impute_method == 'MICE':
            # Perform the MICE algorithm to get an imputed dataset
            lr = LinearRegression()
            imp = IterativeImputer(estimator=lr,missing_values=np.nan, max_iter=1000, verbose=2, imputation_order='roman',random_state=0)
            sub_data_imputed = imp.fit_transform(sub_data)

            # Convert the imputed dataset from an array back into a dataframe
            sub_data_imputed = pd.DataFrame(sub_data_imputed)

        # ------------------------------------------------------------------- #
        # Imputation based on KNN
        # ------------------------------------------------------------------- #
        elif impute_method == 'KNN':
            # Perform the KNN algorithm to get an imputed dataset. We're using the number of nearest neighbours to be '2' here. 
            imputer = KNNImputer(n_neighbors=2)
            sub_data_imputed = imputer.fit_transform(sub_data)

            # Convert the imputed dataset from an array back into a dataframe
            sub_data_imputed = pd.DataFrame(sub_data_imputed)

        # ------------------------------------------------------------------- #
        # Imputation based on MI
        # ------------------------------------------------------------------- #
        elif impute_method == 'MI':
            # Create imputer object to fill in missing values w/ "mean imputation"
            imputer = SimpleImputer(strategy='mean')

            # Create a copy of the data with missing values filled in
            sub_data_imputed = pd.DataFrame(imputer.fit_transform(sub_data))

        # ------------------------------------------------------------------- #
        # Add the imputed column to the overall imputed dataset. 
        # (continues until all columns of 'y' are imputed in the for-loop) 
        # ------------------------------------------------------------------- #
        data_imputed[f"imputed_{i}"] = sub_data_imputed.iloc[:, -1]

    
    # ------------------------------------------------------------------- #
    # Error calculation
    # ------------------------------------------------------------------- #
    # Calculate std deviation of entire data set
    std_sd = data_full.values.std(ddof=1)   

    NANcheck3 = data.isnull()
    rmse_summ = 0
    mae_summ = 0
    N = 0
    for k in range(NANcheck3.shape[0]):         # Iterate over rows
        for j in range(NANcheck3.shape[1]):     # Iterate over columns
            val = NANcheck3.iat[k, j]           # If it had been a missing value here, then count twd error calculation
            if val == True:
                diff = data_full.iat[k, j] - data_imputed.iat[k, j]
                abs_diff = abs(diff)
                N = N + 1
                rmse_summ = rmse_summ + diff**2
                mae_summ = mae_summ + abs_diff

    rmse = math.sqrt(rmse_summ/N) / std_sd
    mae = (mae_summ / N) / std_sd

     



    return data_imputed, mae, rmse, N

