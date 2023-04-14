# Project Files
A repository to contain all project files for Capstone Group 5. The following is a brief directory:

## Datasets 
This folder contains all datasets used for testing purposes. Included are the original datasets, the datasets with 30%, 50%, 70%, and 90% missing values from the defining properties, and the imputed datasets using each of the methods. Each imputed datset contains a final sheet containing error metrics obtained from the imputation as compared to the original dataset. 

## Final Imputation Algorithms
This folder contains the following files of interest: 
  1. final_imputation.py
      - This file defines a function named "final_imputation.py" which accepts a dataset (structured as a python dataframe) which has a given number of physical    properties containing missing values (must be located at the end columns of the dataset). 
      - Various imputation methods can be selected: 
            - 'final': Chooses the imputation method based on lowest error achieved between MF and KNN on a 10% subsample of the dataset's nonmissing values
            - 'MF': Uses the MissForest method
            - 'GAIN': Uses the GAIN method
            - 'MICE': Uses the MICE method
            - 'KNN': Uses the KNN method
            - 'MI': Uses the Mean Imputation method
      - Returns a python dataframe containing imputation results, as well as NRMSE and NMAE information for imputation performed on a 10% subsample of the dataset's nonmissing values
     
  3. test_imputation.py
      - Similar to final_imputation.py, however, can be used for testing purposes when a complete dataset is known. Accepts the complete dataset as well as the dataset with missing values (located in the final columns of the dataset). 
      - Same algorithm options as final_imputation.py

  5. example_final_imputation.py 
      - Contains script which runs the final_imputation.py code on sample datasets, and exports an Excel file containing a sheet listing the original dataset, a sheet containing the imputed dataset results, and a sheet containing error metric information
  
  7. example_test_imputation.py
      - Contains script which runs the test_imputation.py code on the provided sample datasets, and exports an Excel file containing a sheet listing the original dataset, a sheet containing the imputed dataset results, and a sheet containing error metric information
      - Each dataset contains 10 Excel sheets of data with different missing values, for the purposes of testing and averaging error metrics. The script will perform imputation on each of these 10 sheets, and compare the results to the full dataset's known values. 
