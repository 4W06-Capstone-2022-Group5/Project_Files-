import pandas as pd
import os
from final_imputation import final_imputation
from test_imputation import testing_imputation

## ------------------------------------------------------------------- #
## DATASET LINKS: Uncomment a desired dataset to work with it
## ------------------------------------------------------------------- #
##---------------------------------------------------------------------------#
## Steel Fatigue Strength Data
##---------------------------------------------------------------------------#
# Full Steel Fatigue Strength Dataset:
url_full = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Steel_Fatigue_Database.csv'

# Steel Fatigue Strength dataset at different missing value rates:
url_30 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Steel-Fatigue30P-Removed.xlsx'
url_50 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Steel_Data_Sets/Steel-Fatigue50P-Removed.xlsx'
url_70 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Steel_Data_Sets/Steel-Fatigue70P-Removed.xlsx'
url_90 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Steel_Data_Sets/Steel-Fatigue90P-Removed.xlsx'

# Number of columns containing physical properties with missing values
num_properties = 1

##---------------------------------------------------------------------------#
## Nickle Super Alloy Data 
##---------------------------------------------------------------------------#
## Full dataset
#url_full = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Nickel_Superalloy_Dataset.xlsx'
    
## Dataset at different missing value rates
#url_30 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Nickle-SuperAlloy30P-Removed.xlsx'
#url_50 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Nickle-SuperAlloy50P-Removed.xlsx'
#url_70 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Nickle-SuperAlloy70P-Removed.xlsx'
#url_90 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Nickle-SuperAlloy90P-Removed.xlsx'

## Number of columns containing physical properties with missing values
#num_properties = 5

##---------------------------------------------------------------------------#
## Steel Creep Rupture Data
##---------------------------------------------------------------------------#
## Full dataset
#url_full = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Steel_Creep_Rupture.xlsx'
    
## Dataset at different missing value rates
#url_30 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Steel-Creep-Rupture30P-Removed.xlsx'
#url_50 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Steel-Creep-Rupture50P-Removed.xlsx'
#url_70 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Steel-Creep-Rupture70P-Removed.xlsx'
#url_90 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Steel-Creep-Rupture90P-Removed.xlsx'

## Number of columns containing physical properties with missing values
#num_properties = 3

##---------------------------------------------------------------------------#
## Steel UTS Data 
##---------------------------------------------------------------------------#
## Full dataset
#url_full = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Steel_Yield_and_Ultimate_Tensile_Strength.xlsx'
    
## Dataset at different missing value rates
#url_30 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Steel-UTS30P-Removed.xlsx'
#url_50 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Steel-UTS50P-Removed.xlsx'
#url_70 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Steel-UTS70P-Removed.xlsx'
#url_90 = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Data_Sets/Removed_Data/Steel-UTS90P-Removed.xlsx'

## Number of columns containing physical properties with missing values
#num_properties = 4

# ------------------------------------------------------------------- #
# TESTING FUNCTION 
# ------------------------------------------------------------------- #
# Choose which dataset to work with (uncomment section from above that you want)
url = url_30
data_full = pd.read_excel(url, sheet_name=0)

# Function parameters
impute_method = 'MF'

# Begin Excel sheet
writer = pd.ExcelWriter('imputed_data_testing.xlsx', engine='xlsxwriter')
data_full.to_excel(writer, sheet_name = 'Full_data', index=False)

# Perform for all 10 testing sheets
size = 10
rmse_vector = []
mae_vector = []
N_vector = []
for i in range(1,size+1):

    # Print current sheet index to user: 
    print(f'Sheet {i} complete.')

    # Do the imputation
    data = pd.read_excel(url, sheet_name=i)
    data_imputed, mae, rmse, N = testing_imputation(data, data_full, num_properties, impute_method)
    
    # Collect all errors
    rmse_vector.append(rmse)
    mae_vector.append(mae)
    N_vector.append(N)

    # Export results to Excel
    data.to_excel(writer, sheet_name=f'Missing_data_{i}', index=False)
    data_imputed.to_excel(writer, sheet_name=f'Imputed_data_{i}', index=False)

# Export averaged errors to Excel
mean_rmse = sum(rmse_vector) / len(rmse_vector)
mean_mae = sum(mae_vector) / len(mae_vector)

# Put the stats into the Excel sheet
stats_df = pd.DataFrame()
stats_df = stats_df.assign(MAE_Vals=mae_vector)
stats_df = stats_df.assign(RMSE_Vals=rmse_vector)
stats_df["N-MAE_Average"] = mean_mae
stats_df["N-RMSE_Average"] = mean_rmse
stats_df.to_excel(writer, sheet_name='Stats')

# Save excel document
writer.save()
