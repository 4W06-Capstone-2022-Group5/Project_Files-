import pandas as pd
import os
from final_imputation import final_imputation

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




## --------------------------------------------------------------------------------------------------------------- #
## FINAL FUNCTION 
## --------------------------------------------------------------------------------------------------------------- #

# Choose which dataset to work with (uncomment section from above that you want)
url = url_30
data = pd.read_excel(url, sheet_name=1)

# Choose desired imputation method
impute_method = 'MI'

# Perform imputation
data_imputed, mae_vector, rmse_vector = final_imputation(data, num_properties, impute_method)

# Export to Excel
writer = pd.ExcelWriter('imputed_data.xlsx', engine='xlsxwriter')
data.to_excel(writer, sheet_name='Original_data', index=False)
data_imputed.to_excel(writer, sheet_name='Imputed_data', index=False)

# Put the stats into the Excel sheet
mean_rmse = sum(rmse_vector) / len(rmse_vector)
mean_mae = sum(mae_vector) / len(mae_vector)
stats_df = pd.DataFrame()
stats_df = stats_df.assign(MAE_Vals=mae_vector)
stats_df = stats_df.assign(RMSE_Vals=rmse_vector)
stats_df["N-MAE_Average"] = mean_mae
stats_df["N-RMSE_Average"] = mean_rmse
stats_df.to_excel(writer, sheet_name='Stats')

# Save Excel
writer.save()

