# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd

#from data_loader import data_loader
from gain import gain
from utils import rmse_loss

# Getting the steel data with 30% removed outputs
url = 'https://raw.githubusercontent.com/4W06-Capstone-2022-Group5/Project_Files-/main/Steel%20Fatigue%20database.csv'
steel_data = pd.read_csv(url)
steel_data = steel_data.to_numpy()


def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load data and introduce missingness
  # ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)
  
  # Impute missing data
  imputed_data_x = gain(steel_data, gain_parameters)
  
  # Report the RMSE performance
  #rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
  
  #print()
  #print('RMSE Performance: ' + str(np.round(rmse, 4)))
  
  return imputed_data_x

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      default='steel_data',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data = main(args)

  # Convert to Dataframe
  imputed_data = pd.DataFrame(imputed_data) 

  # Export the imputed data set into a CSV file (each run will provide new imputations)
  imputed_data.to_csv('GAIN_30P-Removed.csv', encoding='utf-8')

