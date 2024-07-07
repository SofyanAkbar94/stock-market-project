from scripts.data_repo import DataRepository
from scripts.transform import TransformData
from scripts.train import TrainModel

import pandas as pd
import warnings
def main():

  # SLOWER workflow settings: full update (the latest data, retrain the model, make inference (few mins)
  FETCH_REPO = True
  TRANSFORM_DATA = True
  TRAIN_MODEL = True
  

  # FAST workflow settings (everything comes local_repo/ directory)
  # FETCH_REPO = False
  # TRANSFORM_DATA = False
  # TRAIN_MODEL = False

  # =========================================
  # Step1: Get data
  # =========================================
  print('Step 1: Getting data from APIs or Load from disk')
  repo = DataRepository()

  if FETCH_REPO:
    # Fetch All 3 datasets for all dates from APIs
    repo.fetch()
    # save data to a local dir
    repo.persist(data_dir='local_data/')
  else:
    # OR Load from disk
    repo.load(data_dir='local_data/')  

  # =========================================
  # Step2: Transform data into one dataframe
  # =========================================
  print('Step 2: Making data transformations (combining into one dataset)')

  transformed =  TransformData(repo = repo)

  if TRANSFORM_DATA:
    transformed.transform()
    transformed.persist(data_dir='local_data/')
  else:
    transformed.load(data_dir='local_data/')

  # =========================================
  # Step3: Train/ Load Model
  # =========================================
  print('Step 3: Training the model or loading from disk')

  # Suppress all warnings (not recommended in production unless necessary)
  warnings.filterwarnings("ignore")

  trained = TrainModel(transformed=transformed)

  if TRAIN_MODEL:
    trained.prepare_dataframe() # prepare dataframes
    trained.train_random_forest() # train the model
    trained.persist(data_dir='local_data/') # save the model to disk
  else:
    trained.prepare_dataframe() # prepare dataframes (incl. for inference)
    trained.load(data_dir='local_data/')

  # =========================================
  # Step4: Make Inference
  # =========================================
  print('Step 4: Making inference')

  prediction_name='pred_rf_best'
  trained.make_inference(pred_name=prediction_name)
  COLUMNS = ['Adj Close','Ticker','Date',prediction_name, prediction_name+'_rank']
  
  print('Results of the estimation (last 10 days):')
  # Set display options to prevent truncation
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.max_colwidth', None)

  result_df = trained.df_full[trained.df_full[f'{prediction_name}_rank'] <= 2].sort_values(by=["Date", f'{prediction_name}_rank']).tail(10)[COLUMNS]
  print(result_df)

  # =========================================
  # Step5: Trade Simulation
  # =========================================
  print('Step 5: Performing trade simulation')

  # Trade simulation logic
  for index, row in result_df.iterrows():
      # Access relevant columns for trade simulation
      ticker = row['Ticker']
      date = row['Date']
      prediction = row[prediction_name]

  # Trade logic implementation
  for index, row in repo.ticker_df.iterrows():
  # Buy when SMA10 crosses above SMA20 and predicted growth_future_5d is positive
    if row['SMA10'] > row['SMA20'] and row['is_positive_growth_5d_future'] == 1:
      print(f"Buy {row['Ticker']} on {row['Date']}")

  # Sell when growth_future_10d is negative
    if row['is_positive_growth_10d_future'] == 0:
      print(f"Sell {row['Ticker']} on {row['Date']}")

if __name__ == "__main__":
  main()