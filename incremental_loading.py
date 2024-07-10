#Change this code in train.py in the script folder

import pandas as pd
import numpy as np
import os
import joblib
import bind
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from scripts.transform import TransformData

Base = declarative_base()

class TransformedData(Base):
    __tablename__ = 'transformed_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    Date = Column(Date)
    Ticker = Column(String)
    Open = Column(Float)
    High = Column(Float)
    Low = Column(Float)
    Close = Column(Float)
    Adj_Close = Column(Float)
    Volume = Column(Float)
    # Add other columns as needed

class TrainModel:
    def __init__(self, data_source: str = 'file', file_path: str = '', transformed: TransformData = None, db_url: str = ''):
        self.data_source = data_source
        self.file_path = file_path
        self.db_url = db_url
        self.transformed_df = None

        if self.data_source == 'file':
            self.transformed_df = self._load_from_file(file_path)
        elif self.data_source == 'database':
            self.engine = create_engine(self.db_url)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind(self.engine))
            self.transformed_df = self._load_from_db()
        elif self.data_source == 'download' and transformed is not None:
            self.transformed_df = transformed.transformed_df.copy(deep=True)
        else:
            raise ValueError("Invalid data source or missing transformed data")

        self.transformed_df['ln_volume'] = self.transformed_df.Volume.apply(lambda x: np.log(x) if x > 0 else np.nan)
        self._define_feature_sets()

    def _load_from_file(self, file_path: str) -> pd.DataFrame:
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        else:
            raise FileNotFoundError(f"No file found at {file_path}")

    def _save_to_file(self, df: pd.DataFrame, file_path: str):
        df.to_parquet(file_path, index=False)

    def _load_from_db(self) -> pd.DataFrame:
        session = self.Session()
        data = session.query(TransformedData).all()
        session.close()
        return pd.DataFrame([d.__dict__ for d in data])

    def _save_to_db(self, df: pd.DataFrame):
        session = self.Session()
        df.to_sql('transformed_data', self.engine, if_exists='append', index=False)
        session.commit()
        session.close()

    def _define_feature_sets(self):
        self.GROWTH = [g for g in self.transformed_df if (g.find('growth_') == 0) & (g.find('future') < 0)]
        self.OHLCV = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.CATEGORICAL = ['Month', 'Weekday', 'Ticker']
        self.TO_PREDICT = [g for g in self.transformed_df.keys() if (g.find('future') >= 0)]
        self.MACRO = ['gdppot_us_yoy', 'gdppot_us_qoq', 'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS', 'DGS1', 'DGS5', 'DGS10']
        self.CUSTOM_NUMERICAL = ['vix_adj_close', 'SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative', 'volatility', 'ln_volume']
        self.TECHNICAL_INDICATORS = [
            'adx', 'adxr', 'apo', 'aroon_1', 'aroon_2', 'aroonosc', 'bop', 'cci', 'cmo', 'dx', 'macd', 'macdsignal', 
            'macdhist', 'macd_ext', 'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix', 'macdhist_fix', 
            'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo', 'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 
            'fastk', 'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr', 'ad', 'adosc', 'obv', 'atr', 'natr', 
            'ht_dcperiod', 'ht_dcphase', 'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine', 
            'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice'
        ]
        self.TECHNICAL_PATTERNS = [g for g in self.transformed_df.keys() if g.find('cdl') >= 0]
        self.NUMERICAL = self.GROWTH + self.TECHNICAL_INDICATORS + self.TECHNICAL_PATTERNS + self.CUSTOM_NUMERICAL + self.MACRO
        self.TO_DROP = ['Year', 'Date', 'Month_x', 'Month_y', 'index', 'Quarter', 'index_x', 'index_y'] + self.CATEGORICAL + self.OHLCV
        self.OTHER = [k for k in self.transformed_df.keys() if k not in self.OHLCV + self.CATEGORICAL + self.NUMERICAL + self.TO_DROP + self.TO_PREDICT]

    def _define_dummies(self):
        self.transformed_df.loc[:, 'Month'] = self.transformed_df.Month_x.astype(str)
        self.transformed_df['Weekday'] = self.transformed_df['Weekday'].astype(str)
        dummy_variables = pd.get_dummies(self.transformed_df[self.CATEGORICAL], dtype='int32')
        self.df_full = pd.concat([self.transformed_df, dummy_variables], axis=1)
        self.DUMMIES = dummy_variables.keys().to_list()

    def _perform_temporal_split(self, df: pd.DataFrame, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
        train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
        val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)
        split_labels = ['train' if date <= train_end else 'validation' if date <= val_end else 'test' for date in df['Date']]
        df['split'] = split_labels
        return df

    def _define_dataframes_for_ML(self):
        features_list = self.NUMERICAL + self.DUMMIES
        to_predict = 'is_positive_growth_5d_future'
        self.train_df = self.df_full[self.df_full.split.isin(['train'])].copy(deep=True)
        self.valid_df = self.df_full[self.df_full.split.isin(['validation'])].copy(deep=True)
        self.train_valid_df = self.df_full[self.df_full.split.isin(['train', 'validation'])].copy(deep=True)
        self.test_df = self.df_full[self.df_full.split.isin(['test'])].copy(deep=True)
        self.X_train = self.train_df[features_list + [to_predict]]
        self.X_valid = self.valid_df[features_list + [to_predict]]
        self.X_train_valid = self.train_valid_df[features_list + [to_predict]]
        self.X_test = self.test_df[features_list + [to_predict]]
        self.X_all = self.df_full[features_list + [to_predict]].copy(deep=True)
        self.X_train = self._clean_dataframe_from_inf_and_nan(self.X_train)
        self.X_valid = self._clean_dataframe_from_inf_and_nan(self.X_valid)
        self.X_train_valid = self._clean_dataframe_from_inf_and_nan(self.X_train_valid)
        self.X_test = self._clean_dataframe_from_inf_and_nan(self.X_test)
        self.X_all = self._clean_dataframe_from_inf_and_nan(self.X_all)
        self.y_train = self.X_train[to_predict]
        self.y_valid = self.X_valid[to_predict]
        self.y_train_valid = self.X_train_valid[to_predict]
        self.y_test = self.X_test[to_predict]
        self.y_all = self.X_all[to_predict]
        del self.X_train[to_predict]
        del self.X_valid[to_predict]
        del self.X_train_valid[to_predict]
        del self.X_test[to_predict]
        del self.X_all[to_predict]
        print(f'length: X_train {self.X_train.shape[0]}, X_valid {self.X_valid.shape[0]}, X_train_valid {self.X_train_valid.shape[0]}, X_test {self.X_test.shape[0]}')

    def _clean_dataframe_from_inf_and_nan(self, df):
        for c in df.keys():
            df = df.loc[~df[c].isnull()]
            df = df.loc[~df[c].isin([np.nan, np.inf, -np.inf])]
        return df

    def prepare_dataframe(self):
        self._define_dummies()
        min_date, max_date = self.transformed_df.Date.min(), self.transformed_df.Date.max()
        self.df_full = self._perform_temporal_split(self.df_full, min_date, max_date)
        self._define_dataframes_for_ML()

    def train_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        print('Training Random Forest Classifier')
        self.clf = RandomForestClassifier(
            n_estimators=250, 
            max_depth=10, 
            max_features='sqrt', 
            criterion='gini', 
            n_jobs=-1
        )
        self.clf.fit(self.X_train, self.y_train)
        print('Model Training Completed')

    def persist(self, data_dir: str = 'local_data/'):
        print(f'Persisting model and data to directory: {data_dir}')
        joblib.dump(self.clf, os.path.join(data_dir, 'model.joblib'))
        self.df_full.to_parquet(os.path.join(data_dir, 'full_df.parquet'), index=False)
        print(f'Model and data have been saved in {data_dir}')

# Change this code in main.py and run it
import os
from scripts.train import TrainModel  # Make sure the import path is correct

def main():
    # Define the path to your transformed DataFrame
    transformed_file_path = 'local_data/transformed_df.parquet'
    
    # Ensure the file exists before proceeding
    if not os.path.exists(transformed_file_path):
        raise FileNotFoundError(f"No file found at {transformed_file_path}")
    
    # Initialize the TrainModel with the file path
    model = TrainModel(data_source='file', file_path=transformed_file_path)
    
    # Prepare the data
    model.prepare_dataframe()
    
    # Train the model
    model.train_random_forest()
    
    # Persist the model and the full DataFrame
    data_dir = 'local_data/'
    model.persist(data_dir=data_dir)

if __name__ == '__main__':
    main()