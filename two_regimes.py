#Change this code in train.py in the script folder

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from scripts.transform import TransformData

class TrainModel:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.transformed_df = self._load_from_file(file_path)
        self.transformed_df['ln_volume'] = self.transformed_df.Volume.apply(lambda x: np.log(x) if x > 0 else np.nan)

    def _load_from_file(self, file_path: str) -> pd.DataFrame:
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        else:
            raise FileNotFoundError(f"No file found at {file_path}")

    def _save_to_file(self, df: pd.DataFrame, file_path: str):
        df.to_parquet(file_path, index=False)

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
        print(f'length: X_train {self.X_train.shape},  X_validation {self.X_valid.shape}, X_test {self.X_test.shape}')
        print(f'  X_train_valid = {self.X_train_valid.shape},  all combined: X_all {self.X_all.shape}')

    def _clean_dataframe_from_inf_and_nan(self, df: pd.DataFrame):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def prepare_dataframe(self):
        print("Prepare the dataframe: define feature sets, add dummies, temporal split")
        self._define_feature_sets()
        self._define_dummies()
        min_date_df = self.df_full.Date.min()
        max_date_df = self.df_full.Date.max()
        self._perform_temporal_split(self.df_full, min_date=min_date_df, max_date=max_date_df)
        self._define_dataframes_for_ML()

    def train_random_forest(self, max_depth=17, n_estimators=200):
        print('Training the best model (RandomForest (max_depth=17, n_estimators=200))')
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        self.model = self.model.fit(self.X_train_valid, self.y_train_valid)

    def persist(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        model_filename = 'random_forest_model.joblib'
        path = os.path.join(data_dir, model_filename)
        joblib.dump(self.model, path)
        data_filename = 'df_full.parquet'
        data_path = os.path.join(data_dir, data_filename)
        self._save_to_file(self.df_full, data_path)

    def load(self, data_dir: str):
        model_filename = 'random_forest_model.joblib'
        path = os.path.join(data_dir, model_filename)
        self.model = joblib.load(path)
        data_filename = 'df_full.parquet'
        data_path = os.path.join(data_dir, data_filename)
        self.df_full = self._load_from_file(data_path)
        self.prepare_dataframe()

    def make_inference(self, pred_name: str):
        print('Making inference')
        y_pred_all = self.model.predict_proba(self.X_all)
        y_pred_all_class1 = [k[1] for k in y_pred_all]
        y_pred_all_class1_array = np.array(y_pred_all_class1)
        self.df_full[pred_name] = y_pred_all_class1_array
        self.df_full[f"{pred_name}_rank"] = self.df_full.groupby("Date")[pred_name].rank(method="first", ascending=False)

# Usage example
# For running from file:
model = TrainModel(file_path='local_data/transformed_df.parquet')
model.prepare_dataframe()
model.train_random_forest()
model.make_inference(pred_name='prediction')
model.persist(data_dir='local_data/')

# Change this code in main.py to use and run it
import os
from scripts.train import TrainModel

def main():
    # Define the path to your transformed DataFrame
    transformed_file_path = 'local_data/transformed_df.parquet'
    
    # Ensure the file exists before proceeding
    if not os.path.exists(transformed_file_path):
        raise FileNotFoundError(f"No file found at {transformed_file_path}")
    
    # Initialize the TrainModel with the file path
    model = TrainModel(file_path=transformed_file_path)
    
    # Prepare the data
    model.prepare_dataframe()
    
    # Train the model
    model.train_random_forest()
    
    # Make inference
    pred_name = 'prediction'
    model.make_inference(pred_name=pred_name)
    
    # Persist the model and the full DataFrame
    data_dir = 'local_data/'
    model.persist(data_dir=data_dir)

if __name__ == '__main__':
    main()
