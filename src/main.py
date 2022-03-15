import os
import sys
import uuid
import math
import pickle
import pathlib
import getpass
from platform import uname
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from collections import OrderedDict
from scipy.integrate import cumtrapz
from functools import reduce

import catboost
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingClassifier
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)

class Config(object):
    
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.WARNING)
    FILE_HANDLER = logging.FileHandler('logfile.log')
    FORMATTER    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    FILE_HANDLER.setFormatter(FORMATTER)
    LOGGER.addHandler(file_handler)
    
    DATA = dict(
        # BASE_DIR = pathlib.Path().resolve(),
        DATASET_DIR = pathlib.Path().resolve() / "data/dengue",
        EXPORT_DIR = pathlib.Path().resolve() / "data/dengue/exports",
    )
    
    ANALYSIS_CONFIG = dict(
        OUTLIERS_COLS = ["precipitation_amt_mm", "reanalysis_precip_amt_kg_per_m2", "reanalysis_sat_precip_amt_mm", "station_precip_mm"]
    )
    
    MODELLING_CONFIG = dict(
        TRAIN_COLS = ['year', 'weekofyear', 'ndvi_ne', 'ndvi_nw', 'ndvi_se',
                      'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
                      'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
                      'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
                      'reanalysis_precip_amt_kg_per_m2',
                      'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
                      'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
                      'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
                      'station_min_temp_c', 'station_precip_mm'],
        FEATURE_ENGINEER_COLS = ['low_season', 'rampup_season', 'high_season',
                                 'reanalysis_specific_humidity_g_per_kg_1lag',
                                 'reanalysis_specific_humidity_g_per_kg_2lag',
                                 'reanalysis_specific_humidity_g_per_kg_3lag',
                                 'reanalysis_dew_point_temp_k_1lag', 'reanalysis_dew_point_temp_k_2lag',
                                 'reanalysis_dew_point_temp_k_3lag', 'reanalysis_min_air_temp_k_1lag',
                                 'reanalysis_min_air_temp_k_2lag', 'reanalysis_min_air_temp_k_3lag',
                                 'reanalysis_max_air_temp_k_1lag', 'reanalysis_max_air_temp_k_2lag',
                                 'reanalysis_max_air_temp_k_3lag', 'station_min_temp_c_1lag',
                                 'station_min_temp_c_2lag', 'station_min_temp_c_3lag',
                                 'station_max_temp_c_1lag', 'station_max_temp_c_2lag',
                                 'station_max_temp_c_3lag', 'reanalysis_air_temp_k_1lag',
                                 'reanalysis_air_temp_k_2lag', 'reanalysis_air_temp_k_3lag',
                                 'reanalysis_relative_humidity_percent_1lag',
                                 'reanalysis_relative_humidity_percent_2lag',
                                 'reanalysis_relative_humidity_percent_3lag'],
        TUNING_METHOD = "random_search",
        FEATURE_SELECTION_COLUMNS = ["RF", "Extratrees", "Kbest"],
    )


class Analysis(Config):
    data = {}
    
    def __init__(self, city=["*"]):
        self.city = city
        
    
    def get_data(self):
        logging.info("----------------------------------------------------------- PREPROCESSING ------------------------------------------------------------")
        logging.info("Reading TRAIN Dataset:")
        self.data["train_df"] = pd.read_csv(self.DATA["DATASET_DIR"] / 'merged_train.csv', index_col=0)
        
        logging.info("Initiate Preprocessing of Train Data:")
        logging.info("  - correct the datetime format in Train dataset...")
        self.data["train_df"]["week_start_date"] = pd.to_datetime(self.data["train_df"]["week_start_date"])
        self.data["train_df"] = self.data["train_df"].set_index("week_start_date")
        for col in ["year", "weekofyear"]:
            self.data["train_df"][col] = self.data["train_df"][col].astype(int)
        
        logging.info("  - select 'sj' city data...")
        self.data["sj_train_df"] = self.data["train_df"][self.data["train_df"]["city"]=="sj"]
        self.data["sj_train_df"].reset_index(drop=True, inplace=True)
        
        logging.info("  - fix incorrect maximum 'weekofyear' feature...")
        self.data["sj_train_df"] = self.process_inconsistent(train=True)
        
        logging.info("  - missing values imputation...")
        self.data["sj_train_df"] = self.process_missing_values(train=True)
        
        logging.info("  - outliers removal...")
        self.data["sj_train_df"] = self.process_outliers(train=True)
        
        logging.info("------------------------------------------------------- Done processing model data ------------------------------------------------------")
        
        
    def process_inconsistent(self, train=True):
        if train:
            sj_df = self.data["sj_train_df"]
        else:
            sj_df = self.data["sj_test_df"]
        
        logging.info("  - correct 'weekofyear' column where some year has maximum week number of '53' which are not right, will be corrected to '52'...")
        for year in [2001, 2007, 2013]:
            sj_df.loc[:,'weekofyear'] = np.where(sj_df["year"]==year, sj_df["weekofyear"]+1, sj_df["weekofyear"])
        sj_df.loc[:,'weekofyear'] = np.where(sj_df["weekofyear"]>52, 1, sj_df["weekofyear"])
                
        return sj_df
        
        
    def process_missing_values(self, train=True):
        if train:
            sj_df = self.data["sj_train_df"]
        else:
            sj_df = self.data["sj_test_df"]
            
        logging.info("  - using KNN Imputation model with 'n_neighbors'=5'...")
        imputer = KNNImputer(n_neighbors=5)
        
        num_sj_df = sj_df.select_dtypes(include=[np.number, "float64", "int64"])
        num_sj_df = pd.DataFrame(imputer.fit_transform(num_sj_df), columns=num_sj_df.columns)
        for col in num_sj_df.columns:
            sj_df[col] = num_sj_df[col]
            
        return sj_df
     
     
    def iqr_based_filtering(self, df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound  = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df[col]<lower_bound) |(df[col]>upper_bound)]
        
        return df
       
    
    def process_outliers(self, train=True):
        if train:
            sj_df = self.data["sj_train_df"]
        else:
            sj_df = self.data["sj_test_df"]
            
        logging.info("  - using IQR based filtering to handle outliers...")
        for col in self.ANALYSIS_CONFIG["OUTLIERS_COLS"]:
            sj_outlier_removed_df = self.iqr_based_filtering(sj_df, col)
        
        ol_col_list = ["city", "year"] + self.ANALYSIS_CONFIG["OUTLIERS_COLS"]
        sub_sj_outlier_removed_df = sj_outlier_removed_df[ol_col_list]
        merge_sj_outlier_removed_df = sj_df.merge(sub_sj_outlier_removed_df, 
                                                  on=["city", "year"], 
                                                  how="left", 
                                                  suffixes=["", "_src"], 
                                                  indicator="_join_ind")
        
        for col in self.ANALYSIS_CONFIG["OUTLIERS_COLS"]:
            sj_df.loc[merge_sj_outlier_removed_df["_join_ind"]=="both",col] = merge_sj_outlier_removed_df[col+"_src"]

        return sj_df
    
    
class Train(Analysis):
    data = {}
    
    def __init__(self, target_var, train_df):
        super().__init__()
        self.data["model_df"] = train_df
        self.meta = dict(
            target_var = target_var,
            stime = datetime.now(),
            user = getpass.getuser(),
            sys = uname()[1],
            py = '.'.join(map(str, sys.version_info[:3])),
        )
        
        self.REGRESSION_MODELS = dict(
            LINEAR = dict(alg=LinearRegression()),
            RFR = dict(alg=RandomForestRegressor(), args=dict(randome_state=42, scaled=False),
                    param_grid={
                            "max_depth"         : [None, 5, 10, 20, 35],
                            "max_features"      : [2, 5, "auto"],
                            # "min_samples_leaf"  : [2, 3, 4, 10],
                            "n_estimators"      : [20, 50, 100, 200],
                        }),
            XGBOOST = dict(alg=XGBRegressor(), args=dict(random_state=42, scaled=False),
                        param_grid={
                            "learning_rate":[0.01, 0.05, 0.1, 0.3],
                            "max_depth": [2, 3, 6, 10], # 3
                            "n_estimators": [20, 50, 200], # 100
                        }),
            GRADIENT = dict(alg=GradientBoostingRegressor(), args=dict(random_state=42),
                            param_grid={
                                "n_estimators": [100, 150, 200, 400],
                                "learning_rate": [0.03, 0.1, 0.3],
                                'max_depth': [2, 4, 5, 6, 8],
                            }),
            BAGGING = dict(alg=BaggingClassifier(), args=dict(random_state=42),
                    param_grid={
                        "n_estimators": [10, 30, 50, 100],
                        "max_features": [1, 5, 20, 100],
                        'max_samples': [1, 5, 20, 100],
                    }),
        )
        
    
    def get_model_data(self):
        logging.info("------------------------------------------------------------- MODELLING -------------------------------------------------------------")
        logging.info("Reading Processed Model Dataset:")
        logging.info("Initiate Feature Engineering:")
        logging.info("  - create dengue season categorical features...")
        self.data["sj_train_df"] = self.data["model_df"]
        cutoffs = [11,30]
        self.data["sj_train_df"]['low_season'] = np.where((self.data["sj_train_df"].weekofyear<cutoffs[0]), 1, 0)
        self.data["sj_train_df"]['rampup_season'] = np.where((self.data["sj_train_df"].weekofyear>=cutoffs[0]) &
                                                             (self.data["sj_train_df"].weekofyear<cutoffs[1]), 1, 0)
        self.data["sj_train_df"]['high_season'] = np.where((self.data["sj_train_df"].weekofyear>=cutoffs[1]), 1, 0)
        
        logging.info("  - create lag features of temperature and humidity...")
        to_shift = ["reanalysis_specific_humidity_g_per_kg", "reanalysis_dew_point_temp_k", "reanalysis_min_air_temp_k",
                    "reanalysis_max_air_temp_k", "station_min_temp_c", "station_max_temp_c",
                    "reanalysis_air_temp_k", "reanalysis_relative_humidity_percent"]
            
        for i in to_shift:
            self.data["sj_train_df"][i+"_1lag"] = self.data["sj_train_df"][i].shift(-1)
            self.data["sj_train_df"][i+"_2lag"] = self.data["sj_train_df"][i].shift(-2)
            self.data["sj_train_df"][i+"_3lag"] = self.data["sj_train_df"][i].shift(-3)
        self.data["sj_train_df"] = self.data["sj_train_df"].fillna(method="ffill")
        
        logging.info("1. Run Base Dengue Prediction Model Without Feature Engineering & Feature Selection:")
        logging.info("  - split the data into Train & Test data with original features...")
        self.base_sj_X_train, self.base_sj_X_test, self.base_sj_y_train, self.base_sj_y_test = self.split_data(self.data["sj_train_df"], 
                                                                                                               input_cols=self.MODELLING_CONFIG["TRAIN_COLS"], 
                                                                                                               target="total_cases", 
                                                                                                               ratio=0.20)        
        self.data["base_sj_predict_df"] = pd.DataFrame()
        self.base_sj_metrics_list = []
        self.base_sj_model = []
        for model_name in self.REGRESSION_MODELS:
            if model_name == "LINEAR":
                model, predict, predict_series, metrics = self.run_model(self.base_sj_X_train, self.base_sj_X_test, 
                                                                         self.base_sj_y_train, self.base_sj_y_test, 
                                                                         model_name, 
                                                                         tuning_method=None)
            else:
                model, predict, predict_series, metrics = self.run_model(self.base_sj_X_train, self.base_sj_X_test, 
                                                                         self.base_sj_y_train, self.base_sj_y_test, 
                                                                         model_name, 
                                                                         tuning_method=self.MODELLING_CONFIG["TUNING_METHOD"])
            self.data["base_sj_predict_df"][f"{model_name}_total_cases"] = predict
            self.base_sj_metrics_list.append(metrics)
            self.base_sj_model.append(model)
            
        self.data["base_sj_predict_df"].index = predict_series.index
        self.data["base_sj_predict_df"]["y_test"] = self.base_sj_y_test
        self.data["base_sj_predict_df"].reset_index(inplace=True)
        self.data["base_sj_metrics_df"] = pd.DataFrame(self.base_sj_metrics_list)
        self.data["base_sj_model"] = pd.DataFrame(self.base_sj_model).rename(columns={0: "Algorithm"})
        
        logging.info("2. Run Feature Engineering Dengue Prediction Model:")
        logging.info("  - split the data into Train & Test data with original & new features...")
        self.fe_sj_X_train, self.fe_sj_X_test, self.fe_sj_y_train, self.fe_sj_y_test = self.split_data(self.data["sj_train_df"], 
                                                                                                       input_cols=self.MODELLING_CONFIG["TRAIN_COLS"] + self.MODELLING_CONFIG["FEATURE_ENGINEER_COLS"], 
                                                                                                       target="total_cases", 
                                                                                                       ratio=0.20)
        self.data["fe_sj_predict_df"] = pd.DataFrame()
        self.fe_sj_metrics_list = []
        self.fe_sj_model = []
        for model_name in self.REGRESSION_MODELS:
            if model_name == "LINEAR":
                model, predict, predict_series, metrics = self.run_model(self.fe_sj_X_train, self.fe_sj_X_test, 
                                                                         self.fe_sj_y_train, self.fe_sj_y_test, 
                                                                         model_name, 
                                                                         tuning_method=None)
            else:
                model, predict, predict_series, metrics = self.run_model(self.fe_sj_X_train, self.fe_sj_X_test, 
                                                                         self.fe_sj_y_train, self.fe_sj_y_test, 
                                                                         model_name, 
                                                                         tuning_method=self.MODELLING_CONFIG["TUNING_METHOD"])
            self.data["fe_sj_predict_df"][f"{model_name}_total_cases"] = predict
            self.fe_sj_metrics_list.append(metrics)
            self.fe_sj_model.append(model)
            
        self.data["fe_sj_predict_df"].index = predict_series.index
        self.data["fe_sj_predict_df"]["y_test"] = self.fe_sj_y_test
        self.data["fe_sj_predict_df"].reset_index(inplace=True)
        self.data["fe_sj_metrics_df"] = pd.DataFrame(self.fe_sj_metrics_list)
        self.data["fe_sj_model"] = pd.DataFrame(self.fe_sj_model).rename(columns={0: "Algorithm"})
        
        logging.info("3. Run Feature Selection Dengue Prediction Model:")
        self.sj_top_20_features, self.data["sj_score_table"] = self.feature_selection_scores(self.base_sj_X_train, self.base_sj_y_train)
        
        logging.info("  - split the data into Train & Test data using selected features...")
        self.fs_sj_X_train, self.fs_sj_X_test, self.fs_sj_y_train, self.fs_sj_y_test = self.split_data(self.data["sj_train_df"], 
                                                                                                        input_cols=self.sj_top_20_features, 
                                                                                                        target="total_cases", 
                                                                                                        ratio=0.20)

        self.data["fs_sj_predict_df"] = pd.DataFrame()
        self.fs_sj_metrics_list = []
        self.fs_sj_model = []

        for model_name in self.REGRESSION_MODELS:
            if model_name == "LINEAR":
                model, predict, predict_series, metrics = self.run_model(self.fs_sj_X_train, self.fs_sj_X_test, 
                                                                         self.fs_sj_y_train, self.fs_sj_y_test, 
                                                                         model_name, 
                                                                         tuning_method=None)
            else:
                model, predict, predict_series, metrics = self.run_model(self.fs_sj_X_train, self.fs_sj_X_test, 
                                                                         self.fs_sj_y_train, self.fs_sj_y_test, 
                                                                         model_name, 
                                                                         tuning_method=self.MODELLING_CONFIG["TUNING_METHOD"])
                
            self.data["fs_sj_predict_df"][f"{model_name}_total_cases"] = predict
            self.fs_sj_metrics_list.append(metrics)
            self.fs_sj_model.append(model)
            
        self.data["fs_sj_predict_df"].index = predict_series.index
        self.data["fs_sj_predict_df"]["y_test"] = self.fs_sj_y_test
        self.data["fs_sj_predict_df"].reset_index(inplace=True)
        self.data["fs_sj_metrics_df"] = pd.DataFrame(self.fs_sj_metrics_list)
        self.data["fs_sj_model"] = pd.DataFrame(self.fs_sj_model).rename(columns={0: "Algorithm"})
        
        logging.info("---------------------- Done modelling using LINEAR, RANDOM FOREST, XGBOOST & GRADIENT BOOSTING on 'total_cases' ----------------------")
            
        
    def split_data(self, df, input_cols=[], target="total_cases", ratio=0.30):
        X = df[input_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    
    @staticmethod
    def root_mean_square_error(actual, pred):
        rmse = math.sqrt(mean_absolute_error(actual, pred))
        
        return rmse
    

    def evaluate(self, actual, pred):
        R2 = r2_score(actual, pred)
        MAE = mean_absolute_error(actual, pred)
        RMSE = self.root_mean_square_error(actual, pred)
        
        metrics = dict(MAE=MAE, RMSE=RMSE, R2_Score=R2)
        
        return metrics
    
    
    def run_model(self, x_train, x_test, y_train, y_test, model_name, tuning_method):
        if model_name == "LINEAR":
            model_type = self.REGRESSION_MODELS["LINEAR"]
            alg = model_type["alg"]
        elif model_name == "RFR":
            model_type = self.REGRESSION_MODELS["RFR"]
            alg = model_type["alg"]
        elif model_name == "XGBOOST":
            model_type = self.REGRESSION_MODELS["XGBOOST"]
            alg = model_type["alg"]
        elif model_name == "GRADIENT":
            model_type = self.REGRESSION_MODELS["GRADIENT"]
            alg = model_type["alg"]
        elif model_name == "BAGGING":
            model_type = self.REGRESSION_MODELS["BAGGING"]
            alg = model_type["alg"]
            
        if tuning_method == None:
            model = alg
        elif tuning_method == "grid_search":
            alg_tuned = GridSearchCV(estimator=alg,
                                    param_grid=model_type["param_grid"],
                                    cv=5,
                                    verbose=0)
        elif tuning_method == "random_search":
            alg_tuned = RandomizedSearchCV(estimator=alg,
                                        param_distributions=model_type["param_grid"],
                                        cv=5,
                                        verbose=0)

        if tuning_method == None:
            model.fit(x_train, y_train)
        else:
            alg_tuned.fit(x_train, y_train)
            model = alg.set_params(**alg_tuned.best_params_)
            model.fit(x_train, y_train)

        predict = model.predict(x_test)
        predict_series = pd.Series(predict, index=y_test.index)
        metrics = self.evaluate(y_test, predict)
        metrics["MODEL"] = model_name
        
        return model, predict, predict_series, metrics
    
    
    def random_forest_selection(self, x, y):
        alg = RandomForestRegressor()
        alg.fit(x, y)
        preds = alg.predict(x)
        accuracy = r2_score(preds, y)

        rf_fi = pd.DataFrame(alg.feature_importances_, columns=["RF"], index=x.columns)
        rf_fi = rf_fi.reset_index().sort_values(['RF'],ascending=0)
        
        return rf_fi


    def extratrees_selection(self, x, y):
        alg = ExtraTreesRegressor()
        alg.fit(x, y)

        extratrees_fi = pd.DataFrame(alg.feature_importances_, columns=["Extratrees"], index=x.columns)
        extratrees_fi = extratrees_fi.reset_index().sort_values(['Extratrees'],ascending=0)
        
        return extratrees_fi


    def kbest_selection(self, x, y):
        model = SelectKBest(score_func=chi2, k=5)
        alg = model.fit(x.abs(), y)

        pd.options.display.float_format = '{:.2f}'.format
        kbest_fi = pd.DataFrame(alg.scores_, columns=["Kbest"], index=x.columns)
        kbest_fi = kbest_fi.reset_index().sort_values('Kbest',ascending=0)
        
        return kbest_fi


    def feature_selection_scores(self, x, y):
        try:
            logging.info("- feature selection through Random Forest Regressor...")
            rf_fi = self.random_forest_selection(x, y)
        except MemoryError:
            print("- feature selection through Random Forest Regressor not run due to laptop memory issue...")

        logging.info("- feature selection through Extratrees Regressor...")
        extratrees_fi = self.extratrees_selection(x, y)

        logging.info("- feature selection through K-Best...")
        kbest_fi = self.kbest_selection(x, y)

        logging.info("Creating feature selection table to acquire the right features")
        dfs = [rf_fi, extratrees_fi, kbest_fi]
        features_final_results = reduce(lambda left,right: pd.merge(left, right, on='index'), dfs)

        score_table = pd.DataFrame({},[])
        score_table['index'] = features_final_results['index']
        for i in self.MODELLING_CONFIG["FEATURE_SELECTION_COLUMNS"]:
            score_table[i] = features_final_results['index'].isin(list(features_final_results.nlargest(5,i)['index'])).astype(int)
        score_table['final_score'] = score_table[self.MODELLING_CONFIG["FEATURE_SELECTION_COLUMNS"]].sum(axis=1)
        score_table = score_table.sort_values('final_score',ascending=0)
        top_20_features = score_table.iloc[:20, ]["index"].values
        score_table = score_table.reset_index(drop=True)

        return top_20_features , score_table
        


if __name__ == "__main__":
    analysis = Analysis()
    analysis.get_data()

    train = Train(target_var="total_cases", train_df=analysis.data["sj_train_df"])
    train.get_model_data()
    
    print("Comparing the results of 3 different scenarios will be used to test the prediction of Dengue 'total_cases':")
    print("1st scenario: base model without any feature engineering or feature selection...")
    print(train.data["base_sj_metrics_df"])
    print("2nd scenario: feature engineered model with new features like season of dengue cases and lags features of temperature & humidity...")
    print(train.data["fe_sj_metrics_df"])
    print("3rd scenario: feature selection model using top 20 features selected from Random Forest, Extratree Regressor and K-Best...")
    print(train.data["fs_sj_metrics_df"])
    print(train.sj_top_20_features)
    
    print("We can see that in 'sj' city. GRADIENT BOOSTING model performs the best with lowest MAE, RMSE and highest R2 score.")