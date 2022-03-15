import os
import pathlib
import logging
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlob.gridspec as gridspec
import seaborn as sns
import missingno as msno
from collections import OrderedDict

from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor

import sys
sys.append("../")

from dengai.config.config import Config


class Logger():
    info = print()
    warning = print()
    debug = warning()
    critical = print()

class Preprocessing(Config):
    data = {}
    
    def __init__(self, city=["*"]):
        self.city = city
        
    
    def get_data(self):
        self.logger.info("----------------------------------------------------------- PREPROCESSING ------------------------------------------------------------")
        self.logger.info("Reading TRAIN Dataset:")
        self.data["train_df"] = pd.read_csv(self.DATA["DATASET_DIR"] / 'merged_train.csv', index_col=0)
        
        self.logger.info("Initiate Preprocessing of Train Data:")
        self.logger.info("  - correct the datetime format in Train dataset...")
        self.data["train_df"]["week_start_date"] = pd.to_datetime(self.data["train_df"]["week_start_date"])
        self.data["train_df"] = self.data["train_df"].set_index("week_start_date")
        for col in ["year", "weekofyear"]:
            self.data["train_df"][col] = self.data["train_df"][col].astype(int)
        
        self.logger.info("  - select 'sj' city data...")
        self.data["sj_train_df"] = self.data["train_df"][self.data["train_df"]["city"]=="sj"]
        self.data["sj_train_df"].reset_index(drop=True, inplace=True)
        
        self.logger.info("  - fix incorrect maximum 'weekofyear' feature...")
        self.data["sj_train_df"] = self.process_inconsistent(train=True)
        
        self.logger.info("  - missing values imputation...")
        self.data["sj_train_df"] = self.process_missing_values(train=True)
        
        self.logger.info("  - outliers removal...")
        self.data["sj_train_df"] = self.process_outliers(train=True)
        
        self.logger.info("------------------------------------------------------- Done processing model data ------------------------------------------------------")
        
        
    def process_inconsistent(self, train=True):
        if train:
            sj_df = self.data["sj_train_df"]
        else:
            sj_df = self.data["sj_test_df"]
        
        self.logger.info("  - correct 'weekofyear' column where some year has maximum week number of '53' which are not right, will be corrected to '52'...")
        for year in [2001, 2007, 2013]:
            sj_df.loc[:,'weekofyear'] = np.where(sj_df["year"]==year, sj_df["weekofyear"]+1, sj_df["weekofyear"])
        sj_df.loc[:,'weekofyear'] = np.where(sj_df["weekofyear"]>52, 1, sj_df["weekofyear"])
                
        return sj_df
        
        
    def process_missing_values(self, train=True):
        if train:
            sj_df = self.data["sj_train_df"]
        else:
            sj_df = self.data["sj_test_df"]
            
        self.logger.info("  - using KNN Imputation model with 'n_neighbors'=5'...")
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
            
        self.logger.info("  - using IQR based filtering to handle outliers...")
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