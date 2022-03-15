import os
import sys
import pathlib
import inspect
from datetime import timedelta

import logging
file_handler = logging.FileHandler('logfile.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(name)s : %(message)s')

BASE_DIR = pathlib.Path().resolve()


class Logger():
    info = print()
    warning = print()
    debug = warning()
    critical = print()
    

class Config(object):
    
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