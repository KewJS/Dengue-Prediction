import os
import sys
import pathlib
import inspect
from datetime import timedelta

BASE_DIR = pathlib.Path().resolve()


class Config(object):
    
    QDEBUG = True
    
    CURRENT_TIME = datetime.delta()
    
    NAME = dict(
        FULL = "Dengue Catch",
        SHORT = "DC",
    )
    
    FILES = dict(
        DATASET_DIR = BASE_DIR / "data",
        EXPORT_DIR = DATASET_DIR / "exports",
    )
    
    ANALYSIS_CONFIG = dict(
        
    )
    