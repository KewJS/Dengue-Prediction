import os
import sys
import pathlib
import inspect

BASE_DIR = pathlib.Path().resolve()



class Config(Object):
    
    QDEBUG = True
    
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
    