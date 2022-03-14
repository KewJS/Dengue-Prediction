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

from dengai.config.config import Config
from dengai.analysis.feature_engineer import Feature_Engineer





class Preprocess(Config):
    data = {}
    def __init__(self, city, suffix="", logger=Logger()):
        self.city = city
        self.suffix = suffix
        self.logger = logger