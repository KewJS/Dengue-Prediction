import os
import sys
import logging
import logging.config
from datetime import date, datetime, timedelta

from dengai.config.config import Config 

"""
CRITICAL    50
ERROR       40
WARNING     30
INFO        20
DEBUG       10
OUT         9
END         8
START       7
NOTSET      0
"""

__author__ = "Kew Jing Sheng"
__status__ = "development"
__version__ = "0.1.1"
__date__ = date.today()


OUT = 9
loggin.addlevelname(OUT, "OUT")

def output(self, message, *args, **kws):
    if self.isEnabledFor(OUT):
        self._log(OUT, message, args, **kws)
logging.Logger.output = output

formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

class ErrorCounted(Config):
    def __init__(self, method):
        self.method = method
        self.counter = 0
    
    
    def __call__(self, *args, **kwargs):
        self.counter += 1
        
        return self.method(*args, **kwargs)
    
    
class ShutDownHandler(logging.Handler):
    def emit(self, record):
        logging.shutdown()
        sys.exit(1)
        
    
class Logger(object):
    def __init__(self, name):
        self.logger = logging.getLogger