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
END         8
START       7
NOTSET      0
"""

__author__ = "Kew Jing Sheng"
__status__ = "development"
__version__ = "0.1.1"
__date__ = date.today()

current_time = datetime.now()

OUT = 7
loggin.addlevelname(START, "START")
def output(self, message, *args, **kws):
    if self.isEnabledFor(START):
        self._log(START, message, args, **kws)
logging.Logger.start = start

OUT = 8
loggin.addlevelname(END, "END")
def output(self, message, *args, **kws):
    if self.isEnabledFor(END):
        self._log(END, message, args, **kws)
logging.Logger.end = end

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
        
    
formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")

class Logger(object):
    def __init__(self, name):
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        
        stream_hdlr = logging.StreamHandler(sys.stderr)
        stream_hdlr.setFormatter(formatter)
        self.logger.addHandler(stream_hdlr)
        
        file_hdlr = logging.FileHandler(name)
        file_hdlr.setFormatter(formatter)
        self.logger.addHandler(file_hdlr)
        
        self.logger.setLevel(7)
        self.logger.error = ErrorCounted(self.logger.error)
        self.logger.addHandler(ShutDownHandler(level=50))
        
        def start(self, message):
            global current_time
            current_time = datetime.now()
            self.logger.start(message)
            
        def info(self, message):
            self.logger.info(message)
            
        def warning(self, message):
            self.logger.warning(message)
            
        def error(self, message):
            self.logger.error(message)
            
        def critical(self, message):
            self.logger.critical(message)
            
        def end(self, args):
            tdelta = datetime.now() - current_time
            tdelta -= timedelta(microseconds=tdelta.microseconds)
            if hasattr(args, "odb"):
                self.info("Commiting results to output database ...")
                args.odb.commit()
                args.odb.close()
            self.logger.end("Finished task {} (total time spend: {})".format(args.task.title(), tdelta))