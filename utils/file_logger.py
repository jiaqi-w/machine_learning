import os
import logging
import datetime
import config
from logging.handlers import TimedRotatingFileHandler

__author__ = "Jiaqi"
__version__ = "2"
__date__ = "Oct 15, 2017"

class File_Logger_Helper():

    @staticmethod
    def get_daily_handler(logger_dir, logger_fname="RL_log.log", logger_mode="testing"):
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        logger_fname = os.path.join(logger_dir, logger_fname)
        #FIXME I commented out this log file print as it's annoying from command line usage - kkbowden
        # print("Add log file", logger_fname)
        handler = TimedRotatingFileHandler(logger_fname, when='midnight', interval=1)
        if logger_mode == "testing":
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
        # Formatter
        formatter = logging.Formatter( "%(asctime)s | %(pathname)s:%(lineno)d | %(funcName)s | %(levelname)s | %(message)s ")
        handler.setFormatter(formatter)

        return handler

    @staticmethod
    def get_logger(logger_dir=config.LOG_DIR, logger_fname="my_log.log", logger_mode="testing"):
        if logger_mode == "testing":
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        # if not os.path.exists(logger_dir):
        #     os.makedirs(logger_dir)
        # logger = logging.getLogger(os.path.join(logger_dir, logger_fname))
        logger = logging.getLogger(logger_fname)
        if not len(logger.handlers):
            # FIXME, should check the kind of handler. As it could be email handler as well.
            handler = File_Logger_Helper.get_daily_handler(logger_dir, logger_fname, logger_mode)
            logger.addHandler(handler)
        return logger

if __name__ == "__main__":
    File_Logger_Helper.get_logger()