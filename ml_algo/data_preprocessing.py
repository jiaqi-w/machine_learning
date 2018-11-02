import pandas as pd
import config
from utils.file_logger import File_Logger_Helper

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Oct 28 2018"


class Data_Preprocessing():

    def __init__(self, logger=None):
        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="preprocessing")


    def get_X_y_featurenames_from_file(self, filename, drop_colnames:list=None, label_colname="label"):
        # TODO: Add Data_Preprocessing for data filtering.
        self.logger.info("Read file {}".format(filename))
        df = pd.read_csv(filename)
        if drop_colnames is not None:
            df = df.drop(drop_colnames, axis=1)

        # Get the features.
        X = df.drop(label_colname, axis=1)

        # Get the label.
        # y = df[[label_colname]]
        y = df[label_colname]

        # # Get the header of the features.
        # feature_names = X.columns.values.tolist()

        return X, y

