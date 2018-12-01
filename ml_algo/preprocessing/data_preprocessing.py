import pandas as pd
import config
from utils.file_logger import File_Logger_Helper
from utils.pickel_helper import Pickle_Helper
from sklearn.model_selection import train_test_split
import os

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Oct 28 2018"


class Data_Preprocessing():

    def __init__(self, logger=None):
        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="preprocessing")
        self.model_name = ""
        self.X_train= None
        self.y_train = None
        self.X_test= None
        self.y_test = None

    def load_data_if_exists(self,
                             dump_model_dir=config.PREROCESS_PICKLES_DIR):
        # Load the file is not already done so. If there is no pickle created, train one for it.
        self.logger.info("Load Model")

        self.dump_model_dir = dump_model_dir
        if not os.path.exists(dump_model_dir):
            os.makedirs(dump_model_dir)

        self.dump_X_train_fname = os.path.join(dump_model_dir, "{}_X_train.pickle".format(self.model_name))
        self.dump_y_train_fname = os.path.join(dump_model_dir, "{}_y_train.pickle".format(self.model_name))
        self.dump_X_test_fname = os.path.join(dump_model_dir, "{}_X_test.pickle".format(self.model_name))
        self.dump_y_test_fname = os.path.join(dump_model_dir, "{}_y_test.pickle".format(self.model_name))

        self.X_train = Pickle_Helper.load_model_from_pickle(self.dump_X_train_fname)
        self.y_train = Pickle_Helper.load_model_from_pickle(self.dump_y_train_fname)
        self.X_test = Pickle_Helper.load_model_from_pickle(self.dump_X_test_fname)
        self.y_test = Pickle_Helper.load_model_from_pickle(self.dump_y_test_fname)

    def store_data(self, replace_exists=False):
        if not os.path.exists(self.dump_X_train_fname) or replace_exists is True:
            if self.X_train is not None:
                Pickle_Helper.save_model_to_pickle(self.X_train, self.dump_X_train_fname)
        if not os.path.exists(self.dump_y_train_fname) or replace_exists is True:
            if self.y_train is not None:
                Pickle_Helper.save_model_to_pickle(self.y_train, self.dump_y_train_fname)
        if not os.path.exists(self.dump_X_test_fname) or replace_exists is True:
            if self.X_test is not None:
                Pickle_Helper.save_model_to_pickle(self.X_test, self.dump_X_test_fname)
        if not os.path.exists(self.dump_y_test_fname) or replace_exists is True:
            if self.y_test is not None:
                Pickle_Helper.save_model_to_pickle(self.y_test, self.dump_y_test_fname)

    def get_X_y_featurenames_from_dateframe(self, df, drop_colnames:list=None, label_colname="label", dataset="train", check_exits=False):
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

    def get_X_y_featurenames_from_file(self, filename, drop_colnames:list=None, label_colname="label", dataset="train", check_exits=False):
        self.logger.info("Read file {}".format(filename))
        df = pd.read_csv(filename)
        return self.get_X_y_featurenames_from_dateframe(
            df,
            drop_colnames=drop_colnames,
            label_colname=label_colname,
            dataset=dataset,
            check_exits=check_exits
        )

    def get_X_y_featurenames_from_pickle(self, filename, drop_colnames:list=None, label_colname="label", dataset="train", check_exits=False):
        df = Pickle_Helper.load_model_from_pickle(pickle_fname=filename)
        return self.get_X_y_featurenames_from_dateframe(
            df,
            drop_colnames=drop_colnames,
            label_colname=label_colname,
            dataset=dataset,
            check_exits=check_exits
        )

    def get_df_from_pickle(self, filename):
        ret = Pickle_Helper.load_model_from_pickle(filename)
        df = pd.DataFrame(ret)
        # print("columns", df.columns.values)
        return df

    def split_train_test(self, X, y, test_size=0.1, general_name="data", replace_exists=False,  dump_model_dir=config.PREROCESS_PICKLES_DIR):
        # TODO: check exists, if not, split
        self.model_name = "{}_split{}".format(general_name, test_size)
        self.load_data_if_exists(dump_model_dir=dump_model_dir)

        if self.X_train is None or replace_exists:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
            self.store_data(replace_exists=replace_exists)
        self.logger.info("y_train distribution \n{}".format(self.y_train.value_counts()))
        self.logger.info("y_test distribution \n{}".format(self.y_test.value_counts()))
        return self.X_train, self.X_test, self.y_train, self.y_test




