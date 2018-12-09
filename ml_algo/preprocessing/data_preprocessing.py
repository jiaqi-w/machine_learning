import pandas as pd
import config
from utils.file_logger import File_Logger_Helper
from utils.pickel_helper import Pickle_Helper
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Oct 28 2018"


class Data_Preprocessing():

    def __init__(self,
                 data_name="data",
                 feature_name="f1.f2",
                 target_name="t",
                 num_crossvalidation=10,
                 random_state=312018,
                 test_split=None,
                 replace_exists=False,
                 logger=None):
        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="data_preprocessing")
        self.model_name = ""
        self.data_name= data_name
        self.feature_name= feature_name
        self.target_name= target_name
        self.num_crossvalidation= num_crossvalidation
        self.random_state= random_state
        self.test_split= test_split
        self.replace_exists= replace_exists
        self.y_train = None
        self.X_test= None
        self.y_test = None
        self.kfold = None
        self.kod = 0

        self.load_data_if_exists()


    def generate_model_name(self):

        self.model_name = self.data_name
        if self.feature_name is not None:
            self.model_name = self.model_name + "_{}".format(self.feature_name)
        if self.target_name is not None:
            self.model_name = self.model_name + "_{}".format(self.target_name)

        if self.num_crossvalidation is not None:
            self.model_name = self.model_name + "_{}cv".format(self.num_crossvalidation)
        if self.random_state is not None:
            self.model_name = self.model_name + "_{}rs".format(self.random_state)
        if self.test_split is not None:
            self.model_name = self.model_name + "_{}ts".format(self.test_split)
        self.logger.info("model_name={}".format(self.model_name))
        return self.model_name

    def load_data_if_exists(self,
                             dump_model_dir=config.PREROCESS_PICKLES_DIR):
        # Load the file is not already done so. If there is no pickle created, train one for it.
        self.dump_model_dir = dump_model_dir
        if not os.path.exists(dump_model_dir):
            os.makedirs(dump_model_dir)

        self.generate_model_name()

        self.logger.info("Load data model {}".format(self.model_name))

        self.dump_X_train_fname = os.path.join(dump_model_dir, "{}_X_train.pickle".format(self.model_name))
        self.dump_y_train_fname = os.path.join(dump_model_dir, "{}_y_train.pickle".format(self.model_name))
        self.dump_X_test_fname = os.path.join(dump_model_dir, "{}_X_test.pickle".format(self.model_name))
        self.dump_y_test_fname = os.path.join(dump_model_dir, "{}_y_test.pickle".format(self.model_name))

        if self.replace_exists == False:
            self.X_train = Pickle_Helper.load_model_from_pickle(self.dump_X_train_fname)
            self.y_train = Pickle_Helper.load_model_from_pickle(self.dump_y_train_fname)
            self.X_test = Pickle_Helper.load_model_from_pickle(self.dump_X_test_fname)
            self.y_test = Pickle_Helper.load_model_from_pickle(self.dump_y_test_fname)

        self.dump_kfold_fname = os.path.join(dump_model_dir, "{}_kfold.pickle".format(self.model_name))
        if self.replace_exists == False:
            self.kfold = Pickle_Helper.load_model_from_pickle(self.dump_kfold_fname)

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

        if not os.path.exists(self.dump_kfold_fname) or replace_exists is True:
            if self.kfold is not None:
                Pickle_Helper.save_model_to_pickle(self.kfold, self.dump_kfold_fname)

    def get_X_y_featurenames_from_dateframe(self, df, feature_columns:list=None,
                                            label_colnames:list=("label"),
                                            drop_colnames:list=None):

        # Get the label.
        # y = df[[label_colname]]
        y = df[label_colnames]

        X = None
        if feature_columns is not None:
            X = df[feature_columns]

        if drop_colnames is not None:
            X = df.drop(X, axis=1)

        # # Get the header of the features.
        # feature_names = X.columns.values.tolist()

        return X, y

    def get_X_y_featurenames_from_file(self, filename,
                                       feature_columns:list=None,
                                       label_colnames:list=("label"),
                                       drop_colnames:list=None):
        self.logger.info("Read file {}".format(filename))
        df = pd.read_csv(filename)
        self.logger.info("columns {}".format(df.columns.values))
        self.logger.info("head {}".format(df.values))
        return self.get_X_y_featurenames_from_dateframe(
            df,
            feature_columns=feature_columns,
            label_colnames=label_colnames,
            drop_colnames=drop_colnames,
        )

    def get_X_y_featurenames_from_pickle(self, filename,
                                         feature_columns:list=None,
                                         label_colnames:list=("label"),
                                         drop_colnames:list=None):
        df = Pickle_Helper.load_model_from_pickle(pickle_fname=filename)
        return self.get_X_y_featurenames_from_dateframe(
            df,
            feature_columns=feature_columns,
            label_colnames=label_colnames,
            drop_colnames=drop_colnames
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

    def init_kfold(self):
        # if self.kfold is None or self.replace_exists:
            # random_state=None, shuffle=False
            # self.kfold = KFold(n_splits=n_splits)
            # use random_state to regenerate the same splits.
        self.logger.info("Initial kStratifiedKFold {} nsplit {} random state".format(self.num_crossvalidation,
                                                                                     self.random_state))
        self.kfold = StratifiedKFold(n_splits=self.num_crossvalidation, random_state=self.random_state)

        # for train_index, test_index in self.kfold.split(X, y):
        #     print("TRAIN:", train_index, "TEST:", test_index)
        #     # We don't store the data here, since it's redundant to save so many files.
        #     X_train, X_test = X[train_index], X[test_index]
        #     y_train, y_test = y[train_index], y[test_index]





