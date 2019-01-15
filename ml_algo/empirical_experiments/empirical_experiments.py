import ast
import csv
import os
import time
import re

import numpy as np

import pandas as pd

import config
from ml_algo.deep_learning.convolutional_neural_network.cnn_nlp_model import CNN_NLP_Model
from ml_algo.deep_learning.ensemble_neural_network.cnn_rnn_nlp_model import CNN_RNN_NLP_Model
from ml_algo.deep_learning.recursive_neural_network.rnn_nlp_model import RNN_NLP_Model
from ml_algo.deep_learning.convolutional_neural_network.cnn_merge_nlp_model import CNN_Merge_NLP_Model
from ml_algo.deep_learning.nn_nlp_model import NN_NLP_Model
from ml_algo.evaluation.cv_model_evaluator import CV_Model_Evaluator
from ml_algo.preprocessing.data_preprocessing import Data_Preprocessing
from utils.file_logger import File_Logger_Helper
from ml_algo.preprocessing.feature_processing import Feature_Processing

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 31 2018"

class Empricial_Experiment:

    def __init__(self,
                 setting_file_list,
                 X, y,
                 multi_class=None,
                 data_name="data",
                 text_feature_name_list=("f1", "f2"),
                 other_feature_name_list=("f1", "f2"),
                 target_name_list=("t1", "t2"),
                 num_crossvalidation=10,
                 random_state=312018,
                 test_split=None,
                 evaluate_dir=os.path.join(config.EVALUATE_DATA_DIR, "crossv10"),
                 predict_dir=os.path.join(config.EVALUATE_DATA_DIR, "crossv10"),
                 logger=None
                 ):

        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="exp")
        self.X = X
        self.y = y
        self.y = y
        self.multi_class = multi_class
        self.data_name = data_name
        self.text_feature_name_list = text_feature_name_list
        self.other_feature_name_list = other_feature_name_list
        self.feature_name = []
        if text_feature_name_list is not None:
            self.feature_name += text_feature_name_list
        if other_feature_name_list is not None:
            self.feature_name += other_feature_name_list
        self.target_name_list = target_name_list
        self.num_crossvalidation = num_crossvalidation
        self.random_state = random_state
        self.test_split = test_split
        # self.X_unlable = X_unlable
        self.evaluate_dir = evaluate_dir
        self.predict_dir = predict_dir
        self.mid = 0

        self.cv_evaluate_fname = os.path.join(evaluate_dir, "crossv10_{}.csv".format(time.time()))
        self.setting_file_list = setting_file_list


    def run_cv(self, replace_exists=False):

        feature_name = ".".join(self.feature_name)
        if len(self.feature_name) > 3:
            feature_name = "{}f".format(len(self.feature_name))

        for target in self.target_name_list:
            self.logger.info("Train for target '{}'".format(target))
            data_preprocessing = Data_Preprocessing(
                data_name=self.data_name,
                feature_name=feature_name,
                target_name=target,
                num_crossvalidation=10,
                random_state=312018,
            )

            for setting_fname in self.setting_file_list:
                self.logger.info("Scan setting: {}".format(setting_fname))
                with open(setting_fname, 'r') as setting_file:
                    output_path = os.path.dirname(setting_fname)
                    basename = str(os.path.basename(setting_fname))
                    modelname = re.sub("_experiment_settings.csv", "", basename)
                    evaluate_fname = os.path.join(output_path,
                                                  re.sub(r"\.csv",
                                                         "_evaluate_{}.csv".format(data_preprocessing.model_name),
                                                         basename))
                    cv_evaluate_fname = os.path.join(output_path,
                                                     re.sub(r"\.csv",
                                                            "_cv_evaluate_{}.csv".format(data_preprocessing.model_name),
                                                            basename))

                    with open(evaluate_fname, 'w') as evaluate_file:
                        csv_cv_writer = None
                        with open(cv_evaluate_fname, 'w') as cv_evaluate_file:

                            csv_reader = csv.DictReader(setting_file)

                            evaluate_csv_writer = None
                            origin_header = None
                            for i, row in enumerate(csv_reader):
                                if self.multi_class is not None and self.multi_class > 0:
                                    self.cv_exp_multi(
                                        setting=row,
                                        data_preprocessing=data_preprocessing,
                                        data_id=self.data_name,
                                        target=target,
                                        modelname=modelname,
                                        replace_exists=replace_exists,
                                        csv_cv_writer=csv_cv_writer,
                                        cv_evaluate_file=cv_evaluate_file,
                                        evaluate_csv_writer=evaluate_csv_writer,
                                        evaluate_file=evaluate_file
                                    )
                                else:
                                    self.cv_exp(
                                        setting=row,
                                        data_preprocessing=data_preprocessing,
                                        data_id=self.data_name,
                                        target=target,
                                        modelname=modelname,
                                        replace_exists=replace_exists,
                                        csv_cv_writer=csv_cv_writer,
                                        cv_evaluate_file=cv_evaluate_file,
                                        evaluate_csv_writer=evaluate_csv_writer,
                                        evaluate_file=evaluate_file
                                    )
                                # self.logger.info("Run setting {}".format(row))
                                # if origin_header is None:
                                #     origin_header = row.keys()
                                #
                                # cv_model_evaluator = CV_Model_Evaluator()
                                # data_preprocessing.init_kfold()
                                # d_id = 0
                                # for train_index, test_index in data_preprocessing.kfold.split(self.X, self.y):
                                #     d_id += 1
                                #     data_id = "{}_{}".format(data_preprocessing.model_name, d_id)
                                #     self.logger.info("train_index {}".format(train_index))
                                #     self.logger.info("test_index {}".format(test_index))
                                #
                                #     train_index = train_index.tolist()
                                #     test_index = test_index.tolist()
                                #
                                #     X_train = self.X.iloc[train_index]
                                #     X_test = self.X.iloc[test_index]
                                #     self.logger.info("X_tain.head()={}".format(X_train.head()))
                                #     self.logger.info("X_tain.shape()={}".format(X_train.shape))
                                #     self.logger.info("X_test.head()={}".format(X_test.head()))
                                #     self.logger.info("X_test.shape()={}".format(X_test.shape))
                                #
                                #     X_text_train = None
                                #     if self.text_feature_name_list is not None:
                                #         X_text_train = X_train[self.text_feature_name_list]
                                #         X_text_test = X_test[self.text_feature_name_list]
                                #
                                #     X_feature_train = None
                                #     if self.other_feature_name_list is not None:
                                #         X_feature_train = X_train[self.other_feature_name_list]
                                #         X_feature_test = X_test[self.other_feature_name_list]
                                #
                                #     y_train = self.y[target].iloc[train_index]
                                #     y_test = self.y[target].iloc[test_index]
                                #
                                #     self.logger.info("y_train distribution.csv: \n{}".format(y_train.value_counts()))
                                #     self.logger.info("y_test distribution.csv: \n{}".format(y_test.value_counts()))
                                #
                                #     fieldnames = None
                                #     if modelname == "cnn":
                                #         self.logger.info("Run CNN")
                                #         fieldnames, evaluate_dict, y_pred = self.run_cnn(row, data_id,
                                #                                                          X_text_train, y_train,
                                #                                                          X_text_test, y_test,
                                #                                                          replace_exists)
                                #
                                #     elif modelname == "cnn_merge":
                                #         self.logger.info("Run CNN Merge")
                                #         fieldnames, evaluate_dict, y_pred = self.run_cnn_merge(row, data_id,
                                #                                                                X_text_train, X_feature_train,
                                #                                                                y_train, X_text_test,
                                #                                                                X_feature_test, y_test,
                                #                                                                replace_exists)
                                #
                                #     elif modelname == "cnn_rnn":
                                #         self.logger.info("Run CNN RNN")
                                #         fieldnames, evaluate_dict, y_pred = self.run_cnn_rnn(row, data_id,
                                #                                                              X_text_train, y_train,
                                #                                                              X_text_test, y_test,
                                #                                                              replace_exists)
                                #
                                #     elif modelname.startswith("nn"):
                                #         self.logger.info("Run NN")
                                #         fieldnames, evaluate_dict, y_pred = self.run_nn(row, data_id,
                                #                                                                X_text_train, X_feature_train,
                                #                                                                y_train, X_text_test,
                                #                                                                X_feature_test, y_test,
                                #                                                                replace_exists)
                                #
                                #     elif modelname == "rnn":
                                #         self.logger.info("Run RNN")
                                #         fieldnames, evaluate_dict, y_pred = self.run_cnn_rnn(row, data_id,
                                #                                                              X_text_train, y_train,
                                #                                                              X_text_test, y_test,
                                #                                                              replace_exists)
                                #
                                #     if fieldnames is not None:
                                #
                                #         # predict_df[training.model_name] = y_pred
                                #         # The original model's name is too long, index is easier to read.
                                #         # self.predict_df["m{}_predict".format(i)] = y_pred
                                #         # fieldnames = ["mid"] + csv_reader.fieldnames + fieldnames
                                #         self.mid += 1
                                #         cv_model_evaluator.add_evaluation_metric(self.mid, evaluate_dict)
                                #
                                #         if evaluate_csv_writer is None:
                                #             fieldnames = ["mid"] + list(origin_header) + fieldnames
                                #             self.logger.info("fieldnames={}".format(fieldnames))
                                #             evaluate_csv_writer = csv.DictWriter(evaluate_file, fieldnames=fieldnames)
                                #             evaluate_csv_writer.writeheader()
                                #             evaluate_file.flush()
                                #
                                #         new_row = {}
                                #         new_row.update(row)
                                #         new_row["mid"] = self.mid
                                #         new_row["data_name"] = data_id
                                #         new_row.update(evaluate_dict)
                                #         evaluate_csv_writer.writerow(new_row)
                                #         evaluate_file.flush()
                                #
                                # if csv_cv_writer is None:
                                #     # Save cross validation results.
                                #     fieldnames = list(origin_header) + cv_model_evaluator.get_evaluation_fieldnames()
                                #     csv_cv_writer = csv.DictWriter(cv_evaluate_file,
                                #                                 fieldnames=fieldnames)
                                #     csv_cv_writer.writeheader()
                                #
                                # new_row = cv_model_evaluator.get_evaluation_dict()
                                # new_row.update(row)
                                # new_row["data_name"] = data_id
                                # csv_cv_writer.writerow(new_row)
                                # cv_evaluate_file.flush()

                    print("output crss result in {}".format(self.cv_evaluate_fname))
                    print("Evaluation result saved in {}".format(self.cv_evaluate_fname))
                    # self.predict_df.to_csv(predict_fname)
                    # print("Prediction result saved in {}".format(predict_fname))


    def cv_exp_multi(self, setting, data_preprocessing, data_id, target, modelname, replace_exists,
               csv_cv_writer, cv_evaluate_file, evaluate_csv_writer, evaluate_file):
        self.logger.info("Run setting {}".format(setting))
        origin_header = setting.keys()

        cv_model_evaluator = CV_Model_Evaluator()
        data_preprocessing.init_kfold(is_multi_class=True)
        d_id = 0
        for train_index, test_index in data_preprocessing.kfold.split(self.X):
            d_id += 1
            data_id = "{}_{}".format(data_preprocessing.model_name, d_id)
            self.logger.info("train_index {}".format(train_index))
            self.logger.info("test_index {}".format(test_index))

            train_index = train_index.tolist()
            test_index = test_index.tolist()

            X_train = self.X.iloc[train_index]
            X_test = self.X.iloc[test_index]
            self.logger.info("X_tain.head()={}".format(X_train.head()))
            self.logger.info("X_tain.shape()={}".format(X_train.shape))
            self.logger.info("X_test.head()={}".format(X_test.head()))
            self.logger.info("X_test.shape()={}".format(X_test.shape))

            X_text_train, X_text_test = None, None
            if self.text_feature_name_list is not None:
                X_text_train = X_train[self.text_feature_name_list]
                # if X_text_pseudo is not None:
                #     self.logger.info("X_text_pseudo.shape()={}".format(X_text_pseudo.shape))
                #     X_text_train = X_text_train.append(X_text_pseudo, ignore_index=True)[self.text_feature_name_list]
                #     self.logger.info("X_text_train.reshape()={}".format(X_text_train.shape))
                X_text_test = X_test[self.text_feature_name_list]

            X_feature_train, X_feature_test = None, None
            if self.other_feature_name_list is not None:
                X_feature_train = X_train[self.other_feature_name_list]
                # if X_feature_pseudo is not None:
                #     self.logger.info("X_feature_pseudo.shape()={}".format(X_feature_pseudo.shape))
                #     X_feature_train = X_feature_train.append(X_feature_pseudo, ignore_index=True)[self.other_feature_name_list]
                #     self.logger.info("X_feature_train.reshape()={}".format(X_feature_train.shape))
                X_feature_test = X_test[self.other_feature_name_list]

            y_train = self.y[train_index]

            self.logger.info("y_train={}".format(y_train))
            self.logger.info("y_train.shape()={}".format(y_train.shape))
            # if y_pseudo is not None:
            #     self.logger.info("y_pseudo.shape()={}".format(y_pseudo.shape))
            #     y_train = pd.DataFrame(np.append(y_train.values, y_pseudo.values), columns=[target])[target]
            #     self.logger.info("semi y_train.shape()={}".format(y_train.shape))
            y_test = self.y[test_index]

            # self.logger.info("y_train distribution.csv: \n{}".format(y_train.value_counts()))
            # self.logger.info("y_test distribution.csv: \n{}".format(y_test.value_counts()))

            fieldnames, evaluate_dict, y_pred = self.run_one_validation(setting, data_id, modelname, replace_exists,
                                                                        X_text_train, X_feature_train, y_train,
                                                                        X_text_test, X_feature_test, y_test)

            if fieldnames is not None:

                # predict_df[training.model_name] = y_pred
                # The original model's name is too long, index is easier to read.
                # self.predict_df["m{}_predict".format(i)] = y_pred
                # fieldnames = ["mid"] + csv_reader.fieldnames + fieldnames
                self.mid += 1
                cv_model_evaluator.add_evaluation_metric(self.mid, evaluate_dict)

                if evaluate_csv_writer is None:
                    fieldnames = ["mid"] + list(origin_header) + fieldnames
                    self.logger.info("fieldnames={}".format(fieldnames))
                    evaluate_csv_writer = csv.DictWriter(evaluate_file, fieldnames=fieldnames)
                    evaluate_csv_writer.writeheader()
                    evaluate_file.flush()

                new_row = {}
                new_row.update(setting)
                new_row["mid"] = self.mid
                new_row["data_name"] = data_id
                new_row.update(evaluate_dict)
                evaluate_csv_writer.writerow(new_row)
                evaluate_file.flush()

        if csv_cv_writer is None:
            # Save cross validation results.
            fieldnames = list(origin_header) + cv_model_evaluator.get_evaluation_fieldnames()
            csv_cv_writer = csv.DictWriter(cv_evaluate_file,
                                           fieldnames=fieldnames)
            csv_cv_writer.writeheader()

        new_row = cv_model_evaluator.get_evaluation_dict()
        new_row.update(setting)
        new_row["data_name"] = data_id
        csv_cv_writer.writerow(new_row)
        cv_evaluate_file.flush()

    def cv_exp(self, setting, data_preprocessing, data_id, target, modelname, replace_exists,
               csv_cv_writer, cv_evaluate_file, evaluate_csv_writer, evaluate_file,
               X_text_pseudo=None, X_feature_pseudo=None, y_pseudo=None):
        self.logger.info("Run setting {}".format(setting))
        origin_header = setting.keys()

        cv_model_evaluator = CV_Model_Evaluator()
        if self.multi_class is not None and self.multi_class > 0:
            data_preprocessing.init_kfold(is_multi_class=True)
        else:
            data_preprocessing.init_kfold()
        d_id = 0
        for train_index, test_index in data_preprocessing.kfold.split(self.X, self.y):
            d_id += 1
            data_id = "{}_{}".format(data_preprocessing.model_name, d_id)
            self.logger.info("train_index {}".format(train_index))
            self.logger.info("test_index {}".format(test_index))

            train_index = train_index.tolist()
            test_index = test_index.tolist()

            X_train = self.X.iloc[train_index]
            X_test = self.X.iloc[test_index]
            self.logger.info("X_tain.head()={}".format(X_train.head()))
            self.logger.info("X_tain.shape()={}".format(X_train.shape))
            self.logger.info("X_test.head()={}".format(X_test.head()))
            self.logger.info("X_test.shape()={}".format(X_test.shape))

            X_text_train, X_text_test = None, None
            if self.text_feature_name_list is not None:
                X_text_train = X_train[self.text_feature_name_list]
                if X_text_pseudo is not None:
                    self.logger.info("X_text_pseudo.shape()={}".format(X_text_pseudo.shape))
                    X_text_train = X_text_train.append(X_text_pseudo, ignore_index=True)[self.text_feature_name_list]
                    self.logger.info("X_text_train.reshape()={}".format(X_text_train.shape))
                X_text_test = X_test[self.text_feature_name_list]

            X_feature_train, X_feature_test = None, None
            if self.other_feature_name_list is not None:
                X_feature_train = X_train[self.other_feature_name_list]
                if X_feature_pseudo is not None:
                    self.logger.info("X_feature_pseudo.shape()={}".format(X_feature_pseudo.shape))
                    X_feature_train = X_feature_train.append(X_feature_pseudo, ignore_index=True)[self.other_feature_name_list]
                    self.logger.info("X_feature_train.reshape()={}".format(X_feature_train.shape))
                X_feature_test = X_test[self.other_feature_name_list]

            y_train = self.y.iloc[train_index][target]

            self.logger.info("y_train.head()={}".format(y_train.head()))
            self.logger.info("y_train.shape()={}".format(y_train.shape))
            if y_pseudo is not None:
                self.logger.info("y_pseudo.shape()={}".format(y_pseudo.shape))
                y_train = pd.DataFrame(np.append(y_train.values, y_pseudo.values), columns=[target])[target]
                self.logger.info("semi y_train.shape()={}".format(y_train.shape))
            y_test = self.y[target].iloc[test_index]

            self.logger.info("y_train distribution.csv: \n{}".format(y_train.value_counts()))
            self.logger.info("y_test distribution.csv: \n{}".format(y_test.value_counts()))

            fieldnames, evaluate_dict, y_pred = self.run_one_validation(setting, data_id, modelname, replace_exists,
                                                                        X_text_train, X_feature_train, y_train,
                                                                        X_text_test, X_feature_test, y_test)

            if fieldnames is not None:

                # predict_df[training.model_name] = y_pred
                # The original model's name is too long, index is easier to read.
                # self.predict_df["m{}_predict".format(i)] = y_pred
                # fieldnames = ["mid"] + csv_reader.fieldnames + fieldnames
                self.mid += 1
                cv_model_evaluator.add_evaluation_metric(self.mid, evaluate_dict)

                if evaluate_csv_writer is None:
                    fieldnames = ["mid"] + list(origin_header) + fieldnames
                    self.logger.info("fieldnames={}".format(fieldnames))
                    evaluate_csv_writer = csv.DictWriter(evaluate_file, fieldnames=fieldnames)
                    evaluate_csv_writer.writeheader()
                    evaluate_file.flush()

                new_row = {}
                new_row.update(setting)
                new_row["mid"] = self.mid
                new_row["data_name"] = data_id
                new_row.update(evaluate_dict)
                evaluate_csv_writer.writerow(new_row)
                evaluate_file.flush()

        if csv_cv_writer is None:
            # Save cross validation results.
            fieldnames = list(origin_header) + cv_model_evaluator.get_evaluation_fieldnames()
            csv_cv_writer = csv.DictWriter(cv_evaluate_file,
                                           fieldnames=fieldnames)
            csv_cv_writer.writeheader()

        new_row = cv_model_evaluator.get_evaluation_dict()
        new_row.update(setting)
        new_row["data_name"] = data_id
        csv_cv_writer.writerow(new_row)
        cv_evaluate_file.flush()

    def run_one_validation(self, setting, data_id, modelname, replace_exists,
                           X_text_train, X_feature_train, y_train,
                           X_text_test, X_feature_test, y_test):
        self.logger.info("Run setting {}".format(setting))

        fieldnames, evaluate_dict, y_pred = None, None, None
        if modelname == "cnn" or modelname == "cnn_semi":
            self.logger.info("Run CNN")
            fieldnames, evaluate_dict, y_pred = self.run_cnn(setting, data_id,
                                                             X_text_train, y_train,
                                                             X_text_test, y_test,
                                                             replace_exists)

        elif modelname == "cnn_merge" or modelname == "cnn_merge_semi":
            self.logger.info("Run CNN Merge")
            fieldnames, evaluate_dict, y_pred = self.run_cnn_merge(setting, data_id,
                                                                   X_text_train, X_feature_train,
                                                                   y_train, X_text_test,
                                                                   X_feature_test, y_test,
                                                                   replace_exists)

        elif modelname == "cnn_rnn" or modelname == "cnn_rnn_semi":
            self.logger.info("Run CNN RNN")
            fieldnames, evaluate_dict, y_pred = self.run_cnn_rnn(setting, data_id,
                                                                 X_text_train, y_train,
                                                                 X_text_test, y_test,
                                                                 replace_exists)

        elif modelname.startswith("nn") or modelname == "nn_semi":
            self.logger.info("Run NN")
            fieldnames, evaluate_dict, y_pred = self.run_nn(setting, data_id,
                                                            X_text_train, X_feature_train,
                                                            y_train, X_text_test,
                                                            X_feature_test, y_test,
                                                            replace_exists)

        elif modelname == "rnn" or modelname == "rnn_semi":
            self.logger.info("Run RNN")
            fieldnames, evaluate_dict, y_pred = self.run_cnn_rnn(setting, data_id,
                                                                 X_text_train, y_train,
                                                                 X_text_test, y_test,
                                                                 replace_exists)

        return fieldnames, evaluate_dict, y_pred



    def define_cnn(self, setting, data_id, replace_exists):

        classifier_name = setting["classifier_name"]
        num_words = int(setting["num_words"])
        max_text_len = int(setting["max_text_len"])
        embedding_vector_dimension = int(setting["embedding_vector_dimension"])
        data_name = setting["data_name"]
        data_name = data_id
        self.logger.info("date_name {}".format(data_name))
        feature_name = setting["feature_name"]
        target_name = setting["target_name"]
        num_filter = int(setting["num_filter"])
        keneral_size_list = ast.literal_eval(setting["keneral_size_list"])
        pool_size = int(setting["pool_size"])
        drop_perc = float(setting["drop_perc"])
        l2_constraint = int(setting["l2_constraint"])
        batch_size = int(setting["batch_size"])
        epochs = int(setting["epochs"])
        embedding_fname = setting["embedding_file"]
        if embedding_fname is not None:
            embedding_fname = os.path.join(config.WORD_EMBEDDING_DIR, embedding_fname)

        model = CNN_NLP_Model(
            classifier_name=classifier_name,
            num_words=num_words,
            max_text_len=max_text_len,
            embedding_vector_dimension=embedding_vector_dimension,
            embedding_fname=embedding_fname,
            data_name=data_name,
            num_filter=num_filter,
            keneral_size_list=keneral_size_list,
            pool_size=pool_size,
            drop_perc=drop_perc,
            l2_constraint=l2_constraint,
            batch_size=batch_size,
            epochs=epochs,
            feature_name=feature_name,
            target_name=target_name,
            replace_exists=replace_exists
        )
        return model

    def run_cnn(self, setting, data_id, X_text_train, y_train, X_text_test, y_test, replace_exists):

        model = self.define_cnn(setting, data_id, replace_exists)
        # print("X_text_train.shape", X_text_train.shape)
        model.train(X_text_train, y_train)
        print("y_test distribution.csv", y_test.value_counts())

        fieldnames, evaluate_dict, y_pred = model.evaluate_model(X_text_test, y_test, output_evaluate_dir=None)
        return fieldnames, evaluate_dict, y_pred

    def define_cnn_rnn(self, setting, data_id, replace_exists):
        classifier_name = setting["classifier_name"]
        num_words = int(setting["num_words"])
        max_text_len = int(setting["max_text_len"])
        embedding_vector_dimension = int(setting["embedding_vector_dimension"])
        data_name = setting["data_name"]
        feature_name = setting["feature_name"]
        target_name = setting["target_name"]
        num_filter = int(setting["num_filter"])
        keneral_size = int(setting["keneral_size"])
        pool_size = int(setting["pool_size"])
        drop_perc = float(setting["drop_perc"])
        l2_constraint = int(setting["l2_constraint"])
        batch_size = int(setting["batch_size"])
        epochs = int(setting["epochs"])
        embedding_fname = setting["embedding_file"]
        if embedding_fname is not None:
            embedding_fname = os.path.join(config.WORD_EMBEDDING_DIR, embedding_fname)

        model = CNN_RNN_NLP_Model(
            classifier_name=classifier_name,
            num_words=num_words,
            max_text_len=max_text_len,
            embedding_vector_dimension=embedding_vector_dimension,
            embedding_fname=embedding_fname,
            data_name=data_id,
            num_filter=num_filter,
            keneral_size=keneral_size,
            pool_size=pool_size,
            drop_perc=drop_perc,
            l2_constraint=l2_constraint,
            batch_size=batch_size,
            epochs=epochs,
            feature_name=feature_name,
            target_name=target_name,
            replace_exists=replace_exists
        )
        return model

    def run_cnn_rnn(self, setting, data_id, X_text_train, y_train, X_text_test, y_test, replace_exists):

        model = self.define_cnn_rnn(setting, data_id, replace_exists)
        model.train(X_text_train, y_train)
        print("y_test distribution.csv", y_test.value_counts())
        fieldnames, evaluate_dict, y_pred = model.evaluate_model(X_text_test, y_test, output_evaluate_dir=None)
        return fieldnames, evaluate_dict, y_pred

    def define_cnn_merge(self, setting, data_id, replace_exists):
        classifier_name = setting["classifier_name"]
        num_words = int(setting["num_words"])
        max_text_len = int(setting["max_text_len"])
        embedding_vector_dimension = int(setting["embedding_vector_dimension"])
        data_name = setting["data_name"]
        feature_name = setting["feature_name"]
        target_name = setting["target_name"]
        num_custome_features = int(setting["num_custome_features"])
        num_filter = int(setting["num_filter"])
        keneral_size_list = ast.literal_eval(setting["keneral_size_list"])
        pool_size = int(setting["pool_size"])
        drop_perc = float(setting["drop_perc"])
        l2_constraint = int(setting["l2_constraint"])
        batch_size = int(setting["batch_size"])
        epochs = int(setting["epochs"])
        embedding_fname = setting["embedding_file"]
        if embedding_fname is not None:
            embedding_fname = os.path.join(config.WORD_EMBEDDING_DIR, embedding_fname)

        model = CNN_Merge_NLP_Model(
            classifier_name=classifier_name,
            num_words=num_words,
            max_text_len=max_text_len,
            embedding_vector_dimension=embedding_vector_dimension,
            embedding_fname=embedding_fname,
            data_name=data_id,
            num_custome_features=num_custome_features,
            num_filter=num_filter,
            keneral_size_list=keneral_size_list,
            pool_size=pool_size,
            drop_perc=drop_perc,
            l2_constraint=l2_constraint,
            batch_size=batch_size,
            epochs=epochs,
            feature_name=feature_name,
            target_name=target_name,
            replace_exists=replace_exists
        )
        return model

    def run_cnn_merge(self, setting, data_id, X_text_train, X_feature_train, y_train,
                      X_text_test, X_feature_test, y_test, replace_exists):

        model = self.define_cnn_merge(setting, data_id, replace_exists)
        if self.multi_class is not None and self.multi_class > 0:
            model.num_class = self.multi_class
        model.train(X_text_train, y_train, X_feature_train)
        # print("y_test distribution.csv", y_test.value_counts())

        fieldnames, evaluate_dict, y_pred = model.evaluate_model(X_text_test, y_test, X_feature_test,
                                output_evaluate_dir=None)
        return fieldnames, evaluate_dict, y_pred

    def define_nn(self, setting, data_id, replace_exists):

        classifier_name = setting["classifier_name"]
        num_feature = int(setting["num_feature"])
        data_name = setting["data_name"]
        feature_name = setting["feature_name"]
        target_name = setting["target_name"]
        neuron_unit_list = ast.literal_eval(setting["neuron_unit_list"])
        drop_perc = float(setting["drop_perc"])
        batch_size = int(setting["batch_size"])
        epochs = int(setting["epochs"])

        model = NN_NLP_Model(
            classifier_name=classifier_name,
            data_name=data_id,
            drop_perc=drop_perc,
            num_feature=num_feature,
            neuron_unit_list=neuron_unit_list,
            batch_size=batch_size,
            epochs=epochs,
            feature_name=feature_name,
            target_name=target_name,
            replace_exists=replace_exists
        )
        return model

    def run_nn(self, setting, data_id, X_text_train, X_feature_train, y_train,
               X_text_test, X_feature_test, y_test, replace_exists):

        model = self.define_nn(setting, data_id, replace_exists)
        model.train(X_feature_train, y_train)
        print("y_test distribution.csv", y_test.value_counts())

        fieldnames, evaluate_dict, y_pred = model.evaluate_model(X_feature_test, y_test, output_evaluate_dir=None)
        return fieldnames, evaluate_dict, y_pred

    def define_rnn(self, setting, data_id, replace_exists):
        classifier_name = setting["classifier_name"]
        num_words = int(setting["num_words"])
        max_text_len = int(setting["max_text_len"])
        embedding_vector_dimension = int(setting["embedding_vector_dimension"])
        data_name = setting["data_name"]
        feature_name = setting["feature_name"]
        target_name = setting["target_name"]
        num_class = int(setting["num_class"])
        num_lstm_layer = int(setting["num_lstm_layer"])
        drop_perc = float(setting["drop_perc"])
        # l2_constraint = int(row["l2_constraint"])
        batch_size = int(setting["batch_size"])
        epochs = int(setting["epochs"])
        embedding_fname = setting["embedding_file"]
        if embedding_fname is not None:
            embedding_fname = os.path.join(config.WORD_EMBEDDING_DIR, embedding_fname)

        model = RNN_NLP_Model(
            classifier_name=classifier_name,
            num_words=num_words,
            max_text_len=max_text_len,
            embedding_vector_dimension=embedding_vector_dimension,
            embedding_fname=embedding_fname,
            data_name=data_id,
            num_class=num_class,
            num_lstm_layer=num_lstm_layer,
            drop_perc=drop_perc,
            # l2_constraint=l2_constraint,
            batch_size=batch_size,
            epochs=epochs,
            feature_name=feature_name,
            target_name=target_name,
            replace_exists=replace_exists
        )
        return model

    def run_rnn(self, setting, data_id, X_train, y_train, X_test, y_test, replace_exists):
        model = self.define_rnn(setting, data_id, replace_exists)
        model.train(X_train, y_train)
        print("y_test distribution.csv", y_test.value_counts())
        fieldnames, evaluate_dict, y_pred = model.evaluate_model(X_test, y_test)

        return fieldnames, evaluate_dict, y_pred

    def get_model(self, modelname, setting, data_id, replace_exists=False):

        fieldnames, evaluate_dict, y_pred, model = None, None, None, None
        if modelname == "cnn" or modelname == "cnn_semi":
            self.logger.info("Load CNN")
            model = self.define_cnn(setting, data_id, replace_exists)

        elif modelname == "cnn_merge" or modelname == "cnn_merge_semi":
            self.logger.info("Load CNN Merge")
            model = self.define_cnn_merge(setting, data_id, replace_exists)

        elif modelname == "cnn_rnn" or modelname == "cnn_rnn_semi":
            self.logger.info("Load CNN RNN")
            model = self.define_cnn_rnn(setting, data_id, replace_exists)

        elif modelname.startswith("nn") or modelname == "nn_semi":
            self.logger.info("Load NN")
            model = self.define_nn(setting, data_id, replace_exists)

        elif modelname == "rnn" or modelname == "rnn_semi":
            self.logger.info("Load RNN")
            model = self.define_cnn_rnn(setting, data_id, replace_exists)

        return model

    def run_semi_supervise(self,
                           X_text_unlabled:pd.Series, X_feature_unlabled:pd.Series, pseudo_label_dir, pseudo_label_name,
                           X_text_test:pd.Series, X_feature_test:pd.Series, predict_dir, predict_name, should_train=False
                           ):

        feature_name = ".".join(self.feature_name)
        if len(self.feature_name) > 3:
            feature_name = "{}f".format(len(self.feature_name))
        for target in self.target_name_list:
            data_preprocessing = Data_Preprocessing(
                data_name=self.data_name,
                feature_name=feature_name,
                target_name=target,
                num_crossvalidation=10,
                random_state=312018,
            )

            for setting_fname in self.setting_file_list:
                self.logger.info("Scan setting: {}".format(setting_fname))
                with open(setting_fname, 'r') as setting_file:
                    output_path = os.path.dirname(setting_fname)
                    basename = str(os.path.basename(setting_fname))
                    modelname = re.sub("_experiment_settings.csv", "", basename)
                    evaluate_fname = os.path.join(output_path,
                                                  re.sub(r"\.csv",
                                                         "_evaluate_{}.csv".format(data_preprocessing.model_name),
                                                         basename))
                    cv_evaluate_fname = os.path.join(output_path,
                                                     re.sub(r"\.csv",
                                                            "_cv_evaluate_{}.csv".format(data_preprocessing.model_name),
                                                            basename))
                    new_evaluate_fname = os.path.join(output_path,
                                                      re.sub(r"_evaluate_", "semi_evaluate_", evaluate_fname))

                    new_cv_evaluate_fname = os.path.join(output_path,
                                                  re.sub(r"_cv_evaluate_", "semi_cv_evaluate_", cv_evaluate_fname))


                    with open(new_evaluate_fname, 'w') as new_evaluate_file:
                        new_csv_cv_writer = None
                        with open(new_cv_evaluate_fname, 'w') as new_cv_evaluate_file:

                            new_evaluate_csv_writer = None

                            setting_dict = {}
                            with open(evaluate_fname, 'r') as evaluate_file:
                                csv_reader = csv.DictReader(evaluate_file)
                                for row in csv_reader:
                                    setting_dict[row["mid"]] = row

                            with open(cv_evaluate_fname, 'r') as cv_evaluate_file:
                                csv_reader = csv.DictReader(cv_evaluate_file)
                                for row in csv_reader:
                                    for j in range(self.num_crossvalidation):
                                        # MUST MUST notice that the replace is False since we don't want to replace the best model!!
                                        # use the best macro f1 model
                                        if "best_macro_f1_mid" in row and row["best_macro_f1_mid"] in setting_dict:
                                            setting = setting_dict[row["best_macro_f1_mid"]]
                                            if setting is not None:
                                                self.logger.info("Best macro_f1 {} model setting {}".format(row["best_macro_f1"],
                                                                                                            setting))
                                                model = self.get_model(setting["classifier_name"],
                                                                       setting,
                                                                       setting["data_name"],
                                                                       replace_exists=False)

                                                self.logger.info("Best model {}".format(model.model_name))
                                                y_pseudo = model.predict_y(X_text_pred=X_text_unlabled,
                                                                         X_feature_pred=X_feature_unlabled
                                                                         )
                                                output_fname = os.path.join(pseudo_label_dir, "{}_{}_{}_pseudo_lable.csv".format(target, pseudo_label_name, model.model_name))
                                                self.logger.info("y_pseudo {}".format(np.unique(y_pseudo, return_counts=True)))
                                                df = pd.DataFrame(data=y_pseudo)
                                                df.to_csv(output_fname)
                                                self.logger.info("Output result to {}".format(output_fname))

                                                if should_train:
                                                    row["classifier_name"] = "{}_semi".format(row["classifier_name"])

                                                    # run corss validation experiment on new model

                                                    self.cv_exp(
                                                        setting=row,
                                                        data_preprocessing=data_preprocessing,
                                                        data_id=self.data_name,
                                                        target=target,
                                                        modelname=row["classifier_name"],
                                                        replace_exists=False,
                                                        csv_cv_writer=new_csv_cv_writer,
                                                        cv_evaluate_file=new_cv_evaluate_file,
                                                        evaluate_csv_writer=new_evaluate_csv_writer,
                                                        evaluate_file=new_evaluate_file,
                                                        X_text_pseudo=X_text_unlabled,
                                                        X_feature_pseudo=X_feature_unlabled,
                                                        y_pseudo=y_pseudo
                                                    )

    def predict_by_setting(self, target, setting, X_text_pred, X_feature_pred, predict_dir):
        if setting is not None:
            model = self.get_model(setting["classifier_name"],
                                   setting,
                                   setting["data_name"],
                                   replace_exists=False)

            self.logger.info("Get {} model {}".format(setting["classifier_name"], model.model_name))
            y_predict = model.predict_y(X_text_pred=X_text_pred,
                                       X_feature_pred=X_feature_pred
                                       )
            output_fname = os.path.join(predict_dir,
                                        "{}_{}_predict.csv".format(target, model.model_name))
            self.logger.info("y_pseudo {}".format(np.unique(y_predict, return_counts=True)))
            df = pd.DataFrame(data=y_predict)
            # df.index.name = "Test_Instance"
            df.to_csv(output_fname)
            self.logger.info("Output result to {}".format(output_fname))




