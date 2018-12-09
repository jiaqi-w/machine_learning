import ast
import csv
import os
import time
import re

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

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 31 2018"

class Empricial_Experiment:

    def __init__(self,
                 setting_file_list,
                 X, y,
                 data_name="data",
                 text_feature_name_list=("f1", "f2"),
                 other_feature_name_list=("f1", "f2"),
                 target_name_list=("t1", "t2"),
                 num_crossvalidation=10,
                 random_state=312018,
                 test_split=None,
                 # X_unlable=None,
                 #
                 # data_id_list, X_text_train_list, X_feature_train_list, y_train_list,
                 # X_text_test_list, X_feature_test_list, y_test_list,
                 # X_text_eval, X_feature_eval, y_eval,
                 # setting_file_list,

                 # cnn_setting_fname=os.path.join(config.DATA, "shared_task", "empirical_exp", "cnn_agency",
                 #                                "cnn_merge_experiment_settings.csv"),
                 # cnn_merge_setting_fname=os.path.join(config.DATA, "shared_task", "empirical_exp", "cnn_merge_agency",
                 #                                "cnn_merge_experiment_settings.csv"),
                 # cnn_rnn_setting_fname=os.path.join(config.DATA, "shared_task", "empirical_exp", "cnn_rnn_agency","rnn_experiment_settings.csv"),
                 # nn_setting_fname=os.path.join(config.DATA, "shared_task", "empirical_exp", "rnn_agency", "rnn_experiment_settings.csv"),
                 # rnn_setting_fname=os.path.join(config.DATA, "shared_task", "empirical_exp", "nn_agency", "nn_experiment_settings.csv"),
                 evaluate_dir=os.path.join(config.EVALUATE_DATA_DIR, "crossv10"),
                 predict_dir=os.path.join(config.EVALUATE_DATA_DIR, "crossv10"),
                 logger=None
                 ):

        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="exp")
        self.X = X
        self.y = y
        self.y = y
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

        # self.predict_df = pd.DataFrame()
        # if y_eval is not None:
        #     self.predict_df.append(X_text_eval)
        #     self.predict_df.append(X_feature_eval)
        #     self.predict_df["gold"] = y_eval

        self.cv_evaluate_fname = os.path.join(evaluate_dir, "crossv10_{}.csv".format(time.time()))
        # self.cnn_setting_fname = cnn_setting_fname
        # self.cnn_merge_setting_fname = cnn_merge_setting_fname
        # self.cnn_rnn_setting_fname = cnn_rnn_setting_fname
        # self.nn_setting_fname = nn_setting_fname
        # self.rnn_setting_fname = rnn_setting_fname
        self.setting_file_list = setting_file_list


    def run_cv(self, predict_fname=None, replace_exists=False):

        for target in self.target_name_list:
            self.logger.info("Train for target '{}'".format(target))
            data_preprocessing = Data_Preprocessing(
                data_name=self.data_name,
                feature_name=".".join(self.feature_name),
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
                                self.logger.info("Run setting {}".format(row))
                                if origin_header is None:
                                    origin_header = row.keys()

                                cv_model_evaluator = CV_Model_Evaluator()
                                data_preprocessing.init_kfold()
                                for train_index, test_index in data_preprocessing.kfold.split(self.X, self.y):
                                    data_id = "{}_{}".format(data_preprocessing.model_name, i)
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

                                    X_text_train = None
                                    if self.text_feature_name_list is not None:
                                        X_text_train = X_train[self.text_feature_name_list]
                                        X_text_test = X_test[self.text_feature_name_list]

                                    X_feature_train = None
                                    if self.other_feature_name_list is not None:
                                        X_feature_train = X_train[self.other_feature_name_list]
                                        X_feature_test = X_test[self.other_feature_name_list]

                                    y_train = self.y[target].iloc[train_index]
                                    y_test = self.y[target].iloc[test_index]

                                    self.logger.info("y_train distribution: \n{}".format(y_train.value_counts()))
                                    self.logger.info("y_test distribution: \n{}".format(y_test.value_counts()))

                                    fieldnames = None
                                    if modelname == "cnn":
                                        self.logger.info("Run CNN")
                                        fieldnames, evaluate_dict, y_pred = self.run_cnn(row, data_id,
                                                                                         X_text_train, y_train,
                                                                                         X_text_test, y_test,
                                                                                         replace_exists)

                                    elif modelname == "cnn_merge":
                                        self.logger.info("Run CNN Merge")
                                        fieldnames, evaluate_dict, y_pred = self.run_cnn_merge(row, data_id,
                                                                                               X_text_train, X_feature_train,
                                                                                               y_train, X_text_test,
                                                                                               X_feature_test, y_test,
                                                                                               replace_exists)

                                    elif modelname == "cnn_rnn":
                                        self.logger.info("Run CNN RNN")
                                        fieldnames, evaluate_dict, y_pred = self.run_cnn_rnn(row, data_id,
                                                                                             X_text_train, y_train,
                                                                                             X_text_test, y_test,
                                                                                             replace_exists)

                                    elif modelname.startswith("nn"):
                                        self.logger.info("Run NN")
                                        fieldnames, evaluate_dict, y_pred = self.run_nn(row, data_id,
                                                                                               X_text_train, X_feature_train,
                                                                                               y_train, X_text_test,
                                                                                               X_feature_test, y_test,
                                                                                               replace_exists)

                                    elif modelname == "rnn":
                                        self.logger.info("Run RNN")
                                        fieldnames, evaluate_dict, y_pred = self.run_cnn_rnn(row, data_id,
                                                                                             X_text_train, y_train,
                                                                                             X_text_test, y_test,
                                                                                             replace_exists)

                                    if fieldnames is not None:

                                        # predict_df[training.model_name] = y_pred
                                        # The original model's name is too long, index is easier to read.
                                        # self.predict_df["m{}_predict".format(i)] = y_pred
                                        # fieldnames = ["mid"] + csv_reader.fieldnames + fieldnames
                                        self.mid += 1
                                        cv_model_evaluator.add_evaluation_metric(self.mid, evaluate_dict)

                                        if evaluate_csv_writer is None:
                                            # 'max_text_len', 'keneral_size_list', 'embedding_file', 'id', 'batch_size', 'l2_constraint', 'classifier_name', 'num_words', 'pool_size', 'num_filter', 'mid', 'drop_perc', 'data_name', 'embedding_vector_dimension', 'feature_name', 'target_name', 'epochs'
                                            fieldnames = ["mid"] + list(origin_header) + fieldnames
                                            self.logger.info("fieldnames={}".format(fieldnames))
                                            evaluate_csv_writer = csv.DictWriter(evaluate_file, fieldnames=fieldnames)
                                            evaluate_csv_writer.writeheader()
                                            evaluate_file.flush()

                                        new_row = {}
                                        new_row.update(row)
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
                                new_row.update(row)
                                csv_cv_writer.writerow(new_row)
                                cv_evaluate_file.flush()

                    print("output crss result in {}".format(self.cv_evaluate_fname))
                    print("Evaluation result saved in {}".format(self.cv_evaluate_fname))
                    # self.predict_df.to_csv(predict_fname)
                    # print("Prediction result saved in {}".format(predict_fname))

    def run_cnn(self, setting, data_id, X_text_train, y_train, X_text_test, y_test, replace_exists):

        classifier_name = setting["classifier_name"]
        num_words = int(setting["num_words"])
        max_text_len = int(setting["max_text_len"])
        embedding_vector_dimension = int(setting["embedding_vector_dimension"])
        data_name = setting["data_name"]
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

        training = CNN_NLP_Model(
            classifier_name=classifier_name,
            num_words=num_words,
            max_text_len=max_text_len,
            embedding_vector_dimension=embedding_vector_dimension,
            embedding_fname=embedding_fname,
            data_name=data_id,
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
        # print("X_text_train.shape", X_text_train.shape)
        training.train(X_text_train, y_train)
        print("y_test distribution", y_test.value_counts())

        fieldnames, evaluate_dict, y_pred = training.evaluate_model(X_text_test, y_test, output_evaluate_dir=None)
        return fieldnames, evaluate_dict, y_pred


    def run_cnn_rnn(self, setting, data_id, X_text_train, y_train, X_text_test, y_test, replace_exists):

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

        training = CNN_RNN_NLP_Model(
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
        training.train(X_text_train, y_train)
        print("y_test distribution", y_test.value_counts())
        fieldnames, evaluate_dict, y_pred = training.evaluate_model(X_text_test, y_test, output_evaluate_dir=None)
        return fieldnames, evaluate_dict, y_pred

    @staticmethod
    def run_cnn_merge(setting, data_id, X_text_train, X_feature_train, y_train,
                      X_text_test, X_feature_test, y_test, replace_exists):

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

        training = CNN_Merge_NLP_Model(
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
        training.train(X_text_train, y_train, X_feature_train)
        print("y_test distribution", y_test.value_counts())

        fieldnames, evaluate_dict, y_pred = training.evaluate_model(X_text_test, y_test, X_feature_test,
                                output_evaluate_dir=None)
        return fieldnames, evaluate_dict, y_pred

    @staticmethod
    def run_nn(row, data_id, X_text_train, X_feature_train, y_train,
               X_text_test, X_feature_test, y_test, replace_exists):

        classifier_name = row["classifier_name"]
        num_feature = int(row["num_feature"])
        data_name = row["data_name"]
        feature_name = row["feature_name"]
        target_name = row["target_name"]
        neuron_unit_list = ast.literal_eval(row["neuron_unit_list"])
        drop_perc = float(row["drop_perc"])
        batch_size = int(row["batch_size"])
        epochs = int(row["epochs"])

        training = NN_NLP_Model(
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
        training.train(X_feature_train, y_train)
        print("y_test distribution", y_test.value_counts())

        fieldnames, evaluate_dict, y_pred = training.evaluate_model(X_feature_test, y_test, output_evaluate_dir=None)
        return fieldnames, evaluate_dict, y_pred

    @staticmethod
    def run_rnn(setting, data_id, X_train, y_train, X_test, y_test, replace_exists):
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

        training = RNN_NLP_Model(
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
        training.train(X_train, y_train)
        print("y_test distribution", y_test.value_counts())
        fieldnames, evaluate_dict, y_pred = training.evaluate_model(X_test, y_test)

        return fieldnames, evaluate_dict, y_pred