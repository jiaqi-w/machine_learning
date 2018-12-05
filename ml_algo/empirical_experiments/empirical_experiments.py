import ast
import csv
import os

import pandas as pd

import config
from ml_algo.deep_learning.convolutional_neural_network.cnn_nlp_model import CNN_NLP_Model
from ml_algo.deep_learning.ensemble_neural_network.cnn_rnn_nlp_model import CNN_RNN_NLP_Model
from ml_algo.deep_learning.recursive_neural_network.rnn_nlp_model import RNN_NLP_Model

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 31 2018"


def run_cnn(cnn_setting_fname, X_train, y_train, X_test:pd.Series, y_test:pd.Series,
            evaluate_fname, predict_fname, replace_exists):

    with open(evaluate_fname, 'w', newline='') as evaluate_file:
        with open(cnn_setting_fname, 'r') as setting_file:
            csv_reader = csv.DictReader(setting_file)

            evaluate_csv_writer = None
            predict_df = pd.DataFrame(data=X_test)
            predict_df["gold"] = y_test

            for i, row in enumerate(csv_reader):
                #classifier_name,num_words,max_text_len,embedding_vector_dimension,data_name,feature_name,target_name,num_filter,keneral_size_list,pool_size,drop_perc,l2_constraint,batch_size,epochs
                classifier_name = row["classifier_name"]
                num_words = int(row["num_words"])
                max_text_len = int(row["max_text_len"])
                embedding_vector_dimension = int(row["embedding_vector_dimension"])
                data_name = row["data_name"]
                feature_name = row["feature_name"]
                target_name = row["target_name"]
                num_filter = int(row["num_filter"])
                keneral_size_list = ast.literal_eval(row["keneral_size_list"])
                pool_size = int(row["pool_size"])
                drop_perc = float(row["drop_perc"])
                l2_constraint = int(row["l2_constraint"])
                batch_size = int(row["batch_size"])
                epochs = int(row["epochs"])
                embedding_fname = row["embedding_file"]
                if embedding_fname is not None:
                    embedding_fname = os.path.join(config.WORD_EMBEDDING_DIR, embedding_fname)

                training = CNN_NLP_Model(
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
                training.train(X_train, y_train)
                print("y_test distribution", y_test.value_counts())

                fieldnames, evaluate_dict, y_pred = training.evaluate_model(X_test, y_test, output_evaluate_dir=None)
                # predict_df[training.model_name] = y_pred
                # The original model's name is too long, index is easier to read.
                predict_df["m{}_predict".format(i)] = y_pred
                fieldnames = ["id"] + csv_reader.fieldnames + fieldnames

                if evaluate_csv_writer is None:
                    evaluate_csv_writer = csv.DictWriter(evaluate_file, fieldnames=fieldnames)
                    evaluate_csv_writer.writeheader()
                    evaluate_file.flush()

                new_row = {}
                new_row.update(row)
                new_row["id"] = i
                new_row.update(evaluate_dict)
                evaluate_csv_writer.writerow(new_row)
                evaluate_file.flush()
        print("Evaluation result saved in {}".format(evaluate_fname))
        predict_df.to_csv(predict_fname)
        print("Prediction result saved in {}".format(predict_fname))

def run_cnn_rnn(cnn_setting_fname, X_train, y_train, X_test, y_test, evaluate_fname, predict_fname, replace_exists):

    with open(evaluate_fname, 'w', newline='') as evaluate_file:
        with open(cnn_setting_fname, 'r') as setting_file:
            csv_reader = csv.DictReader(setting_file)

            evaluate_csv_writer = None
            predict_df = pd.DataFrame(data=X_test)
            predict_df["gold"] = y_test

            for i, row in enumerate(csv_reader):
                #classifier_name,num_words,max_text_len,embedding_vector_dimension,data_name,feature_name,target_name,num_filter,keneral_size_list,pool_size,drop_perc,l2_constraint,batch_size,epochs
                classifier_name = row["classifier_name"]
                num_words = int(row["num_words"])
                max_text_len = int(row["max_text_len"])
                embedding_vector_dimension = int(row["embedding_vector_dimension"])
                data_name = row["data_name"]
                feature_name = row["feature_name"]
                target_name = row["target_name"]
                num_filter = int(row["num_filter"])
                keneral_size = int(row["keneral_size"])
                pool_size = int(row["pool_size"])
                drop_perc = float(row["drop_perc"])
                l2_constraint = int(row["l2_constraint"])
                batch_size = int(row["batch_size"])
                epochs = int(row["epochs"])
                embedding_fname = row["embedding_file"]
                if embedding_fname is not None:
                    embedding_fname = os.path.join(config.WORD_EMBEDDING_DIR, embedding_fname)

                training = CNN_RNN_NLP_Model(
                    classifier_name=classifier_name,
                    num_words=num_words,
                    max_text_len=max_text_len,
                    embedding_vector_dimension=embedding_vector_dimension,
                    embedding_fname=embedding_fname,
                    data_name=data_name,
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
                training.train(X_train, y_train)
                print("y_test distribution", y_test.value_counts())
                fieldnames, evaluate_dict, y_pred = training.evaluate_model(X_test, y_test, output_evaluate_dir=None)
                # predict_df[training.model_name] = y_pred
                # The original model's name is too long, index is easier to read.
                predict_df["m{}_predict".format(i)] = y_pred
                fieldnames = ["id"] + csv_reader.fieldnames + fieldnames

                if evaluate_csv_writer is None:
                    evaluate_csv_writer = csv.DictWriter(evaluate_file, fieldnames=fieldnames)
                    evaluate_csv_writer.writeheader()
                    evaluate_file.flush()

                new_row = {}
                new_row.update(row)
                new_row["id"] = i
                new_row.update(evaluate_dict)
                evaluate_csv_writer.writerow(new_row)
                evaluate_file.flush()
            print("Evaluation result saved in {}".format(evaluate_fname))
            predict_df.to_csv(predict_fname)
            print("Prediction result saved in {}".format(predict_fname))


def run_rnn(cnn_setting_fname, X_train, y_train, X_test, y_test, evaluate_fname, replace_exists):

    with open(evaluate_fname, 'w', newline='') as evaluate_file:
        with open(cnn_setting_fname, 'r') as setting_file:
            csv_reader = csv.DictReader(setting_file)

            fieldnames = csv_reader.fieldnames + ["time", 'macro_prec', 'macro_recall', 'macro_f1',
                                     'micro_prec', 'micro_recall', 'micro_f1',
                                     "weighted_prec", "weighted_recall", "weighted_f1"]
            csv_writer = csv.DictWriter(evaluate_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            for row in csv_reader:
                #classifier_name,num_words,max_text_len,embedding_vector_dimension,data_name,feature_name,target_name,num_filter,keneral_size_list,pool_size,drop_perc,l2_constraint,batch_size,epochs
                classifier_name = row["classifier_name"]
                num_words = int(row["num_words"])
                max_text_len = int(row["max_text_len"])
                embedding_vector_dimension = int(row["embedding_vector_dimension"])
                data_name = row["data_name"]
                feature_name = row["feature_name"]
                target_name = row["target_name"]
                num_class = int(row["num_class"])
                num_lstm_layer = int(row["num_lstm_layer"])
                drop_perc = float(row["drop_perc"])
                # l2_constraint = int(row["l2_constraint"])
                batch_size = int(row["batch_size"])
                epochs = int(row["epochs"])
                embedding_fname = row["embedding_file"]
                if embedding_fname is not None:
                    embedding_fname = os.path.join(config.WORD_EMBEDDING_DIR, embedding_fname)

                training = RNN_NLP_Model(
                    classifier_name=classifier_name,
                    num_words=num_words,
                    max_text_len=max_text_len,
                    embedding_vector_dimension=embedding_vector_dimension,
                    embedding_fname=embedding_fname,
                    data_name=data_name,
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
                evaluate_dict = training.evaluate_model(X_test, y_test)

                new_row = {}
                new_row.update(row)
                new_row.update(evaluate_dict)
                csv_writer.writerow(new_row)
                evaluate_file.flush()
