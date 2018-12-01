import csv
import ast
from ml_algo.convolutional_neural_network.cnn_nlp_binary_model import CNN_NLP_Binary_Model
import os
import config

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 31 2018"


def run(cnn_setting_fname, X_train, y_train, X_test, y_test, evaluate_fname, replace_exists):

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
                num_filter = int(row["num_filter"])
                keneral_size_list = ast.literal_eval(row["keneral_size_list"])
                pool_size = int(row["pool_size"])
                drop_perc = float(row["drop_perc"])
                l2_constraint = int(row["l2_constraint"])
                batch_size = int(row["batch_size"])
                epochs = int(row["epochs"])

                training = CNN_NLP_Binary_Model(
                    classifier_name=classifier_name,
                    num_words=num_words,
                    max_text_len=max_text_len,
                    embedding_vector_dimension=embedding_vector_dimension,
                    glove_fname=os.path.join(config.GLOVE_SIXB, 'glove.6B.100d.txt'),
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
                )
                training.train(X_train, y_train, replace_exists=replace_exists)
                print("y_test distribution", y_test.value_counts())
                evaluate_dict = training.evaluate_model(X_test, y_test)

                new_row = {}
                new_row.update(row)
                new_row.update(evaluate_dict)
                csv_writer.writerow(new_row)
                evaluate_file.flush()
