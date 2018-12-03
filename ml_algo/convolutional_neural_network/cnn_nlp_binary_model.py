import config
import os, re
from utils.file_logger import File_Logger_Helper

import numpy as np
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.constraints import max_norm
from keras.layers import Convolution1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn import metrics

from ml_algo.preprocessing.feature_processing import Feature_Processing
from ml_algo.preprocessing.word_embedding import Word_Embedding
from ml_algo.evaluation.model_evaluator import Model_Evaluator

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 1 2018"

class CNN_NLP_Binary_Model():

    # TODO: change this class into multi classifier
    # TODO: create deep learning abstract model.

    def __init__(self,
                 classifier_name="cnn",
                 num_words=10000,
                 max_text_len=1600,
                 embedding_vector_dimension=100,
                 glove_fname=os.path.join(config.GLOVE_SIXB, 'glove.6B.100d.txt'),
                 data_name="data",
                 feature_name="f1.f2",
                 target_name="t",
                 num_filter=2,
                 keneral_size_list=(2,3,4),
                 pool_size=1,
                 drop_perc=0.5,
                 l2_constraint=3,
                 batch_size=100,
                 epochs=10,
                 logger=None):

        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="CNN.log")
        self.feature_preprocessing = Feature_Processing()
        self.classifier_name = classifier_name
        self.num_words = num_words
        self.max_text_len = max_text_len
        self.num_filter = num_filter
        self.keneral_size_list = keneral_size_list
        self.pool_size = pool_size
        self.drop_perc = drop_perc
        self.weight_decay = 1e-4
        self.l2_constraint = l2_constraint
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

        # Initial the embedding layer.
        if glove_fname is not None:

            self.embedding_helper = Word_Embedding(embedding_fname=glove_fname)
            if embedding_vector_dimension != self.embedding_helper.embedding_vector_dimension:
                self.logger.error(
                    "Error, the embedding vector dimension should be {} instead of {}. Fix embedding_vector_dimension to {}".format(
                        self.embedding_helper.embedding_vector_dimension,
                        embedding_vector_dimension,
                        self.embedding_helper.embedding_vector_dimension,
                    ))

            self.embedding_vector_dimension = self.embedding_helper.embedding_vector_dimension
            self.embedding_name = "{}_{}_{}_{}".format(re.sub(r"\.txt", "_", os.path.basename(glove_fname)), data_name, feature_name, target_name)
        else:
            # If the embedding is not specified, we would use the plain token vector.
            self.embedding_helper = Word_Embedding()
            self.embedding_vector_dimension = embedding_vector_dimension
            self.embedding_name = "{}_{}_{}_{}".format("token_vector", data_name, feature_name, target_name)

        # FIXME: simplify the name stuff and move it into embedding class.
        preprocess_name = self.embedding_helper.generate_model_name(
            general_name=self.embedding_name,
            num_words=num_words,
            embedding_vector_dimension=embedding_vector_dimension,
            max_text_len=max_text_len)

        self.load_model_if_exists(classifier_name=classifier_name,
                                  preprocess_name=preprocess_name)

    def reset_max_text_len(self, max_text_len=1600):
        self.max_text_len = max_text_len
        preprocess_name =self.embedding_helper.generate_model_name(general_name=self.embedding_name,
                            num_words=self.num_words,
                            embedding_vector_dimension=self.embedding_vector_dimension,
                            max_text_len=max_text_len)
        self.load_model_if_exists(classifier_name=self.classifier_name,
                                  preprocess_name=preprocess_name)

    def generate_model_name(self,
                            general_name,
                            preprocess_name=None,
                            ):

        model_name = "{}_{}numfilter_{}kernal_{}pool_{}drop_{}norm_{}batch_{}epoch".format(general_name,
                                                                                              self.num_filter,
                                                                                              re.sub(r"\s+", "", str(self.keneral_size_list)),
                                                                                              self.pool_size,
                                                                                              round(self.drop_perc, 2),
                                                                                              self.l2_constraint,
                                                                                              self.batch_size,
                                                                                              self.epochs)
        if preprocess_name is not None:
            model_name = "{}_{}".format(model_name, preprocess_name)
        self.model_name = model_name
        self.logger.info("model_name={}".format(model_name))
        return model_name

    def load_model_if_exists(self,
                             classifier_name="general",
                             preprocess_name="general",
                             dump_model_dir=config.DEEP_MODEL_PICKLES_DIR):
        # Load the file is not already done so. If there is no pickle created, train one for it.
        self.logger.info("Load Model")

        self.dump_model_dir = dump_model_dir
        if not os.path.exists(dump_model_dir):
            os.makedirs(dump_model_dir)

        self.model_name = self.generate_model_name(classifier_name, preprocess_name=preprocess_name)
        self.dump_model_fname = os.path.join(dump_model_dir, "{}.h5".format(self.model_name))

        if os.path.exists(self.dump_model_fname):
            self.model = load_model(self.dump_model_fname)

    def store_model(self, replace_exists=False):
        if not os.path.exists(self.dump_model_fname) or replace_exists is True:
            self.model.save(self.dump_model_fname)

    # Reference: "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification"
    def train(self, X_train:pd.Series, y_train:pd.Series, replace_exists=False):
        # Initial the embedding layer. Don't replace the embedding.
        self.embedding_layer = self.embedding_helper.init_embedding_layer(X_train.values,
                                                                          num_words=self.num_words,
                                                                          embedding_vector_dimension=self.embedding_vector_dimension,
                                                                          max_text_len=self.max_text_len,
                                                                          general_name=self.embedding_name,
                                                                          replace_exists=False
                                                                          )

        # Pad the sequence to the same length
        X_train = self.embedding_helper.encode_X(X_train, max_text_len=self.max_text_len)
        # if isinstance(y_train[0], str):
        y_train = self.feature_preprocessing.encode_y(y_train)

        if self.model == None or replace_exists:
            self.logger.info("Training model {}".format(self.model_name))
            inputs = []
            input = Input(shape=(self.max_text_len,))
            embedding = self.embedding_layer(input)
            univariate_vectors = []
            for filter_size in self.keneral_size_list:
                # channel i
                # input = Input(shape=(self.max_text_len,))
                # embedding = self.embedding_layer(input)
                conv1d = Conv1D(filters=self.num_filter, kernel_size=filter_size, activation='relu')(embedding)
                # dropout to avoid overfitting
                drop = Dropout(self.drop_perc)(conv1d)
                pool1d = MaxPooling1D(pool_size=self.pool_size)(drop)
                flat = Flatten()(pool1d)

                inputs.append(input)
                univariate_vectors.append(flat)

            merged = concatenate(univariate_vectors)
            # regularization
            # dense_regularize = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(merged)
            num_dense_units = self.num_filter * len(self.keneral_size_list)
            if self.l2_constraint == 0:
                dense_regularize = Dense(num_dense_units, activation='relu')(merged)
            else:
                dense_regularize = Dense(num_dense_units, activation='relu', kernel_constraint=max_norm(self.l2_constraint))(merged)
            outputs = Dense(1, activation='sigmoid')(dense_regularize)
            # Please note that we are not using a sequencial model here
            # self.model = Model(inputs=[inputs], outputs=outputs)
            # self.model = Model(inputs=[input], outputs=outputs)
            self.model = Sequential()
            self.model.add(Model(inputs=[input], outputs=outputs))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.logger.info("summarize:\n{}".format(self.model.summary()))

            # Log to tensorboard
            tensorBoardCallback = TensorBoard(log_dir=config.LOG_DIR, write_graph=True)

            self.logger.info("X_train={}".format(X_train))
            self.logger.info("y_train={}".format(y_train))
            # batch_size https://keras.io/getting-started/sequential-model-guide/
            self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, callbacks=[tensorBoardCallback])
            self.store_model(replace_exists=replace_exists)
        else:
            self.logger.info("Trained model {}".format(self.model_name))

    def evaluate_model(self, X_test:pd.Series, y_test:pd.Series, output_evaluate_dir=config.EVALUATE_DATA_DIR):
        if self.model == None:
            self.logger.error("Please train the model first. There is no model for {}".format(self.model_name))
        self.logger.info("Evalute model {}".format(self.model_name))
        # self.logger.info("X_test={}".format(X_test))

        X_encode = self.embedding_helper.encode_X(X_test, max_text_len=self.max_text_len)

        # accuracy
        # scores = self.model.evaluate(X_test, y_test, verbose=0)
        # self.logger.info("Accuracy: %.2f%%" % (scores[-1] * 100))

        y_pred = self.model.predict_classes(X_encode)
        # y_pred = self.model.predict(X_test)
        # y_pred = y_pred.argmax(axis=-1)
        self.logger.info("y_pred {}".format(y_pred))

        y_test = self.feature_preprocessing.encode_y(y_test)
        self.logger.info("y_test {}".format(y_test))

        model_evaluator = Model_Evaluator(y_gold=y_test.tolist(), y_pred=y_pred.flatten().tolist(), X_gold=X_test)

        fieldnames = model_evaluator.get_evaluation_fieldnames()

        evaluate_fname, predict_fname, cm_fname = None, None, None
        if output_evaluate_dir is not None:
            evaluate_fname = os.path.join(output_evaluate_dir, "{}_evaluate.csv".format(self.model_name))
            predict_fname = os.path.join(output_evaluate_dir, "{}_predict.csv".format(self.model_name))
            cm_fname = os.path.join(output_evaluate_dir, "{}_cm.csv".format(self.model_name))

        evaluate_dict = model_evaluator.get_evaluation_dict(evaluation_fname=evaluate_fname,
                                                            predict_fname=predict_fname,
                                                            cm_fname=cm_fname,
                                                            show_cm=False)

        return fieldnames, evaluate_dict, y_pred

