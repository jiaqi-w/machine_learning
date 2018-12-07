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
from sklearn.utils import class_weight
from sklearn.utils import compute_class_weight
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

from ml_algo.preprocessing.feature_processing import Feature_Processing
from ml_algo.preprocessing.word_embedding import Word_Embedding
from ml_algo.evaluation.model_evaluator import Model_Evaluator as cm
from keras.layers import LSTM
from keras.optimizers import Adam
from ml_algo.evaluation.model_evaluator import Model_Evaluator

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 31 2018"

class RNN_NLP_Model():

    def __init__(self,
                 classifier_name="cnn",
                 num_words=10000,
                 max_text_len=1600,
                 embedding_vector_dimension=100,
                 glove_fname=os.path.join(config.WORD_EMBEDDING_DIR, 'glove.6B.100d.txt'),
                 data_name="data",
                 feature_name="f1.f2",
                 target_name="t",
                 num_class=1,
                 kernel_initializer='glorot_uniform',
                 num_lstm_layer=5,
                 drop_perc=0.1,
                 learning_rate=1e-3,
                 weight_decate_rate=0.7,
                 l2_constraint=0,
                 batch_size=100,
                 epochs=10,
                 logger=None):

        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="CNN.log")
        self.feature_preprocessing = Feature_Processing()
        self.classifier_name = classifier_name
        self.num_class = num_class
        self.kernel_initializer = kernel_initializer
        self.num_words = num_words
        self.num_steps = max_text_len
        self.num_lstm_layer = num_lstm_layer
        self.drop_perc = drop_perc
        self.learning_rate = learning_rate
        self.weight_decate_rate = weight_decate_rate
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

        preprocess_name =self.embedding_helper.generate_model_name(embedding_name=self.embedding_name,
                                                                   num_words=num_words,
                                                                   embedding_vector_dimension=embedding_vector_dimension,
                                                                   max_text_len=max_text_len)

        self.load_model_if_exists(classifier_name=classifier_name,
                                  preprocess_name=preprocess_name)

    def reset_max_text_len(self, max_text_len=1600):
        self.num_steps = max_text_len
        preprocess_name =self.embedding_helper.generate_model_name(embedding_name=self.embedding_name,
                                                                   num_words=self.num_words,
                                                                   embedding_vector_dimension=self.embedding_vector_dimension,
                                                                   max_text_len=max_text_len)
        self.load_model_if_exists(classifier_name=self.classifier_name,
                                  preprocess_name=preprocess_name)

    def generate_model_name(self,
                            general_name,
                            preprocess_name=None,
                            ):

        model_name = "{}_{}class_{}layer_{}drop_{}lr_{}dr_{}norm_{}ki_{}batch_{}epoch".format(general_name,
                                                                                  self.num_class,
                                                                                  self.num_lstm_layer,
                                                                                  round(self.drop_perc, 2),
                                                                                  self.learning_rate,
                                                                                  self.weight_decate_rate,
                                                                                  self.l2_constraint,
                                                                                  self.kernel_initializer,
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
                             dump_model_dir=config.PREROCESS_PICKLES_DIR):
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

    def train(self, X_train:pd.Series, y_train:pd.Series, replace_exists=False):
        """
        Reference
        Dialogue Act Classification in Domain-Independent Conversations Using a Deep Recurrent Neural Network
        """
        # Initial the embedding layer. Don't replace the embedding.
        self.embedding_layer = self.embedding_helper.init_embedding_layer(X_train.values)

        # Pad the sequence to the same length
        X_train = self.embedding_helper.encode_X(X_train)
        # if isinstance(y_train[0], str):
        y_train = self.feature_preprocessing.encode_y(y_train)

        if self.model == None or replace_exists:
            self.logger.info("Training model {}".format(self.model_name))
            self.model = Sequential()
            self.model.add(self.embedding_layer)
            self.model.add(LSTM(self.embedding_vector_dimension))

            # self.model.add(Dropout(self.drop_perc))
            # for i in range(1, self.num_lstm_layer):
            #     self.model.add(LSTM(self.embedding_vector_dimension, return_sequences=True, dropout=self.drop_perc))

            # # 256
            # self.model.add(Dense(256, activation='relu', name='FC1'))
            # self.model.add(Dropout(self.drop_perc))

            # adam = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            adam = Adam(lr=self.learning_rate, decay=self.weight_decate_rate)
            if self.num_class == 1:
                # for the imbalanced data. kernel_initializer='uniform',
                # samples are drawn from a uniform distribution within [-limit, limit], with limit = sqrt(3 * scale / n)
                # self.model.add(Dense(self.num_class, activation='softmax', kernel_initializer='uniform'))
                # "sigmoid", ""logistic function
                self.model.add(Dense(1, activation='sigmoid', kernel_initializer=self.kernel_initializer))
                self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
                # self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
            else:
                self.model.add(Dense(self.num_class, activation='softmax', kernel_initializer=self.kernel_initializer))
                self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

            self.logger.info("summarize:\n{}".format(self.model.summary()))

            # Log to tensorboard
            tensorBoardCallback = TensorBoard(log_dir=config.LOG_DIR, write_graph=True)

            self.logger.info("X_train={}".format(X_train))
            self.logger.info("y_train={}".format(y_train))
            # batch_size https://keras.io/getting-started/sequential-model-guide/
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
            self.logger.info("check balance class_weights for imbalance data {}".format(class_weights))
            unique, counts = np.unique(y_train, return_counts=True)
            # class_weights = dict(zip(unique, counts))
            class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
            self.logger.info("add class_weights for imbalance data {}".format(class_weights))
            self.model.fit(X_train, y_train, class_weight=class_weights,
                           batch_size=self.batch_size, epochs=self.epochs, callbacks=[tensorBoardCallback])
            self.store_model(replace_exists=replace_exists)
        else:
            self.logger.info("Trained model {}".format(self.model_name))

    def vanilla_rnn_model(self):
        # disavantage vanishing gradient problem for long sentence.
        pass


    def evaluate_model(self, X_test:pd.Series, y_test:pd.Series, output_evaluate_dir=config.EVALUATE_DATA_DIR):
        if self.model == None:
            self.logger.error("Please train the model first. There is no model for {}".format(self.model_name))
        self.logger.info("Evalute model {}".format(self.model_name))
        # self.logger.info("X_test={}".format(X_test))

        X_encode = self.embedding_helper.encode_X(X_test)

        # accuracy
        # scores = self.model.evaluate(X_test, y_test, verbose=0)
        # self.logger.info("Accuracy: %.2f%%" % (scores[-1] * 100))

        y_pred = self.model.predict_classes(X_encode)
        # y_pred = self.model.predict(X_test)
        # y_pred = y_pred.argmax(axis=-1)
        self.logger.info("y_pred {}".format(y_pred))

        y_test = self.feature_preprocessing.encode_y(y_test)
        self.logger.info("y_test {}".format(y_test))

        model_evaluator = Model_Evaluator(y_gold=list(y_test.tolist()), y_pred=y_pred.flatten().tolist(), X_gold=X_test)

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