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

from ml_algo.preprocessing.feature_processing import Feature_Processing
from ml_algo.preprocessing.word_embedding import Word_Embedding
from ml_algo.evaluation.confusion_matrix_helper import Confusion_Matrix_Helper as cm
from keras.layers import LSTM
from keras.optimizers import Adam

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 31 2018"

class RNN_NLP_Model():

    def __init__(self,
                 classifier_name="cnn",
                 num_words=10000,
                 max_text_len=1600,
                 embedding_vector_dimension=100,
                 glove_fname=os.path.join(config.GLOVE_SIXB, 'glove.6B.100d.txt'),
                 data_name="data",
                 feature_name="f1.f2",
                 target_name="t",
                 num_class=1,
                 num_lstm_layer=10,
                 drop_perc=0.5,
                 l2_constraint=3,
                 batch_size=100,
                 epochs=10,
                 logger=None):

        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="CNN.log")
        self.feature_preprocessing = Feature_Processing()
        self.classifier_name = classifier_name
        self.num_class = num_class
        self.num_words = num_words
        self.max_text_len = max_text_len
        self.num_lstm_layer = num_lstm_layer
        self.drop_perc = drop_perc
        self.weight_decay = 1e-4
        self.l2_constraint = l2_constraint
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

        # Initial the embedding layer.
        if glove_fname is not None:

            self.embedding_helper = Word_Embedding(glove_fname=glove_fname)
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

        preprocess_name =self.embedding_helper.generate_model_name(general_name=self.embedding_name,
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

        model_name = "{}_{}class_{}layer_{}drop_{}norm_{}batch_{}epoch".format(general_name,
                                                                               self.num_class,
                                                                               self.num_lstm_layer,
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
        # Initial the embedding layer.
        self.embedding_layer = self.embedding_helper.init_embedding_layer(X_train.values,
                                                                          num_words=self.num_words,
                                                                          embedding_vector_dimension=self.embedding_vector_dimension,
                                                                          max_text_len=self.max_text_len,
                                                                          general_name=self.embedding_name,
                                                                          replace_exists=replace_exists
                                                                          )

        # Pad the sequence to the same length
        X_train = self.embedding_helper.encode_X(X_train, max_text_len=self.max_text_len)
        # if isinstance(y_train[0], str):
        y_train = self.feature_preprocessing.encode_y(y_train)

        if self.model == None or replace_exists:
            self.logger.info("Training model {}".format(self.model_name))
            self.model = Sequential()
            self.model.add(self.embedding_layer)
            self.model.add(Dropout(self.drop_perc))

            for i in range(1, self.num_lstm_layer):
                self.model.add(LSTM(self.embedding_vector_dimension, return_sequences=True, dropout=self.drop_perc))
            self.model.add(LSTM(self.embedding_vector_dimension))

            # num_class = 1
            self.model.add(Dense(self.num_class, activation='softmax'))

            adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            if self.num_class == 1:
                self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
            else:
                self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

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

    def evaluate_model(self, X_test:pd.Series, y_test:pd.Series, predict_fname=None, cm_fname=None, evaluate_fname=None):
        if self.model == None:
            self.logger.error("Please train the model first. There is no model for {}".format(self.model_name))
        self.logger.info("Evalute model {}".format(self.model_name))
        self.logger.info("X_test={}".format(X_test))

        X_test = self.embedding_helper.encode_X(X_test, max_text_len=self.max_text_len)

        # accuracy
        # scores = self.model.evaluate(X_test, y_test, verbose=0)
        # self.logger.info("Accuracy: %.2f%%" % (scores[-1] * 100))

        y_pred = self.model.predict_classes(X_test)
        # y_pred = self.model.predict(X_test)
        # y_pred = y_pred.argmax(axis=-1)
        self.logger.info("y_pred {}".format(y_pred))

        y_test = self.feature_preprocessing.encode_y(y_test)
        self.logger.info("y_test {}".format(y_test))

        # TODO: save the evaluation results in the future.
        evaluate_dict = {}
        precision, recall, F1, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
        evaluate_dict["macro_prec"] = round(precision, 4)
        evaluate_dict["macro_recall"] = round(recall, 4)
        evaluate_dict["macro_f1"] = round(F1, 4)
        self.logger.info("macro precision={}, recall={}, f1={}, support={}".format(round(precision, 4), round(recall, 4), round(F1, 4), support))
        precision, recall, F1, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
        evaluate_dict["micro_prec"] = round(precision, 4)
        evaluate_dict["micro_recall"] = round(recall, 4)
        evaluate_dict["micro_f1"] = round(F1, 4)
        self.logger.info("micro precision={}, recall={}, f1={}, support={}".format(round(precision, 4), round(recall, 4), round(F1, 4), support))
        precision, recall, F1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        evaluate_dict["weighted_prec"] = round(precision, 4)
        evaluate_dict["weighted_recall"] = round(recall, 4)
        evaluate_dict["weighted_f1"] = round(F1, 4)
        self.logger.info("weighted precision={}, recall={}, f1={}, support={}".format(round(precision, 4), round(recall, 4), round(F1, 4), support))

        target_names = self.feature_preprocessing.get_target_names(y_test)

        report = classification_report(y_test, y_pred, target_names=target_names)
        self.logger.info("report:\n{}".format(report))

        # TODO: confusion matrix.
        cm.evaluate(y_test.tolist(), y_pred.flatten().tolist(), cm_outfname="{}_cm.csv".format(self.model_name))

        # TODO: output results.
        if predict_fname is not None:
            with open(predict_fname, "w") as predict_file:
                df = pd.DataFrame(data=X_test)
                df["predict"] = y_pred
                df.to_csv(predict_file)
                self.logger.info("Save prediction results to {}".format(predict_fname))

        return evaluate_dict
