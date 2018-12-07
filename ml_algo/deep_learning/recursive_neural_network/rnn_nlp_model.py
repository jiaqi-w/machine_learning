import config
import os, re

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


from ml_algo.deep_learning.deep_nlp_abstract_class import Deep_NLP_Abstract_Class

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 31 2018"

class RNN_NLP_Model(Deep_NLP_Abstract_Class):

    def __init__(self,
                 classifier_name="cnn",
                 num_words=10000,
                 max_text_len=1600,
                 embedding_vector_dimension=100,
                 embedding_fname=os.path.join(config.WORD_EMBEDDING_DIR, 'glove.6B.100d.txt'),
                 data_name="data",
                 feature_name="f1.f2",
                 target_name="t",
                 num_class=1,
                 kernel_initializer='glorot_uniform',
                 num_lstm_layer=5,
                 drop_perc=0.1,
                 learning_rate=1e-3,
                 weight_decate_rate=0.7,
                 # l2_constraint=0,
                 batch_size=100,
                 epochs=10,
                 model_learning_rate=1e-3,
                 model_weight_decate_rate=0.7,
                 model_weight_imbalance_class=False,
                 replace_exists=False,
                 logger=None):



        self.num_lstm_layer = num_lstm_layer
        self.drop_perc = drop_perc
        # self.l2_constraint = l2_constraint
        self.learning_rate = learning_rate
        self.weight_decate_rate = weight_decate_rate

        # the super() has to follow the parameter init since the get_custom_name() is invoked with the require value.

        super().__init__(
            classifier_name=classifier_name,
            num_words=num_words,
            max_text_len=max_text_len,
            embedding_vector_dimension=embedding_vector_dimension,
            embedding_fname=embedding_fname,
            data_name=data_name,
            feature_name=feature_name,
            target_name=target_name,
            num_class=num_class,
            kernel_initializer=kernel_initializer,
            batch_size=batch_size,
            epochs=epochs,
            model_learning_rate=model_learning_rate,
            model_weight_decate_rate=model_weight_decate_rate,
            model_weight_imbalance_class=model_weight_imbalance_class,
            replace_exists=replace_exists,
            logger=logger
        )

    def get_custom_name(self):
        # return custom name for the define model.

        model_name = None
        if self.num_lstm_layer is not None:
            if model_name is None:
                model_name = "{}layer".format(self.num_lstm_layer)
            else:
                model_name = "{}_{}numfilter".format(model_name, self.num_lstm_layer)

        drop = round(self.drop_perc, 2)
        if drop is not None:
            if model_name is None:
                model_name = "{}drop".format(drop)
            else:
                model_name = "{}_{}drop".format(model_name, drop)

        return model_name

    def define_model(self):

        self.logger.info("Define model {}".format(self.model_name))
        self.model = Sequential()
        self.model.add(self.embedding_layer)
        self.model.add(Dropout(self.drop_perc))

        if self.num_lstm_layer < 1:
            self.logger.error("Please initial the LSTM with at lest one layer {}".format(self.num_lstm_layer))

        for i in range(1, self.num_lstm_layer):
            self.model.add(LSTM(self.embedding_helper.embedding_vector_dimension, return_sequences=True, dropout=self.drop_perc))

        # Do we want to add the drop out in this layer?
        self.model.add(LSTM(self.embedding_helper.embedding_vector_dimension))

        # # 256
        # self.model.add(Dense(256, activation='relu', name='FC1'))
        # self.model.add(Dropout(self.drop_perc))


    def vanilla_rnn_model(self):
        # disavantage vanishing gradient problem for long sentence.
        pass
