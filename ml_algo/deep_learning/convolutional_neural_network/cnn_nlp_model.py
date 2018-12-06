import config
import os, re

from keras.constraints import max_norm
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Sequential

from ml_algo.deep_learning.deep_nlp_abstract_class import Deep_NLP_Abstract_Class

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 1 2018"

class CNN_NLP_Model(Deep_NLP_Abstract_Class):

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
                 num_filter=2,
                 keneral_size_list=(2,3,4),
                 pool_size=1,
                 drop_perc=0.5,
                 l2_constraint=3,
                 model_learning_rate=1e-3,
                 model_weight_decate_rate=0.7,
                 model_weight_imbalance_class=False,
                 batch_size=100,
                 epochs=10,
                 replace_exists=False,
                 logger=None):


        self.num_filter = num_filter
        self.keneral_size_list = keneral_size_list
        self.pool_size = pool_size
        self.drop_perc = drop_perc
        self.l2_constraint = l2_constraint

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
        if self.num_filter is not None:
            if model_name is None:
                model_name = "{}numfilter".format(self.num_filter)
            else:
                model_name = "{}_{}numfilter".format(model_name, self.num_filter)

        kernal_size_name = re.sub(r"\s+", "", str(self.keneral_size_list))
        if kernal_size_name is not None:
            if model_name is None:
                model_name = "{}kernal".format(kernal_size_name)
            else:
                model_name = "{}_{}kernal".format(model_name, kernal_size_name)

        if self.pool_size is not None:
            if model_name is None:
                model_name = "{}pool".format(self.pool_size)
            else:
                model_name = "{}_{}pool".format(model_name, self.pool_size)

        drop = round(self.drop_perc, 2)
        if drop is not None:
            if model_name is None:
                model_name = "{}drop".format(drop)
            else:
                model_name = "{}_{}drop".format(model_name, drop)

        if self.num_filter is not None:
            if model_name is None:
                model_name = "{}norm".format(self.l2_constraint)
            else:
                model_name = "{}_{}norm".format(model_name, self.l2_constraint)
        return model_name


    def define_model(self):
        '''
        Reference: "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification"

        :return:
        '''
        self.model = Sequential()

        input = Input(shape=(self.max_text_len,))
        embedding = self.embedding_layer(input)
        univariate_vectors = []
        for filter_size in self.keneral_size_list:
            # channel i
            conv1d = Conv1D(filters=self.num_filter, kernel_size=filter_size, activation='relu')(embedding)
            # dropout to avoid overfitting
            drop = Dropout(self.drop_perc)(conv1d)
            pool1d = MaxPooling1D(pool_size=self.pool_size)(drop)
            flat = Flatten()(pool1d)
            univariate_vectors.append(flat)
        merged = concatenate(univariate_vectors)

        self.model.add(Model(inputs=[input], outputs=merged))

        # regularization
        # dense_regularize = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(merged)
        # TODO: tune this parameter in the future.
        num_dense_units = self.num_filter * len(self.keneral_size_list)
        if self.l2_constraint == 0:
            # dense_regularize = Dense(num_dense_units, activation='relu')(merged)
            self.model.add(Dense(num_dense_units, activation='relu', name="dense_regularize"))
        else:
            # dense_regularize = Dense(num_dense_units, activation='relu', kernel_constraint=max_norm(self.l2_constraint))(merged)
            self.model.add(Dense(num_dense_units, activation='relu', kernel_constraint=max_norm(self.l2_constraint)))


