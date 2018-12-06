import config
import os, re

from keras.constraints import max_norm
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import InputLayer
from keras.models import Model
from keras.layers.merge import concatenate
from keras.models import Sequential

from ml_algo.deep_learning.deep_nlp_abstract_class import Deep_NLP_Abstract_Class

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Dec 5 2018"

class NN_NLP_Model(Deep_NLP_Abstract_Class):

    def __init__(self,
                 classifier_name="nn",
                 data_name="data",
                 feature_name="f1.f2",
                 target_name="t",
                 num_class=1,
                 kernel_initializer='glorot_uniform',
                 num_feature=73,
                 neuron_unit_list=(64,32),
                 drop_perc=0.5,
                 model_learning_rate=1e-3,
                 model_weight_decate_rate=0.7,
                 model_weight_imbalance_class=False,
                 batch_size=100,
                 epochs=10,
                 replace_exists=False,
                 logger=None):

        self.num_feature = num_feature
        self.neuron_unit_list = list(neuron_unit_list)
        self.drop_perc = drop_perc

        # the super() has to follow the parameter init since the get_custom_name() is invoked with the require value.

        super().__init__(
            classifier_name=classifier_name,
            num_words=None,
            max_text_len=None,
            embedding_vector_dimension=None,
            embedding_fname=None,
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
        neuron_unit_list_name = re.sub(r"\s+", "", str(self.neuron_unit_list))
        if self.neuron_unit_list is not None:
            if model_name is None:
                model_name = "{}nunit".format(neuron_unit_list_name)
            else:
                model_name = "{}_{}nunit".format(model_name, neuron_unit_list_name)


        drop = round(self.drop_perc, 2)
        if drop is not None:
            if model_name is None:
                model_name = "{}drop".format(drop)
            else:
                model_name = "{}_{}drop".format(model_name, drop)

        return model_name


    def define_model(self):
        '''
        Simple feed forward neural network
        :return:
        '''
        self.model = Sequential()

        self.model.add(InputLayer(input_shape=(self.num_feature,)))
        for num_neuron_unit in self.neuron_unit_list:
            self.model.add(Dense(units=num_neuron_unit, activation="relu",
                                 kernel_initializer=self.kernel_initializer, bias_initializer="zeros"))
            self.model.add(Dropout(self.drop_perc))



