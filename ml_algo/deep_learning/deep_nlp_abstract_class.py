import config
import os, re
from utils.file_logger import File_Logger_Helper

import numpy as np
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import load_model
import pandas as pd

from ml_algo.preprocessing.feature_processing import Feature_Processing
from ml_algo.preprocessing.word_embedding import Word_Embedding
from ml_algo.evaluation.model_evaluator import Model_Evaluator
import abc

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Dec 5 2018"

class Deep_NLP_Abstract_Class(abc.ABC):

    # TODO: change this class into multi classifier

    def __init__(self,
                 classifier_name="nn",
                 num_words=10000,
                 max_text_len=1600,
                 embedding_vector_dimension=100,
                 embedding_fname=os.path.join(config.WORD_EMBEDDING_DIR, 'glove.6B.100d.txt'),
                 data_name="data",
                 feature_name="f1.f2",
                 target_name="t",
                 num_class=1,
                 kernel_initializer='glorot_uniform',
                 batch_size=100,
                 epochs=10,
                 model_learning_rate=1e-3,
                 model_weight_decate_rate=0.7,
                 model_weight_imbalance_class=False,
                 replace_exists=False,
                 logger=None):

        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="NN.log")
        self.feature_preprocessing = Feature_Processing()
        self.classifier_name = classifier_name
        self.num_words = num_words
        self.max_text_len = max_text_len
        self.weight_decay = 1e-4
        self.model_learning_rate = model_learning_rate
        self.model_weight_decate_rate = model_weight_decate_rate
        self.model_weight_imbalance_class = model_weight_imbalance_class
        self.model_weight_imbalance_name = "b"
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_name = data_name
        self.feature_name = feature_name
        self.target_name = target_name
        self.kernel_initializer = kernel_initializer
        self.num_class = num_class
        self.replace_exists = replace_exists
        self.model = None

        # Initial the embedding layer.
        if embedding_fname is not None:

            self.embedding_helper = Word_Embedding(embedding_fname=embedding_fname)
            if embedding_vector_dimension != self.embedding_helper.embedding_vector_dimension:
                self.logger.error(
                    "Error, the embedding vector dimension should be {} instead of {}. Fix embedding_vector_dimension to {}".format(
                        self.embedding_helper.embedding_vector_dimension,
                        embedding_vector_dimension,
                        self.embedding_helper.embedding_vector_dimension,
                    ))

            self.embedding_vector_dimension = self.embedding_helper.embedding_vector_dimension
            # TODO: fix me in the future
            self.embedding_name = re.sub(r"\.txt", "", os.path.basename(embedding_fname))
        else:
            # If the embedding is not specified, we would use the plain token vector.
            self.embedding_helper = Word_Embedding()
            self.embedding_vector_dimension = embedding_vector_dimension
            self.embedding_name = "token_vector"

        # FIXME: simplify the name stuff and move it into embedding class.
        self.preprocessing_name = self.embedding_helper.generate_model_name(
            embedding_name=self.embedding_name,
            num_words=num_words,
            embedding_vector_dimension=embedding_vector_dimension,
            max_text_len=max_text_len,
            data_name=data_name,
            feature_name=feature_name,
            target_name=target_name,
        )

        self.load_model_if_exists(classifier_name=classifier_name,
                                  preprocess_name=self.preprocessing_name,
                                  replace_exists=replace_exists)

    def reset_max_text_len(self, max_text_len=1600):
        self.max_text_len = max_text_len
        self.load_model_if_exists(classifier_name=self.classifier_name,
                                  preprocess_name=self.preprocessing_name)

    @abc.abstractmethod
    def get_custom_name(self):
        return None

    def generate_model_name(self,
                            general_name,
                            preprocess_name=None,
                            ):

        model_name = general_name
        if self.num_class is not None:
            model_name = "{}_{}c".format(model_name, self.num_class)
        custom_name = self.get_custom_name()
        if custom_name is not None:
            model_name = "{}_{}".format(model_name, custom_name)
        if self.model_learning_rate is not None:
            model_name = "{}_{}lr".format(model_name, self.model_learning_rate)
        if self.model_weight_decate_rate is not None:
            model_name = "{}_{}wdecate".format(model_name, self.model_weight_decate_rate)
        if self.model_weight_imbalance_name is not None:
            model_name = "{}_{}".format(model_name, self.model_weight_imbalance_name)
        if self.batch_size is not None:
            model_name = "{}_{}batch".format(model_name, self.batch_size)
        if self.epochs is not None:
            model_name = "{}_{}epoch".format(model_name, self.epochs)

        if preprocess_name is not None:
            model_name = "{}_{}".format(model_name, preprocess_name)
        self.model_name = model_name
        self.logger.info("model_name={}".format(model_name))
        return model_name

    def load_model_if_exists(self,
                             classifier_name="general",
                             preprocess_name="general",
                             replace_exists=False,
                             dump_model_dir=config.DEEP_MODEL_PICKLES_DIR):
        # Load the file is not already done so. If there is no pickle created, train one for it.
        self.dump_model_dir = dump_model_dir
        if not os.path.exists(dump_model_dir):
            os.makedirs(dump_model_dir)

        self.model_name = self.generate_model_name(classifier_name, preprocess_name=preprocess_name)
        self.dump_model_fname = os.path.join(dump_model_dir, "{}.h5".format(self.model_name))

        if os.path.exists(self.dump_model_fname) and replace_exists == False:
            # Reduce loading time if replacing the old model.
            self.logger.info("Load Existed Model {}".format(self.model_name))
            self.model = load_model(self.dump_model_fname)

    def store_model(self):
        if not os.path.exists(self.dump_model_fname) or self.replace_exists is True:
            self.model.save(self.dump_model_fname)

    @abc.abstractmethod
    def define_model(self):
        # TODO implement this method for subclass.
        self.model = Sequential()

    def configure_nn_learning_process(self):
        # TODO implement this method
        # adam = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        adam = Adam(lr=self.model_learning_rate, decay=self.model_weight_decate_rate)
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

    # Reference: "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification"
    def train(self, X_train:pd.Series, y_train:pd.Series):
        if self.replace_exists == False and self.model is None:
            self.logger.error("The self.replace_exists is False. Please make sure you don't want to store/replace the model after training. Please set self.replace_exists to true if you would prefer to replace the old model {}.".format_map(self.model_name))
            return
        # Initial the embedding layer. Don't replace the embedding since it could be shared between different models.
        self.embedding_layer = self.embedding_helper.init_embedding_layer(X_train.values,
                                                                          num_words=self.num_words,
                                                                          embedding_vector_dimension=self.embedding_vector_dimension,
                                                                          max_text_len=self.max_text_len,
                                                                          embedding_name=self.embedding_name,
                                                                          feature_name=self.feature_name,
                                                                          target_name=self.target_name,
                                                                          replace_exists=False
                                                                          )

        # Pad the sequence to the same length
        X_train = self.embedding_helper.encode_X(X_train, max_text_len=self.max_text_len)
        y_train = self.feature_preprocessing.encode_y(y_train)

        if self.model == None:
            self.logger.info("Training model {}".format(self.model_name))

            self.define_model()
            self.configure_nn_learning_process()

            self.logger.info("summarize:\n{}".format(self.model.summary()))

            # Log to tensorboard
            tensorBoardCallback = TensorBoard(log_dir=config.LOG_DIR, write_graph=True)

            self.logger.info("X_train={}".format(X_train))
            self.logger.info("y_train={}".format(y_train))
            # batch_size https://keras.io/getting-started/sequential-model-guide/
            self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, callbacks=[tensorBoardCallback])
            self.store_model()
        else:
            self.logger.info("Trained model {}".format(self.model_name))

    def evaluate_model(self, X_test:pd.Series, y_test:pd.Series, output_evaluate_dir=config.EVALUATE_DATA_DIR):
        if self.model == None:
            self.logger.error("Please train the model first. There is no model for {}".format(self.model_name))
            return
        self.logger.info("Evalute model {}".format(self.model_name))
        # self.logger.info("X_test={}".format(X_test))

        X_encode = self.embedding_helper.encode_X(X_test, max_text_len=self.max_text_len)

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