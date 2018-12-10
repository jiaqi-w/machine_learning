import config
import os, re

from sklearn.preprocessing import LabelEncoder

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
from keras.layers.merge import Concatenate
from keras.layers import InputLayer

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
from sklearn.utils import compute_class_weight

from ml_algo.preprocessing.feature_processing import Feature_Processing
from ml_algo.preprocessing.word_embedding import Word_Embedding
from ml_algo.evaluation.model_evaluator import Model_Evaluator

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 1 2018"

class CNN_Merge_NLP_Model(Deep_NLP_Abstract_Class):

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
                 num_custome_features=0,
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


        self.num_custome_features = num_custome_features
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

        if self.num_custome_features is not None and self.num_custome_features != 0:
            if model_name is None:
                model_name = "{}custfeature".format(self.num_custome_features)
            else:
                model_name = "{}_{}custfeature".format(model_name, self.num_custome_features)
        return model_name

    def define_model(self):
        pass

    def train(self, X_text:pd.Series, y_train:pd.Series, X_features:pd.Series=None):
        '''
        Reference: "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification"
        https://keras.io/getting-started/functional-api-guide/
        :return:
        '''
        # Initial the embedding layer. Don't replace the embedding since it could be shared between different models.
        if self.embedding_helper is not None:
            self.embedding_layer = self.embedding_helper.init_embedding_layer(X_text)
            # Pad the sequence to the same length
            X_text = self.embedding_helper.encode_X(X_text)

        if X_features is not None and X_features.shape[1] > 0:
            # Merge the features
            # X_features = X_features.values
            self.logger.info("X_text shape {}".format(X_text.shape))
            self.logger.info("X_features shape {}".format(X_features.shape))
            self.logger.info("X_features type {}".format(type(X_features)))

            # X_text = np.reshape(X_text, (X_text.shape[0], X_text.shape[1], 1))
            # X_features = np.reshape(X_features, (X_features.shape[0], X_features.shape[1], 1))
            # X_train = [X_text, X_features]
            X_train = {"text_input" : X_text, "feature_input": X_features}
            # self.logger.info("X_train shape {}".format(X_train.shape))
            self.logger.info("Concatenate features X_train {}".format(X_train))
        else:
            X_train = X_text

        y_train = self.feature_preprocessing.encode_y(y_train)

        if self.model == None:
            input_layers = []
            input_text = Input(shape=(self.max_text_len,), name="text_input")
            input_layers.append(input_text)

            embedding = self.embedding_layer(input_text)
            univariate_vectors = []
            for filter_size in self.keneral_size_list:
                # channel i
                print("filter_size", filter_size)
                conv1d = Conv1D(filters=self.num_filter, kernel_size=filter_size, activation='relu')(embedding)
                # dropout to avoid overfitting
                drop = Dropout(self.drop_perc)(conv1d)
                pool1d = MaxPooling1D(pool_size=self.pool_size)(drop)
                flat = Flatten()(pool1d)
                print("flat.shape: {}".format(flat._keras_shape))
                univariate_vectors.append(flat)


            print("input_layers[0].shape:", input_layers[0].shape)
            #  # TODO if num_custome_features == 0, don't add it, same for fix.
            if self.num_custome_features is not None and self.num_custome_features > 0:
                input_features = Input(shape=(self.num_custome_features,), name="feature_input")
                print("input_features.shape:", input_features._keras_shape)

                input_layers.append(input_features)
                univariate_vectors.append(input_features)

                # dense_feature = Dense(self.num_custome_features, activation='linear', name="linear")(input_features)
                # univariate_vectors.append(dense_feature)

            merged = concatenate(univariate_vectors, name="merge_vector")
            # print("merged.shape:", merged._keras_shape)

            # Please note that this input layers must be consistant with the model.fit()
            # self.model.add(Model(inputs=input_layers, outputs=merged, name="input_encoder"))

            # regularization
            # dense_regularize = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(merged)
            # TODO: tune this parameter in the future.
            num_dense_units = self.num_filter * len(self.keneral_size_list)
            if self.l2_constraint == 0:
                regular_layer = Dense(num_dense_units, activation='relu', name="regularization")(merged)
            else:
                regular_layer = Dense(num_dense_units, activation='relu',
                                     kernel_constraint=max_norm(self.l2_constraint), name="regularization")(merged)

            # adam = Adam(lr=self.model_learning_rate, decay=self.model_weight_decate_rate)
            if self.num_class == 1:
                # for the imbalanced data. kernel_initializer='uniform',
                # samples are drawn from a uniform distribution within [-limit, limit], with limit = sqrt(3 * scale / n)
                # self.model.add(Dense(self.num_class, activation='softmax', kernel_initializer='uniform'))
                # "sigmoid", ""logistic function
                # And add a logistic regression on top.
                output = Dense(1, activation='sigmoid', kernel_initializer=self.kernel_initializer)(regular_layer)
                self.model = Model(inputs=input_layers, outputs=output, name="output_layer")
                # self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
                self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                # self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
            else:
                output = Dense(self.num_class, activation='softmax', kernel_initializer=self.kernel_initializer)(regular_layer)
                self.model = Model(inputs=input_layers, outputs=output, name="output_layer")
                # self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
                self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            self.logger.info("summarize:\n{}".format(self.model.summary()))

            # Log to tensorboard
            tensorBoardCallback = TensorBoard(log_dir=config.LOG_DIR, write_graph=True)

            self.logger.info("X_train={}".format(X_text))
            self.logger.info("y_train={}".format(y_train))
            # batch_size https://keras.io/getting-started/sequential-model-guide/
            if self.model_weight_imbalance_class:
                class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
            else:
                class_weight = None

            self.model.fit(X_train, y_train, class_weight=class_weight,
                           batch_size=self.batch_size, epochs=self.epochs,
                           callbacks=[tensorBoardCallback])
            self.store_model()
        else:
            self.logger.info("Trained model {}".format(self.model_name))


    def evaluate_model(self, X_text_test:pd.Series, y_test:pd.Series,
                       X_feature_test:pd.Series=None,
                       output_evaluate_dir=config.EVALUATE_DATA_DIR):
        if self.model == None:
            self.logger.error("Please train the model first. There is no model for {}".format(self.model_name))
            return
        self.logger.info("Evalute model {}".format(self.model_name))
        # self.logger.info("X_test={}".format(X_test))

        if self.embedding_helper is not None:
            X_encode = self.embedding_helper.encode_X(X_text_test)
        else:
            X_encode = X_text_test

        if X_feature_test is not None and X_feature_test.shape[1] > 0:
            # Merge the features
            # X_encode = [X_encode, X_feature_test]
            X_encode = {"text_input": X_encode, "feature_input": X_feature_test}

        # loss, acc = self.model.evaluate(X_encode, y_test, verbose=0)
        # print('Test Loss: %f' % (loss))
        # print('Test Accuracy: %f' % (acc * 100))

        # y_pred = self.model.predict_classes(X_encode)
        y_pred = np.asarray(self.model.predict(X_encode))
        self.logger.info("y_pred {}".format(y_pred))

        if self.num_class == 1:
            pred = []
            for p in y_pred:
                if p > 0.5:
                    pred.append(1)
                else:
                    pred.append(0)
            y_pred = pred
        else:
            y_pred = y_pred.argmax(axis=-1)
            y_pred = np.argmax(y_pred, axis=1)
        self.logger.info("y_pred {}".format(y_pred))

        y_test = self.feature_preprocessing.encode_y(y_test)
        self.logger.info("y_test {}".format(y_test))

        # model_evaluator = Model_Evaluator(y_gold=list(y_test.flatten().tolist()), y_pred=list(y_pred.flatten().tolist()), X_gold=X_text_test)
        model_evaluator = Model_Evaluator(y_gold=list(y_test.flatten().tolist()), y_pred=y_pred, X_gold=X_text_test)

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

        # # if self.feature_preprocessing.label_encoder is None:
        # #     self.feature_preprocessing.label_encoder = LabelEncoder()
        # self.feature_preprocessing.label_encoder = LabelEncoder()
        #
        # self.logger.info("label inverse_transform")
        # # TODO: fix me pleassssssse
        # self.feature_preprocessing.label_encoder.fit(["yes", "no"])
        # y_pred = pd.DataFrame(self.feature_preprocessing.label_encoder.inverse_transform(y_pred))
        #
        # self.logger.info("y_pred {}".format(y_pred))

        return fieldnames, evaluate_dict, y_pred



    def predict_y(self, X_text_pred:pd.Series, X_feature_pred:pd.Series):

        if self.embedding_helper is not None:
            X_encode = self.embedding_helper.encode_X(X_text_pred)
        else:
            X_encode = X_text_pred

        if X_feature_pred is not None and X_feature_pred.shape[1] > 0:
            # Merge the features
            X_encode = [X_encode, X_feature_pred]


        y_pred = self.model.predict(X_encode)
        if self.num_class == 1:
            pred = []
            for p in y_pred:
                if p > 0.5:
                    pred.append(1)
                else:
                    pred.append(0)
            y_pred = pred
        else:
            y_pred = y_pred.argmax(axis=-1)
            y_pred = np.argmax(y_pred, axis=1)
        self.logger.info("y_pred {}".format(y_pred))
        # print("self.feature_preprocessing.name", self.feature_preprocessing.data_lable_name)
        print("label_encoder", self.feature_preprocessing.label_encoder)


        # if self.feature_preprocessing.label_encoder is None:
        #     self.feature_preprocessing.label_encoder = LabelEncoder()
        self.feature_preprocessing.label_encoder = LabelEncoder()

        self.logger.info("label inverse_transform")
        # TODO: fix me pleassssssse
        self.feature_preprocessing.label_encoder.fit(["yes", "no"])
        y_pred = pd.DataFrame(self.feature_preprocessing.label_encoder.inverse_transform(y_pred))

        self.logger.info("y_pred {}".format(y_pred))

        return y_pred