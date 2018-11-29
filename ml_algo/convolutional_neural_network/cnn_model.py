import config
import os, re
from utils.file_logger import File_Logger_Helper

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Convolution1D, Flatten, Dropout
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from ml_algo.preprocessing.feature_processing import Feature_Processing
from ml_algo.preprocessing.word_embedding import Word_Embedding

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 1 2018"

class CNN_Model():

    def __init__(self,
                 classifier_name="cnn",
                 num_words=10000,
                 max_text_len=1600,
                 embedding_vector_dimension=100,
                 glove_fname=os.path.join(config.GLOVE_SIXB, 'glove.6B.100d.txt'),
                 data_name="data",
                 feature_name="f1.f2",
                 target_name="t",
                 logger=None):

        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="CNN.log")
        self.preprocessing = Feature_Processing()
        self.classifier_name = classifier_name
        self.num_words = num_words
        self.max_text_len = max_text_len
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
                            preprocess_name,
                            ):

        model_name = general_name
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
        print("Load Model")

        self.dump_model_dir = dump_model_dir
        if not os.path.exists(dump_model_dir):
            os.makedirs(dump_model_dir)

        self.model_name = self.generate_model_name(classifier_name, preprocess_name)
        self.dump_model_fname = os.path.join(dump_model_dir, "{}.h5".format(self.model_name))

        if os.path.exists(self.dump_model_fname):
            self.model = load_model(self.dump_model_fname)

    def store_model(self, replace_exists=False):
        if not os.path.exists(self.dump_model_fname) or replace_exists is True:
            self.model.save(self.dump_model_fname)

    def train(self, X_train, y_train, replace_exists=False,
              general_name="feature_name"):

        # Initial the embedding layer.

        ### Create sequence

        # tokenizer = Tokenizer(num_words=self.num_words)
        # tokenizer.fit_on_texts(X.values.ravel())

        # Split the data.
        # X_train, X_test, y_train, y_test = preprocessing.split_train_test(X["moment"], y, test_size=test_size)
        # TODO: Figure out why ravel will work?? And its dimension is still the same???
        # X_train, X_test, y_train, y_test = self.preprocessing.split_train_test(X.values.ravel(), y, test_size=test_size)

        self.embedding_layer = self.embedding_helper.init_embedding_layer(X_train,
                                                                          num_words=self.num_words,
                                                                          embedding_vector_dimension=self.embedding_vector_dimension,
                                                                          max_text_len=self.max_text_len,
                                                                          general_name=self.embedding_name
                                                                          )

        # Pad the sequence to the same length
        X_train = self.embedding_helper.encode_X(X_train, max_text_len=self.max_text_len)
        # X_test = self.embedding_helper.encode_X(X_test, max_text_len=self.max_text_len)

        if self.model == None or replace_exists:
            # Using embedding from Keras
            self.model = Sequential()
            # input_dim = self.num_words
            # output_dim = self.embedding_vecor_length
            # self.embedding = Embedding(self.num_words, self.embedding_vecor_length, input_length=self.max_text_len)
            self.model.add(self.embedding_layer)

            # Convolutional model (3x conv, flatten, 2x dense)
            self.model.add(Convolution1D(64, 3, padding='same'))
            self.model.add(Convolution1D(32, 3, padding='same'))
            self.model.add(Convolution1D(16, 3, padding='same'))
            self.model.add(Flatten())
            # Avoid overfitting.
            self.model.add(Dropout(0.2))
            self.model.add(Dense(180, activation='sigmoid'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1, activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Log to tensorboard
            tensorBoardCallback = TensorBoard(log_dir=config.LOG_DIR, write_graph=True)

            print("X_train", X_train)
            print("y_train", y_train)
            self.model.fit(X_train, y_train, epochs=3, callbacks=[tensorBoardCallback], batch_size=64)
            self.store_model(replace_exists=replace_exists)

    def evaluate_model(self, X_test, y_test):
        if self.model == None:
            self.logger.error("Please train the model first. There is no model for {}".format(self.model_name))

        # TODO: Update the code to do evaluation.
        # Evaluation on the test set

        # scores = self.model.evaluate(X_test, y_test, verbose=0)
        # print("Accuracy: %.2f%%" % (scores[-1] * 100))

        y_pred = self.model.predict_classes(X_test)
        # TODO: output results.

        precision, recall, F1, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
        print("macro", round(precision, 4), round(recall, 4), round(F1, 4), support)
        precision, recall, F1, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
        print("micro", round(precision, 4), round(recall, 4), round(F1, 4), support)
        precision, recall, F1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print("weighted", round(precision, 4), round(recall, 4), round(F1, 4), support)

        target_names = self.preprocessing.label_encoder.inverse_transform(np.sort(np.unique(y_test)).tolist())

        report = classification_report(y_test, y_pred, target_names=target_names)
        print(report)

