from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from ml_algo.feature_processing import Feature_Processing
import os, config
from utils.pickel_helper import Pickle_Helper
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import numpy as np


class CNN_Training():

    def __init__(self, num_words=10000, max_text_len=1600, embedding_vecor_length=300):
        self.num_words = num_words
        self.max_text_len = max_text_len
        self.embedding_vecor_length = embedding_vecor_length

        self.embedding = None

        self.preprocessing = Feature_Processing()
        self.model = None

        self.load_model_if_exists(classifier_name="cnn",
                                  preprocess_name="nw{}_maxlen{}_emveclen{}".format(self.num_words, self.max_text_len,
                                                                                    self.embedding_vecor_length))

    def load_model_if_exists(self,
                             classifier_name="general",
                             preprocess_name="general",
                             dump_model_dir=config.PREROCESS_PICKLES_DIR):
        # Load the file is not already done so. If there is no pickle created, train one for it.
        print("Load Model")

        self.dump_model_dir = dump_model_dir
        if not os.path.exists(dump_model_dir):
            os.makedirs(dump_model_dir)

        self.model_name = "{}_{}".format(classifier_name, preprocess_name)

        self.dump_model_fname = os.path.join(dump_model_dir, "{}.h5".format(self.model_name))

        if os.path.exists(self.dump_model_fname):
            self.model = load_model(self.dump_model_fname)

    def store_model(self, replace_exists=False):
        if not os.path.exists(self.dump_model_fname) or replace_exists is True:
            self.model.save(self.dump_model_fname)

    def train(self, X, y, test_size=0.1, replace_exists=False):
        # TODO: store the tokenizer in feature preprocessing, add label name and classifier name in the dump
        ### Create sequence
        tokenizer = Tokenizer(num_words=self.num_words)
        tokenizer.fit_on_texts(X.values.ravel())

        # Split the data.
        # X_train, X_test, y_train, y_test = preprocessing.split_train_test(X["moment"], y, test_size=test_size)
        # TODO: Figure out why ravel will work?? And its dimension is still the same???
        X_train, X_test, y_train, y_test = self.preprocessing.split_train_test(X.values.ravel(), y, test_size=test_size)

        # Pad the sequence to the same length
        print("X_train", X_train)
        X_train = tokenizer.texts_to_sequences(X_train)
        print("sequance", X_train)
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_text_len)
        print("padding", X_train)

        X_test = tokenizer.texts_to_sequences(X_test)
        X_test = sequence.pad_sequences(X_test, maxlen=self.max_text_len)

        if self.model == None or replace_exists:
            # Using embedding from Keras
            self.model = Sequential()
            # input_dim = self.num_words
            # output_dim = self.embedding_vecor_length
            self.embedding = Embedding(self.num_words, self.embedding_vecor_length, input_length=self.max_text_len)
            self.model.add(self.embedding)

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

