import os
import config
from utils.file_logger import File_Logger_Helper
from utils.pickel_helper import Pickle_Helper
import numpy as np
import datetime
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pandas as pd

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Oct 24 2018"

class Word_Embedding():

    def __init__(self,
                 embedding_fname=os.path.join(config.WORD_EMBEDDING_DIR, 'glove.6B.100d.txt'),
                 embedding_name="token_vector",
                 num_words:int=10000,
                 embedding_vector_dimension:int=100,
                 max_text_len:int=100,
                 data_name="data",
                 feature_name="f1.f2",
                 target_name="t",
                 replace_exists=False,
                 logger=None):
        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="word_embedding")
        self.embedding_fname = embedding_fname
        self.embedding_name = embedding_name
        self.num_words = num_words
        self.embedding_vector_dimension = embedding_vector_dimension
        self.max_text_len = max_text_len
        self.data_name = data_name
        self.feature_name = feature_name
        self.target_name = target_name
        self.replace_exists = replace_exists
        self.tokenizer = None
        self.embedding_matrix = None
        self.dump_model_dir = config.PREROCESS_PICKLES_DIR
        self.load_init_model()


    def generate_model_name(self):
        self.model_name = self.embedding_name

        if self.data_name is not None:
            self.model_name = self.model_name + "_{}".format(self.data_name)
        if self.feature_name is not None:
            self.model_name = self.model_name + "_{}".format(self.feature_name)
        if self.target_name is not None:
            self.model_name = self.model_name + "_{}".format(self.target_name)

        if self.num_words is not None:
            self.model_name = self.model_name + "_{}tkn".format(self.num_words)
        if self.embedding_vector_dimension is not None:
            self.model_name = self.model_name + "_{}emb".format(self.embedding_vector_dimension)
        if self.max_text_len is not None:
            self.model_name = self.model_name + "_{}len".format(self.max_text_len)

        self.logger.info("model_name={}".format(self.model_name))
        return self.model_name

    def load_init_model(self):

        if not os.path.exists(self.dump_model_dir):
            os.makedirs(self.dump_model_dir)

        self.generate_model_name()

        # Load the file is not already done so. If there is no pickle created, train one for it.
        self.logger.info("Load Model {}".format(self.model_name))

        self.dump_tokenizer_fname = os.path.join(self.dump_model_dir, "{}_tokenizer.pickle".format(self.model_name))
        self.dump_embmatrix_fname = os.path.join(self.dump_model_dir, "{}_embmatrix.pickle".format(self.model_name))
        if self.replace_exists is False:
            self.tokenizer = Pickle_Helper.load_model_from_pickle(self.dump_tokenizer_fname)
            self.embedding_matrix = Pickle_Helper.load_model_from_pickle(self.dump_embmatrix_fname)

    def load_embedding_weight_vector_dict(self):
        # TODO: extract this part of code and make it more general.

        self.embeddings_index = {}

        if self.embedding_fname is not None and os.path.exists(self.embedding_fname):
            start = datetime.datetime.now()
            with open(self.embedding_fname) as embedding_file:
                for line in embedding_file:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    if self.embedding_vector_dimension is None:
                        self.embedding_vector_dimension = len(coefs)
                    self.embeddings_index[word] = coefs

            end = datetime.datetime.now()
            self.logger.info("It takes {}s to load {} word vectors from embedding file {}"
                             .format((end - start).total_seconds(), len(self.embeddings_index), self.embedding_fname))

    def store_tokenzier(self, replace_exists=False):
        if not os.path.exists(self.dump_tokenizer_fname) or replace_exists is True:
            if self.tokenizer is not None:
                Pickle_Helper.save_model_to_pickle(self.tokenizer, self.dump_tokenizer_fname)

    def store_embedding_matrix(self, replace_exists=False):
        if not os.path.exists(self.dump_embmatrix_fname) or replace_exists is True:
            if self.embedding_matrix is not None:
                Pickle_Helper.save_model_to_pickle(self.embedding_matrix, self.dump_embmatrix_fname)

    def check_dimension(self, embedding_vector_dimension):
        if embedding_vector_dimension is None:
            # Get dimension from the embedding matrix
            embedding_vector_dimension = self.embedding_vector_dimension
        elif self.embedding_vector_dimension is None:
            # If the embedding matrix is null. This is the "token_vector" case.
            self.embedding_vector_dimension = embedding_vector_dimension
        elif embedding_vector_dimension != self.embedding_vector_dimension:
            self.logger.error("Error, the embedding vector dimension should be {} instead of {}. Auto fix to {}".format(self.embedding_vector_dimension, embedding_vector_dimension, self.embedding_vector_dimension))
            # If the user set a different dimension that is different from the embedding matrix.
            embedding_vector_dimension = self.embedding_vector_dimension

        return embedding_vector_dimension

    def get_embedding_weight_vector(self, word):
        embedding_vector = self.embeddings_index.get(word)
        return embedding_vector

    def init_embedding_layer(self, X:np.ndarray):
        # TODO: just deal with one column
        # The simplest way to do it is to execute by columns.

        if self.tokenizer is None or self.replace_exists:
            self.logger.info('New tokenizer with {} number of words.'.format(self.num_words))
            self.tokenizer = Tokenizer(num_words=self.num_words)
            self.tokenizer.fit_on_texts(X.ravel())
            self.store_tokenzier(replace_exists=self.replace_exists)
            # self.tokenizer.fit_on_texts(X.values.ravel())
        self.logger.info('Tokenizer:\n {}'.format(self.tokenizer))

        word_index = self.tokenizer.word_index
        self.logger.info('Found {} unique tokens with {} num_words'.format(len(word_index), self.num_words))
        self.logger.info("word_index={}".format(word_index))

        # Reference: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        self.embedding_matrix = None
        if "token_vector" in self.embedding_name:
            self.logger.info("in_dim={}, out_dim={}, text_length={}".format(self.num_words,
                                                                            self.embedding_vector_dimension,
                                                                            self.max_text_len))
            embedding_layer = Embedding(self.num_words,
                                        self.embedding_vector_dimension,
                                        input_length=self.max_text_len)

        else:
            if self.embedding_matrix is None or self.replace_exists == True:
                # Only load the pretrained embedding weight when the embedding matrix is None.
                self.load_embedding_weight_vector_dict()
                # Add one more dim for the bias. The bias will be the las row of the embedding matrix.
                self.embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_vector_dimension))
                for word, i in word_index.items():
                    # Assign the pre-trained weight to the embedding vector.
                    embedding_vector = self.get_embedding_weight_vector(word)
                    if embedding_vector is not None:
                        # words not found in embedding index will be all-zeros.
                        self.embedding_matrix[i] = embedding_vector
                self.logger.info("Initial embedding matrix {}".format(self.embedding_matrix))
                # print("sum matrix", np.matrix(self.embedding_matrix).sum())
                self.store_embedding_matrix(self.replace_exists)

            self.logger.info(
                "in_dim={}, out_dim={}, text_length={}".format(len(word_index) + 1,
                                                               self.embedding_vector_dimension,
                                                               self.max_text_len))
            embedding_layer = Embedding(len(word_index) + 1,
                                        self.embedding_vector_dimension,
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_text_len,
                                        trainable=False)

        return embedding_layer


    def encode_X(self, X:np.ndarray):
        if self.tokenizer is None:
            self.logger.error("Please initial the embedding by Word_Embedding().init_embedding_layer first")
            return None

        self.logger.info("X.head={}".format(X.head(5)))
        X = self.tokenizer.texts_to_sequences(X)
        self.logger.info("sequance X {}".format(X))
        max_text_len = self.max_text_len
        X = sequence.pad_sequences(X, maxlen=max_text_len)
        self.logger.info("padding X {}".format(X))
        return X



