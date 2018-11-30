import os
import config
from utils.file_logger import File_Logger_Helper
from utils.pickel_helper import Pickle_Helper
import numpy as np
import datetime
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Oct 24 2018"

class Word_Embedding():

    def __init__(self, logger=None, glove_fname=os.path.join(config.GLOVE_SIXB, 'glove.6B.100d.txt')):
        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="word_embedding")
        self.tokenizer = None
        self.init_dict(glove_fname=glove_fname)

    def init_dict(self, glove_fname):

        self.embeddings_index = {}
        self.embedding_vector_dimension = None

        start = datetime.datetime.now()
        with open(glove_fname) as glove_file:
            for line in glove_file:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                if self.embedding_vector_dimension is None:
                    self.embedding_vector_dimension = len(coefs)
                self.embeddings_index[word] = coefs

        end = datetime.datetime.now()
        self.logger.info("It takes {}s to load {} word vectors from GloVe {}"
                         .format((end - start).total_seconds(), len(self.embeddings_index), glove_fname))

    def generate_model_name(self,
                             general_name,
                             num_words:int,
                             embedding_vector_dimension:int,
                             max_text_len:int):

        model_name = general_name
        if num_words is not None:
            model_name = model_name + "_{}tkn".format(num_words)
        if embedding_vector_dimension is not None:
            model_name = model_name + "_{}emb".format(embedding_vector_dimension)
        if max_text_len is not None:
            model_name = model_name + "_{}len".format(max_text_len)

        self.model_name = model_name
        self.logger.info("model_name={}".format(model_name))
        return model_name

    def load_model_if_exists(self,
                             general_name,
                             num_words:int,
                             embedding_vector_dimension:int,
                             max_text_len:int,
                             dump_model_dir=config.PREROCESS_PICKLES_DIR):
        # Load the file is not already done so. If there is no pickle created, train one for it.
        self.logger.info("Load Model")

        self.dump_model_dir = dump_model_dir
        if not os.path.exists(dump_model_dir):
            os.makedirs(dump_model_dir)

        self.generate_model_name(general_name=general_name,
                                 num_words=num_words,
                                 embedding_vector_dimension=embedding_vector_dimension,
                                 max_text_len=max_text_len)

        self.dump_tokenizer_fname = os.path.join(dump_model_dir, "{}_tokenizer.pickle".format(self.model_name))
        self.tokenizer = Pickle_Helper.load_model_from_pickle(self.dump_tokenizer_fname)

    def store_model(self, replace_exists=False):
        if not os.path.exists(self.dump_tokenizer_fname) or replace_exists is True:
            if self.tokenizer is not None:
                Pickle_Helper.save_model_to_pickle(self.tokenizer, self.dump_tokenizer_fname)

    def init_embedding_layer(self, X:np.ndarray,
                             num_words:int,
                             embedding_vector_dimension:int,
                             max_text_len:int,
                             general_name="glove"):
        # TODO: just deal with one column
        # The simplest way to do it is to execute by columns.
        if embedding_vector_dimension is None:
            embedding_vector_dimension = self.embedding_vector_dimension

        self.load_model_if_exists(general_name=general_name,
                                  num_words=num_words,
                                  embedding_vector_dimension=embedding_vector_dimension,
                                  max_text_len=max_text_len)
        self.num_words = num_words
        self.max_text_len = max_text_len

        if self.tokenizer is None:
            self.logger.info('New tokenizer with {} number of words.'.format(num_words))
            self.tokenizer = Tokenizer(num_words=num_words)
            self.tokenizer.fit_on_texts(X.ravel())
            # self.tokenizer.fit_on_texts(X.values.ravel())
        self.logger.info('Tokenizer:\n {}'.format(self.tokenizer))

        word_index = self.tokenizer.word_index
        self.logger.info('Found {} unique tokens with {} num_words'.format(len(word_index), self.num_words))
        self.logger.info("word_index={}".format(word_index))

        # Reference: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        self.embedding_matrix = None
        if "token_vector" in general_name:
            self.logger.info("in_dim={}, out_dim={}, text_length={}".format(num_words, embedding_vector_dimension, max_text_len))
            embedding_layer = Embedding(num_words, embedding_vector_dimension, input_length=max_text_len)

        else:
            # embedding_matrix = np.zeros((len(word_index) + 1, embedding_vector_dimension))
            if embedding_vector_dimension != self.embedding_vector_dimension:
                self.logger.error("Error, the embedding vector dimension should be {} instead of {}".format(self.embedding_vector_dimension, embedding_vector_dimension))
                embedding_vector_dimension = self.embedding_vector_dimension
            self.embedding_matrix = np.zeros((len(word_index) + 1, embedding_vector_dimension))
            for word, i in word_index.items():
                # Assign the pre-trained weight to the embedding vector.
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    self.embedding_matrix[i] = embedding_vector

            self.logger.info(
                "in_dim={}, out_dim={}, text_length={}".format(len(word_index) + 1, embedding_vector_dimension, max_text_len))
            embedding_layer = Embedding(len(word_index) + 1,
                                        embedding_vector_dimension,
                                        weights=[self.embedding_matrix],
                                        input_length=max_text_len,
                                        trainable=False)

        return embedding_layer


    def encode_X(self, X, max_text_len=None):
        if self.tokenizer is None:
            self.logger.error("Please initial the embedding by Word_Embedding().init_embedding_layer first")
            return None

        self.logger.info("X={}".format(X))
        X = self.tokenizer.texts_to_sequences(X)
        self.logger.info("sequance X {}".format(X))
        if max_text_len == None:
            max_text_len = self.max_text_len
        X = sequence.pad_sequences(X, maxlen=max_text_len)
        self.logger.info("padding X {}".format(X))
        return X



