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
from ml_algo.preprocessing.word_embedding import Word_Embedding
from sklearn.feature_extraction.text import CountVectorizer
import abc


__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Dec 5 2018"

class Customized_Embedding_ABC(Word_Embedding, abc.ABC):

    def __init__(self,
                 embedding_fname=None,
                 embedding_name="customize",
                 custom_feature_list=("c1", "c2", "c3"),
                 custom_feature_binary=True,
                 num_words:int=10000,
                 embedding_vector_dimension:int=100,
                 max_text_len:int=100,
                 data_name="data",
                 feature_name="f1.f2",
                 target_name="t",
                 replace_exists=False,
                 logger=None):

        super().__init__(embedding_fname=embedding_fname,
                         embedding_name=embedding_name,
                         num_words=num_words,
                         embedding_vector_dimension=embedding_vector_dimension,
                         max_text_len=max_text_len,
                         data_name=data_name,
                         feature_name=feature_name,
                         target_name=target_name,
                         replace_exists=replace_exists,
                         logger=logger
                         )

        self.custom_feature_list = list(custom_feature_list)
        self.custom_feature_binary = custom_feature_binary

        self.load_feature_bin_vector_model()

    def load_feature_bin_vector_model(self):

        self.dump_catbinvector_fname = os.path.join(self.dump_model_dir, "{}_embd_feavector.pickle".format(self.model_name))
        if self.replace_exists is False:
            self.custom_feature_vector = Pickle_Helper.load_model_from_pickle(self.dump_catbinvector_fname)

        if self.custom_feature_vector is None:
            start = datetime.datetime.now()
            end = datetime.datetime.now()

            feature = self.custom_feature_list
            self.custom_feature_vector = CountVectorizer(binary=self.custom_feature_binary)
            self.custom_feature_vector.fit(feature)

            self.logger.info("It takes {}s to load {} features.".format((end - start).total_seconds(),
                                                                        len(self.custom_feature_vector.vocabulary_)))
            self.embedding_vector_dimension = len(self.custom_feature_list)
            self.logger.info("The actual embedding_vector_dimension is {}".format(self.embedding_vector_dimension))
            self.store_feature_bin_vector()

    def store_feature_bin_vector(self):
        if not os.path.exists(self.dump_catbinvector_fname) or self.replace_exists is True:
            if self.custom_feature_vector is not None:
                Pickle_Helper.save_model_to_pickle(self.custom_feature_vector, self.dump_catbinvector_fname)

    @abc.abstractmethod
    def get_word_custom_feature_list(self, word) -> list:
        # TODO: convert word to the corresponding feature list.
        return []

    def get_embedding_weight_vector(self, word):
        word_feature_list = self.get_word_custom_feature_list(word)
        embedding_vector = self.custom_feature_vector.transform([" ".join(word_feature_list)])
        bias = [0]
        embedding_vector = np.append(embedding_vector.toarray().reshape(-1), bias)
        return embedding_vector
