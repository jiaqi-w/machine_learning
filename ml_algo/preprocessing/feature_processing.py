from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy import sparse
import os, re
from os.path import basename
import pandas as pd
from collections import Counter

import config
from utils.file_logger import File_Logger_Helper
from utils.pickel_helper import Pickle_Helper
import string
import numpy as np
import math
from tensorflow.contrib import learn
from sklearn.pipeline import Pipeline
from nltk.stem import SnowballStemmer
from nltk import word_tokenize

import matplotlib.pyplot as plt

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Oct 24 2018"


class Feature_Processing():

    def __init__(self, logger=None):
        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="preprocessing")

        self.stemmer = SnowballStemmer('english')

        # Fixme: we should use dictionary to save the features to vector
        self.standard_scaler = None
        self.one_hot_encoder = None
        self.dictionary = None
        self.counter_vector = None
        self.vocab_processor = None
        self.label_encoder = None

        self.model_name = None

    def generate_model_name(self, in_fname,
                            is_sparse=True,
                            standardize=False,
                            one_hot_encode=False,
                            convert_bool=False,
                            convert_row_percentage=False,
                            normalize_text=True,
                            use_stem=False,
                            min_tokens:int=None,
                            bag_of_word=False,
                            max_token_number:int=None,
                            counter_ngram:int=None,
                            embedding=False, sentence_size_percentage:float=1, min_word_freq=1):

        model_name = "preprocess"
        if in_fname is not None:
            model_name = re.sub(r"\.\w+$", "", basename(in_fname))

        if is_sparse:
            model_name = model_name + "_spr"
        if standardize:
            model_name = model_name + "_std"
        if convert_bool:
            model_name = model_name + "_bool"
        if convert_row_percentage:
            model_name = model_name + "_rpct"
        if normalize_text:
            model_name = model_name + "_ntext"
        if use_stem:
            model_name = model_name + "_stem"
        if min_tokens:
            model_name = model_name + "_mint{}".format(min_tokens)
        if bag_of_word:
            model_name = model_name + "_bow"
        if max_token_number:
            model_name = model_name + "_tfidf{}".format(max_token_number)
        if counter_ngram is not None:
            model_name = model_name + "_n{}".format(counter_ngram)
        if embedding:
            model_name = model_name + "_embd"
        if sentence_size_percentage:
            model_name = model_name + "_senlenth{}".format(round(sentence_size_percentage,2))
        if min_word_freq:
            model_name = model_name + "_minfreq{}".format(min_word_freq)

        self.model_name = model_name
        self.logger.info("model_name={}".format(model_name))


    def load_model_if_exists(self,
                             in_fname,
                             is_sparse=True,
                             standardize=False,
                             one_hot_encode=False,
                             convert_bool=False,
                             convert_row_percentage=False,
                             normalize_text=True,
                             use_stem=False,
                             min_tokens:int=None,
                             bag_of_word=False,
                             max_token_number:int=None,
                             counter_ngram:int=None,
                             embedding=False, sentence_size_percentage:float=1, min_word_freq=1,
                             dump_model_dir=config.PREROCESS_PICKLES_DIR):
        # Load the file is not already done so. If there is no pickle created, train one for it.
        self.logger.info("Load Model")

        self.dump_model_dir = dump_model_dir
        if not os.path.exists(dump_model_dir):
            os.makedirs(dump_model_dir)

        self.generate_model_name(in_fname=in_fname,
                                 is_sparse=is_sparse,
                                 standardize=standardize,
                                 one_hot_encode=one_hot_encode,
                                 convert_bool=convert_bool,
                                 convert_row_percentage=convert_row_percentage,
                                 normalize_text=normalize_text,
                                 use_stem=use_stem,
                                 min_tokens=min_tokens,
                                 bag_of_word=bag_of_word,
                                 max_token_number=max_token_number,
                                 counter_ngram=counter_ngram,
                                 embedding=embedding, sentence_size_percentage=sentence_size_percentage,
                                 min_word_freq=min_word_freq)

        self.dump_standard_scaler_fname = os.path.join(dump_model_dir, "{}_standard_scaler.pickle".format(self.model_name))
        self.dump_one_hot_encode_fname = os.path.join(dump_model_dir, "{}_onehot_encoder.pickle".format(self.model_name))
        self.dump_dictionary_fname = os.path.join(dump_model_dir, "{}_dictionary.pickle".format(self.model_name))
        self.dump_counter_vec_fname = os.path.join(dump_model_dir, "{}_countvec.pickle".format(self.model_name))
        self.dump_label_encoder_fname = os.path.join(dump_model_dir, "{}_label.pickle".format(self.model_name))
        self.dump_vocab_processor_fname = os.path.join(dump_model_dir, "{}_embedding.pickle".format(self.model_name))

        self.standard_scaler = Pickle_Helper.load_model_from_pickle(self.dump_standard_scaler_fname)
        self.one_hot_encoder = Pickle_Helper.load_model_from_pickle(self.dump_one_hot_encode_fname)
        self.dictionary = Pickle_Helper.load_model_from_pickle(self.dump_dictionary_fname)
        self.counter_vector = Pickle_Helper.load_model_from_pickle(self.dump_counter_vec_fname)
        self.label_encoder = Pickle_Helper.load_model_from_pickle(self.dump_label_encoder_fname)
        self.vocab_processor = Pickle_Helper.load_model_from_pickle(self.dump_vocab_processor_fname)

    def store_model(self, replace_exists=False):
        if not os.path.exists(self.dump_standard_scaler_fname) or replace_exists is True:
            if self.standard_scaler is not None:
                Pickle_Helper.save_model_to_pickle(self.standard_scaler, self.dump_standard_scaler_fname)
        if not os.path.exists(self.dump_one_hot_encode_fname) or replace_exists is True:
            if self.one_hot_encoder is not None:
                Pickle_Helper.save_model_to_pickle(self.one_hot_encoder, self.dump_one_hot_encode_fname)
        if not os.path.exists(self.dump_dictionary_fname) or replace_exists is True:
            if self.dictionary is not None:
                Pickle_Helper.save_model_to_pickle(self.dictionary, self.dump_dictionary_fname)
        if not os.path.exists(self.dump_counter_vec_fname) or replace_exists is True:
            if self.counter_vector is not None:
                Pickle_Helper.save_model_to_pickle(self.counter_vector, self.dump_counter_vec_fname)
        if not os.path.exists(self.dump_label_encoder_fname) or replace_exists is True:
            if self.label_encoder is not None:
                Pickle_Helper.save_model_to_pickle(self.label_encoder, self.dump_label_encoder_fname)
        if not os.path.exists(self.dump_vocab_processor_fname) or replace_exists is True:
            if self.vocab_processor is not None:
                Pickle_Helper.save_model_to_pickle(self.vocab_processor, self.dump_vocab_processor_fname)

    def get_X_y_featurenames_from_file(self, filename, drop_colnames:list=None, label_colname="label"):
        # TODO: Add Data_Preprocessing for data filtering.
        self.logger.info("Read file {}".format(filename))
        df = pd.read_csv(filename)
        if drop_colnames is not None:
            df = df.drop(drop_colnames, axis=1)

        # Get the features.
        X = df.drop(label_colname, axis=1)

        # Get the label.
        # y = df[[label_colname]]
        y = df[label_colname]

        # # Get the header of the features.
        # feature_names = X.columns.values.tolist()

        return X, y

    def get_text_length_for_embedding(self, X_col, text_length_percentage:float=1, show_plot=False):
        # TODO: We might want to have varying sentence lenghth in the future.
        self.logger.info("Observe data distribution.")
        text_length = pd.DataFrame({'length': X_col.apply(lambda x: len(x.split()))})
        self.logger.info("max={}".format(text_length["length"].values.max()))
        self.logger.info("min={}".format(text_length["length"].values.min()))
        self.logger.info("mean={}".format(text_length["length"].values.mean()))
        self.logger.info("std={}".format(text_length["length"].values.std()))
        mean = text_length["length"].values.mean()
        std = text_length["length"].values.std()

        self.logger.info("To include 68% data, we recommend to take length within [{}, {}]."
                         .format(max(0, math.floor(mean - std)), math.ceil(mean + std)))
        self.logger.info("To include 95% data, we recommend to take length within [{}, {}]."
                         .format(max(0, math.floor(mean - 2 * std)), math.ceil(mean + 2 * std)))

        z_score = stats.norm.ppf(text_length_percentage)
        x = z_score * std + mean
        if x == math.inf or x < 25:
            # When 100% is taken, we need to return a maximum value.
            # When the sentence length is no larger than 25, we need to return a maximum value.
            x = math.ceil(text_length["length"].values.max())
        else:
            x = math.ceil(x)

        self.logger.info("We take the text length within [{}, {}] for {}% of the length.".format(0, x, round(text_length_percentage * 100, 2)))


        if show_plot is True:
            length_count = text_length.groupby(["length"])[['length']].size().reset_index(name='count')
            plt.figure()
            length_count.plot(x='length', y='count')
            plt.show()
        return x

    def normalize_text_row(self, text_row, use_stem):
        norm_text_row = []
        for text in text_row:
            norm_text_row.append(self.normalize_text(text, use_stem=use_stem))
        return norm_text_row

    def normalize_text(self, text, use_stem):
        norm_text = None
        if text is not None:
            text = text.lower()
            if use_stem:
                text = self.stemmer.stem(text)
            norm_text = ""
            # Remove punctuation and numbers
            for char in text:
                if char not in string.punctuation and char not in '0123456789':
                    norm_text = norm_text + char

            # Trim extra whitespace
            norm_text = re.sub(r"\s+", " ", norm_text).strip()
        return norm_text

    def preprocess_X(self, X, convert_bool=False,
                     convert_row_percentage=False,
                     normalize_text=False,
                     use_stem=False,
                     min_tokens:int=None):
        # customize preprocessing for features. Inherit this method to do the converstion.
        if convert_bool:
            # Convert the feature value to bool value.
            X = X.apply(lambda x: x > 0).astype(int)

        elif convert_row_percentage:
            # Sum the columns
            # Normalize the feature value according to the row sum.
            # This is used when there is only one type of features.
            # axis=1 means for each column
            X = X.apply(lambda x: x / x.sum(), axis=1)

        elif normalize_text:
            # Normalize the text in all the text columns.
            # str_X = X.select_dtypes(include=['object'])
            # col_names = str_X.columns.values.tolist()
            # X[col_names] = X[col_names].fillna('')
            # X[col_names] = X[col_names].apply(self.normalize_text_row, axis=1)

            # For single column.
            X = X.fillna('')
            X = X.apply(lambda x : self.normalize_text(x, use_stem=use_stem))

        if min_tokens is not None:
            # Filter out the data has fewer tokens than min_tokens.
            X = X.apply(lambda x: x if len(x.split()) >= min_tokens else None).notnull()

        print("X", X)

        return X

    def encode_X(self, X,
                 is_sparse=True,
                 standardize=False,
                 one_hot_encode=False,
                 convert_bool=False,
                 convert_row_percentage=False,
                 normalize_text=True,
                 use_stem=False,
                 min_tokens:int=None,
                 bag_of_word=False,
                 max_token_number:int=None,
                 counter_ngram:int=None,
                 embedding=False, sentence_size_percentage:float=1, min_word_freq=1,
                 show_plot=False,
                 replace_exists=False):
        # The simplest way to do it is to execute by columns.
        feature_coomatrix_columns = []
        feature_names = []
        # TODO: automatically encode the features. For example, use bag of word or counter for the text column. But one hot encoder for the categorical columns.
        # FIXME: if embedding is used, the multicolumn might not work properlly since each column has different max length for the text.
        for col_name in X.columns.values.tolist():
            self.logger.info("Preprocess column '{}'".format(col_name))

            X_col = self.preprocess_X(X[col_name],
                                      convert_bool=convert_bool,
                                      convert_row_percentage=convert_row_percentage,
                                      normalize_text=normalize_text,
                                      use_stem=use_stem,
                                      min_tokens=min_tokens
                                      )
            feature_name = col_name

            if standardize is True:
                # z-score normalize the scale of the feature
                if self.standard_scaler is None or replace_exists:
                    with_mean = not is_sparse
                    # When the data is sparse, do it without mean so that the 0 entry will stay 0.
                    self.standard_scaler = StandardScaler(with_mean=with_mean, with_std=True)
                self.logger.info("Model: {}".format(self.dictionary))
                X_values = self.standard_scaler.fit_transform(X_col)

            elif one_hot_encode is True:
                # categorical_features could declare the columns to be one hot encoder. sparse = True/False
                if self.one_hot_encoder is None or replace_exists:
                    self.one_hot_encoder = OneHotEncoder(handle_unknown=False)
                self.logger.info("Model: {}".format(self.one_hot_encoder))
                # For a single column.
                X_values = self.one_hot_encoder.fit_transform(X_col)
                self.logger.info("Shape of matrix '{}' is {}".format(col_name, X_values.shape))
                # self.logger.info("matrix {}".format(X_values))
                self.logger.info("The feature size is {}".format(len(self.one_hot_encoder.get_feature_names())))
                feature_name = ["{}_{}".format(col_name, fn) for fn in list(self.one_hot_encoder.get_feature_names())]

            # TODO: try whether X.ravel() vs. X
            elif bag_of_word is True:
                if self.dictionary is None or replace_exists:
                    # Some times we might want to keep the stop word.
                    self.dictionary = TfidfVectorizer(tokenizer=word_tokenize,
                                                      stop_words='english',
                                                      max_features=max_token_number)
                self.logger.info("Model: {}".format(self.dictionary))
                # For a single column.
                X_values = self.dictionary.fit_transform(X_col)
                self.logger.info("Shape of matrix '{}' is {}".format(col_name, X_values.shape))
                # self.logger.info("matrix {}".format(X_values))
                self.logger.info("The vocabulary size is {}".format(len(self.dictionary.vocabulary_)))
                feature_name = ["{}_{}".format(col_name, fn) for fn in list(self.dictionary.vocabulary_.keys())]

            elif counter_ngram is not None:
                if self.counter_vector is None or replace_exists:
                    # Some times we might want to keep the stop word.
                    self.counter_vector = CountVectorizer(tokenizer=word_tokenize,
                                                          ngram_range=(1, max(1, counter_ngram)),
                                                          stop_words='english',
                                                          max_features=max_token_number)
                self.logger.info("Model: {}".format(self.counter_vector))
                # For a single column.
                X_values = self.counter_vector.fit_transform(X_col)

                freq_distribution = Counter(dict(zip(self.counter_vector.get_feature_names(), X_values.sum(axis=0).A1)))
                self.logger.info("The most frequent words: {}".format(freq_distribution.most_common(50)))
                self.logger.info("Shape of matrix '{}' is {}".format(col_name, X_values.shape))
                # self.logger.info("matrix {}".format(X_values))
                self.logger.info("The feature size is {}".format(len(self.counter_vector.vocabulary_)))
                feature_name = ["{}_{}".format(col_name, fn) for fn in list(self.counter_vector.vocabulary_.keys())]

                if show_plot is True:
                    count_vect_df = pd.DataFrame(X_values.todense(), columns=self.counter_vector.get_feature_names())
                    # X.sum(axis=0).A1
                    frequency_count = count_vect_df.sum().reset_index(name="sum").groupby("sum").size().reset_index(name='count')
                    plt.figure()
                    frequency_count.plot(x='sum', y='count')
                    plt.xlabel('word frequency for {}'.format(col_name))
                    plt.ylabel('count')
                    plt.title('Word Frequency Distribution')
                    plt.show()

            elif embedding is True:
                # Setup vocabulary processor

                sentence_size = self.get_text_length_for_embedding(X_col, sentence_size_percentage, show_plot=show_plot)

                if self.vocab_processor is None or replace_exists:
                    self.vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)
                self.logger.info("Embedding Model: {}".format(self.vocab_processor))

                # Have to fit transform to get length of unique words.
                # X_col = vocab_processor.fit_transform(X_col.values.ravel())
                X_values = np.array(list(self.vocab_processor.fit_transform(X_col.values)))
                self.logger.info("Shape of matrix '{}' is {}".format(col_name, X_values.shape))
                self.logger.info("matrix {}".format(X_values))
                embedding_size = len(self.vocab_processor.vocabulary_)
                self.logger.info("The sentence size is {}".format(sentence_size))
                self.logger.info("The embedding size is {}".format(embedding_size))

                vocab_dict = self.vocab_processor.vocabulary_._mapping
                feature_name = ["{}_{}".format(col_name, fn) for fn in list(vocab_dict.keys())]

            else:
                X_values = X.values

            feature_coomatrix_columns.append(sparse.coo_matrix(X_values))
            if isinstance(feature_name, list):
                # Noted: There might be same feature names for different feature
                feature_names += feature_name
            else:
                feature_names.append(feature_name)

        if is_sparse is True:
            # horizontal append. Return a sparse matrix.
            feature_matrix = sparse.hstack(feature_coomatrix_columns)
            self.logger.info("Shape of all feature matrix is {}".format(feature_matrix.shape))
        else:
            # horizontal append. Return a dense matrix.
            feature_matrix = np.hstack(feature_coomatrix_columns)
            self.logger.info("Shape of all feature matrix is {}".format(feature_matrix.shape))
        self.logger.info("feature_names = {}".format(feature_names))
        return feature_matrix, feature_names

    def preprocess_y(self, label_list):
        # customize preprocessing for label. Inherit this method to do the converstion.
        return label_list

    def encode_y(self, y:pd.Series=None):
        # Used when the task is to classify discrete target.
        if y is None or len(y) == 0:
            return y

        self.logger.info("distribution of y:\n{}".format(y.value_counts()))

        y = self.preprocess_y(y)
        # Check whether the label is number
        is_num = False

        if np.issubdtype(y.dtype, np.number):
            is_num = True

        if is_num is False:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            # y = self.label_encoder.fit_transform(y.ravel())

            # class_names = self.label_encoder.inverse_transform(label_list)
            # print(class_names)
        else:
            y = y.values

        self.logger.info("y={}".format(y))
        return y

    def get_target_names(self, encoded_y):
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(np.sort(np.unique(encoded_y)).tolist())
        else:
            return None
            # return list(set(encoded_y))

    def preprocess_X_y_featurenames(self, in_fname,
                                    drop_colnames:list=None,
                                    label_colname="label",
                                    is_sparse=True,
                                    standardize=False,
                                    one_hot_encode=False,
                                    convert_bool=False,
                                    convert_row_percentage=False,
                                    normalize_text=True,
                                    use_stem=False,
                                    min_tokens:int=None,
                                    bag_of_word=False,
                                    max_token_number:int=None,
                                    counter_ngram:int=None,
                                    embedding=False, sentence_size_percentage:float=1, min_word_freq=1,
                                    replace_exists=False):

        self.load_model_if_exists(
            in_fname=in_fname,
            is_sparse=is_sparse,
            standardize=standardize,
            one_hot_encode=one_hot_encode,
            convert_bool=convert_bool,
            convert_row_percentage=convert_row_percentage,
            normalize_text=normalize_text,
            use_stem=use_stem,
            min_tokens=min_tokens,
            bag_of_word=bag_of_word,
            max_token_number=max_token_number,
            counter_ngram=counter_ngram,
            embedding=embedding, sentence_size_percentage=sentence_size_percentage,
            min_word_freq=min_word_freq)

        X, y = self.get_X_y_featurenames_from_file(in_fname,
                                                   drop_colnames=drop_colnames,
                                                   label_colname=label_colname)

        X, feature_names = self.encode_X(X,
                          is_sparse=is_sparse,
                          standardize=standardize,
                          one_hot_encode=one_hot_encode,
                          convert_bool=convert_bool,
                          convert_row_percentage=convert_row_percentage,
                          normalize_text=normalize_text,
                          use_stem=use_stem,
                          min_tokens=min_tokens,
                          bag_of_word=bag_of_word,
                          max_token_number=max_token_number,
                          counter_ngram=counter_ngram,
                          embedding=embedding, sentence_size_percentage=sentence_size_percentage, min_word_freq=min_word_freq,
                          replace_exists=replace_exists
                          )

        y = self.encode_y(y)

        # Set this to false if it takes a long time to generate the model.
        self.store_model(replace_exists=True)

        return X, y, feature_names

    # def split_train_test(self, X, y, test_size=0.1):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    #     return X_train, X_test, y_train, y_test

