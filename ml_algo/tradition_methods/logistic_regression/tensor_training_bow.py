import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config
from ml_algo.preprocessing.feature_processing import Feature_Processing
from utils.file_logger import File_Logger_Helper
from utils.pickel_helper import Pickle_Helper

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Oct 27 2018"




class Training():

    def __init__(self, logger=None):
        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="training")

        self.preprocessor = Feature_Processing()

        self.model_name = None
        self.model = None

    def load_model_if_exists(self,
                             classifier_name="general",
                             preprocess_name="general",
                             dump_model_dir=config.PREROCESS_PICKLES_DIR):
        # Load the file is not already done so. If there is no pickle created, train one for it.
        self.logger.info("Load Model")

        self.dump_model_dir = dump_model_dir
        if not os.path.exists(dump_model_dir):
            os.makedirs(dump_model_dir)

        self.model_name = "{}_{}".format(classifier_name, preprocess_name)

        self.dump_model_fname = os.path.join(dump_model_dir, "{}.pickle".format(self.model_name))

        self.model = Pickle_Helper.load_model_from_pickle(self.dump_model_fname)

    def store_model_if_not_exits(self, replace_exists=False):
        if not os.path.exists(self.dump_model_fname) or replace_exists is True:
            Pickle_Helper.save_model_to_pickle(self.model, self.dump_model_fname)

    def train(self, in_fname,
              drop_colnames: list = None,
              label_colname="label",
              is_sparse=True,
              standardize=False,
              convert_bool=False,
              convert_row_percentage=False,
              normalize_text=True,
              use_stem=False,
              bag_of_word=False,
              max_token_number:int=None,
              counter_ngram: int = None,
              embedding=False, sentence_size_percentage: float = 1, min_word_freq=1):

        X, y, feature_names = self.preprocessor.preprocess_X_y_featurenames(
            in_fname=in_fname,
            drop_colnames=drop_colnames,
            label_colname=label_colname,
            is_sparse=is_sparse,
            standardize=standardize,
            convert_bool=convert_bool,
            convert_row_percentage=convert_row_percentage,
            normalize_text=normalize_text,
            use_stem=use_stem,
            bag_of_word=bag_of_word,
            max_token_number=max_token_number,
            counter_ngram=counter_ngram,
            embedding=embedding, sentence_size_percentage=sentence_size_percentage,
            min_word_freq=min_word_freq)

        self.load_model_if_exists(
            classifier_name="logistic_regression", preprocess_name=self.preprocessor.model_name
            )

         # Train the model here
        # print("X", X)
        X_train, X_test, y_train, y_test = self.preprocessor.split_train_test(X, y, test_size=0.1)
        # print("X_train", X_train)
        # X_train = list(X_train)
        # X_test = list(X_test)
        # y_train = list(y_train)
        # y_test = list(y_test)


        # reference : https://github.com/nfmcclure/tensorflow_cookbook/blob/master/07_Natural_Language_Processing
        # Setup Index Matrix for one-hot-encoding
        with tf.Session() as sess:
            embedding_size = len(self.preprocessor.vocab_processor.vocabulary_)
            print("embedding_size", embedding_size)
            sentence_size = self.preprocessor.vocab_processor.max_document_length
            print("sentence_size", sentence_size)

            # Setup Index Matrix for one-hot-encoding
            identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

            # Create variables for logistic regression
            A = tf.Variable(tf.random_normal(shape=[embedding_size, 1]))
            b = tf.Variable(tf.random_normal(shape=[1, 1]))

            # Initialize placeholders
            x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
            y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)

            # Text-Vocab Embedding
            x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
            x_col_sums = tf.reduce_sum(x_embed, 0)

            # Declare model operations
            x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
            # y = AX + b
            model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)

            # Declare loss function (Cross Entropy loss)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

            # Prediction operation
            prediction = tf.sigmoid(model_output)

            # Declare optimizer
            my_opt = tf.train.GradientDescentOptimizer(0.001)
            train_step = my_opt.minimize(loss)

            # Intitialize Variables
            init = tf.global_variables_initializer()
            sess.run(init)

            # Start Logistic Regression
            # print('Starting Training Over {} Sentences.'.format(len(X_train)))
            loss_vec = []
            train_acc_all = []
            train_acc_avg = []
            for ix, t in enumerate(X_train.toarray()):
                y_data = [[y_train[ix]]]

                # Run through each observation for training
                print("data_vector", t)
                print("target", y_data)
                sess.run(train_step, feed_dict={x_data: t, y_target: y_data})
                temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
                loss_vec.append(temp_loss)

                if (ix + 1) % 10 == 0:
                    print('Training Observation #{}, Loss = {}'.format(ix + 1, temp_loss))

                # Keep trailing average of past 50 observations accuracy
                # Get prediction of single observation
                [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
                # Get True/False if prediction is accurate
                train_acc_temp = y_train[ix] == np.round(temp_pred)
                train_acc_all.append(train_acc_temp)
                if len(train_acc_all) >= 50:
                    train_acc_avg.append(np.mean(train_acc_all[-50:]))

            # Get test set accuracy
            # print('Getting Test Set Accuracy For {} Sentences.'.format(len(X_test)))
            test_acc_all = []
            for ix, t in enumerate(X_test.toarray()):
                y_data = [[y_test[ix]]]

                if (ix + 1) % 50 == 0:
                    print('Test Observation #{}'.format(str(ix + 1)))

                # Keep trailing average of past 50 observations accuracy
                # Get prediction of single observation
                [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
                # Get True/False if prediction is accurate
                test_acc_temp = y_test[ix] == np.round(temp_pred)
                test_acc_all.append(test_acc_temp)

            print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))

            # Plot training accuracy over time
            plt.plot(range(len(train_acc_avg)), train_acc_avg, 'k-', label='Train Accuracy')
            plt.title('Avg Training Acc Over Past 50 Generations')
            plt.xlabel('Generation')
            plt.ylabel('Training Accuracy')
            plt.show()

            self.store_model_if_not_exits(replace_exists=True)

