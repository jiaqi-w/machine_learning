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
__date__ = "Oct 28 2018"


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
              max_token_number:int = None,
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
            # Implementing TF-IDF
            # ---------------------------------------
            #
            # Here we implement TF-IDF,
            #  (Text Frequency - Inverse Document Frequency)
            #  for the spam-ham text data.
            #
            # We will use a hybrid approach of encoding the texts
            #  with sci-kit learn's TFIDF vectorizer.  Then we will
            #  use the regular TensorFlow logistic algorithm outline.

            #Do not use tf.reset_default_graph() to clear nested graphs. If you need a cleared graph, exit the nesting and create a new graph.
            # ops.reset_default_graph()

            # TODO: do we want to observe the distribution before using this?
            max_features = len(self.preprocessor.dictionary.vocabulary_)

            # Create variables for logistic regression
            A = tf.Variable(tf.random_normal(shape=[max_features, 1]))
            b = tf.Variable(tf.random_normal(shape=[1, 1]))

            # Initialize placeholders
            x_data = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
            y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

            # Declare logistic model (sigmoid in loss function)
            model_output = tf.add(tf.matmul(x_data, A), b)

            # Declare loss function (Cross Entropy loss)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

            # Actual Prediction
            prediction = tf.round(tf.sigmoid(model_output))
            predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
            accuracy = tf.reduce_mean(predictions_correct)

            # Declare optimizer
            my_opt = tf.train.GradientDescentOptimizer(0.0025)   #why the learning rate is different here?
            train_step = my_opt.minimize(loss)

            # Intitialize Variables
            init = tf.global_variables_initializer()
            sess.run(init)

            # Start Logistic Regression
            train_loss = []
            test_loss = []
            train_acc = []
            test_acc = []
            i_data = []
            batch_size = 200
            X_train = X_train.toarray()
            # y_train = y_train.toarray()
            X_test = X_test.toarray()
            # y_test = y_test.toarray()
            for i in range(10000):

                # Randomly pick example. Why?
                rand_index = np.random.choice(X_train.shape[0], size=batch_size)
                # rand_x = X_train[rand_index].todense()
                rand_x = X_train[rand_index]
                rand_y = np.transpose([y_train[rand_index]])

                sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

                # Only record loss and accuracy every 100 generations
                if (i + 1) % 100 == 0:
                    i_data.append(i + 1)
                    train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
                    train_loss.append(train_loss_temp)

                    test_loss_temp = sess.run(loss, feed_dict={x_data: X_test,
                                                               y_target: np.transpose([y_test])})
                    test_loss.append(test_loss_temp)

                    train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
                    train_acc.append(train_acc_temp)

                    test_acc_temp = sess.run(accuracy, feed_dict={x_data: X_test,
                                                                  y_target: np.transpose([y_test])})
                    test_acc.append(test_acc_temp)
                if (i + 1) % 500 == 0:
                    acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
                    acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
                    print(
                        'Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(
                            *acc_and_loss))

            # Plot loss over time
            plt.plot(i_data, train_loss, 'k-', label='Train Loss')
            plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
            plt.title('Cross Entropy Loss per Generation')
            plt.xlabel('Generation')
            plt.ylabel('Cross Entropy Loss')
            plt.legend(loc='upper right')
            plt.show()

            # Plot train and test accuracy
            plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
            plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
            plt.title('Train and Test Accuracy')
            plt.xlabel('Generation')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            plt.show()

            self.store_model_if_not_exits(replace_exists=True)
