import csv, argparse, os
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from utils.filters import Filters
from utils.pickel_helper import Pickle_Helper
import config
# from flask import current_app
from utils.file_logger import File_Logger_Helper


__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Sept 14 2018"

class Logistic_Regression():

    def __init__(self, dump_model_dir, filter_stopword=True, use_stemm=False, label_dict=None, logger=None):
        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="bow_logistic_regression")
        self.classifier_name = "logistic_regression_bow"
        self.use_stem = use_stemm
        self.stemmer = PorterStemmer()
        self.filter_stopword = filter_stopword
        self.stopwrods = stopwords.words('english')
        self.label_dict = label_dict
        if not os.path.exists(dump_model_dir):
            os.makedirs(dump_model_dir)
        self.dump_model_fname = os.path.join(dump_model_dir, "bow_logistic_regression_model.pickle")
        self.dump_dictionary_fname = os.path.join(dump_model_dir, "bow_logistic_regression_dictionary.pickle")
        self.dump_label_encoder_fname = os.path.join(dump_model_dir, "bow_logistic_regression_label_encoder.pickle")
        self.model = None
        self.dictionary = None
        self.label_encoder = None
        self.load_model()

    def read_text_list_label_list(self, filename):
        self.logger.info("Read file" + filename)
        with open(filename) as infile:
            reader = csv.DictReader(infile)
            text_list = []
            label_list = []
            for row in reader:
                text_list.append(self.pre_processing(row["text"]))
                label_list.append(self.label_mapper(row["label"]))
            return np.array(text_list), np.array(label_list)

    def pre_processing(self, text):
        tokens = []
        if text is not None:
            filtered_text = Filters.replace_non_ascii(text.lower())
            for token in word_tokenize(filtered_text):
                # Use Stemmer
                if self.use_stem is True:
                    updated_token = self.stemmer.stem(token)
                else:
                    updated_token = token
                # Filter stopwords
                if self.filter_stopword is not True or token not in self.stopwrods:
                    tokens.append(updated_token)
        return " ".join(tokens)

    def label_mapper(self, label):
        # If you need to convert the labels.
        # return label
        if self.label_dict and label in self.label_dict:
        # if self.label_dict is not None and label in self.label_dict:
            return self.label_dict[label]
        return label

    def get_features_array_label_array(self, text_list, label_list=None, is_training=False):
        # label_list = np.array(label_list)
        if is_training is True:
            self.dictionary = TfidfVectorizer(stop_words='english')
            self.dictionary.fit(text_list.ravel())
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(label_list.ravel())
        bow_features_array = self.dictionary.transform(text_list.ravel())
        features_array = sparse.hstack([bow_features_array]).tocsr()
        if label_list is not None:
            encoded_label_array = self.label_encoder.transform(label_list.ravel())
            # class_names = self.label_encoder.inverse_transform(encoded_label_array)
            # print(class_names)
        else:
            encoded_label_array = None
        return features_array, encoded_label_array

    def get_features_array_label_array_from_file(self, in_fname, is_training=False):
        text_list, label_list = self.read_text_list_label_list(in_fname)
        return self.get_features_array_label_array(text_list, label_list, is_training)

    def train_model(self, train_file):
        # training
        self.logger.info("Training Model")
        features_array, label_array = self.get_features_array_label_array_from_file(train_file, is_training=True)
        self.model = LogisticRegression(solver='lbfgs',multi_class='multinomial',class_weight='balanced')
        self.model.fit(features_array, label_array)

        Pickle_Helper.save_model_to_pickle(self.model, self.dump_model_fname)
        Pickle_Helper.save_model_to_pickle(self.dictionary, self.dump_dictionary_fname)
        Pickle_Helper.save_model_to_pickle(self.label_encoder, self.dump_label_encoder_fname)

    def load_model(self):
        # Load the file is not already done so. If there is no pickle created, train one for it.
        self.logger.info("Load Model")
        if self.model is None:
            self.model = Pickle_Helper.load_model_from_pickle(self.dump_model_fname)
            self.dictionary = Pickle_Helper.load_model_from_pickle(self.dump_dictionary_fname)
            self.label_encoder = Pickle_Helper.load_model_from_pickle(self.dump_label_encoder_fname)

        if self.model is None:
            self.train_model(config.WASHINGTON_TOPIC_DATA)

    def cross_validation(self, data_file, K=10, results_file=None):
        self.logger.info(str(K) + "Cross Validation")

        features_array, label_array = self.get_features_array_label_array_from_file(data_file, is_training=True)

        kf = KFold(n_splits=K, shuffle=True, random_state=1)

        precision_sum = None
        recall_sum = None
        f1_sum = None
        micro_f1_sum = 0

        header = ['data_source', 'data_type', 'classifier', 'features', "label_name"
             'precision', 'recall', 'f1', 'macro_f1']

        out_file = None
        csv_writer = None
        if results_file is not None:
            out_file = open(results_file, "w")
            csv_writer = csv.writer(out_file)
            csv_writer.writerow(header)

        print(header)
        n = 0
        for train, test in kf.split(label_array):

            self.model = LogisticRegression(solver='lbfgs',multi_class='multinomial',class_weight='balanced')
            self.model.fit(features_array[train,:],label_array[train])

            gold_labels = label_array[test]
            predict_results = self.model.predict(features_array[test,:])

            self.logger.info("Calculate Metrics")
            precision, recall, f1, micro_f1 = self.calculate_metrics(gold_labels, predict_results, n, csv_writer)
            n += 1

            if precision_sum is None:
                precision_sum = np.zeros(len(precision))
                recall_sum = np.zeros(len(precision))
                f1_sum = np.zeros(len(precision))

            precision_sum += np.add(precision_sum, precision)
            recall_sum += np.add(recall_sum, recall)
            f1_sum += np.add(f1_sum, f1)
            micro_f1_sum += micro_f1

        # Pickle_Helper.save_model_to_pickle(self.model, self.dump_model_fname)

        K_precision = precision_sum / K
        K_recall = recall_sum / K
        K_f1 = f1_sum / K
        K_micro_f1 = micro_f1_sum / K

        for i in range (len(K_precision)):
            label_name = self.label_encoder.inverse_transform([i])[0]
            print_row = ["washington_news", "{}-cross-validation".format(K), "logistic_regression", "bow", label_name,
                         "{0:.4f}".format(round(K_precision[i], 4)), "{0:.4f}".format(round(K_recall[i], 4)),
                         "{0:.4f}".format(round(K_f1[i], 4)), "{0:.4f}".format(round(K_micro_f1, 4))]
            self.logger.info(print_row)
            if csv_writer is not None:
                csv_writer.writerow(print_row)

        if csv_writer is not None:
            out_file.close()

    def calculate_metrics(self, X_label, Y_label, K=1, csv_writer=None):
        precision = metrics.precision_score(X_label, Y_label, average=None)
        recall = metrics.recall_score(X_label, Y_label, average=None)
        f1 = metrics.f1_score(X_label, Y_label, average=None)
        micro_f1 = metrics.f1_score(X_label, Y_label, average='micro')

        for i in range (len(precision)):
            label_name = self.label_encoder.inverse_transform([i])[0]
            print_row = ["washington_news", "test_{}".format(K), "logistic_regression", "bow", label_name,
                   "{0:.4f}".format(round(precision[i], 4)), "{0:.4f}".format(round(recall[i], 4)),
                   "{0:.4f}".format(round(f1[i], 4)), "{0:.4f}".format(round(micro_f1, 4))]
            self.logger.info(print_row)
            if csv_writer is not None:
                csv_writer.writerow(print_row)

        return precision, recall, f1, micro_f1

    def predict_results(self, test_file, result_file=None):
        self.logger.info("Predict Results")
        if self.model == None:
            self.logger.error("Please train the model before testing")

        features_array, label_array = self.get_features_array_label_array_from_file(test_file)
        # TODO: save the prediction results as well.
        predict_results = self.model.run_semi_supervise(features_array)
        self.calculate_metrics(label_array, predict_results)

    def predict(self, text):
        features_array, label_array = self.get_features_array_label_array(np.array([text]))
        predict_results = self.model.run_semi_supervise(features_array)
        if predict_results is not None and len(predict_results) > 0:
            class_names = self.label_encoder.inverse_transform(predict_results)
            return class_names[0]
        else:
            self.logger.error("Faile to predict for text ", text)

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-train", "--train_fname", help="train file", type=str, required=True)
    # parser.add_argument("-test", "--test_fname", help="test file", type=str, required=True)
    # args = parser.parse_args()
    #
    # # parameters
    # train_file = args.train_fname
    # test_file = args.test_fname

    from app_settings import create_app
    app = create_app("testing")
    # classifier = BoW_Logistic_Regression(config.BOW_GENERAL_LOGISTIC_REGRESSION_MODEL_DIR)
    # classifier.train_model(train_file)
    # classifier.predict_results(test_file)

    # classifier.cross_validation(config.WASHINGTON_TOPIC_DATA,
    #                             results_file=os.path.join(config.BOW_TOPIC_LOGISTIC_REGRESSION_BOW_MODEL_DIR,
    #                                                       "cross_validation_results.csv"))

    classifier = BoW_Logistic_Regression(config.BOW_SWA_LOGISTIC_REGRESSION_BOW_MODEL_DIR,
                                         label_dict={'%': "PRE", '+': "PLUS", '^2': "UP2", '^g': "UPG",
                                                     '^h': "UPH", '^q': "UPQ", 'aap_am': "AAPAM",
                                                     'arp_nd': "ARPND", 'b^m': "BUPM", 'fo_o_fw_"_by_bc': "FOO",
                                                     'oo_co_cc': "OOCOCC", 'qw^d': "QWUPD", 'qy^d': "QYUPD"})
    classifier.cross_validation(os.path.join(config.DATA, "swda.csv"),
                                results_file=os.path.join(config.BOW_SWA_LOGISTIC_REGRESSION_BOW_MODEL_DIR,
                                                          "cross_validation_results.csv"))







