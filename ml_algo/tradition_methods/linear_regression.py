import csv, argparse, os
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from utils.pickel_helper import Pickle_Helper
import config
# from flask import current_app
from utils.file_logger import File_Logger_Helper
import pandas as pd
from sklearn.metrics import mean_squared_error
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Sept 14 2018"


class Linear_Regression():
    def __init__(self, dump_model_dir, train_file=None, is_bool_value=False, standardize=False, model_name="gen",
                 logger=None):
        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="linear_regression")
        self.classifier_name = "linear_regression"
        if not os.path.exists(dump_model_dir):
            os.makedirs(dump_model_dir)
        general_model_name = "lassocv10_linear_regression"
        self.dump_model_fname = os.path.join(dump_model_dir,
                                             "{}_{}_model.pickle".format(model_name, general_model_name))
        self.dump_dictionary_fname = os.path.join(dump_model_dir,
                                                  "{}_{}_dictionary.pickle".format(model_name, general_model_name))
        self.dump_standard_scaler_fname = os.path.join(dump_model_dir, "{}_{}_standard_scaler.pickle".format(model_name,
                                                                                                             general_model_name))
        self.out_coef_fname = os.path.join(config.ML_OUTPUT_DIR,
                                           "{}_{}_coeficient.csv".format(model_name, general_model_name))
        self.model_name = "{}_{}".format(model_name, general_model_name)
        self.model = None
        self.dictionary = None
        self.standard_scaler = None
        self.load_model(train_file, is_bool_value, standardize)

    # def read_X_list_label_list(self, filename, feature_column="signatures", label_colname="rating", is_bool_value=False):
    #     # Please note that feature column could only has one in this script, else it wont work.
    #     self.logger.info("Read file" + filename)
    #     vocabulary_set = set()
    #     with open(filename) as infile:
    #         reader = csv.DictReader(infile, delimiter='\t')
    #         X_list = []
    #         label_list = []
    #         count = 0
    #         for row in reader:
    #             voc_concated = self.pre_processing(row[feature_column])
    #             if voc_concated is None or voc_concated.strip() == "":
    #                 continue
    #             voc_list = voc_concated.split(",")
    #
    #             if isinstance(voc_list, list) and len(voc_list) > 1:
    #                 for voc in voc_list:
    #                     if voc == "":
    #                         continue
    #                     vocabulary_set.add(voc)
    #             else:
    #                 vocabulary_set.add(voc_concated)
    #
    #             if is_bool_value:
    #                 uniq_voc_list = []
    #                 if isinstance(voc_list, list) and len(voc_list) > 1:
    #                     for voc in voc_list:
    #                         if voc in uniq_voc_list:
    #                             continue
    #                         else:
    #                             uniq_voc_list.append(voc)
    #                     voc_concated = ",".join(uniq_voc_list)
    #                 else:
    #                     uniq_voc_list.append(voc_concated)
    #
    #             X_list.append(voc_concated)
    #             count += 1
    #             label_list.append(self.label_mapper(row[label_colname]))
    #
    #         print("vocabulary_set", vocabulary_set)
    #         print("num_feature={}".format(len(vocabulary_set)))
    #         print("There are {} data".format(count))
    #         return np.array(X_list), np.array(label_list), np.array(list(vocabulary_set))

    def read_X_list_label_list(self, filename, id_column="session_id", feature_column="signatures",
                               label_colname="rating", is_bool_value=False):
        # Please note that feature column could only has one in this script, else it wont work.
        self.logger.info("Read file" + filename)
        prev_id = None
        vocabulary_set = set()
        with open(filename) as infile:
            reader = csv.DictReader(infile, delimiter='\t')
            X_list = []
            id_feature_list = None
            label_list = []
            count = 0
            for row in reader:
                cur_id = row[id_column]
                if cur_id != prev_id:
                    if id_feature_list is not None and len(id_feature_list) > 0:
                        X_list.append(",".join(id_feature_list))
                        count += 1
                        label_list.append(self.label_mapper(row[label_colname]))
                    id_feature_list = []

                feature = self.pre_processing(row[feature_column])
                vocabulary_set.add(feature)

                if is_bool_value is False or feature not in id_feature_list:
                    id_feature_list.append(feature)

                prev_id = cur_id

            print("vocabulary_set", vocabulary_set)
            print("num_feature={}".format(len(vocabulary_set)))
            print("There are {} data".format(count))
            return np.array(X_list), np.array(label_list), np.array(list(vocabulary_set))

    def pre_processing(self, feature):
        # Override this function if we need to convert the features
        # This feature must be a continues number
        return feature

    def label_mapper(self, label: float):
        # Override this function if we need to convert the label
        return round(float(label))

    def fit_transform_features_array_label_array(self, X_list, label_list=None, is_training=False, vocabulary=None,
                                                 standardize=False):
        # FIXME: do the standardize later.
        # label_list = np.array(label_list)
        # FIXME: Please remember to do this
        if is_training is True:
            if vocabulary is not None:
                self.dictionary = CountVectorizer(vocabulary=vocabulary)
            else:
                self.dictionary = CountVectorizer()
            self.dictionary.fit(X_list.ravel())

        features_array = self.dictionary.transform(X_list.ravel())

        # Normalize the features
        # Cannot center sparse matrices: pass `with_mean=False` instead.
        if standardize == True:
            self.standard_scaler = StandardScaler(with_mean=False, with_std=True)
            features_array = self.standard_scaler.fit_transform(features_array)

        # print("features_array", features_array)
        # A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
        features_array = sparse.hstack([features_array]).tocsr()
        # print(features_array)
        # features_array = features_array.toarray()
        return features_array, label_list

    def get_features_array_label_array_from_file(self, in_fname, is_training=False, is_bool_value=False,
                                                 standardize=False):
        X_list, label_list, vocabulary = self.read_X_list_label_list(in_fname, is_bool_value=is_bool_value)
        return self.fit_transform_features_array_label_array(X_list, label_list, is_training, vocabulary, standardize)

    def train_model(self, train_file, is_bool_value=False, standardize=False):
        # training
        self.logger.info("Get Features")
        features_array, label_array = self.get_features_array_label_array_from_file(train_file, is_training=True,
                                                                                    is_bool_value=is_bool_value,
                                                                                    standardize=standardize)
        # TODO: check about the different parameters.
        # self.model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        # self.model = linear_model.Lasso(alpha = 0.1)
        self.model = linear_model.LassoCV(cv=10, normalize=False, verbose=True, n_jobs=2)
        print(self.model)
        self.logger.info("Training Model")
        self.model.fit(features_array, label_array)

        Pickle_Helper.save_model_to_pickle(self.model, self.dump_model_fname)
        Pickle_Helper.save_model_to_pickle(self.dictionary, self.dump_dictionary_fname)
        Pickle_Helper.save_model_to_pickle(self.standard_scaler, self.dump_standard_scaler_fname)
        self.print_linear_regression_formular()

    def print_linear_regression_formular(self, pass_vocabulary=None, out_fname=None):
        # print("self.model.coef_", self.model.coef_)
        fname = out_fname or self.out_coef_fname
        vocabulary = pass_vocabulary or self.dictionary.vocabulary_
        # print("vocabulary", vocabulary)
        with open(fname, "w") as out_file:
            fieldnames = ["feature",
                          "coefficient",
                          "removed",
                          "(MSE={},alpha={},intercept={})".format(self.model.mse_path_.mean(), self.model.alpha_,
                                                                  self.model.intercept_)]
            csv_writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for idx, col_name in enumerate(vocabulary):
                # print("col_name", col_name)
                # print("The coefficient for {} is {}".format(col_name, self.model.coef_[0][idx]))
                # elements.append("{}*{}".format(self.model.coef_[0][idx], col_name))
                coef = self.model.coef_[idx]
                new_row = {}
                new_row["feature"] = col_name
                new_row["coefficient"] = coef
                if abs(coef - 0) < 0.0000001:
                    new_row["removed"] = True
                else:
                    print("The coefficient for {} is {}".format(col_name, self.model.coef_[idx]))
                csv_writer.writerow(new_row)

        print("Model", self.model_name)
        print(self.model)
        print("Number of vocabulary is {}".format(len(vocabulary)))
        print("Number of coefficient is {}".format(len(self.model.coef_)))
        print("alpha_ = {}".format(self.model.alpha_))
        print("intercept = {}".format(self.model.intercept_))
        print("mse_mean = {}".format(self.model.mse_path_.mean()))

        # Display results: http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
        # Feature Selection: http://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_boston.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-boston-py
        # Display results
        # print("self.model.mse_path_", self.model.mse_path_)
        # print("self.model.alphas_", self.model.alphas_)
        # avg_mse = sum_mse/(len(self.model.mse_path_) * 10)
        # print("self.model.mse_path_.mean() {}".format(self.model.mse_path_.mean()))
        # print("self.model.mse_path_.mean(axis=-1) {}".format(self.model.mse_path_.mean(axis=-1)))

        # m_log_alphas = -np.log10(self.model.alphas_)
        #
        # plt.figure()
        # ymin, ymax = (min(self.model.mse_path_.mean(axis=-1)) - 0.2), (max(self.model.mse_path_.mean(axis=-1)) + 0.2)
        # plt.plot(m_log_alphas, self.model.mse_path_, ':')
        # sum_mse = 0
        # for mse_10 in self.model.mse_path_:
        #     # print("mse", mse_10)
        #     for mse in mse_10:
        #         sum_mse += mse
        #
        # plt.plot(m_log_alphas, self.model.mse_path_.mean(axis=-1), 'k',
        #          label='Average across the folds', linewidth=2)
        # plt.axvline(-np.log10(self.model.alpha_), linestyle='--', color='k',
        #             label='alpha: CV estimate')
        #
        # plt.legend()
        #
        # plt.xlabel('-log(alpha)')
        # plt.ylabel('Mean square error')
        # plt.title('Mean square error on each fold: coordinate descent')
        # plt.axis('tight')
        # plt.ylim(ymin, ymax)
        # plt.show()

    def load_model(self, train_file, is_bool_value, standardize):
        # Load the file is not already done so. If there is no pickle created, train one for it.
        self.logger.info("Load Model")
        if self.model is None:
            self.model = Pickle_Helper.load_model_from_pickle(self.dump_model_fname)
            self.dictionary = Pickle_Helper.load_model_from_pickle(self.dump_dictionary_fname)
            self.standard_scaler = Pickle_Helper.load_model_from_pickle(self.dump_standard_scaler_fname)

        if self.model is None:
            self.train_model(train_file, is_bool_value, standardize)

    def predict_results(self, test_file, result_file=None):
        self.logger.info("Predict Results")
        if self.model == None:
            self.logger.error("Please train the model before testing")

        self.print_linear_regression_formular()

        features_array, label_array = self.get_features_array_label_array_from_file(test_file)
        # TODO: save the prediction results as well.
        predict_results = self.model.predict(features_array)
        # TODO: implement the R^2 score for evaluation.
        self.calculate_metrics(label_array, predict_results)

    def predict(self, text):
        # FIXME: add all the parameter to the fit_ function
        features_array, label_array = self.fit_transform_features_array_label_array(np.array([text]))
        predict_results = self.model.predict(features_array)
        if predict_results is not None and len(predict_results) > 0:
            class_names = self.label_encoder.inverse_transform(predict_results)
            return class_names[0]
        else:
            self.logger.error("Faile to predict for text ", text)


def convert_file_format(infname, outfname):
    fieldnames = set()
    with open(infname) as infile:
        csv_reader = csv.DictReader(infile, delimiter='\t')
        for row in csv_reader:
            fieldnames.add(row["signatures"])

    with open(infname) as infile:
        csv_reader = csv.DictReader(infile, delimiter='\t')
        with open(outfname, "w") as outfile:
            # Fixme: this list might need to be stored.
            print("Begin converting the {} to {}".format(infname, outfname))
            csv_writer = csv.DictWriter(outfile, fieldnames=list(fieldnames))
            csv_writer.writeheader()
            print("write header")
            for row in csv_reader:
                new_row = collections.defaultdict(lambda: 0)
                new_row[row["signatures"]] = 1
                csv_writer.writerow(new_row)
                outfile.flush()
            print("Finish converting the {} to {}".format(infname, outfname))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-train", "--train_fname", help="train file", type=str, required=True)
    # parser.add_argument("-test", "--test_fname", help="test file", type=str, required=True)
    # args = parser.parse_args()
    #
    # # parameters
    # train_file = args.train_fname
    # test_file = args.test_fname

    # infname = os.path.join(config.DATA, "ml_input", "signature_turn_ratings.csv")
    infname1 = os.path.join(config.DATA, "ml_input", "signature_turn_aug_ratings.csv")
    # infname = os.path.join(config.DATA, "ml_input", "signature_conv_ratings_aug.csv")


    classifier = Linear_Regression(dump_model_dir=config.LINEAR_REGRESSION_ML_PICKLES_DIR,
                                   train_file=infname1,
                                   is_bool_value=True,
                                   standardize=True,
                                   model_name="sigconv_zscore_aug")

    # classifier.print_linear_regression_formular()

    classifier = Linear_Regression(dump_model_dir=config.LINEAR_REGRESSION_ML_PICKLES_DIR,
                                   train_file=infname1,
                                   is_bool_value=True,
                                   standardize=False,
                                   model_name="sigconv_bool_aug")


    # classifier.print_linear_regression_formular()








