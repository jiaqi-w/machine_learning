import csv, argparse, os, re
from os.path import basename
import pandas as pd

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Jan 12 2019"

class SVM():
    
    def __init__(self, train_file=None, model_name="gen", test_size=None, cv=10, normalize=False, is_bool_value=False, is_percentage=False):
        self.classifier_name = "linear_regression"
        if test_size == None:
            general_model_name = "lassocv{}_regression".format(cv)
        else:
            general_model_name = "lasso_regression_split{}".format(test_size)
        if model_name is None and train_file is not None:
            model_name = re.sub(r"\.\w+$", "", basename(train_file))
        self.model_name = "{}_{}".format(model_name, general_model_name)
        self.out_coef_fname = os.path.join(os.path.dirname(train_file), "{}_coeficient.csv".format(self.model_name))
        self.model = None
        self.standard_scaler = None
        self.load_model(train_file, test_size, cv, normalize, is_bool_value, is_percentage)


    def read_X_list_label_list(self, filename, label_colname="conv_rate", is_bool_value=False, is_percentage=False):
        df = pd.read_csv(filename)
        df = df.drop('session_id', axis=1)
        X = df.drop(label_colname, axis=1)
        if is_bool_value:
            X = X.apply(lambda x : x > 0).astype(int)
        elif is_percentage:
            # Sum the columns
            X = X.apply(lambda x: x / x.sum(), axis=1)
        # print("X", X)
        y = df[[label_colname]]
        return X, y, X.columns.values.tolist()

    def pre_processing(self, feature):
        # TODO: Override this function if we need to convert the features
        # This feature must be a continues number
        return feature

    def label_mapper(self, label:float):
        # TODO: Override this function if we need to convert the label
        return round(float(label))

    def fit_transform_features_array_label_array(self, X_list, label_list=None, feature_names=None, normalize=False):
        if normalize == True:
            # TODO: please set with_mean = True if the data is not sparse.
            self.standard_scaler = StandardScaler(with_mean=False, with_std=True)
            features_array = self.standard_scaler.fit_transform(X_list)
        else:
            features_array = X_list.values

        # A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
        # features_array = sparse.hstack([features_array]).tocsr()
        return features_array, label_list, feature_names

    def get_features_array_label_array_from_file(self, in_fname, normalize=False, is_bool_value=False, is_percentage=False):
        X_list, label_list, feature_names = self.read_X_list_label_list(in_fname, is_bool_value=is_bool_value, is_percentage=is_percentage)
        return self.fit_transform_features_array_label_array(X_list, label_list, feature_names, normalize)

    def train_model_cv(self, train_file, normalize, is_bool_value, is_percentage, cv=10, save_model=False):
        # training
        features_array, label_array, feature_names = self.get_features_array_label_array_from_file(train_file,
                                                                                                normalize=normalize,
                                                                                                is_bool_value=is_bool_value,
                                                                                                is_percentage=is_percentage)
        # TODO: you can change the model here. Now we are using 10-cross valication for the model.
        # self.model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        # self.model = linear_model.Lasso(alpha = 0.1)
        self.model = linear_model.LassoCV(cv=cv, normalize=False, verbose=True, max_iter=10000)
        print("Model Settings:", self.model)
        self.model.fit(features_array, label_array)

        self.print_linear_regression_formular(feature_names)

    def train_model(self, train_file, normalize, is_bool_value, is_percentage, test_size=0.10, alpha=0.1):
        # training
        features_array, label_array, feature_names = self.get_features_array_label_array_from_file(train_file,
                                                                                                normalize=normalize,
                                                                                                is_bool_value=is_bool_value,
                                                                                                is_percentage=is_percentage)

        X_train, X_test, y_train, y_test = train_test_split(features_array, label_array, test_size=test_size)

        # TODO: you can change the model here. Now we aplit the training set and test set.
        # self.model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        # self.model = linear_model.LassoCV(cv=10, normalize=False, verbose=True, max_iter=10000)
        # We could use the best alpha learnt from cross validation.
        self.model = linear_model.Lasso(alpha=alpha)
        print("Model Settings:", self.model)
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        print("R score", score)

        y_predict = self.model.predict(X_test)
        regression_model_mse = mean_squared_error(y_predict, y_test)
        print("alpha", self.model.alpha)
        print("mse", regression_model_mse)

        self.print_linear_regression_formular(feature_names)

    def print_linear_regression_formular(self, feature_names=None, filename=None, is_cross_validation=False):
        if filename is not None and feature_names is None:
            df = pd.read_csv(filename)
            df = df.drop('session_id', axis=1)
            X = df.drop("conv_rate", axis=1)
            feature_names = X.columns.values.tolist()
        if feature_names == None:
            return
        print("vocabulary", feature_names)

        print("Number of vocabulary is {}".format(len(feature_names)))
        print("self.model.coef_", self.model.coef_)
        print("self.model.coef_ len", len(self.model.coef_))

        with open(self.out_coef_fname, "w") as out_file:
            fieldnames = ["feature",
                          "coefficient",
                          "removed",
                          "(intercept={})".format(self.model.intercept_)]
            csv_writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for idx, col_name in enumerate(feature_names):
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
        print("Scaler", self.standard_scaler)
        print("Settings", self.model)
        print("Number of vocabulary is {}".format(len(feature_names)))
        print("Number of coefficient is {}".format(len(self.model.coef_)))
        print("intercept = {}".format(self.model.intercept_))
        if is_cross_validation:
            print("alpha_ = {}".format(self.model.alpha_))
            print("mse_mean = {}".format(self.model.mse_path_.mean()))

    def load_model(self, train_file, test_size, cv, normalize, is_bool_value, is_percentage):
        # Load the file is not already done so. If there is no pickle created, train one for it.

        if self.model is None:
            if test_size is None:
                # Cross validation
                self.train_model_cv(train_file, normalize, is_bool_value, is_percentage, cv)
            else:
                # Otherwise
                self.train_model(train_file, normalize, is_bool_value, is_percentage)


if __name__ == "__main__":
    # python3 lasso_regression_stub.py -in data/lasso_data/toy_example.csv -split 0.1
    # python3 lasso_regression_stub.py -in data/lasso_data/toy_example.csv -cv 10
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--in_fname", help="train/testing file", type=str, required=True)
    parser.add_argument("-split", "--test_size", help="The split rate of training and testing set", type=float, required=False, default=None)
    parser.add_argument("-cv", "--cv", help="The number of cross validation", type=int, default=10)
    parser.add_argument("--normalize", help="normalize the scale of the feature", action="store_true")
    parser.add_argument("--bool", help="convert the feature value to bool value", action="store_true")
    parser.add_argument("--is_percentage", help="normalize the feature value to percentage for each row", action="store_true")

    parser.add_argument("-name", "--model_name", help="save the model with name", type=str, default=None)
    args = parser.parse_args()

    # parameters
    in_fname = args.in_fname
    test_size = args.test_size
    cv = args.cv
    normalize = args.normalize
    is_bool_value = args.bool
    is_percentage = args.is_percentage

    # Optional
    model_name = args.model_name

    classifier = Lasso_Regression(train_file=in_fname,
                                  test_size=test_size,
                                  cv=cv,
                                  normalize=normalize,
                                  is_bool_value=is_bool_value,
                                  is_percentage=is_percentage,
                                  model_name=model_name)