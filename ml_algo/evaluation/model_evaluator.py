import config
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import csv
from sklearn import metrics
from utils.file_logger import File_Logger_Helper
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 9 2018"

class Model_Evaluator():
    
    def __init__(self, y_gold:list, y_pred:list, X_gold:pd.Series=None, is_multi_class=False, logger=None):
        # Please note that the list of gold and predict should have the original label when they pass in.
        self.X_gold = X_gold
        self.y_gold = y_gold
        self.y_pred = y_pred
        self.is_multi_class = is_multi_class
        if is_multi_class is True:
            self.class_names = list(range(self.y_gold.shape[1]))
        else:
            self.class_names = list(set(self.y_gold + self.y_pred))
        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="evaluate.log")

    def get_evaluation_fieldnames(self, with_cm=True):
        fieldnames = []
        for class_name in self.class_names:
            # evaluation metric header
            fieldnames += ["{}_prec".format(class_name), "{}_recall".format(class_name), "{}_f1".format(class_name), "{}_support".format(class_name)]
        fieldnames += ['accuracy', 'roc_auc',
                       'macro_prec', 'macro_recall', 'macro_f1',
                       'micro_prec', 'micro_recall', 'micro_f1',
                       "weighted_prec", "weighted_recall", "weighted_f1"]

        if with_cm is True:
            for class_name in self.class_names:
                for predict_class_name in self.class_names:
                    fieldnames.append("TP_{}_P_{}".format(class_name, predict_class_name))
        return fieldnames

    def get_evaluation_dict(self, evaluation_fname=None, predict_fname=None, cm_fname=None, show_cm=False):
        evaluation_dict = {}
        evaluation_dict.update(self.get_evaluation_metric(evaluation_fname=evaluation_fname, predict_fname=predict_fname))
        if self.is_multi_class is not True:
            evaluation_dict.update(self.get_confusion_matrix(cm_fname=cm_fname, show_plot=show_cm))
        return evaluation_dict


    def get_evaluation_metric(self, evaluation_fname=os.path.join(config.EVALUATE_DATA_DIR, "evaluate.csv"),
                              predict_fname=os.path.join(config.EVALUATE_DATA_DIR, "prediction.csv")):

        # TODO: save the evaluation results in the future.
        metric_dict = {}

        print("self.y_gold", self.y_gold)
        print("self.y_pred", self.y_pred)

        # Compute Area Under the Curve (AUC) using the trapezoidal rule
        # fpr, tpr, thresholds = metrics.roc_curve(self.y_gold, self.y_pred, pos_label=2)
        # print("auc", metrics.auc(fpr, tpr))

        # default average='macro'
        roc_auc = roc_auc_score(self.y_gold, self.y_pred)
        metric_dict["roc_auc"] = round(roc_auc, 4)
        self.logger.info("roc_auc={}".format(round(roc_auc, 4)))

        accuracy = accuracy_score(self.y_gold, self.y_pred)
        metric_dict["accuracy"] = round(accuracy, 4)
        self.logger.info("accuracy={}".format(round(accuracy, 4)))

        precision, recall, F1, support = precision_recall_fscore_support(self.y_gold, self.y_pred, average='macro')
        metric_dict["macro_prec"] = round(precision, 4)
        metric_dict["macro_recall"] = round(recall, 4)
        metric_dict["macro_f1"] = round(F1, 4)
        self.logger.info(
            "macro precision={}, recall={}, f1={}, support={}".format(round(precision, 4), round(recall, 4),
                                                                      round(F1, 4), support))
        precision, recall, F1, support = precision_recall_fscore_support(self.y_gold, self.y_pred, average='micro')
        metric_dict["micro_prec"] = round(precision, 4)
        metric_dict["micro_recall"] = round(recall, 4)
        metric_dict["micro_f1"] = round(F1, 4)
        self.logger.info(
            "micro precision={}, recall={}, f1={}, support={}".format(round(precision, 4), round(recall, 4),
                                                                      round(F1, 4), support))
        precision, recall, F1, support = precision_recall_fscore_support(self.y_gold, self.y_pred, average='weighted')
        metric_dict["weighted_prec"] = round(precision, 4)
        metric_dict["weighted_recall"] = round(recall, 4)
        metric_dict["weighted_f1"] = round(F1, 4)
        self.logger.info(
            "weighted precision={}, recall={}, f1={}, support={}".format(round(precision, 4), round(recall, 4),
                                                                         round(F1, 4), support))

        # For specific class names.
        print("self.class_names", self.class_names)
        if self.is_multi_class is True:
            for i in self.class_names:
                precision, recall, F1, support = precision_recall_fscore_support(self.y_gold[:, i], self.y_pred[:, i],
                                                                                 average='macro')
                metric_dict["{}_prec".format(i)] = round(precision, 4)
                metric_dict["{}_recall".format(i)] = round(recall, 4)
                metric_dict["{}_f1".format(i)] = round(F1, 4)
                metric_dict["{}_support".format(i)] = support
                self.logger.info(
                    "class_name={}, macro precision={}, recall={}, f1={}, support={}".format(i, round(precision, 4), round(recall, 4),
                                                                              round(F1, 4), support))
        else:
            precision = metrics.precision_score(self.y_gold, self.y_pred, labels=self.class_names, average=None)
            recall = metrics.recall_score(self.y_gold, self.y_pred, average=None)
            F1 = metrics.f1_score(self.y_gold, self.y_pred, average=None)
            for i, class_name in enumerate(self.class_names):
                print("class_name", class_name)
                metric_dict["{}_prec".format(class_name)] = round(precision[i], 4)
                metric_dict["{}_recall".format(class_name)] = round(recall[i], 4)
                metric_dict["{}_f1".format(class_name)] = round(F1[i], 4)

        report = classification_report(self.y_gold, self.y_pred)
        self.logger.info("report:\n{}".format(report))

        if evaluation_fname is not None:
            with open(evaluation_fname, "w") as evaluate_file:
                csv_writer = csv.DictWriter(evaluate_file, fieldnames=self.get_evaluation_fieldnames(with_cm=False))
                csv_writer.writeheader()
                csv_writer.writerow(metric_dict)
                evaluate_file.flush()
                self.logger.info("Save evaluation results to {}".format(evaluation_fname))

        # TODO: output results.
        if predict_fname is not None:
            with open(predict_fname, "w") as predict_file:
                if self.X_gold is not None:
                    df = pd.DataFrame(data=self.X_gold)
                else:
                    df = pd.DataFrame()
                df["predict"] = self.y_pred
                df.to_csv(predict_file)
                self.logger.info("Save prediction results to {}".format(predict_fname))

        return metric_dict

    def get_confusion_matrix(self, cm_fname=os.path.join(config.EVALUATE_DATA_DIR, "confusion_matrix.csv"),
                             show_plot=False):

        cm_dict = {}

        cnf_matrix = confusion_matrix(self.y_gold, self.y_pred, self.class_names)
        self.logger.info("self.class_names={}".format(self.class_names))
        self.logger.info("The confusion matrix is \n {} {}".format("TP\P ", self.class_names))
        for i, row in enumerate(cnf_matrix):
            self.logger.info("{}".format([self.class_names[i]] + row.tolist()))
            for j, item in enumerate(row):
                # i == j is the correct predicion
                cm_dict["TP_{}_P_{}".format(self.class_names[i], self.class_names[j])] = item

        if cm_fname is not None:
            with open(cm_fname, "w") as out_file:
                csv_writer = csv.writer(out_file)
                csv_writer.writerow(["TP\P"] + self.class_names)
                for i, row in enumerate(cnf_matrix):
                    csv_writer.writerow([self.class_names[i]] + row.tolist())
                    out_file.flush()
                self.logger.info("Save confusion matrix in file {}".format(cm_fname))

        if show_plot is True:
            # TODO: we can store the plot as well.
            np.set_printoptions(precision=2)
            # Plot non-normalized confusion matrix
            # plt.figure()
            plt.figure()
            # from matplotlib.pyplot import figure
            # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

            Model_Evaluator.plot_confusion_matrix(cnf_matrix, classes=self.class_names,
                                                  title='Confusion matrix')
            plt.show()

        return cm_dict

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        Reference from sklearn documentation.
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        # plt.xticks(tick_marks, classes, rotation=45)
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()




