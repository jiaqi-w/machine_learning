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

class CV_Model_Evaluator():
    
    def __init__(self, logger=None):
        self.logger = logger or File_Logger_Helper.get_logger(logger_fname="cv.log")
        self.mid_list = []
        self.roc_auc_list = []
        self.accuracy_list = []

        self.micro_precision_list = []
        self.micro_recall_list = []
        self.micro_f1_list = []
        self.macro_precision_list = []
        self.macro_recall_list = []
        self.macro_f1_list = []
        self.weighted_precision_list = []
        self.weighted_recall_list = []
        self.weighted_f1_list = []

        self.best_macro_f1 = None
        self.best_macro_f1_mid = None
        self.best_micro_f1 = None
        self.best_micro_f1_mid = None
        self.best_weighted_f1 = None
        self.best_weighted_f1_mid = None

    def get_evaluation_fieldnames(self):
        fieldnames = []
        fieldnames += ['cv_accuracy', 'cv_roc_auc',
                       'cv_macro_prec', 'cv_macro_recall', 'cv_macro_f1',
                       'cv_micro_prec', 'cv_micro_recall', 'cv_micro_f1',
                       'cv_weighted_prec', 'cv_weighted_recall', 'cv_weighted_f1',
                       'best_roc_auc','best_roc_auc_mid',
                       'best_accuracy','best_accuracy_mid',
                       'best_macro_f1','best_macro_f1_mid',
                       'best_micro_f1','best_micro_f1_mid',
                       'best_weighted_f1','best_weighted_f1_mid',
                       ]
        return fieldnames

    def add_evaluation_metric(self, mid, metric_dict):
        self.mid_list.append(mid)
        # self.roc_auc_list.append(metric_dict["roc_auc"])
        self.accuracy_list.append(metric_dict["accuracy"])

        self.micro_precision_list.append(metric_dict["micro_prec"])
        self.micro_recall_list.append(metric_dict["micro_recall"])
        self.micro_f1_list.append(metric_dict["micro_f1"])
        self.macro_precision_list.append(metric_dict["macro_prec"])
        self.macro_recall_list.append(metric_dict["macro_recall"])
        self.macro_f1_list.append(metric_dict["macro_f1"])
        self.weighted_precision_list.append(metric_dict["weighted_prec"])
        self.weighted_recall_list.append(metric_dict["weighted_recall"])
        self.weighted_f1_list.append(metric_dict["weighted_f1"])

    def get_evaluation_dict(self, evaluation_fname=None):
        cv_metric_dict = {}
        if len(self.accuracy_list) == 0:
            return cv_metric_dict
        cv_metric_dict["cv_accuracy"] = round(np.average(self.accuracy_list), 4)
        cv_metric_dict["cv_roc_auc"] = round(np.average(self.roc_auc_list), 4)
        cv_metric_dict["cv_macro_prec"] = round(np.average(self.macro_precision_list), 4)
        cv_metric_dict["cv_macro_recall"] = round(np.average(self.macro_recall_list), 4)
        cv_metric_dict["cv_macro_f1"] = round(np.average(self.macro_f1_list), 4)
        cv_metric_dict["cv_micro_prec"] = round(np.average(self.micro_precision_list), 4)
        cv_metric_dict["cv_micro_recall"] = round(np.average(self.micro_recall_list), 4)
        cv_metric_dict["cv_micro_f1"] = round(np.average(self.micro_f1_list), 4)
        cv_metric_dict["cv_weighted_prec"] = round(np.average(self.weighted_precision_list), 4)
        cv_metric_dict["cv_weighted_recall"] = round(np.average(self.weighted_recall_list), 4)
        cv_metric_dict["cv_weighted_f1"] = round(np.average(self.weighted_f1_list), 4)
        # cv_metric_dict["best_roc_auc"] = np.max(self.roc_auc_list)
        # cv_metric_dict["best_roc_auc_mid"] = self.mid_list[np.argmax(self.roc_auc_list)]
        cv_metric_dict["best_accuracy"] = np.max(self.accuracy_list)
        cv_metric_dict["best_accuracy_mid"] = self.mid_list[np.argmax(self.accuracy_list)]
        cv_metric_dict["best_macro_f1"] = np.max(self.macro_f1_list)
        cv_metric_dict["best_macro_f1_mid"] = self.mid_list[np.argmax(self.macro_f1_list)]
        cv_metric_dict["best_micro_f1"] = np.max(self.micro_f1_list)
        cv_metric_dict["best_micro_f1_mid"] = self.mid_list[np.argmax(self.micro_f1_list)]
        cv_metric_dict["best_weighted_f1"] = np.max(self.weighted_f1_list)
        cv_metric_dict["best_weighted_f1_mid"] = self.mid_list[np.argmax(self.weighted_f1_list)]
        if evaluation_fname is not None:
            with open(evaluation_fname, "w") as evaluate_file:
                csv_writer = csv.DictWriter(evaluate_file, fieldnames=self.get_evaluation_fieldnames())
                csv_writer.writeheader()
                csv_writer.writerow(cv_metric_dict)
                evaluate_file.flush()
        return cv_metric_dict




