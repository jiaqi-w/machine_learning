import config
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import collections
import operator
import matplotlib.pyplot as plt
import itertools
import csv

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 9 2018"

class Confusion_Matrix_Helper():

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

    @staticmethod
    def evaluate(y_gold:list, y_pred:list, cm_outfname, show_plot=False):

        label_set = set(y_gold + y_pred)
        class_names = list(label_set)

        # TODO: change the location of the output.
        if cm_outfname is None:
            cm_outfname = "prediction_confusion_matrix.csv"
        with open(cm_outfname, "w") as out_file:
            csv_writer = csv.writer(out_file)

            cnf_matrix = confusion_matrix(y_gold, y_pred, class_names)
            print("class_names={}".format(class_names))
            print("The confusion matrix is \n {} {}".format("TP\P ", class_names))
            csv_writer.writerow(["TP\P"] + class_names)
            for i, row in enumerate(cnf_matrix):
                csv_writer.writerow([class_names[i]] + row.tolist())
                print(print("{}".format([class_names[i]] + row.tolist())))
                out_file.flush()

            np.set_printoptions(precision=2)

            if show_plot is True:
                # Plot non-normalized confusion matrix
                # plt.figure()
                plt.figure()
                # from matplotlib.pyplot import figure
                # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

                Confusion_Matrix_Helper.plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
                plt.show()


        precision, recall, F1, support = precision_recall_fscore_support(y_gold, y_pred, average='macro')
        print("macro", round(precision, 4), round(recall, 4), round(F1, 4), support)
        precision, recall, F1, support = precision_recall_fscore_support(y_gold, y_pred, average='micro')
        print("micro", round(precision, 4), round(recall, 4), round(F1, 4), support)
        precision, recall, F1, support = precision_recall_fscore_support(y_gold, y_pred, average='weighted')
        print("weighted", round(precision, 4), round(recall, 4), round(F1, 4), support)

        report = classification_report(y_gold, y_pred)
        print(report)



