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

class Annotation_Evaluation():

    def plot_confusion_matrix(self, cm, classes,
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

    def evaluate_kapa_correlation(self,
                                  in_dir=config.EVALUATE_DATA_DIR,
                                  predict_colname="predict",
                                  annotation_colnames=("annotation_1", "annotation_2", "annotation_3"),
                                  overlap=99
                                  ):

        # one_hot_encoder = OneHotEncoder(handle_unknown=False)

        label_encoder = LabelEncoder()


        for f in os.listdir(in_dir):
            fname = os.path.join(in_dir, f)
            if os.path.isfile(fname) and fname.endswith(".csv"):
                print("Read file {}".format(fname))
                df = pd.read_csv(fname)

                label_set = set(list(df[list(annotation_colnames)].astype(str).values.ravel()))
                print("label_set", label_set)

                label_encoder.fit(list(label_set))

                out_file = open("annotation_confusion_matrix.csv", "w")
                csv_writer = csv.writer(out_file)


                for i, annotation_colname in enumerate(annotation_colnames):
                    for j in range(i+1, len(annotation_colnames)):
                        y_label1 = df[annotation_colname].astype(str).head(overlap)
                        y_label2 = df[annotation_colnames[j]].astype(str).head(overlap)
                        y1 = label_encoder.transform(y_label1)
                        y2 = label_encoder.transform(y_label2)
                        # print(annotation_colname, df[annotation_colname].astype(str).head(overlap))
                        print(annotation_colname, y1)

                        # print(annotation_colnames[j], df[annotation_colnames[j]].astype(str).head(overlap))
                        print(annotation_colnames[j], y2)
                        agreement_score = cohen_kappa_score(y1, y2)
                        print("The Kappa Correlation is {} for {} and {}"
                              .format(round(agreement_score, 2), annotation_colname, annotation_colnames[j]))

                        # y true


                        class_names = list(label_set)
                        cnf_matrix = confusion_matrix(y_label1, y_label2, class_names)
                        print("class_names={}".format(class_names))
                        print("The confusion matrix is {}".format(cnf_matrix))
                        comm = "The Kappa Correlation is {} for {} and {}".format(round(agreement_score, 2), annotation_colname, annotation_colnames[j])
                        csv_writer.writerow([comm])
                        csv_writer.writerow(["{}\{}".format(annotation_colname, annotation_colnames[j])] + class_names)
                        for i, row in enumerate(cnf_matrix):
                            csv_writer.writerow([class_names[i]] + row.tolist())
                            out_file.flush()

                        csv_writer.writerow([])

                        np.set_printoptions(precision=2)

                        # Plot non-normalized confusion matrix
                        # plt.figure()
                        plt.figure()
                        # from matplotlib.pyplot import figure
                        # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

                        self.plot_confusion_matrix(cnf_matrix, classes=class_names,
                                              title='Confusion matrix for {} and {}'.format(annotation_colname, annotation_colnames[j]))


                        plt.show()

                        print()

                df = df.where((pd.notnull(df)), None)
                df["gold_label"] = df[list(annotation_colnames)].apply(lambda row: self.get_majority_vote(row), axis=1)
                df.to_csv(fname)
                # Only consider the ones have overlap
                df = df.head(overlap)
                # Remove the onces that doesn't have agreement
                df = df[df["gold_label"].notnull()]
                # self.evaluate(df["gold_label"].astype(str).head(overlap), df[predict_colname].astype(str).head(overlap))
                self.evaluate(df["gold_label"].astype(str), df[predict_colname].astype(str))
                out_file.close()

    def get_majority_vote(self, row):

        label_count_dict = collections.defaultdict(int)
        for label in row:
            if label == None:
                continue
            label_count_dict[label] += 1

        if len(label_count_dict) == 0:
            return None

        sorted_label_count = sorted(label_count_dict.items(), key=operator.itemgetter(1), reverse=True)

        if len(sorted_label_count) == 1:
            return sorted_label_count[0][0]

        if sorted_label_count[0][1] > sorted_label_count[1][1]:
            return sorted_label_count[0][0]
        else:
            # No one wins
            return None


    def evaluate(self, y_gold, y_pred):

        label_encoder = LabelEncoder()

        label_set = set(y_gold.tolist() + y_pred.tolist())
        class_names = list(label_set)

        with open("prediction_confusion_matrix.csv", "w") as out_file:
            csv_writer = csv.writer(out_file)

            cnf_matrix = confusion_matrix(y_gold, y_pred, class_names)
            print("class_names={}".format(class_names))
            print("The confusion matrix is {}".format(cnf_matrix))
            csv_writer.writerow(["TP\P"] + class_names)
            for i, row in enumerate(cnf_matrix):
                csv_writer.writerow([class_names[i]] + row.tolist())
                out_file.flush()

            np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
            # plt.figure()
            plt.figure()
            # from matplotlib.pyplot import figure
            # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

            self.plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
            plt.show()


        # agreement_score = cohen_kappa_score(y_gold, y_pred)
        # print("The 'Kappa Correlation' is {} for gold and predict".format(round(agreement_score, 2)))

        label_encoder.fit(class_names)
        y_gold = label_encoder.transform(y_gold)
        y_pred = label_encoder.transform(y_pred)

        precision, recall, F1, support = precision_recall_fscore_support(y_gold, y_pred, average='macro')
        print("macro", round(precision, 4), round(recall, 4), round(F1, 4), support)
        precision, recall, F1, support = precision_recall_fscore_support(y_gold, y_pred, average='micro')
        print("micro", round(precision, 4), round(recall, 4), round(F1, 4), support)
        precision, recall, F1, support = precision_recall_fscore_support(y_gold, y_pred, average='weighted')
        print("weighted", round(precision, 4), round(recall, 4), round(F1, 4), support)

        target_names = label_encoder.inverse_transform(list(set(y_gold.tolist() + y_pred.tolist())))

        report = classification_report(y_gold, y_pred, target_names=target_names)
        print(report)


if __name__ == "__main__":
    Annotation_Evaluation().evaluate_kapa_correlation()

