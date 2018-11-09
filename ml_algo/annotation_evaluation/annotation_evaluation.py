import config
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import collections
import operator

__author__ = "Jiaqi"
__version__ = "1"
__date__ = "Nov 9 2018"

class Annotation_Evaluation():


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
            if os.path.isfile(fname):
                print("Read file {}".format(fname))
                df = pd.read_csv(fname)

                for i, annotation_colname in enumerate(annotation_colnames):
                    for j in range(i+1, len(annotation_colnames)):
                        y1 = label_encoder.fit_transform(df[annotation_colname].astype(str).head(overlap))
                        y2 = label_encoder.fit_transform(df[annotation_colnames[j]].astype(str).head(overlap))
                        print(annotation_colname, y1)
                        print(annotation_colnames[j], y2)
                        agreement_score = cohen_kappa_score(y1, y2)
                        print("The Kappa Correlation is {} for {} and {}"
                              .format(round(agreement_score, 2), annotation_colname, annotation_colnames[j]))

                        print()
                df = df.where((pd.notnull(df)), None)
                df["gold_label"] = df[list(annotation_colnames)].apply(lambda row: self.get_majority_vote(row), axis=1)
                df = df[df["gold_label"] != None]
                self.evaluate(df["gold_label"].astype(str), df[predict_colname].astype(str))

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
        print(label_set)
        label_encoder.fit(list(label_set))
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

