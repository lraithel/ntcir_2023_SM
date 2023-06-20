import argparse
import pandas as pd

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score


import numpy as np


"""
Evaluation script for the NTCIR'17 Social Media Shared Task

Authors: Hui-Syuan Yeh, Lisa Raithel

"""


def confusion_matrix():
    """..."""

    """
    y_true = np.array([[1, 0, 1],
                       [0, 1, 0]])
    y_pred = np.array([[1, 0, 0],
                       [0, 1, 1]])
    multilabel_confusion_matrix(y_true, y_pred)
    """
    pass


def convert_to_binary(row):
    if sum(row) >= 1:
        return 1
    return 0


def load_data(file_name):
    """Load a CSV file into a pandas dataframe and return the predictions."""
    return pd.read_csv(file_name, sep=",")


def drop_non_class_cols(df):
    """Drop all columns that do not contain class names."""
    return df.drop(columns=["train_id", "text"])


def get_per_class_scores(gold_df, pred_df):
    """..."""
    gold_df = drop_non_class_cols(gold_df)
    pred_df = drop_non_class_cols(pred_df)

    # get the class names
    classes = list(gold_df)

    # get only the labels
    golds = gold_df.values.tolist()
    preds = pred_df.values.tolist()

    print(classification_report(golds, preds, target_names=classes, zero_division=0))


def get_binary_scores(gold_df, pred_df):
    """..."""

    # create a separate dataframe for the content (not needed for scores)
    content_df = gold_df.filter(items=["train_id", "text"])

    gold_df = drop_non_class_cols(gold_df)
    pred_df = drop_non_class_cols(pred_df)

    # add a column with the binary label for each df
    gold_df["binary_gold"] = gold_df.apply(convert_to_binary, axis=1)
    # print(gold_df)
    pred_df["binary_pred"] = pred_df.apply(convert_to_binary, axis=1)

    # get only the labels
    golds = gold_df.values.tolist()
    preds = pred_df.values.tolist()

    # merge the "content" and the binary labels for a better analysis
    # TODO: add #TPs, #FPs, #TN, #FN
    merged_binary_df = pd.concat(
        [content_df, gold_df["binary_gold"], pred_df["binary_pred"]], axis=1
    )

    print(
        classification_report(
            golds, preds, labels=[0, 1], target_names=["no ADR", "ADR"], zero_division=0
        )
    )


def get_balanced_accuracy(gold_df, pred_df):
    """..."""

    gold_df = drop_non_class_cols(gold_df)
    pred_df = drop_non_class_cols(pred_df)

    # get only the labels
    golds = gold_df.values.tolist()
    preds = pred_df.values.tolist()

    # full

    gold_wo1 = len(golds)
    balanced_acc = 0
    acc_0, acc_1 = 0, 0
    exact_match = 0

    for i in range(len(golds)):
        # balanced_acc
        balanced_acc += balanced_accuracy_score(golds[i], preds[i])
        # acc_0
        acc_0 += recall_score(golds[i], preds[i], average=None, zero_division=0)[0]
        # acc_1
        try:
            acc_1 += recall_score(golds[i], preds[i], average=None, zero_division=0)[1]
        except:
            gold_wo1 -= 1

        # exact match
        if golds[i] == preds[i]:
            exact_match += 1

    if gold_wo1 == 0:
        acc_1_overall = 0

    else:
        acc_1_overall = acc_1 / gold_wo1

    print("acc_0: {}".format(acc_0 / len(golds)))
    print("acc_1: {}".format(acc_1_overall))
    print("balanced_acc: {}".format(balanced_acc / len(golds)))
    print("exact_acc: {}".format(exact_match / len(golds)))
    # balanced_acc is averaging, so if example has gold=[0,0,0], we only consider the acc_0
    # applying to the same logic, we only averaging accross the examples having 1


def main(gold_csv, pred_csv):
    """..."""
    gold_df = load_data(gold_csv)  # .values.tolist()
    pred_df = load_data(pred_csv)  # .values.tolist()

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nBinary Scores:")

    get_binary_scores(gold_df, pred_df)

    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nPer Class Scores:")

    get_per_class_scores(gold_df, pred_df)

    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nBalanced Accuracy:")
    get_balanced_accuracy(gold_df, pred_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-gold_file", default=None, type=str)
    parser.add_argument("-prediction_file", default=None, type=str)

    args = parser.parse_args()

    main(gold_csv=args.gold_file, pred_csv=args.prediction_file)
