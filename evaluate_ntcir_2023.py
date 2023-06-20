import argparse
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import f1_score

import numpy as np


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


def load_data_binary(prediction_file, gold_file):
    """..."""
    df_pred = pd.read_csv(prediction_file, sep=",")
    df_pred = df_pred.drop(columns=["train_id", "text"])

    df_pred["binary_class"] = df_pred.apply(convert_to_binary, axis=1)

    preds = df_pred["binary_class"].tolist()

    df_gold = pd.read_csv(gold_file, sep=",")
    df_gold = df_gold.drop(columns=["train_id", "text"])
    df_gold["binary_class"] = df_gold.apply(convert_to_binary, axis=1)
    golds = df_gold["binary_class"].tolist()

    return preds, golds


def load_data_per_class(prediction_file, gold_file):
    """..."""
    df_pred = pd.read_csv(prediction_file, sep=",")
    df_pred = df_pred.drop(columns=["train_id", "text"])
    preds = df_pred.values.tolist()

    df_gold = pd.read_csv(gold_file, sep=",")
    df_gold = df_gold.drop(columns=["train_id", "text"])
    classes = list(df_gold)
    golds = df_gold.values.tolist()

    return preds, golds, classes


def get_per_class_scores(golds, preds, classes):
    """..."""

    print(classification_report(golds, preds, target_names=classes, zero_division=0))


def get_binary_scores(golds, preds):
    """..."""

    print(
        classification_report(
            golds, preds, labels=[0, 1], target_names=["no ADR", "ADR"], zero_division=0
        )
    )


def main(prediction_file, gold_file):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nBinary Scores:")

    preds, golds = load_data_binary(prediction_file, gold_file)

    get_binary_scores(preds, golds)

    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nPer Class Scores:")

    preds, golds, classes = load_data_per_class(prediction_file, gold_file)

    get_per_class_scores(golds, preds, classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("prediction_file", default=None)
    parser.add_argument("gold_file", default=None)

    args = parser.parse_args()

    main(args.prediction_file, args.gold_file)
