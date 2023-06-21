import argparse
import pandas as pd

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score


import numpy as np


"""
Evaluation script for the NTCIR'17 Social Media Shared Task

Authors: Hui-Syuan Yeh, Lisa Raithel

"""


def convert_to_binary(row):
    if sum(row) >= 1:
        return 1
    return 0


def count_tns_bin(row):
    if row["binary_gold"] == 0 and row["binary_pred"] == 0:
        return 1
    return 0


def count_tps_bin(row):
    if row["binary_gold"] == 1 and row["binary_pred"] == 1:
        return 1
    return 0


def count_fps_bin(row):
    if row["binary_gold"] == 0 and row["binary_pred"] == 1:
        return 1
    return 0


def count_fns_bin(row):
    if row["binary_gold"] == 1 and row["binary_pred"] == 0:
        return 1
    return 0


def count_tps(row, classes):
    """..."""
    tps = 0
    for class_ in classes:
        if row[class_ + "_gold"] == 1 and row[class_ + "_pred"] == 1:
            tps += 1

    return tps


def count_fps(row, classes):
    """..."""
    fps = 0
    for class_ in classes:
        if row[class_ + "_gold"] == 0 and row[class_ + "_pred"] == 1:
            fps += 1

    return fps


def count_tns(row, classes):
    """..."""
    tns = 0
    for class_ in classes:
        if row[class_ + "_gold"] == 0 and row[class_ + "_pred"] == 0:
            tns += 1

    return tns


def count_fns(row, classes):
    """..."""
    fns = 0
    for class_ in classes:
        if row[class_ + "_gold"] == 1 and row[class_ + "_pred"] == 0:
            fns += 1

    return fns


def load_data(file_name):
    """Load a CSV file into a pandas dataframe and return the predictions."""
    return pd.read_csv(file_name, sep=",")


def drop_non_class_cols(df):
    """Drop all columns that do not contain class names."""
    return df.drop(columns=["train_id", "text"])


def get_per_class_scores(gold_df, pred_df, to_csv=True):
    """..."""
    content_df = gold_df.filter(items=["train_id", "text"])
    # get rid of train ID and text for a cleaner overview
    gold_df = drop_non_class_cols(gold_df)
    # get the class names
    classes = list(gold_df)
    # add a suffix to distinguish the column headers from those of the preds
    gold_df = gold_df.add_suffix("_gold")

    pred_df = drop_non_class_cols(pred_df).add_suffix("_pred")

    # Concatenate the gold and prediction dataframes.
    # you can also add the IDs and texts for a better analysis:
    # merged_df = pd.concat([content_df, gold_df, pred_df], axis=1)
    merged_df = pd.concat([gold_df, pred_df], axis=1)

    merged_df["#TPs"] = merged_df.apply(count_tps, axis=1, args=(classes,))
    merged_df["#FPs"] = merged_df.apply(count_fps, axis=1, args=(classes,))
    merged_df["#TNs"] = merged_df.apply(count_tns, axis=1, args=(classes,))
    merged_df["#FNs"] = merged_df.apply(count_fns, axis=1, args=(classes,))
    merged_df.loc["Total"] = merged_df.sum(numeric_only=True)

    # get only the labels
    golds = gold_df.values.tolist()
    preds = pred_df.values.tolist()

    print(classification_report(golds, preds, target_names=classes, zero_division=0))

    if to_csv:
        merged_df.to_csv("overview_predictions_per_class.csv")


def get_binary_scores(gold_df, pred_df, to_csv=True):
    """Convert the gold and predicted classes to binary format and calculate scores.

    Returns classification report and a CSV with counts for true positives (TPs),
    false positives (FPs), true negatives (TNs) and false negatives (FNs).
    """

    # create a separate dataframe for the content (not needed for scores)
    content_df = gold_df.filter(items=["train_id", "text"])

    gold_df = drop_non_class_cols(gold_df)
    pred_df = drop_non_class_cols(pred_df)

    # add a column with the binary label for each df
    gold_df["binary_gold"] = gold_df.apply(convert_to_binary, axis=1)
    pred_df["binary_pred"] = pred_df.apply(convert_to_binary, axis=1)

    # merge the "content" and the binary labels for a better analysis
    merged_binary_df = pd.concat(
        [content_df, gold_df["binary_gold"], pred_df["binary_pred"]], axis=1
    )

    merged_binary_df["#TPs"] = merged_binary_df.apply(count_tps_bin, axis=1)
    merged_binary_df["#FPs"] = merged_binary_df.apply(count_fps_bin, axis=1)
    merged_binary_df["#TNs"] = merged_binary_df.apply(count_tns_bin, axis=1)
    merged_binary_df["#FNs"] = merged_binary_df.apply(count_fns_bin, axis=1)

    # get only the binary labels
    golds = merged_binary_df["binary_gold"].values.tolist()
    preds = merged_binary_df["binary_pred"].values.tolist()

    # add total of counts
    merged_binary_df.loc["Total"] = merged_binary_df.sum(numeric_only=True)

    # print(f"Confusion matrix:\n{confusion_matrix(golds, preds)}")

    print(
        classification_report(
            golds, preds, labels=[0, 1], target_names=["no ADE", "ADE"], zero_division=0
        )
    )

    if to_csv:
        merged_binary_df.to_csv("overview_predictions_binary.csv")


def get_per_label_scores(gold_df, pred_df):
    """..."""

    gold_df = drop_non_class_cols(gold_df)
    pred_df = drop_non_class_cols(pred_df)

    # get only the labels
    golds = gold_df.values.tolist()
    preds = pred_df.values.tolist()

    golds_flat = [item for sublist in golds for item in sublist]
    preds_flat = [item for sublist in preds for item in sublist]

    print(classification_report(golds_flat, preds_flat, labels=[0, 1], zero_division=0))

    exact_match = 0

    for i in range(len(golds)):
        if golds[i] == preds[i]:
            exact_match += 1

    print(f"Exact accuracy: {exact_match / len(golds)}\n")


def main(gold_csv, pred_csv):
    """Calculate the different scores.

    Binary: Calculates the performance of classifying a document into the classes
            "contains ADE" vs. "does not contain ADE". A document is considered
            to contain an ADE if a least one symptom (class) is positive (1).
    Per class: Calculates precision, recall and F1 score for each class (symptom)
    Exact match accuracy: Calculates the percentage of exact matches across all
                          samples.
    (Full) per label : Calculates precision, recall and F1 score for each label
                       (0 and 1) across samples and classes.
    """
    gold_df = load_data(gold_csv)
    pred_df = load_data(pred_csv)

    print(
        "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nBinary Scores (ADE vs. no ADE):\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    )

    get_binary_scores(gold_df, pred_df)

    print(
        "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n(Individual) Per Class Scores:\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    )

    get_per_class_scores(gold_df, pred_df)

    print(
        "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n(Full) Per Label Scores:\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    )

    get_per_label_scores(gold_df, pred_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-gold_file", default=None, type=str)
    parser.add_argument("-prediction_file", default=None, type=str)

    args = parser.parse_args()

    main(gold_csv=args.gold_file, pred_csv=args.prediction_file)
