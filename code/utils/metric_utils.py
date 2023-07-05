from copy import deepcopy

import numpy as np
import pandas as pd
from rapidfuzz.distance.Levenshtein import distance as levenshtein


def compute_accuracy(df, true_col, pred_col):
    df = deepcopy(df)
    correct_predictions = df[df[true_col] == df[pred_col]]
    accuracy = len(correct_predictions) / len(df)
    return accuracy


def sigmoid(x):
    return 2 - 2 / (1 + np.exp(-x))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.subtract(y_true, y_pred))))


def normalized_rmse(y_true, y_pred):
    numerator = rmse(y_true, y_pred)
    denominator = rmse(y_true, np.mean(y_true))

    if denominator == 0:
        # force finite
        if numerator == 0:
            ret = 1.0
        else:
            ret = 0.0
    else:
        ret = sigmoid(numerator/denominator)
    return ret


def normalized_levenshtein_score(y_true, y_pred):
    total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def _compute_metric(truths, preds, d_type):
    if len(truths) != len(preds):
        return 0.

    if d_type == "categorical":
        # cast datatypes --
        truths = [str(t) for t in truths]
        preds = [str(p) for p in preds]

        return normalized_levenshtein_score(truths, preds)

    elif d_type == "numerical":
        truths = [float(t) for t in truths]
        preds = [float(p) for p in preds]
        return normalized_rmse(truths, preds)
    else:
        raise ValueError


def compute_metrics_counts(true_df, pred_df):
    """
    Evaluate predictions using Benetech - Making Graphs Accessible metric

    :param true_df: ground truth dataframe
    :type true_df: pd.DataFrame
    :param pred_df: _description_
    :type pred_df: pd.DataFrame
    :return: custom metric
    :rtype: float
    """
    true_df = deepcopy(true_df)
    pred_df = deepcopy(pred_df)

    gt_required_cols = ["id", "data_series", "chart_type"]

    for col in gt_required_cols:
        assert col in true_df.columns, f"{col} must be there in true_df"
    true_df = true_df[gt_required_cols].copy()

    true_df = true_df.rename(
        columns={
            "data_series": "true_data_series",
            "chart_type": "true_chart_type",
        }
    )

    true_df["true_data_series"] = true_df["true_data_series"].apply(lambda x: [elem for elem in x if elem == elem])
    true_df["true_count"] = true_df["true_data_series"].apply(lambda x: len(x))

    pred_df = pred_df[["id", "count", "chart_type"]].copy()
    pred_df = pred_df.reset_index(drop=True)

    pred_df = pred_df.rename(
        columns={
            "count": "pred_count",
            "chart_type": "pred_chart_type",
        }
    )

    df = pd.merge(true_df, pred_df, on="id", how="left")

    chart_type_accuracy = compute_accuracy(df, "true_chart_type", "pred_chart_type")
    count_accuracy = compute_accuracy(df, "true_count", "pred_count")

    return_dict = dict()
    return_dict["lb"] = count_accuracy
    return_dict['chart_type_accuracy'] = chart_type_accuracy
    return_dict["count_accuracy"] = count_accuracy

    return return_dict


def compute_metrics(true_df, pred_df):
    """
    Evaluate predictions using Benetech - Making Graphs Accessible metric

    :param true_df: ground truth dataframe
    :type true_df: pd.DataFrame
    :param pred_df: _description_
    :type pred_df: pd.DataFrame
    :return: custom metric
    :rtype: float
    """
    true_df = deepcopy(true_df)
    pred_df = deepcopy(pred_df)

    gt_required_cols = ["id", "source", "data_series", "chart_type", "data_type"]
    for col in gt_required_cols:
        assert col in true_df.columns, f"{col} must be there in true_df"
    true_df = true_df[gt_required_cols].copy()

    true_df = true_df.rename(
        columns={
            "data_series": "true_data_series",
            "chart_type": "true_chart_type",
        }
    )

    pred_df = pred_df[["id", "data_series", "chart_type"]].copy()
    pred_df = pred_df.reset_index(drop=True)

    pred_df = pred_df.rename(
        columns={
            "data_series": "pred_data_series",
            "chart_type": "pred_chart_type",
        }
    )

    df = pd.merge(true_df, pred_df, on="id", how="left")
    df["pred_data_series"] = df["pred_data_series"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    df["pred_chart_type"] = df["pred_chart_type"].apply(
        lambda x: x if isinstance(x, str) else "NotAChart"
    )

    df = df.reset_index(drop=True)
    mga_lb, scores = _get_score(df)

    return_dict = dict()
    return_dict["lb"] = mga_lb
    return_dict['scores'] = scores

    # chart-wise scores
    chart_options = [
        "horizontal_bar",
        "dot",
        "scatter",
        "line",
        "vertical_bar",
    ]

    for ct in chart_options:
        tmp_df = df[df["true_chart_type"] == ct].copy()
        tmp_df = tmp_df.reset_index(drop=True)
        s, _ = _get_score(tmp_df)
        return_dict[f"{ct}_score"] = s

    return return_dict


def _get_score(df):
    df = deepcopy(df)
    if len(df) == 0:
        return -1, []

    scores = []

    for _, row in df.iterrows():
        if row["pred_chart_type"] != row["true_chart_type"]:
            score = 0.0
        else:
            # check for nan in truths ---
            truths = [t for t in row["true_data_series"] if t == t]
            preds = row["pred_data_series"]

            try:
                score = _compute_metric(truths, preds, row["data_type"])
            except Exception as e:
                print(e)
                score = 0.

        scores.append(score)

    mga_lb = np.mean(scores)

    return mga_lb, scores
