import argparse
import glob
import json
from itertools import chain
from operator import itemgetter

import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf


# --- Output utils ----------------------------------------------------------------------#
def num2str(num):
    try:
        num = "{:.2e}".format(num)
    except Exception as e:
        pass
    return num


def _build_output(fp):
    """process JSON files to prepare output

    :param fp: file path
    :type fp: str
    :return: parsed annotation
    :rtype: dict
    """

    # read annotations ---
    with open(fp, "r") as f:
        anno = json.load(f)

    # store necessary data for labels ---
    chart_id = fp.split("/")[-1].split(".")[0]
    chart_type = anno['chart-type']
    chart_data = anno['data-series']
    x_dtype = anno['axes']['x-axis']['values-type']
    y_dtype = anno['axes']['y-axis']['values-type']

    x_series = [d['x'] for d in chart_data]
    y_series = [d['y'] for d in chart_data]

    output_elems = []
    header = f"Row 0: {chart_type}"
    output_elems.append(header)

    for idx, (x, y) in enumerate(zip(x_series, y_series)):
        row = f"Row {idx + 1}: {num2str(x)} | {num2str(y)}"
        output_elems.append(row)

    ret = {
        "id": chart_id,
        "output": "\n".join(output_elems),
    }

    return ret


def build_outputs(cfg, num_jobs=8):
    data_dir = cfg.competition_dataset.data_dir.rstrip("/")
    anno_paths = glob.glob(f"{data_dir}/train/annotations/*.json")
    annotations = Parallel(n_jobs=num_jobs, verbose=1)(
        delayed(_build_output)(file_path) for file_path in anno_paths)
    output_df = pd.DataFrame(annotations)
    return output_df


# --- Label utils -----------------------------------------------------------------------#

def _process_json(fp):
    """process JSON files with annotations

    :param fp: file path
    :type fp: str
    :return: parsed annotation
    :rtype: dict
    """

    # read annotations ---
    with open(fp, "r") as f:
        anno = json.load(f)

    # store necessary data for labels ---
    chart_id = fp.split("/")[-1].split(".")[0]
    chart_source = anno["source"]
    chart_type = anno['chart-type']
    chart_data = anno['data-series']

    if chart_type == "scatter":
        chart_data = sorted(chart_data, key=itemgetter('x', 'y'))

    x_dtype = anno['axes']['x-axis']['values-type']
    y_dtype = anno['axes']['y-axis']['values-type']

    x_series = [d['x'] for d in chart_data]
    y_series = [d['y'] for d in chart_data]

    x_id = f"{chart_id}_x"
    y_id = f"{chart_id}_y"

    # store labels ---
    # x label
    labels = []

    labels.append(
        {
            "id": x_id,
            "source": chart_source,
            "data_series": x_series,
            "chart_type": chart_type,
            "data_type": x_dtype,
        }
    )

    # y label
    labels.append(
        {
            "id": y_id,
            "source": chart_source,
            "data_series": y_series,
            "chart_type": chart_type,
            "data_type": y_dtype,
        }
    )

    return labels


def process_annotations(cfg, num_jobs=8):
    data_dir = cfg.competition_dataset.data_dir.rstrip("/")
    anno_paths = glob.glob(f"{data_dir}/train/annotations/*.json")
    annotations = Parallel(n_jobs=num_jobs, verbose=1)(
        delayed(_process_json)(file_path) for file_path in anno_paths)
    labels_df = pd.DataFrame(list(chain(*annotations)))
    return labels_df

# def process_annotations(cfg, num_jobs=8):
#     anno_dir = cfg.competition_dataset.train.annotation_dir.rstrip("/")
#     anno_paths = glob.glob(f"{anno_dir}/*.json")
#     annotations = Parallel(n_jobs=num_jobs, verbose=1)(
#         delayed(_process_json)(file_path) for file_path in anno_paths)
#     labels_df = pd.DataFrame(list(chain(*annotations)))
#     return labels_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", type=str, required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config_path)

    labels_df = process_annotations(cfg)
