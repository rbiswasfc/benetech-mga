import argparse
import glob
import json
import os
from itertools import chain

import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold


# ---
def print_line():
    prefix, unit, suffix = "#", "--", "#"
    print(prefix + unit*50 + suffix)


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

    labels = []

    labels.append(
        {
            "id": chart_id,
            "source": chart_source,
            "chart_type": chart_type,
        }
    )

    return labels


def process_annotations(cfg, num_jobs=8):
    data_dir = cfg.competition_dataset.data_dir.rstrip("/")
    anno_paths = glob.glob(f"{data_dir}/train/annotations/*.json")
    annotations = Parallel(n_jobs=num_jobs, verbose=1)(delayed(_process_json)(file_path) for file_path in anno_paths)
    labels_df = pd.DataFrame(list(chain(*annotations)))
    return labels_df


def create_cv_folds(cfg):
    """Create Folds for the MGA task

    :param args: config file
    :type args: dict
    """
    print_line()
    print("creating folds ...")
    fold_df = process_annotations(cfg)
    fold_df = fold_df[["id", "source", "chart_type"]].copy()
    fold_df = fold_df.drop_duplicates()
    fold_df = fold_df.reset_index(drop=True)

    # ------
    skf = StratifiedKFold(
        n_splits=cfg.fold_metadata.n_folds,
        shuffle=True,
        random_state=cfg.fold_metadata.seed
    )

    for f, (t_, v_) in enumerate(skf.split(fold_df, fold_df["chart_type"].values)):
        fold_df.loc[v_, "kfold"] = f
    fold_df["kfold"] = fold_df["kfold"].astype(int)

    # allocate fold 99 to synthetic data
    fold_df["kfold"] = fold_df[["kfold", "source"]].apply(
        lambda x: x[0] if x[1] == "extracted" else 99, axis=1,
    )

    print(fold_df["kfold"].value_counts())

    # save fold split ---
    save_dir = cfg.fold_metadata.fold_dir
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"cv_map_{cfg.fold_metadata.n_folds}_folds.parquet")
    fold_df = fold_df[["id", "kfold"]].copy()
    fold_df = fold_df.reset_index(drop=True)
    fold_df.to_parquet(save_path)
    print("done!")
    print_line()
    # ---


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", type=str, required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config_path)
    create_cv_folds(cfg)
