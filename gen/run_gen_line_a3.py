import argparse
import glob
import json
import os
import random
import sys
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from src.generator_utils import generate_random_string
from src.line_plot_patch_v1 import BasicLinePlot
from src.line_xy_generation import generate_from_synthetic, generate_from_wiki
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


STOPWORDS = [
    "ISBN",
    "exit",
    "edit",
]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def generate_plot_data(cfg, wiki_generator, synthetic_generator):
    generator = random.choices(
        [wiki_generator, synthetic_generator],
        weights=[0.1, 0.9],
        k=1,
    )[0]

    try:
        data = next(generator)
    except Exception as e:
        data = next(synthetic_generator)

    x_series = list(deepcopy(data['x_series']))
    y_series = list(deepcopy(data['y_series']))

    # process underlying data ---
    x_series = [str(x) for x in x_series]
    x_series = [x_val[:cfg.max_chars] for x_val in x_series]

    # uniqueness
    seen = set()
    new_x_series = []
    new_y_series = []

    for idx, x_val in enumerate(x_series):
        if x_val not in seen:
            new_x_series.append(x_val)
            seen.add(x_val)
            new_y_series.append(y_series[idx])
    x_series = deepcopy(new_x_series)
    y_series = deepcopy(new_y_series)

    # x_series = list(set(x_series))
    # y_series = y_series[:len(x_series)]

    if len(x_series) < 2:
        x_series.append(generate_random_string())
        y_series.append(random.random()*100)

   # max data points in a plot
    x_series = x_series[:cfg.max_points]
    y_series = y_series[:cfg.max_points]

    # pre-post padding ---
    x_pre = []
    x_post = []

    if (len(x_series) >= 6) & (random.random() >= 0.2):
        # ---
        n_pre = 1
        n_post = 1

        if random.random() >= 0.8:
            n_post = 2

        p = random.random()
        if p <= 0.1:
            n_pre = 0
        elif p <= 0.2:
            n_post = 0

        n_mid = len(x_series) - n_pre - n_post

        # ---
        x_org = deepcopy(x_series)

        x_pre = deepcopy(x_org[:n_pre])

        x_series = deepcopy(x_org[n_pre:n_pre + n_mid])
        y_series = deepcopy(y_series[n_pre:n_pre + n_mid])

        x_post = deepcopy(x_org[n_pre+n_mid:])

    if random.random() >= 0.1:
        for idx in range(len(x_pre)):
            if random.random() >= 0.75:
                x_pre[idx] = ' '
        for idx in range(len(x_post)):
            if random.random() >= 0.75:
                x_post[idx] = ' '

    data['x_series'] = list(x_series)
    data['y_series'] = list(y_series)
    data['x_pre'] = list(x_pre)
    data['x_post'] = list(x_post)

    return data


def generate_annotation(data):
    data = deepcopy(data)
    x_mga = data['x_series']
    y_mga = data['y_series']

    chart_type = 'line'
    x_type = 'categorical'
    y_type = 'numerical'

    data_series = []
    for xi, yi in zip(x_mga, y_mga):
        data_series.append(
            {
                'x': xi,
                'y': yi,
            }
        )

    annotation = dict()
    annotation['chart-type'] = chart_type

    annotation['axes'] = dict()
    annotation['axes']['x-axis'] = dict()
    annotation['axes']['x-axis']['values-type'] = x_type

    annotation['axes']['y-axis'] = dict()
    annotation['axes']['y-axis']['values-type'] = y_type

    annotation['data-series'] = data_series
    return annotation


def main(args, cfg):
    with open(args.wiki_path, 'r') as f:
        wiki_bank = json.load(f)
    stem_df = pd.read_pickle(args.stem_path)
    stem_bank = dict(zip(stem_df["title"], stem_df["keywords"]))

    # process stem bank
    processed_stem_bank = dict()
    for key, values in stem_bank.items():
        key = key.replace("_", " ")
        values = [v for v in values if not v.startswith("[")]
        values = [v for v in values if not v in STOPWORDS]

        if len(values) >= 4:
            processed_stem_bank[key] = list(set(values))

    print(f"wiki bank size: {len(wiki_bank)}")
    print(f"stem bank size: {len(processed_stem_bank)}")

    wiki_generator = generate_from_wiki(wiki_bank)
    synthetic_generator = generate_from_synthetic(processed_stem_bank)

    # -- input/output ---
    os.makedirs(cfg.output.image_dir, exist_ok=True)
    os.makedirs(cfg.output.annotation_dir, exist_ok=True)
    # texture_files = glob.glob(f"{args.texture_dir}/*/*.jpg")
    texture_files = glob.glob(f"{args.texture_dir}/*/*.png")
    print(f"# texture files: {len(texture_files)}")

    p_bar = tqdm(range(cfg.num_images))
    for image_num in range(cfg.num_images):
        base_image_id = f'syn_line_{generate_random_string()}'
        the_example = generate_plot_data(cfg, wiki_generator, synthetic_generator)

        # cast in the format of MGA
        mga_anno = generate_annotation(the_example)
        anno_path = os.path.join(cfg.output.annotation_dir, f"{base_image_id}.json")
        image_id = f"{base_image_id}"
        try:
            BasicLinePlot(cfg, the_example, texture_files=texture_files).make_line_plot(image_id)
            with open(anno_path, "w") as f:
                json.dump(mga_anno, f, cls=NpEncoder)

        except Exception as e:
            print(e)
            print("--"*40)
            print(the_example)
            print("--"*40)
        p_bar.update()
    p_bar.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--wiki_path', type=str, required=True)
    ap.add_argument('--stem_path', type=str, required=True)
    ap.add_argument('--conf_path', type=str, required=True)
    ap.add_argument('--texture_dir', type=str, required=True)

    args = ap.parse_args()
    cfg = OmegaConf.load(args.conf_path)

    main(args, cfg)
