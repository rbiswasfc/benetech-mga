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
from src.hbar_plot_advanced import HorizontalBarPlot
from src.hbar_xy_generation import generate_from_synthetic, generate_from_wiki
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
        weights=[0.50, 0.50],
        k=1,
    )[0]

    try:
        data = next(generator)
    except Exception as e:
        print(e)
        data = next(synthetic_generator)

    x_series = deepcopy(data['x_series'])
    y_series = deepcopy(data['y_series'])

    # process underlying data ---
    y_series = [str(y) for y in y_series]
    y_series = [y_val[:cfg.max_chars] for y_val in y_series]

   # max data points in a plot
    x_series = x_series[:cfg.max_points]
    y_series = y_series[:cfg.max_points]

    if (abs(min(x_series)) < 1e-3) & (abs(max(x_series)) < 1e-3):  # fixing constant value
        idx = random.randint(0, len(x_series)-1)
        x_series[idx] = random.uniform(0.01, 0.99)

    # ---
    data['x_series'] = list(x_series)
    data['y_series'] = list(y_series)
    return data


def generate_annotation(data):
    data = deepcopy(data)
    x_mga = data['x_series']
    y_mga = data['y_series']

    chart_type = 'horizontal_bar'
    x_type = 'numerical'
    y_type = 'categorical'

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
    texture_files = glob.glob(f"{args.texture_dir}/*.png")
    print(f"# texture files: {len(texture_files)}")

    p_bar = tqdm(range(cfg.num_images))
    for _ in range(cfg.num_images):
        base_image_id = f'syn_hbar_{generate_random_string()}'
        the_example = generate_plot_data(cfg, wiki_generator, synthetic_generator)

        # cast in the format of MGA
        mga_anno = generate_annotation(the_example)
        anno_path = os.path.join(cfg.output.annotation_dir, f"{base_image_id}.json")
        image_id = f"{base_image_id}"
        try:
            HorizontalBarPlot(cfg, the_example, texture_files=texture_files).make_horizontal_bar_plot(image_id)
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
