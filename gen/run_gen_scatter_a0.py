import argparse
import glob
import json
import os
import warnings
from copy import deepcopy
from operator import itemgetter

import numpy as np
from omegaconf import OmegaConf
from src.generator_utils import generate_random_string
from src.scatter_plot_advanced import BasicScatterPlot
from src.scatter_xy_generation import generate_from_synthetic
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def generate_plot_data(cfg):

    data = generate_from_synthetic()

    x_series = list(deepcopy(data['x_series']))
    y_series = list(deepcopy(data['y_series']))

    # process underlying data ---
    data_series = [{'x': x_val, 'y': y_val} for x_val, y_val in zip(x_series, y_series)]
    data_series = sorted(data_series, key=itemgetter('x', 'y'))

    x_series = [d['x'] for d in data_series]
    y_series = [d['y'] for d in data_series]

   # max data points in a plot ---
    x_series = x_series[:cfg.max_points]
    y_series = y_series[:cfg.max_points]

    data['x_series'] = list(x_series)
    data['y_series'] = list(y_series)
    return data


def generate_annotation(data):
    data = deepcopy(data)
    x_mga = data['x_series']
    y_mga = data['y_series']

    chart_type = 'scatter'
    x_type = 'numerical'
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
    # -- input/output ---
    os.makedirs(cfg.output.image_dir, exist_ok=True)
    os.makedirs(cfg.output.annotation_dir, exist_ok=True)
    texture_files = glob.glob(f"{args.texture_dir}/*.png")
    print(f"# texture files: {len(texture_files)}")

    p_bar = tqdm(range(cfg.num_images))
    for _ in range(cfg.num_images):
        base_image_id = f'syn_scatter_{generate_random_string()}'
        the_example = generate_plot_data(cfg)

        # cast in the format of MGA
        mga_anno = generate_annotation(the_example)
        anno_path = os.path.join(cfg.output.annotation_dir, f"{base_image_id}.json")
        image_id = f"{base_image_id}"
        try:
            BasicScatterPlot(cfg, the_example, texture_files=texture_files).make_scatter_plot(image_id)
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
    ap.add_argument('--conf_path', type=str, required=True)
    ap.add_argument('--texture_dir', type=str, required=True)

    args = ap.parse_args()
    cfg = OmegaConf.load(args.conf_path)

    main(args, cfg)
