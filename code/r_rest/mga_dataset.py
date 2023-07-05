import json
import os
import pdb
from copy import deepcopy
from operator import itemgetter

import albumentations as A
import numpy as np
import torch
from PIL import Image
from tokenizers import AddedToken
from torch.utils.data import Dataset
from transformers import Pix2StructProcessor

# -- token map --#
TOKEN_MAP = {
    "line": "[<lines>]",
    "vertical_bar": "[<vertical_bar>]",
    "scatter": "[<scatter>]",
    "dot": "[<dot>]",
    "horizontal_bar": "[<horizontal_bar>]",
    "histogram": "[<histogram>]",

    "c_start": "[<c_start>]",
    "c_end": "[<c_end>]",
    "x_start": "[<x_start>]",
    "x_end": "[<x_end>]",
    "y_start": "[<y_start>]",
    "y_end": "[<y_end>]",

    "p_start": "[<p_start>]",
    "p_end": "[<p_end>]",

    "bos_token": "[<mga>]",
}

# -----


def fix_data_series(graph_id, x_series, y_series):
    if graph_id in [
        'a80688cb2101',
        'c48de2fcb4a4',
        '4566b5627dfc',
        '0df1338d2df7',
        'icdar22_line_PMC5456760___materials-09-00461-g005b',
        'icdar22_line_PMC5445839___materials-03-02447-g004',
        'icdar22_line_PMC2803785___1471-2350-10-142-1',
    ]:
        x_series = x_series[1:]
        y_series = y_series[1:]

    if graph_id in [
        'icdar22_line_PMC5666930___materials-10-01124-g015',
        'icdar22_line_PMC5344571___materials-10-00038-g009',
    ]:
        x_series = x_series[:-1]
        y_series = y_series[:-1]

    if graph_id in [
        'icdar22_line_PMC3201928___1471-2458-11-782-1',
    ]:
        x_series = ['1'] + x_series
        y_series = [0.87] + y_series

    if graph_id in [
        'icdar22_line_PMC6293241___3',
    ]:
        x_series = ['0'] + x_series
        y_series = [0.27] + y_series

    if graph_id in [
        'icdar22_line_PMC5448636___materials-04-01034-g005',
        'icdar22_line_PMC3673829___1471-2458-13-507-3',
        'icdar22_line_PMC5632326___IJPH-46-1237-g001',
        'icdar22_line_PMC5741214___g003',
        'icdar22_line_PMC4542101___12889_2015_2121_Fig12_HTML',
    ]:
        y_series = [y_val*100 for y_val in y_series]

    return x_series, y_series

# ref: https://www.kaggle.com/code/nbroad/donut-train-benetech


def is_nan(val):
    return val != val


def get_processor(cfg):
    """
    load the processor
    """
    processor_path = cfg.model.backbone_path
    print(f"loading processor from {processor_path}")
    processor = Pix2StructProcessor.from_pretrained(processor_path)
    processor.image_processor.is_vqa = False
    processor.image_processor.patch_size = {
        "height": cfg.model.patch_size,
        "width": cfg.model.patch_size
    }

    # NEW TOKENS
    print("adding new tokens...")
    new_tokens = []
    for _, this_tok in TOKEN_MAP.items():
        new_tokens.append(this_tok)
    new_tokens = sorted(new_tokens)

    tokens_to_add = []
    for this_tok in new_tokens:
        tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))

    processor.tokenizer.add_tokens(tokens_to_add)

    return processor


class MGADataset(Dataset):
    """Dataset class for MGA dataset
    """

    def __init__(self, cfg, graph_ids, transform=None):

        self.cfg = cfg
        self.data_dir = cfg.competition_dataset.data_dir.rstrip("/")
        self.image_dir = os.path.join(self.data_dir, "train", "images")
        self.annotation_dir = os.path.join(self.data_dir, "train", "annotations")

        self.syn_data_dir = cfg.competition_dataset.syn_dir.rstrip("/")
        self.syn_image_dir = os.path.join(self.syn_data_dir, "images")
        self.syn_annotation_dir = os.path.join(self.syn_data_dir, "annotations")

        self.pl_data_dir = cfg.competition_dataset.pl_dir.rstrip("/")
        self.pl_image_dir = os.path.join(self.pl_data_dir, "images")
        self.pl_annotation_dir = os.path.join(self.pl_data_dir, "annotations")

        self.icdar_data_dir = cfg.competition_dataset.icdar_dir.rstrip("/")
        self.icdar_image_dir = os.path.join(self.icdar_data_dir, "images")
        self.icdar_annotation_dir = os.path.join(self.icdar_data_dir, "annotations")

        self.graph_ids = deepcopy(graph_ids)
        self.transform = transform

        # load processor
        self.load_processor()

    def load_processor(self):
        self.processor = get_processor(self.cfg)

    def load_image(self, graph_id):
        if ("syn_" in graph_id) | ("ext_" in graph_id):
            try:
                image_path = os.path.join(self.syn_image_dir, f"{graph_id}.jpg")
                image = Image.open(image_path)  # .convert('RGBA')
                image = image.convert('RGB')

            except Exception as e:
                image_path = os.path.join(self.syn_image_dir, f"{graph_id}_v0.jpg")
                image = Image.open(image_path)  # .convert('RGBA')
                image = image.convert('RGB')

        elif "pl_" in graph_id:
            image_path = os.path.join(self.pl_image_dir, f"{graph_id}.jpg")
            image = Image.open(image_path)  # .convert('RGBA')
            image = image.convert('RGB')

        elif "icdar22_" in graph_id:
            image_path = os.path.join(self.icdar_image_dir, f"{graph_id}.jpg")
            image = Image.open(image_path)  # .convert('RGBA')
            image = image.convert('RGB')

        else:
            image_path = os.path.join(self.image_dir, f"{graph_id}.jpg")
            image = Image.open(image_path)
            image = image.convert('RGB')
        return image

    def process_point(self, val, d_type, chart_type):
        """
        process the x/y point in a suitable format
        """
        # handling of numerical points ---
        if d_type == "numerical":
            if chart_type != "dot":
                val = "{:.2e}".format(float(val))
            else:
                val = str(int(float(val)))  # only counts for dot charts

        elif d_type == "categorical":
            val = val  # val.strip()

        else:
            raise TypeError

        return val

    def build_output(self, graph_id):
        if ("syn_" in graph_id) | ("ext_" in graph_id):
            annotation_path = os.path.join(self.syn_annotation_dir, f"{graph_id}.json")
        elif "pl_" in graph_id:
            annotation_path = os.path.join(self.pl_annotation_dir, f"{graph_id}.json")
        elif "icdar22_" in graph_id:
            annotation_path = os.path.join(self.icdar_annotation_dir, f"{graph_id}.json")
        else:
            annotation_path = os.path.join(self.annotation_dir, f"{graph_id}.json")

        # read annotations ---
        with open(annotation_path, "r") as f:
            annotation = json.load(f)

        # store necessary data for labels ---
        chart_type = annotation['chart-type']
        chart_data = annotation['data-series']

        # sort chart data in case of scatter plot ---
        if chart_type == "scatter":
            chart_data = sorted(chart_data, key=itemgetter('x', 'y'))

        x_dtype = annotation['axes']['x-axis']['values-type']
        y_dtype = annotation['axes']['y-axis']['values-type']

        x_series = [d['x'] for d in chart_data if not is_nan(d['x'])]
        y_series = [d['y'] for d in chart_data if not is_nan(d['y'])]

        # detect histogram --
        if len(x_series) != len(y_series):
            chart_type = "histogram"

        x_series, y_series = fix_data_series(graph_id, x_series, y_series)

        num_x = len(x_series)
        num_y = len(y_series)

        # x_max = None
        # if x_dtype == "numerical":
        #     x_max = max([abs(float(x)) for x in x_series])

        # y_max = None
        # if y_dtype == "numerical":
        #     y_max = max([abs(float(y)) for y in y_series])

        x_series = [self.process_point(x, x_dtype, chart_type) for x in x_series]
        y_series = [self.process_point(y, y_dtype, chart_type) for y in y_series]

        c_string = TOKEN_MAP["c_start"] + TOKEN_MAP[chart_type] + TOKEN_MAP["c_end"]
        p_string = TOKEN_MAP["p_start"] + f"{num_x}|{num_y}" + TOKEN_MAP["p_end"]
        x_string = TOKEN_MAP["x_start"] + "|".join(x_series) + TOKEN_MAP["x_end"]
        y_string = TOKEN_MAP["y_start"] + "|".join(y_series) + TOKEN_MAP["y_end"]
        e_string = self.processor.tokenizer.eos_token

        text = f"{c_string}{p_string}{x_string}{y_string}{e_string}"
        # text = f"{c_string}{x_string}{y_string}{e_string}"

        return text, chart_type

    def __str__(self):
        string = 'MGA Dataset'
        string += f'\tlen = {len(self)}\n'
        return string

    def __len__(self):
        return len(self.graph_ids)

    def __getitem__(self, index):
        graph_id = self.graph_ids[index]
        image = self.load_image(graph_id)

        if self.transform is not None:
            image = np.array(image)
            image = self.transform(image=image)["image"]

        try:
            text, chart_type = self.build_output(graph_id)
        except Exception as e:
            print(f"Error in {graph_id}")
            print(e)  # e
            text, chart_type = 'error', 'error_chart'

        # image processor ---
        p_img = self.processor(
            images=image,
            max_patches=self.cfg.model.max_patches,
            add_special_tokens=True,
        )

        # process text
        p_txt = self.processor(
            text=text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.cfg.model.max_length,
        )

        r = {}
        r['id'] = graph_id
        r['chart_type'] = chart_type
        r['image'] = image
        r['text'] = text
        r['flattened_patches'] = p_img['flattened_patches']
        r['attention_mask'] = p_img['attention_mask']

        try:
            r['decoder_input_ids'] = p_txt['decoder_input_ids']
        except KeyError:
            r['decoder_input_ids'] = p_txt['input_ids']

        try:
            r['decoder_attention_mask'] = p_txt['decoder_attention_mask']
        except KeyError:
            r['decoder_attention_mask'] = p_txt['attention_mask']

        return r


# # ---- transforms ----#

def create_train_transforms():
    """
    Returns transformations.

    Returns:
        albumentations transforms: transforms.
    """

    transforms = A.Compose(
        [
            A.OneOf(
                [
                    A.RandomToneCurve(scale=0.3),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.2),
                        contrast_limit=(-0.4, 0.5),
                        brightness_by_max=True,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=(-20, 20),
                        sat_shift_limit=(-30, 30),
                        val_shift_limit=(-20, 20)
                    )
                ],
                p=0.5,
            ),

            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.GaussNoise(var_limit=(3.0, 9.0)),
                ],
                p=0.5,
            ),

            A.Downscale(always_apply=False, p=0.1, scale_min=0.90, scale_max=0.99),
        ],

        p=0.5,
    )
    return transforms
