
import json
import os
import random
import sys
from copy import deepcopy

import numpy as np
import pandas as pd

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
    from constants import YEAR_TITLES
    from function_generator import generate_y
    from generator_utils import has_non_latin_chars
    from metadata_generator import generate_thematic_metadata
except ImportError:
    raise ImportError('Importing failed.')


def detect_year(input_series):
    input_series = deepcopy(input_series)
    input_series = [str(x) for x in input_series]

    try:
        datetime_series = np.array(input_series, dtype='datetime64[Y]')
    except ValueError:
        return False

    if np.any(np.isnat(datetime_series)):
        return False
    if np.any(datetime_series < np.datetime64('1800')):
        return False
    if np.any(datetime_series >= np.datetime64('2100')):
        return False
    return True


def is_numeric(input_list):
    try:
        for elem in input_list:
            float(elem)
        return True
    except ValueError:
        return False


def process_x_series(x_series, x_title):
    x_series = deepcopy(x_series)
    if detect_year(x_series):
        x_title = random.choice(YEAR_TITLES)
        x_series = [str(x) for x in x_series]

    if is_numeric(x_series):
        x_series = [str(round(x, 2)) for x in x_series]
    return x_series, x_title


# -- wiki generator ---
def update_wiki_series(
    x_series,
    y_series,
    max_char_allowed=256,
    max_char_per_element=64,
):
    # copy the series ---
    x_series = deepcopy(x_series)
    y_series = deepcopy(y_series)

    x_series = [str(x) for x in x_series]
    # ignore total 50% of the times ---
    if 'total' in x_series[-1].lower():
        if random.random() >= 0.25:
            x_series = x_series[:-1]
            y_series = y_series[:-1]

    # tracker ---
    char_count = 0
    for idx, x_val in enumerate(x_series):
        char_count += min(len(x_val), max_char_per_element)
        if char_count >= max_char_allowed:
            break

    x_series = x_series[:idx + 1]
    y_series = y_series[:idx + 1]

    # normalization ---
    threshold = 5e5
    y_series = np.array(y_series)
    max_val = np.absolute(y_series).max()

    if max_val > threshold:
        y_series = y_series * (random.uniform(threshold*0.5, threshold*1.5) / max_val)
    if random.random() >= 0.10:
        y_series = np.absolute(y_series)
    y_series = list(y_series)

    return x_series, y_series


def generate_from_wiki(wiki_bank):
    example = random.choice(wiki_bank)

    for example in wiki_bank:

        cat_series_list = []
        num_series_list = []
        plot_title = example[0]['plot-title']

        for field in example:
            if field['data-type'] == 'categorical':
                # check if its unique ---
                if len(set(field['data-series'])) == len(field['data-series']):
                    if not has_non_latin_chars(field['data-series']):  # FIXME: this checks for non-ascii instead
                        cat_series_list.append(field)
            else:
                num_series_list.append(field)

        if (len(cat_series_list) > 0) & (len(num_series_list) > 0):

            xs = random.choice(cat_series_list)
            ys = random.choice(num_series_list)

            # prune series ---
            x_series = xs['data-series']
            y_series = ys['data-series']
            x_series, y_series = update_wiki_series(x_series, y_series)

            anno = {
                'plot_title': plot_title,

                'x_title': xs['series-name'],
                'y_title': ys['series-name'],

                'x_series': x_series,
                'y_series': y_series,
            }

            yield anno


# --- generate x series -------------------------------------------------------#


def generate_years(n_points, uniform=True, **kwargs):
    """generate a list of years
    """

    # start and end years ---
    start_year = random.randint(1900, 2020)
    end_year = random.randint(start_year + 20, 2100)

    default_interval = random.choices(
        [1, 2, 3, 4, 5, 10],
        weights=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
        k=1,
    )[0]

    # generate the years with a constant/variable interval
    ret = [start_year]

    for _ in range(n_points - 1):
        if uniform:
            interval = default_interval
        else:
            interval = random.randint(1, 10)
        next_year = ret[-1] + interval

        if next_year > end_year:
            break
        ret.append(next_year)

    return ret


def generate_integers(n_points, uniform=True, length_scale=25, **kwargs):
    """generate a list of integers
    """
    if length_scale <= 1:
        print(f'Updating length scale from {length_scale} to 2')
        length_scale = 2

    start = random.randint(-length_scale, length_scale)
    default_interval = random.randint(1, length_scale)

    # generate the years with a constant/variable interval
    points = [start]
    for _ in range(n_points - 1):
        prev = points[-1]
        if uniform:
            interval = default_interval
        else:
            interval = random.randint(1, length_scale)
        points.append(prev + interval)
    return points


def generate_integer_for_shared(n_points, **kwargs):
    """generate a list of integers
    """
    length_scale = random.randint(1, 10)

    start = 0  # random.choice([0, length_scale])
    interval = random.randint(1, length_scale)

    # generate the years with a constant/variable interval
    points = [start]
    for _ in range(n_points - 1):
        prev = points[-1]
        points.append(prev + interval)
    return points


def generate_floats(n_points, uniform=True, length_scale=25.0, **kwargs):
    """generate a list of floats    
    """
    assert length_scale > 0
    # restrict to 24 points for floats

    # number of decimal places
    n_d = random.choices([1, 2], weights=[0.9, 0.1], k=1)[0]

    if n_d == 2:
        n_points = min(n_points, 12)

    start = random.uniform(-length_scale, length_scale)
    default_interval = random.uniform(0.1, length_scale)

    points = [start]
    for _ in range(n_points - 1):
        prev = points[-1]
        if uniform:
            interval = default_interval
        else:
            interval = random.uniform(0.1, length_scale)
        points.append(prev + interval)

    points = [round(point, n_d) for point in points]

    return points


def sample_cat_list(category_dict, n_points):
    """generate a list of random categories
    """
    key = random.choice(list(category_dict.keys()))
    kv = list(category_dict[key])
    n = min(n_points, len(kv))
    x_cats = random.sample(kv, n)
    x_title = key
    return x_title, x_cats


def generate_x(category_dict):
    """
    main function to generate x series
    """
    p = random.random()
    n_points = random.randint(4, 36)
    uniformity = random.random() >= 0.25

    if p >= 0.75:
        func_list = [generate_years, generate_integers, generate_floats]
        func = random.choices(func_list, weights=[0.30, 0.50, 0.20], k=1)[0]
        length_scale = random.randint(10, 100)
        x_title, x_values = "", func(n_points, uniform=uniformity, length_scale=length_scale)
    else:
        x_title, x_values = sample_cat_list(category_dict, n_points)
    return x_title, x_values


# --- generate data -----------------------------------------------------------#
def update_syn_series(
    x_series,
    y_series,
    max_char_allowed=256,
    max_char_per_element=64,
):
    # copy the series ---
    x_series = deepcopy(x_series)
    y_series = deepcopy(y_series)

    x_series = [str(x) for x in x_series]

    # tracker ---
    char_count = 0
    for idx, x_val in enumerate(x_series):
        char_count += min(len(x_val), max_char_per_element)
        if char_count >= max_char_allowed:
            break

    x_series = x_series[:idx + 1]
    y_series = y_series[:idx + 1]
    return x_series, y_series


def generate_xy(category_dict):
    x_title, x_values = generate_x(category_dict)
    y_values = generate_y(x_values)

    metadata = generate_thematic_metadata()
    plot_title = metadata['title']

    # cast x axis to string
    x_values = [str(x) for x in x_values]
    if (len(x_title) <= 1) & (not x_title.startswith('page')):
        x_title = metadata['xlabel']

    if detect_year(x_values):
        x_title = random.choice(YEAR_TITLES)

    y_title = metadata['ylabel']

    x_values, y_values = update_syn_series(x_values, y_values)

    # -----
    to_return = {
        'plot_title': plot_title,

        'x_title': x_title,
        'y_title': y_title,

        'x_series': x_values,
        'y_series': y_values,
    }
    # ---
    return to_return


def generate_from_synthetic(category_dict):
    while True:
        yield generate_xy(category_dict)

# --------


def generate_shared():
    n_points = random.randint(6, 12)
    x_values = generate_integer_for_shared(n_points)
    y_values = generate_y(x_values)
    x_values = [str(x) for x in x_values]

    if random.random() >= 0.1:
        y_values = list(np.abs(np.array(y_values)))

    metadata = generate_thematic_metadata()
    plot_title = metadata['title']
    x_title = metadata['xlabel']
    y_title = metadata['ylabel']

    # -----
    to_return = {
        'plot_title': plot_title,

        'x_title': x_title,
        'y_title': y_title,

        'x_series': x_values,
        'y_series': y_values,
    }
    # ---
    return to_return


def generate_from_shared():
    while True:
        yield generate_shared()
