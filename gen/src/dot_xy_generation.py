
import os
import random
import sys
from copy import deepcopy

import numpy as np

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
    from constants import YEAR_TITLES
    from generator_utils import detect_year, is_numeric
    from metadata_generator import generate_thematic_metadata
except ImportError:
    raise ImportError('Importing failed.')


def process_x_series(x_series, x_title):
    x_series = deepcopy(x_series)
    if detect_year(x_series):
        x_title = random.choice(YEAR_TITLES)
        x_series = [str(x) for x in x_series]

    if is_numeric(x_series):
        x_series = [str(round(x, 2)) for x in x_series]
    return x_series, x_title


# --- generate x series -------------------------------------------------------#
def generate_years(n_points):
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
        interval = default_interval
        next_year = ret[-1] + interval
        if next_year > end_year:
            break
        ret.append(next_year)
    return ret


def generate_integers(n_points):
    """generate a list of integers
    """
    length_scale = random.randint(1, 1000)

    if random.random() >= 0.25:
        length_scale = random.randint(1, 5)
    elif random.random() >= 0.1:
        length_scale = random.randint(1, 10)
    elif random.random() >= 0.05:
        length_scale = random.randint(1, 100)
    else:
        length_scale = random.randint(1, 1000)

    if random.random() >= 0.9:
        start = random.randint(-n_points*length_scale, n_points*length_scale)
    else:
        start = random.randint(0, n_points*length_scale)

    interval = random.randint(1, length_scale+1)

    # generate the years with a constant/variable interval
    points = [start]
    for _ in range(n_points - 1):
        prev = points[-1]
        points.append(prev + interval)
    return points


def generate_floats(n_points):
    """generate a list of floats    
    """
    length_scale = random.choice(
        [0.1, 0.2, 0.4, 0.5, 0.8]
    )

    n_d = 1

    if random.random() >= 0.5:
        start = random.randint(0, 100)
    else:
        start = random.randint(0, 10)

    interval = random.choice(
        [0.1, 0.2, 0.4, 0.5, 0.8]
    )

    points = [start]
    for _ in range(n_points - 1):
        prev = points[-1]
        points.append(prev + interval)
    points = [round(point, n_d) for point in points]
    return points


def sample_cat_list(category_dict, n_points):
    """
    generate a list of random categories
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
    n_points = random.randint(4, 24)

    if p >= 0.5:
        func_list = [generate_years, generate_integers, generate_floats]
        func = random.choices(func_list, weights=[0.15, 0.80, 0.05], k=1)[0]
        x_title, x_values = "", func(n_points)
    else:
        x_title, x_values = sample_cat_list(category_dict, n_points)
    return x_title, x_values


def count_generator(x):
    p = random.random()

    if p <= 0.02:
        max_val = 25
    elif p <= 0.05:
        max_val = 16
    elif p <= 0.5:
        max_val = 12
    else:
        max_val = 8

    A = random.randint(8, max_val)

    min_val = 0
    y_values = [random.randint(min_val, A) for _ in range(len(x))]

    if random.random() >= 0.95:
        y_values = sorted(y_values, reverse=random.random() >= 0.5)

    if max(y_values) <= 0:
        idx = random.randint(0, len(y_values)-1)
        y_values[idx] = random.randint(1, 10)

    count_sum = np.sum(y_values)
    if count_sum <= 4:
        for idx in range(len(y_values)):
            y_values[idx] += random.randint(0, 2)

    return y_values

# --- generate data -----------------------------------------------------------#


def update_syn_series(
    x_series,
    y_series,
    max_char_allowed=64,
    max_char_per_element=16,
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

    #
    return x_series, y_series


def generate_xy(category_dict):
    x_title, x_values = generate_x(category_dict)
    y_values = count_generator(x_values)

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

    y_values = [int(y) for y in y_values]

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
