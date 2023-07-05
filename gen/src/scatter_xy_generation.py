import os
import random
import sys

import numpy as np

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
    from constants import YEAR_TITLES
    from function_generator import generate_y
    from generator_utils import detect_year
    from metadata_generator import generate_thematic_metadata
except ImportError:
    raise ImportError('Importing failed.')


# --- generate x series -------------------------------------------------------#


def generate_years(n_points, uniform=True, **kwargs):
    """generate a list of years
    """

    # start and end years ---
    start_year = random.randint(1900, 2020)
    end_year = random.randint(start_year + 20, 2200)
    repeat_val = random.random() >= 0.75

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
            if (repeat_val) & (random.random() >= 0.8):
                interval = 0  # repeat point
        next_year = ret[-1] + interval

        if next_year > end_year:
            break
        ret.append(next_year)

    return ret


def generate_integers(n_points, uniform=True, length_scale=25, **kwargs):
    """generate a list of integers
    """
    assert length_scale > 0

    p = random.random()
    if p <= 0.1:
        start = random.randint(0, 2)
    elif p <= 0.5:
        start = random.randint(0, length_scale*n_points)
    else:
        start = random.randint(-length_scale*n_points, length_scale*n_points)

    repeat_val = random.random() >= 0.75

    default_interval = random.randint(1, length_scale)

    # generate the years with a constant/variable interval
    points = [start]
    for _ in range(n_points - 1):
        prev = points[-1]
        if uniform:
            interval = default_interval
        else:
            interval = random.randint(1, length_scale)
            if (repeat_val) & (random.random() >= 0.8):
                interval = 0  # repeat point

        points.append(prev + interval)
    return points


def generate_floats(n_points, uniform=True, length_scale=25.0, **kwargs):
    """generate a list of floats    
    """
    assert length_scale > 0

    # number of decimal places
    n_d = random.choices([1, 2, 3], weights=[0.75, 0.20, 0.05], k=1)[0]

    p = random.random()
    if p <= 0.1:
        start = random.uniform(0, 2)
    elif p <= 0.5:
        start = random.uniform(0, length_scale*n_points)
    else:
        start = random.uniform(-length_scale*n_points, length_scale*n_points)

    repeat_val = random.random() >= 0.75
    default_interval = random.uniform(0.1, length_scale)

    points = [start]
    for _ in range(n_points - 1):
        prev = points[-1]
        if uniform:
            interval = default_interval
        else:
            interval = random.uniform(0, length_scale)
            if (repeat_val) & (random.random() >= 0.75):
                interval = random.uniform(0, 0.05)

        points.append(prev + interval)

    points = [round(point, n_d) for point in points]

    return points


def generate_exponentials(n_points, **kwargs):
    """generate a list of exponential numbers
    """
    max_val = 1_000_000
    base = random.choice([2, 3, 4, 5, 10])
    # exponents = np.arange(n_points)
    nums = [random.random() for i in range(n_points)]
    exponents = []
    sum = 0
    for num in nums:
        sum += num
        exponents.append(sum)

    points = []
    shift = random.uniform(0, 10)

    for e in exponents:
        p = shift + np.power(base, e)
        if p > max_val:
            break
        else:
            points.append(p)
    return points


def generate_logarithmics(n_points, uniform=True, length_scale=25.0, **kwargs):
    assert length_scale > 0, f'length scale must be positive: current = {length_scale}'

    base = random.choice([2, 3, 4, 5, 10])
    nums = [random.random() for i in range(n_points)]
    logs = []
    sum = 0
    for num in nums:
        sum += num
        logs.append(sum)

    points = []
    shift = random.uniform(0, 10)

    for log in logs:
        p = shift + np.log(log * length_scale) / np.log(base)
        if p > 0:
            points.append(p)
    return points


def generate_gaussian(n_points, **kwargs):
    mu = random.uniform(-10, 10)
    sigma = random.uniform(1, 25)
    return list(np.random.normal(mu, sigma, n_points))


def generate_beta(n_points, **kwargs):
    alpha = random.uniform(0.5, 5)
    beta = random.uniform(0.5, 5)
    return list(np.random.beta(alpha, beta, n_points))


def generate_poisson(n_points, **kwargs):
    lam = random.uniform(1, 50)
    return list(np.random.poisson(lam, n_points))


def generate_random(n_points, **kwargs):
    shift = random.uniform(-10, 10)
    x_vals = [shift + random.random() for _ in range(n_points)]
    return x_vals


def insert_outliers(data, multiple_std=3):

    num_outliers = random.randint(1, 5)

    # Convert the list to a numpy array
    data_arr = np.array(data)

    # Calculate the mean and standard deviation
    mean = np.mean(data_arr)
    std_dev = np.std(data_arr)

    # Generate the desired number of outliers
    outliers = []
    for _ in range(num_outliers):
        # Add or subtract a multiple of the standard deviation from the mean
        outlier = mean + (multiple_std * std_dev) * random.choice([-1, 1]) * (1. + random.random())
        outliers.append(outlier)

    # Randomly insert the outliers into the original data
    data = data_arr.tolist()
    data.extend(outliers)
    return data

# modifiers -------


def generate_noise(points, noise_type="gaussian", noise_scale=0.1):
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_scale, len(points))
    else:
        raise ValueError("Invalid noise type.")
    return list(np.array(points) + noise)

# main function -------


def _generate_x():
    """
    main function to generate x series
    """
    # control variables ---
    if random.random() <= 0.8:
        n_points = random.randint(8, 20)
    elif random.random() <= 0.9:
        n_points = random.randint(4, 8)
    else:
        n_points = random.randint(20, 64)

    uniformity = random.random() > 0.8

    range_category = random.choices(
        ["narrow", "normal", "wide"], weights=[0.1, 0.85, 0.05], k=1
    )[0]

    if range_category == "narrow":
        length_scale = random.uniform(0.01, 0.1)
    elif range_category == "wide":
        length_scale = random.uniform(1e3, 1e4)
    else:
        length_scale = random.uniform(1, 10)

    # generation function ---
    func_list = [
        generate_years,
        generate_integers,
        generate_floats,
        generate_exponentials,
        generate_logarithmics,
        generate_gaussian,
        generate_beta,
        generate_poisson,
        generate_random,
    ]

    func_weights = [
        1.0,
        2.0,
        5.0,
        0.25,
        0.25,
        2.0,
        0.5,
        0.5,
        1.0,
    ]

    total_weight = sum(func_weights)
    func_weights = [w/total_weight for w in func_weights]

    func = random.choices(func_list, weights=func_weights, k=1)[0]

    if func in [generate_years, generate_integers]:
        length_scale = max(1, int(length_scale))

    x_vals = func(n_points, uniform=uniformity, length_scale=length_scale)

    # modifiers ---
    if random.random() > 0.8:
        noise = generate_noise(x_vals, noise_type="gaussian", noise_scale=length_scale/10.0)
        x_vals += noise

    if random.random() > 0.99:
        x_vals = insert_outliers(x_vals, multiple_std=2.0)

    x_vals = sorted(x_vals)
    return x_vals


def generate_x(max_n=48):
    '''
    main api to generate a series of x values
    '''
    num_sources = 1  # random.choices([1, 2, 3], weights=[0.98, 0.01, 0.01], k=1)[0]

    x_vals = []
    for _ in range(num_sources):
        x_vals += _generate_x()
    x_vals = sorted(x_vals)[:max_n]
    return x_vals


# --- generate data -----------------------------------------------------------#
def generate_from_synthetic():
    while True:
        try:
            xs = generate_x()
            ys = generate_y(xs)
            break
        except Exception as e:
            print(e)
            continue

    metadata = generate_thematic_metadata()
    plot_title = metadata['title']

    x_title = metadata['xlabel']
    y_title = metadata['ylabel']
    if detect_year(xs):
        x_title = random.choice(YEAR_TITLES)

    # -----
    to_return = {
        'plot_title': plot_title,
        'x_title': x_title,
        'y_title': y_title,
        'x_series': xs,
        'y_series': ys,
    }
    # ---
    return to_return
