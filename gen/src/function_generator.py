import random

import numpy as np
from scipy.special import factorial, gamma, zeta


# utils ---------------------------------------------------------------------------------#
def get_random_params():
    prob = random.random()
    if prob <= 0.2:
        a, r = 1.0, 0.1
    elif prob <= 0.4:
        a, r = 0.1, 1.0
    else:
        a, r = 1.0, 1.0

    y_start = np.random.uniform(-a, a)
    y_range = np.random.uniform(0.01, r)

    prob = random.random()
    if prob <= 0.05:
        m = 1e-4
    elif prob <= 0.10:
        m = 1e-2
    elif prob <= 0.30:
        m = 1e+0
    elif prob <= 0.50:
        m = 1e+1
    elif prob <= 0.80:
        m = 1e+2
    elif prob <= 0.95:
        m = 1e+4
    else:
        m = 1e+5

    y_start *= m
    y_range *= m

    return y_start, y_range


def vector_scaling_at_random(y):
    y_start, y_range = get_random_params()
    y_min = np.min(y)
    y_max = np.max(y)
    return y_start + (y - y_min) / (y_max - y_min) * y_range


# functions ------------------------------------------------------------------------------#
def percentages(x):
    n_points = len(x)
    ret = [random.randint(0, 100) for _ in range(n_points)]
    return ret


def random_fractions(x):
    n_points = len(x)
    shift = random.uniform(-10, 10)
    ret = [shift + random.random() for _ in range(n_points)]
    return ret


# generator ------------------------------------------------------------------------------#

def fn_generation(x):

    length = len(x)
    length_scale = random.uniform(1, 10)

    x = np.linspace(random.uniform(-length_scale, 0), random.uniform(0, length_scale), length)

    # Define the relationship between x and y
    relationships = [
        lambda x: x**2 * random.uniform(-10, 10),  # quadratic
        lambda x: x**3 * random.uniform(-10, 10),  # cubic
        lambda x: np.sin(x * random.uniform(1, 5)),  # sinusoidal
        lambda x: np.cos(x * random.uniform(1, 5)),  # cosinusoidal
        lambda x: np.exp(x / random.uniform(5, 15)),  # exponential
        lambda x: np.log(np.abs(x) + 1),  # logarithmic
        lambda x: np.sqrt(np.abs(x)),  # square root
        lambda x: np.abs(x),  # absolute value
        lambda x: np.tanh(x),  # hyperbolic tangent
        lambda x: 1 / (x + 1),  # reciprocal
        lambda x: np.sinh(x / random.uniform(5, 15)),  # hyperbolic sine
        lambda x: np.cosh(x / random.uniform(5, 15)),  # hyperbolic cosine
        lambda x: np.sinc(x / random.uniform(5, 15)),  # sinc function
        lambda x: np.sign(x) * np.sqrt(np.abs(x)),  # signed square root
        lambda x: x * np.sin(x),  # x*sin(x)
        lambda x: x * np.cos(x),  # x*cos(x)
        lambda x: np.arcsin(np.clip(x / 10, -1, 1)),  # arcsine
        lambda x: np.arccos(np.clip(x / 10, -1, 1)),  # arccosine
        lambda x: np.heaviside(x, 0.5),  # heaviside step function
        lambda x: np.piecewise(x, [x < 0, x >= 0], [lambda x: x**3, lambda x: x**2]),  # piecewise function
        lambda x: np.polyval([random.uniform(-1, 1) for _ in range(5)], x),  # 4th degree polynomial with random coefficients
        lambda x: np.exp(-x**2 / (2 * random.uniform(1, 5)**2)),  # Gaussian
        lambda x: np.floor(x),  # floor function
        lambda x: np.ceil(x),  # ceil function
        lambda x: np.log10(np.abs(x) + 1),  # log10
        lambda x: np.arctan(x),  # arctangent
        lambda x: np.arctanh(np.clip(x / 10, -0.999, 0.999)),  # inverse hyperbolic tangent
        lambda x: np.exp(-np.abs(x)),  # decaying exponential
        lambda x: np.exp(-x**2 / (2 * random.uniform(1, 5)**2)) * np.cos(x * random.uniform(1, 5)),  # Gabor function
        lambda x: np.sqrt(np.abs(x)) * np.sign(x),  # square root with original sign
        lambda x: np.exp(x) / (1 + np.exp(x)),  # logistic (sigmoid) function
        lambda x: 1 / (1 + np.exp(-x)),  # logistic (sigmoid) function (alternative)
        lambda x: x * np.log(np.abs(x) + 1),  # x * log(x)
        lambda x: np.cbrt(x),  # cube root
        lambda x: np.around(x, decimals=random.randint(0, 2)),  # rounding with random decimals
        lambda x: gamma(x),  # Gamma function
        lambda x: factorial(np.floor(x)),  # Factorial function
        lambda x: zeta(x),  # Riemann Zeta function
        lambda x: np.power(random.random(), x),  # Exponential function with base 2
    ]

    # Generate y values based on the relationship
    while True:
        chosen_relationships = random.sample(relationships, random.randint(1, 5))

        # ----------
        y = np.zeros_like(x)
        for f in chosen_relationships:
            y += random.uniform(-1, 1) * f(x)

        if not np.isnan(y).any():
            break

    # apply ranges ---
    y = vector_scaling_at_random(y)

    # flip
    if random.random() <= 0.10:
        y = np.flip(y)

    # add noise ---
    if random.random() < 0.25:
        range_y = max(y) - min(y)
        noise = np.random.normal(0, range_y / 10, len(y))
        y = y + noise

    return y


def generate_y(x):
    """generate y series
    """
    func_list = [percentages, random_fractions, fn_generation]
    func = random.choices(func_list, weights=[0.05, 0.05, 0.90], k=1)[0]
    y = func(x)

    # force positive ---
    p = random.random()
    if p <= 0.2:
        if random.random() >= 0.10:
            y = np.array(y) - np.amin(y)  # start from 0
        else:
            y = np.absolute(y)  # absolute value
    elif p <= 0.4:
        y = y - np.mean(y)  # center around 0

    return list(y)
