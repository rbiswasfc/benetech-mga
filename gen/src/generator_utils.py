import os
import random
import re
import string
import textwrap
import traceback
from copy import deepcopy

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from PIL import Image
from scipy.interpolate import BSpline, interp1d, make_interp_spline


def generate_random_string():
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(8))


def is_numeric(input_list):
    try:
        for elem in input_list:
            float(elem)
        return True
    except ValueError:
        return False


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


def has_non_latin_chars(input_list):
    """has non ascii chars in input list
    """
    for elem in input_list:
        if any(ord(character) > 127 for character in elem):
            return True
        # if bool(re.search(r'[^\x00-\x7F]', elem)):
        #     return True
    return False


def is_constant(input_list):
    max_, min_ = max(input_list), min(input_list)
    if max_ - min_ <= 1e-6:
        return True
    return False


def get_random_equation():
    # List of templates for LaTeX equations
    templates = [
        r'$\alpha_{} > \beta_{}$',
        r'$\sum_{{i=0}}^\infty x_{}^{}$',
        r'$\frac{{{}x^2}}{{{}y^2}}$',
        r'$\sqrt[{}]{{2}}$',
        r'$\hat{{y}} = {}x + {}$',
        r'$e^{{{}x}}=\sum_{{i=0}}^\infty \frac{{1}}{{i!}}x^i$',
        r'$\int_0^1 x^{} dx$',
        r'$\frac{{d}}{{dx}}(x^{})$',
        r'$\lim_{{x \to {}}} x^{}$',
        r'$\frac{{d^{}y}}{{dx^{}}}$',
        r'$\int_{{-{}x}}^{{{}x}} {}x^{} dx$',
        r'$\lim_{{x \to {}}} \frac{{1}}{{x^{}}}$',
        r'$\prod_{{i=1}}^{{n}} {}i^{}$',
        r'$\frac{{\partial^{} f}}{{\partial x^{}}}$',
        r'$\int e^{{-{}x^2}} dx$',
    ]

    # Choose a random template
    template = random.choice(templates)

    # Generate random values for the placeholders
    values = [random.randint(1, 10) for _ in range(template.count('{}'))]

    # Insert the values into the template
    equation = template.format(*values)

    return equation


def generate_series_name():
    """
    Generate a single diverse, imaginative, and creative series name.

    Returns:
        A string representing a diverse, imaginative, and creative series name.
    """
    prefixes = [
        "Serendipitous",
        "Surreal",
        "Whimsical",
        "Quirky",
        "Fanciful",
        "Fantastic",
        "Mythical",
        "Mystical",
        "Enigmatic",
        "Cryptic",
        "Arcane",
        "Ethereal",
        "Celestial",
        "Stellar",
        "Nebulous",
        "Hypnotic",
        "Euphoric",
        "Luminous",
        "Vibrant",
        "Electric",
        "Magnetic",
        "Galactic",
        "Cosmic",
        "Epic",
        "Legendary",
        "Mythic",
        "Chimeric",
        "Futuristic",
        "Ambiguous",
        "Abstract",
        "Intriguing",
        "Puzzling",
        "Mysterious",
        "Amorphous",
        "Unusual",
        "Rare",
        "Exotic",
        "Outlandish",
        "Eclectic",
        "Whimsical",
        "Unconventional",
        "Unorthodox",
        "Offbeat",
        "Divergent",
        "Innovative",
        "Avant-Garde",
        "Experimental",
        "Radical",
        "Revolutionary",
        "Series",
        "Column",
        "Line",
        "Curve",
        "Plot",
        "Graph",
        "Trace",
        "Function",
        "Distribution",
        "Histogram",
        "Scatter",
        "Bar",
        "Area",
        "Bubble",
        "Heatmap",
        "Boxplot",
        "Violin",
        "Radar",
        "Polar",
        "Network",
        "Tree",
        "Sankey",
        "Sunburst",
        "Chord",
        "Wordcloud",
        "Map",
        "Globe",
        "Surface",
        "Contour",
        "Isosurface",
        "Streamline",
        "Vector",
        "Glyph",
        "Tensor",
        "Eigen",
        "Kernel",
        "Wavelet",
        "Fourier",
        "Hilbert",
        "Laplacian",
        "Cauchy",
        "Gamma",
        "Beta",
        "Weibull",
        "Lognormal",
        "Poisson",
        "Exponential",
        "Gaussian",
        "Uniform",
        "Bernoulli",
        "Binomial",
        "Hypergeometric",
        "Multinomial",
        "Dirichlet",
        "Beta-Binomial",
        "Poisson-Gamma",
        "Negative Binomial",
        "Geometric",
        "Empirical",
    ]

    suffixes = [
        "A",
        "B",
        "C",
        "1",
        "2",
        "3",
        "I",
        "II",
        "III",
        "IV",
        "V",
        "VI",
        "VII",
        "VIII",
        "IX",
        "X",
        "XI",
        "XII",
        "XIII",
        "XIV",
        "XV",
        "Prime",
        "Double",
        "Triple",
        "Quad",
        "Plus",
        "Minus",
        "Delta",
        "Gamma",
        "Alpha",
        "Beta",
        "Chi",
        "Omega",
        "Sigma",
        "Lambda",
        "Nu",
        "Rho",
        "Tau",
        "Xi",
        "Zeta",
        "Kappa",
        "Eta",
        "Theta",
        "Psi",
        "Phi",
        "Upsilon",
        "Epsilon",
        "Pi",
        "Upsilondiaeresis",
        "Omicron",
        "Eta",
        "Iota",
        "Digamma",
        "Nabla",
        "Aleph",
        "Beth",
        "Gimel"
    ]
    return random.choice(prefixes) + " " + random.choice(suffixes)


def generate_range(n_points, **kwargs):
    points = [random.randint(0, 100) for _ in range(n_points+1)]
    points = sorted(list(set(points)))

    operators = ['<', '>', '<=', '>=', '==', '!=',
                 'greater than', 'less than', 'equal to', 'not equal to']

    prefixes = [
        'pre', 'post', 'mid', 'early', 'late', 'mid', 'early', 'late', 'mid', 'early', 'late',
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa',
        'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi',
        'chi', 'psi', 'omega', 'capital', 'corner', 'upper', 'lower', 'left', 'right', 'top',
        'diagonal', 'horizontal', 'vertical', 'upper left', 'upper right', 'lower left', 'lower right',
        'age', 'year', 'month', 'day', 'hour', 'minute', 'second', 'quarter', 'decade', 'century',
        'millennium', 'epoch', 'era', 'period', 'eon', 'eonothem', 'eon', 'era', 'period', 'epoch',
        'month', 'calorie', 'resistance', 'capacitance', 'inductance', 'conductance', 'admittance',
        'power', 'work done', 'energy', 'force', 'pressure', 'stress', 'frequency', 'speed', 'velocity',
        'acceleration', 'angular velocity', 'angular acceleration', 'angular momentum', 'momentum',
        'impulse', 'torque', 'moment of inertia', 'electric charge', 'electric current', 'voltage',
        'electric field', 'magnetic field', 'magnetic flux', 'magnetic flux density', 'magnetic moment',
        'magnetic pole', 'magnetomotive force', 'magnetic vector potential', 'magnetic scalar potential',
        'magnetic dipole moment', 'magnetic susceptibility', 'magnetic permeability', 'magnetic reluctance',
        'heat', 'heat capacity', 'entropy', 'specific heat capacity', 'specific entropy', 'specific energy',
        'specific volume', 'specific weight', 'specific charge', 'specific force', 'specific impulse',
        'specific power', 'specific energy consumption', 'specific absorption rate', 'specific surface area',
        'molecular weight', 'molar mass', 'molar volume', 'molar heat capacity', 'molar entropy',
        'molar energy', 'molar refractivity', 'molar conductivity', 'molar magnetic susceptibility',
        'molar volume', 'molar concentration', 'molar absorptivity', 'molar attenuation coefficient',
        'angle', 'solid angle', 'frequency', 'wavenumber', 'angular frequency', 'luminous flux',
        'luminous energy', 'luminance', 'illuminance', 'luminous intensity', 'luminous efficacy',
        'wave number', 'angular wave number', 'angular wavenumber', 'angular frequency', 'angular velocity',
        'degree of rotation', 'angular displacement', 'angular acceleration', 'angular momentum',
    ]

    use_prefix = random.random() >= 0.3

    cats = []
    for i in range(len(points)):
        if use_prefix:
            op = random.choice(operators)
            prefix = random.choice(prefixes)
            if random.random() <= 0.25:
                prefix = prefix.title()

            cat = f'{prefix} {op} {points[i]}'
            cats.append(cat)
        else:
            if i == 0:
                op = random.choice(['<', '<=', 'less than', 'less than or equal to'])
            elif i == len(points) - 1:
                op = random.choice(['>', '>=', 'greater than', 'greater than or equal to'])
            else:
                op = '-'

            if random.random() <= 0.25:
                op = op.title()
            elif random.random() <= 0.5:
                op = op.upper()
            elif random.random() <= 0.75:
                op = op.lower()

            if i == 0:
                cat = f'{op} {points[i]}'
            elif i == len(points) - 1:
                cat = f'{op} {points[i]}'
            else:
                cat = f'{points[i-1]} {op} {points[i]}'
            cats.append(cat)
    return cats
