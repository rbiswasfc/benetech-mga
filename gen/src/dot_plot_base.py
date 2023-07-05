import io
import os
import random
import sys
import textwrap
import traceback
from copy import deepcopy

import matplotlib
import matplotlib.cm as cm
import matplotlib.font_manager
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.ticker import MaxNLocator
from numpy.polynomial import Polynomial
from PIL import Image
from scipy.interpolate import BSpline, interp1d, make_interp_spline

matplotlib.use('Agg')

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
except Exception as e:
    sys.path.append("/kaggle/input/gen-utils-easy")

try:
    from constants import RANDOM_LABELS
    from generator_utils import (generate_series_name, get_random_equation,
                                 is_numeric)
except ImportError:
    raise ImportError('Importing failed.')

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
except Exception as e:
    sys.path.append("/kaggle/input/gen-utils-easy")


# ---------------------------------------------------------------------------------------#
# params
# ---------------------------------------------------------------------------------------#
FONT_FAMILY = [
    'DejaVu Sans',
    'Arial',
    'Times New Roman',
    'Courier New',
    'Helvetica',
    'Verdana',
    'Trebuchet MS',
    'Palatino',
    'Georgia',
    'MesloLGS NF',
    'Lucida Grande',
]

MARKER_OPTIONS = [
    'o',
    'p',
    '*',
    'd',
]

CUSTOM_MARKERS = [
    r'$\clubsuit$',
    r'$\diamondsuit$',
    r'$\heartsuit$',
    r'$\spadesuit$',
    r'$\otimes$'
]

COLOR_OPTIONS = [
    'b',
    'g',
    'r',
    'c',
    'm',
    'y',
    'k',
]

SPINE_STYLES = [
    'center',
    'zero'
]

CMAP_OPTIONS = [
    'viridis',
    'inferno',
    'cool',
    'spring',
    'summer',
    'winter',
]

FONT_WEIGHTS = [
    'normal',
    'bold',
    'light',
    'ultralight',
    'heavy',
    'black',
    'semibold',
    'demibold',
]


def generate_random_params():
    # theme = 'ticks'
    theme = random.choices(
        ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'],
        weights=[0.025, 0.025, 0.025, 0.025, 0.9],
        k=1
    )[0]

    # font weight ---
    if random.random() >= 0.75:
        font_weight = random.choice(FONT_WEIGHTS)
    else:
        font_weight = 'normal'

    # rotation ---
    rotation_x = random.choices(
        [0, 90, 270], weights=[0.90, 0.05, 0.05], k=1
    )[0]
    rotation_y = random.choices(
        [0, 45, 90],
        weights=[0.9, 0.05, 0.05], k=1
    )[0]

    p = random.random()
    if p <= 0.8:
        lift_factor = 0.5
    else:
        lift_factor = random.uniform(0.0, 0.5)

    params = {
        'sns_theme': theme,
        'plt_style': random.choice(plt.style.available),

        'font_family': random.choice(FONT_FAMILY),
        'font_size': random.randint(10, 12),
        'font_weight': font_weight,
        'font_color': random.choice(COLOR_OPTIONS),

        'dpi': 100,
        'width': random.uniform(4, 6),
        'height': random.uniform(4, 6),
        'dot_size': random.randint(8, 10),
        'color': random.choice(COLOR_OPTIONS),
        'marker': random.choice(MARKER_OPTIONS),

        'lift_factor': lift_factor,


        'x_tick': random.random() >= 0.10,
        'y_tick': random.random() >= 0.10,

        'xtick_label_rotation': rotation_x,
        'ytick_label_rotation': rotation_y,

        'num_x_ticks': random.randint(4, 8),
        'num_y_ticks': random.randint(4, 8),

        'aux_spine': random.random() >= 0.95,
        'left_spine': random.random() >= 0.25,

        'custom_spine': random.random() >= 1.99,  # dummy
        'spine_style': random.choice(SPINE_STYLES),

        'x_grid': random.random() >= 0.95,
        'y_grid': random.random() >= 0.95,
        'x_minor_grid': random.random() >= 0.99,
        'y_minor_grid': random.random() >= 0.99,

        'add_legend': random.random() >= 0.95,
    }

    if random.random() >= 0.9:
        params['marker'] = random.choice(CUSTOM_MARKERS)
    # params['marker'] = 'x'

    return params


def adjust_params(params):
    params = deepcopy(params)

    width = max(4, params['n_points'] / 4)

    h_factor = 4.0
    if params['max_count'] >= 12:
        h_factor = 3.0

    if params['max_count'] <= 4:
        h_factor = 6.0

    height = max(2.4, params['max_count'] / h_factor)

    if params['max_x_tick_chars'] >= 8:
        params['width'] = max(5.0, params['width'])
        params['height'] = max(5.0, params['height'])

    params['width'] = width
    params['height'] = height

    # rotations ---
    if (params['n_points'] >= 12) | (params['mean_x_chars'] >= 5) | (params['max_x_tick_chars'] >= 10):
        params['font_size'] = 6  # random.randint(6, 8)

        # handle rotation --
        if (random.random() >= 0.80) & (params['mean_x_chars'] <= 6):
            params['xtick_label_rotation'] = random.uniform(45, 90)
        else:
            params['xtick_label_rotation'] = 90

    if (params['max_x_tick_chars'] <= 4) & (params['n_points'] <= 12):
        params['xtick_label_rotation'] = 0
        params['font_size'] = max(8, params['font_size'])

    elif (params['max_x_tick_chars'] <= 6) & (params['n_points'] <= 8):
        params['xtick_label_rotation'] = 0
        params['font_size'] = max(12, params['font_size'])

    # color ---
    if params['plt_style'] == 'dark_background':
        params['font_color'] = 'w'
        params['color'] = 'w'

    if params['numeric_x']:
        params['sns_theme'] = 'ticks'

    if 'bold' in params['font_weight']:
        params['font_size'] = 6

    return params

# ---------------------------------------------------------------------------------------#
# data handling
# ---------------------------------------------------------------------------------------#


def parse_example(cfg, the_example):
    # get underlying data ---
    plot_title = the_example['plot_title']

    x_title = the_example['x_title']
    y_title = the_example['y_title']

    x_values = the_example['x_series']
    y_values = the_example['y_series']

    titles = {
        'x_title': x_title,
        'y_title': y_title,
        'plot_title': plot_title,
    }

    data_series = {
        'x_values': x_values,
        'y_values': y_values,
    }

    return titles, data_series


# ---------------------------------------------------------------------------------------#
# dot plot
# ---------------------------------------------------------------------------------------#

class BasicDotPlot:
    def __init__(self, cfg, the_example, texture_files=None, debug=False):
        self.cfg = cfg
        self.example = deepcopy(the_example)
        self.params = generate_random_params()
        self.debug = debug
        self.texture_files = texture_files

        # create data dependent params ---
        self.params['n_points'] = len(the_example['x_series'])
        self.params['max_x_tick_chars'] = max([len(str(x)) for x in the_example['x_series']])
        self.params['mean_x_chars'] = np.mean([len(str(x)) for x in the_example['x_series']])
        self.params['max_count'] = max(the_example['y_series'])
        self.params['numeric_x'] = is_numeric(the_example['x_series'])

        # adjust params ---
        self.params = adjust_params(self.params)
        # print(self.params)

        # stats ---
        self.num_x = len(the_example['x_series'])
        self.num_y = len(the_example['y_series'])
        self.y_min = min(the_example['y_series'])
        self.y_max = max(the_example['y_series'])

        # configure style & create figure ---
        self.configure_style()
        self.fig, self.ax = self.get_figure_handles()

    def configure_style(self):
        plt.rcParams.update(plt.rcParamsDefault)  # reset to defaults

        sns.set_style(style=self.params['sns_theme'])
        plt.style.use(self.params['plt_style'])

        plt.rcParams['font.family'] = self.params['font_family']
        plt.rcParams['font.size'] = self.params['font_size']
        plt.rcParams['font.weight'] = self.params['font_weight']

        # tick parameters---
        tick_size = random.uniform(3.0, 6.0)
        tick_width = random.uniform(0.5, 1.5)

        tick_size_minor = random.uniform(2.0, tick_size)
        tick_width_minor = random.uniform(0.5, tick_width)

        plt.rcParams['xtick.major.size'] = tick_size
        plt.rcParams['ytick.major.size'] = tick_size

        plt.rcParams['xtick.major.width'] = tick_width
        plt.rcParams['ytick.major.width'] = tick_width

        plt.rcParams['xtick.minor.size'] = tick_size_minor
        plt.rcParams['ytick.minor.size'] = tick_size_minor

        plt.rcParams['xtick.minor.width'] = tick_width_minor
        plt.rcParams['ytick.minor.width'] = tick_width_minor

    def get_figure_handles(self):
        width = self.params['width']
        height = self.params['height']
        dpi = self.params['dpi']

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        return fig, ax

    # -----------------------------------------------------------------------------------#
    # axis limits
    # -----------------------------------------------------------------------------------#
    def set_axis_limits(self):
        # data ranges - x axis
        if not is_numeric(self.example['x_series']):
            x_lb, x_ub = 0, len(self.x_values) - 1  # categorical x values
            x_lb -= random.uniform(0.25, 0.5)
            x_ub += random.uniform(0.25, 0.5)
            self.ax.set_xlim(x_lb, x_ub)
        else:
            x_lb, x_ub = min(self.x_values), max(self.x_values)  # categorical x values
            range_x = x_ub - x_lb
            f = random.uniform(0.02, 0.1)
            x_lb -= f*range_x
            x_ub += f*range_x
            self.ax.set_xlim(x_lb, x_ub)

        p = random.random()
        y_lb, y_ub = -1, max(self.y_values) + 1

        y_lb += self.params['lift_factor']
        y_ub += self.params['lift_factor']

        # squeeze  effect
        y_ub *= random.uniform(0.9, 1.0)

        self.ax.set_ylim(y_lb, y_ub)

    # -----------------------------------------------------------------------------------#
    # ticks
    # -----------------------------------------------------------------------------------#
    def configure_ticks(self):
        # tick parameters --
        direction = random.choice(['in', 'out', 'inout'])
        x_top = (random.random() >= 0.6) & (self.params['aux_spine'])
        x_labeltop = (random.random() >= 0.8) & (x_top) & (self.params['max_x_tick_chars'] <= 5)
        self.x_labeltop = x_labeltop

        y_left = (self.params['left_spine']) | (random.random() >= 0.98)
        y_labelleft = y_left

        y_right = (random.random() >= 0.6) & (self.params['aux_spine'])
        y_labelright = (random.random() >= 0.8) & (y_right)

        self.ax.minorticks_on()  # turn on minor ticks

        # -----------------------------------------
        # set x ticks --------
        if not is_numeric(self.example['x_series']):
            x_tick_positions = np.arange(self.num_x)
            self.ax.set_xticks(x_tick_positions)
            x_tick_labels = ["\n".join(textwrap.wrap(c, width=random.randint(6, 8))) for c in self.x_values]
            self.ax.set_xticklabels(x_tick_labels, minor=False)
        else:
            n = len(self.x_values)
            if n <= 5:
                interval = 1
            elif n <= 10:
                interval = random.randint(1, 2)
            else:
                interval = random.randint(2, 4)

            # try to include the last point
            if (n-1) % interval != 0:
                for div in reversed(range(1, n-1)):
                    if (n-1) % div == 0:
                        interval = div
                        break

            x_major_tick_positions = [self.x_values[i] for i in range(len(self.x_values)) if i % interval == 0]
            x_major_tick_labels = [str(x) for x in x_major_tick_positions]

            self.ax.set_xticks(x_major_tick_positions, minor=False)
            self.ax.set_xticklabels(x_major_tick_labels, minor=False)

            x_minor_tick_positions = [self.x_values[i] for i in range(len(self.x_values)) if i % interval != 0]
            self.ax.set_xticks(x_minor_tick_positions, minor=True)

            x_formatter = ticker.FuncFormatter(
                lambda x, pos: "{:.7f}".format(x).rstrip('0').rstrip('.')
            )
            self.ax.xaxis.set_major_formatter(x_formatter)

        self.ax.tick_params(
            axis='x',
            which='both',
            rotation=self.params['xtick_label_rotation'],
            direction=direction,
            top=x_top,  # ---
            labeltop=x_labeltop,
            zorder=5,
        )

        if not is_numeric(self.x_values):
            self.ax.xaxis.set_tick_params(which='minor', bottom=False, top=False)
        elif random.random() >= 0.98:
            self.ax.xaxis.set_tick_params(which='minor', bottom=False, top=False)

        # -----------------------------------------
        y_formatter = ticker.FuncFormatter(
            lambda x, pos: f'{int(x)}' if x.is_integer() else f'{x:.2f}'
        )
        self.ax.yaxis.set_major_formatter(y_formatter)

        # y axis ---
        self.ax.tick_params(
            axis='y',
            which='both',
            rotation=self.params['ytick_label_rotation'],
            direction=direction,
            left=y_left,  # ---
            right=y_right,  # ---
            labelleft=y_labelleft,
            labelright=y_labelright,
            zorder=5,
        )

        if random.random() >= 0.5:
            self.ax.yaxis.set_tick_params(which='minor', left=False, right=False)

    # -----------------------------------------------------------------------------------#
    # gridlines
    # -----------------------------------------------------------------------------------#

    def configure_gridlines(self):
        # ----
        major_linestyle = random.choice(['-', '--', ':', '-.'])
        minor_linestyle = random.choice(['-', '--', ':', '-.'])

        major_linewidth = random.uniform(0.5, 1.0)
        minor_linewidth = random.uniform(0.1, major_linewidth)

        major_color = random.choice(COLOR_OPTIONS)
        minor_color = random.choice(COLOR_OPTIONS)

        major_alpha = random.uniform(0.4, 1.0)
        minor_alpha = random.uniform(0.1, major_alpha)

        # set x axis gridlines ---
        if self.params['x_grid']:

            self.ax.xaxis.grid(
                True,
                linestyle=major_linestyle,
                linewidth=major_linewidth,
                color=major_color,
                alpha=major_alpha,
                which='major',
            )
            if self.params['x_minor_grid']:
                self.ax.xaxis.grid(
                    True,
                    linestyle=minor_linestyle,
                    linewidth=minor_linewidth,
                    color=minor_color,
                    alpha=minor_alpha,
                    which='minor',
                )
        else:
            self.ax.xaxis.grid(visible=False, which='both')

        # set y axis gridlines ---
        if self.params['y_grid']:
            self.ax.yaxis.grid(
                True,
                linestyle=major_linestyle,
                linewidth=major_linewidth,
                color=major_color,
                alpha=major_alpha,
                which='major',
            )
            if self.params['y_minor_grid']:
                self.ax.yaxis.grid(
                    True,
                    linestyle=minor_linestyle,
                    linewidth=minor_linewidth,
                    color=minor_color,
                    alpha=minor_alpha,
                    which='minor',
                )
        else:
            self.ax.yaxis.grid(visible=False, which='both')

    # ------------------------------------------------------------------------------------#
    # titles
    # ------------------------------------------------------------------------------------#

    def configure_titles(self):
        # bounding box ---
        box_color = random.choice(['c', 'k', 'gray', 'r', 'b', 'g', 'm', 'y'])
        padding = random.uniform(0, 1)
        boxstyle = random.choice(['round', 'square'])

        if random.random() >= 0.90:
            bbox_props = {
                'facecolor': box_color,
                'alpha': random.uniform(0.1, 0.2),
                'pad': padding,
                'boxstyle': boxstyle
            }
        else:
            bbox_props = None

        # font dict --
        if self.params['font_color'] == 'y':  # no yellow font color
            self.params['font_color'] = 'k'

        font_dict = {
            'family': self.params['font_family'],
            'color': self.params['font_color'],
            'weight': self.params['font_weight'],
            'size': self.params['font_size'],
        }

        axes_font_dict = deepcopy(font_dict)
        axes_font_dict['size'] = self.params['font_size'] - random.randint(0, 2)

        x_location = 'center'
        y_location = 'center'

        if random.random() >= 0.85:
            x_location = random.choice(['left', 'right', 'center'])
            y_location = random.choice(['top', 'bottom', 'center'])

        if (self.params['custom_spine']) & (self.params['spine_style'] == 'center'):
            x_location = 'right'
            y_location = 'top'

        if random.random() >= 0.25:
            self.ax.set_xlabel(
                self.title_dict['x_title'][:12],
                fontdict=axes_font_dict,
                loc=x_location,
                labelpad=random.uniform(4, 10),
                bbox=bbox_props,
            )

            self.ax.set_ylabel(
                self.title_dict['y_title'][:12],
                fontdict=axes_font_dict,
                loc=y_location,
                labelpad=random.uniform(4, 10),
                bbox=bbox_props,
            )

        # figure title ---
        fig_width_inch = self.fig.get_figwidth()
        fig_dpi = self.fig.dpi
        fig_width_pixel = fig_width_inch * fig_dpi
        char_width_pixel = 12
        num_chars = int(fig_width_pixel / random.uniform(1, 2)*char_width_pixel)

        # Set a long title and use textwrap to automatically wrap the text
        title = '\n'.join(textwrap.wrap(self.title_dict['plot_title'], num_chars))
        title_font_dict = deepcopy(font_dict)
        title_font_dict['size'] = self.params['font_size'] + random.randint(-2, 2)
        title_loc = random.choice(['left', 'right', 'center'])

        if (self.params['custom_spine']) & (self.params['spine_style'] == 'center'):
            title_loc = random.choice(['left', 'right'])

        if self.x_labeltop:
            title_pad = - 32
        elif self.params['aux_spine']:
            title_pad = 24
        else:
            title_pad = random.randint(6, 24)
            if random.random() >= 0.8:
                title_pad = - title_pad

        if random.random() >= 0.25:
            self.ax.set_title(
                title,
                fontdict=title_font_dict,
                loc=title_loc,
                pad=title_pad,
                bbox=bbox_props
            )

    # -----------------------------------------------------------------------------------#
    # spine
    # -----------------------------------------------------------------------------------#
    def configure_spine(self):
        # spine visibility ---
        active_spines = ['left', 'bottom', 'right', 'top']

        self.ax.spines['left'].set_visible(True)
        self.ax.spines['bottom'].set_visible(True)

        if not self.params['aux_spine']:
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['top'].set_visible(False)
            active_spines.remove('right')
            active_spines.remove('top')

        if not self.params['left_spine']:
            self.ax.spines['left'].set_visible(False)
            active_spines.remove('left')

        # set linewidth ---
        linewidth = random.uniform(1.0, 3.0)
        for spine in active_spines:
            self.ax.spines[spine].set_linewidth(linewidth)

        # set color ---
        color = random.choice(['r', 'g', 'b', 'k', 'gray'])
        if random.random() >= 0.75:
            color = 'k'
        if self.params['plt_style'] == 'dark_background':
            color = 'w'

        for spine in active_spines:
            self.ax.spines[spine].set_color(color)

        # set spine positions ---
        picked_style = self.params['spine_style']

        if self.params['custom_spine'] & (not self.params['aux_spine']):  # apply styles
            self.params['x_grid'] = False
            self.params['y_grid'] = False
            self.params['y_minor_grid'] = False
            self.params['num_y_ticks'] = min(4, self.params['num_y_ticks'])

            if picked_style == 'zero':
                if (np.amin(self.y_values) < 0) & (np.amax(self.y_values) > 0):
                    self.ax.spines[active_spines].set_position('zero')

            if picked_style == 'center':
                self.ax.spines[active_spines].set_position('center')
        else:
            if 'bottom' in active_spines:
                self.ax.spines['bottom'].set_position('zero')

        # arrowheads ---
        if (random.random() >= 0.00) & (not self.params['left_spine']):
            if is_numeric(self.x_values):
                ub = np.amax(self.x_values)
                lb = np.amin(self.x_values)
                offset = self.x_values[1] - self.x_values[0]

            else:
                ub = len(self.x_values) - 1
                lb = 0
                offset = 1

            # lb -= offset
            # ub += offset

            offset *= 2

            # Create an arrow at both ends of the bottom spine
            arrow_left = mpatches.FancyArrowPatch((lb, 0), (lb-offset, 0),
                                                  mutation_scale=24,
                                                  fc='k',
                                                  arrowstyle='-|>')  # Style of the arrow

            arrow_right = mpatches.FancyArrowPatch((ub, 0), (ub+offset, 0),
                                                   mutation_scale=24,
                                                   fc='k',
                                                   arrowstyle='-|>')

            self.ax.add_patch(arrow_left)
            self.ax.add_patch(arrow_right)

    # -----------------------------------------------------------------------------------#
    # Plot
    # -----------------------------------------------------------------------------------#

    def make_basic_plot(self, x, y):
        x = deepcopy(x)
        y = deepcopy(y)

        squeeze = 0.9

        if not is_numeric(x):
            x = list(np.arange(len(x)))

        # if random.random() >= 0.9:
        #     if max(y) <= 8:
        #         self.params['marker'] = 'x'

        factor = self.params['lift_factor']

        for xi, yi in zip(x, y):
            self.ax.plot(
                [xi]*yi,
                [i*squeeze + factor for i in (range(yi))],
                marker=self.params['marker'],
                ms=self.params['dot_size'],
                linestyle='',
                color=self.params['color'],
            )

    def fancy_plot_v1(self, x, y):
        x = deepcopy(x)
        y = deepcopy(y)

        if not is_numeric(x):
            x = list(np.arange(len(x)))

        factor = self.params['lift_factor']
        squeeze = 0.9

        for xi, yi in zip(x, y):
            self.ax.plot(
                [xi]*yi,
                [i * squeeze + factor for i in (range(yi))],
                color=random.choice(['r', 'g', 'b', 'k']),
                marker=self.params['marker'],
                ms=self.params['dot_size'],
                linestyle='',
            )

    def fancy_plot_v2(self, x, y):
        x = deepcopy(x)
        y = deepcopy(y)

        if not is_numeric(x):
            x = list(np.arange(len(x)))

        factor = self.params['lift_factor']
        squeeze = 0.9

        for xi, yi in zip(x, y):
            self.ax.plot(
                [xi]*yi,
                [i*squeeze + factor for i in (range(yi))],
                color='k',
                alpha=random.uniform(0.2, 1.0),
                marker=self.params['marker'],
                ms=self.params['dot_size'],
                linestyle='',
            )

    def fancy_plot_v3(self, x, y):
        x = deepcopy(x)
        y = deepcopy(y)

        if not is_numeric(x):
            x = list(np.arange(len(x)))
        factor = self.params['lift_factor']

        # ---
        # Color gradient based on y values
        cmap = plt.cm.get_cmap('viridis')  # choose a color map
        color = [cmap(val/max(y)) for val in y]

        # Size gradient based on y values
        size = [50 + 30*val/max(y) for val in y]
        squeeze = 0.9

        # -----
        for xi, yi, ci, si in zip(x, y, color, size):

            self.ax.scatter(
                [xi]*yi,
                [i*squeeze + factor for i in range(yi)],
                marker=self.params['marker'],
                s=si,
                c=ci,
                alpha=random.uniform(0.8, 1.0),
                edgecolors='k',
            )

    def fancy_plot_v4(self, x, y):
        x = deepcopy(x)
        y = deepcopy(y)

        if not is_numeric(x):
            x = list(np.arange(len(x)))

        factor = self.params['lift_factor']

        squeeze = random.uniform(0.8, 0.9)
        marker = random.choice(['o', '*'])

        for xi, yi in zip(x, y):
            self.ax.plot(
                [xi]*yi,
                [i*squeeze + factor for i in (range(yi))],
                color=random.choice(['k', 'r']),
                alpha=random.uniform(0.2, 1.0),
                marker=marker,
                ms=self.params['dot_size'],
                linestyle='',
            )

    # -----------------------------------------------------------------------------------#
    # dot plot
    # -----------------------------------------------------------------------------------#

    def create_plot(self):
        plot_options = [
            self.make_basic_plot,
            self.fancy_plot_v1,
            self.fancy_plot_v2,
            self.fancy_plot_v3,
            self.fancy_plot_v4,
        ]

        weights = [
            3.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]

        plot_fn = random.choices(plot_options, weights=weights, k=1)[0]
        if self.params['plt_style'] == 'dark_background':
            plot_fn = self.make_basic_plot

        if plot_fn != self.make_basic_plot:
            self.params['left_spine'] = False

        plot_fn(self.x_values, self.y_values)

    # -----------------------------------------------------------------------------------#
    # stats
    # -----------------------------------------------------------------------------------#

    def add_stat_info(self):
        if random.random() >= 0.98:
            # Add statistical information
            avg = np.mean(self.y_values)
            min_val = round(np.min(self.y_values), 2)
            max_val = round(np.max(self.y_values), 2)

            stats_text = f'Average: {avg:.2f}\nMin: {min_val}\nMax: {max_val}'

            self.ax.text(
                random.uniform(0.02, 0.2),
                random.uniform(0.8, 0.95),
                stats_text,
                transform=self.ax.transAxes,
                verticalalignment='top',
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=random.uniform(0.2, 0.4),
                ),
                fontdict={'size': 6},
            )

    # -----------------------------------------------------------------------------------#
    # draw circle
    # -----------------------------------------------------------------------------------#

    def add_circle(self):
        if random.random() >= 0.98:
            idx = random.randint(0, len(self.x_values) - 1)
            if not is_numeric(self.x_values):
                xi = idx
            else:
                xi = self.x_values[idx]
            yi = self.y_values[idx]
            r = 0.5

            circle = Circle((xi, yi/4), r, fill=False, edgecolor='red', lw=2)
            self.ax.add_patch(circle)

    # -----------------------------------------------------------------------------------#
    # draw rectangle
    # -----------------------------------------------------------------------------------#

    def add_rectangle(self):
        if random.random() >= 0.98:
            idx = random.randint(0, len(self.x_values) - 1)

            if not is_numeric(self.x_values):
                xi = idx
                width = 0.6
            else:
                xi = self.x_values[idx]
                width = (self.x_values[1] - self.x_values[0])*0.6

            yi = self.y_values[idx]
            rectangle = Rectangle(
                (xi - width/2, -0.5),
                width,
                yi+0.5,
                fill=False,
                edgecolor=random.choice(COLOR_OPTIONS),
                lw=random.uniform(0.5, 2.5),
            )
            self.ax.add_patch(rectangle)
            # Add an annotation with an arrow pointing to the rectangle
            self.ax.annotate(
                random.choice(RANDOM_LABELS),
                xy=(xi, yi),
                xytext=(xi + width, yi + 2),
                arrowprops=dict(facecolor='black', shrink=0.05)
            )

    # ------------------------------------------------------------------------------------#
    # main api
    # ------------------------------------------------------------------------------------#

    def make_dot_plot(self, graph_id):
        """main function to make a line plot

        :param the_example: underlying data
        :type the_example: dict
        """
        try:
            the_example = deepcopy(self.example)
            title_dict, data_series = parse_example(self.cfg, the_example)
            self.title_dict = title_dict
            self.x_values = data_series['x_values']
            self.y_values = data_series['y_values']

            # create the plot ---
            self.create_plot()

            # # error bars ---
            # self.plot_error_bars()

            # # limits ---
            self.set_axis_limits()

            # # spine ---
            self.configure_spine()

            # # ticks ---
            self.configure_ticks()

            # # gridlines ---
            self.configure_gridlines()

            # # titles --
            self.configure_titles()

            # # legend ---
            # self.configure_legend()

            # # stats ---
            self.add_stat_info()

            self.add_circle()
            self.add_rectangle()

            # --- SAVING ----------------------------------------------------------#
            save_path = os.path.join(self.cfg.output.image_dir, f'{graph_id}.jpg')

            if random.random() >= 0.75:
                self.fig.tight_layout()

                # Save the figure to a memory buffer in RGBA format
                buf = io.BytesIO()
                self.fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)

                # Load the image from the buffer using PIL
                img = Image.open(buf).convert('RGB')

                # add textures ---
                if self.texture_files is not None:
                    if random.random() <= 0.25:
                        texture = Image.open(random.choice(self.texture_files)).convert('RGB').resize(img.size)
                        img = Image.blend(img, texture, alpha=random.uniform(0.05, 0.15))

                if random.random() <= 0.1:  # grayscale
                    img = img.convert('L')

                img.save(save_path)
                buf.close()

                # plt.show()
                plt.close(self.fig)
                plt.close('all')
            else:
                self.fig.savefig(save_path, format='jpg', bbox_inches='tight')
                plt.close(self.fig)
                plt.close('all')

        except Exception as e:
            plt.close(self.fig)
            traceback.print_exc()
