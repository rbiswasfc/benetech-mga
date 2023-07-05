import io
import os
import random
import string
import sys
import textwrap
import traceback
from copy import deepcopy

import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import numpy as np
import seaborn as sns
from faker import Faker
from matplotlib.ticker import LogLocator, MaxNLocator
from PIL import Image
from scipy.stats import gaussian_kde

fake = Faker()

# --
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
except Exception as e:
    sys.path.append("/kaggle/input/gen-utils-easy")

try:
    from constants import RANDOM_LABELS, SYMBOLS, UNITS
    from generator_utils import (generate_series_name, get_random_equation,
                                 is_constant, is_numeric)
except ImportError:
    raise ImportError('Importing failed.')

# This function will be used to format the ticks


def custom_formatter(x, pos):
    if abs(x) < 1e-8:
        return '0'

    if (abs(x) < 100) & (abs(x) > 0.1):
        return '{:0.2f}'.format(x).rstrip('0').rstrip('.')

    x = '{:0.2e}'.format(x)
    # remove leading '+' and unnecessary zeros
    coeff, exponent = x.split('e')
    exponent = exponent.lstrip('+0')
    exponent = exponent.replace('-0', '-')
    # if exponent is empty after stripping, it should be zero
    coeff = round(float(coeff), 2)
    exponent = exponent if exponent else '0'
    return '{}e{}'.format(coeff, exponent)


def add_legend_name(title_dict):
    p = random.random()

    if p <= 0.3:
        return title_dict['y_title']
    elif p <= 0.6:
        return random.choice(RANDOM_LABELS)
    else:
        return generate_series_name()


def generate_annotations(length=1):
    if random.random() <= 0.5:
        return fake.word()

    p = random.random()
    chars = string.ascii_lowercase
    digits = string.digits

    if p >= 0.5:
        anno = "".join(random.choices(chars, k=length))
    else:
        anno = "".join(random.choices(digits, k=length))
    return anno


def is_log_scale_feasible(input_list):
    input_list = deepcopy(input_list)
    min_value = min(input_list)
    range_value = max(input_list) - min(input_list)
    if (min_value > 0) & (range_value > 1e3):
        return True
    return False


def get_formatter(input_list):
    input_list = deepcopy(input_list)
    input_range = max(input_list) - min(input_list)

    unit = random.choice(UNITS)
    currency_symbol = random.choice(SYMBOLS)

    # Apply custom formatter
    formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x)}' if x.is_integer() else f'{round(x, 2):.2f}')

    # formatter = ticker.ScalarFormatter()  # default formatter

    other_formatters = [
        ticker.FuncFormatter(lambda x, pos: f'{int(x)}%' if x.is_integer() else f'{x:.2f}%'),
        ticker.FuncFormatter(lambda x, pos: f'{int(x)}{unit}' if x.is_integer() else f'{x:.2f}{unit}'),
        ticker.FuncFormatter(lambda x, pos: f'{currency_symbol}{int(x)}' if x.is_integer() else f'{currency_symbol}{x:.2f}'),
        custom_formatter,
    ]

    prob = random.random()
    if input_range >= 1e5:  # use integer formatting for large numbers
        formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x)}')
    elif input_range <= 0.1:  # use float formatting for small numbers
        formatter = ticker.FuncFormatter(lambda x, pos: "{:.7f}".format(x).rstrip('0').rstrip('.'))
    elif (prob >= 0.975) & (input_range <= 1e5):  # use other formatters
        formatter = random.choice(other_formatters)
    # sci_formatter = custom_formatter
    # formatter = random.choice([formatter, sci_formatter])
    return formatter

# ------


def parse_example(the_example):
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

# ---- params -----------#


# ---- custom markers ----#
# Number of vertices for each shape
def generate_custom_markers():
    num_vertices = 5
    num_markers = 100
    my_markers = []

    # Generate the markers
    for _ in range(num_markers):
        # Randomly generate vertices
        verts = [(random.random(), random.random()) for _ in range(num_vertices)]
        verts.append(verts[0])  # Close the polygon

        # Define the path codes
        codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO]*(num_vertices-1) + [mpath.Path.CLOSEPOLY]
        # Create the marker and add it to the list
        my_marker = mpath.Path(verts, codes)
        my_markers.append(my_marker)
    return my_markers


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
    'v',
    's',
    'p',
    'P',
    '*',
    '+',
    'x',
    r'$\clubsuit$',
    r'$\diamondsuit$',
    r'$\heartsuit$',
    r'$\spadesuit$',
    r'$\otimes$',
]

CUSTOM_MARKERS = generate_custom_markers()

COLOR_OPTIONS = ['b', 'g', 'r', 'c', 'm', 'k']

FONT_WEIGHTS = [
    'normal', 'bold', 'light', 'ultralight', 'heavy', 'black', 'semibold', 'demibold'
]

SPINE_STYLES = ['center', 'zero']  # , 'zero']


def generate_random_params():
    theme = random.choices(
        ['dark', 'white', 'ticks'],
        weights=[0.05, 0.05, 0.9],
        k=1
    )[0]
    # theme = 'ticks'

    if random.random() >= 0.75:
        font_weight = random.choice(FONT_WEIGHTS)
    else:
        font_weight = 'normal'

    # rotations ---
    rotation_x = random.choices([0, 45, 90], weights=[0.95, 0.025, 0.025], k=1)[0]
    rotation_y = random.choices([0, 45, 90], weights=[0.85, 0.10, 0.05], k=1)[0]

    params = {
        'sns_theme': theme,
        'plt_style': random.choice(plt.style.available),

        'marker': random.choice(MARKER_OPTIONS),  # random.choice(MARKER_OPTIONS),
        'marker_size': random.uniform(8, 16),
        'color': random.choice(COLOR_OPTIONS),

        'font_family': random.choice(FONT_FAMILY),
        'font_size': random.randint(6, 10),
        'font_weight': font_weight,
        'font_color': random.choice(COLOR_OPTIONS),

        'dpi': 100,
        'width': random.uniform(3, 7),
        'height': random.uniform(2, 7),

        'max_pixels_w': 800,
        'max_pixels_h': 800,

        'xtick_label_rotation': rotation_x,
        'ytick_label_rotation': rotation_y,

        'aux_spine': random.random() >= 0.70,
        'custom_spine': random.random() >= 0.975,

        'x_grid': random.random() >= 0.90,
        'y_grid': random.random() >= 0.90,
        'x_minor_grid': random.random() >= 0.95,
        'y_minor_grid': random.random() >= 0.95,

        'add_legend': random.random() >= 0.9,

    }

    p = random.random()

    if p >= 0.95:  # wide
        params['width'] = random.uniform(8, 10)
        params['height'] = random.uniform(2, 4)
    elif p >= 0.90:  # tall
        params['width'] = random.uniform(4, 5)
        params['height'] = random.uniform(4, 6)
    else:
        pass

    if params['plt_style'] == 'dark_background':
        params['font_color'] = 'w'
        params['color'] = 'w'

    if params['custom_spine']:
        params['font_size'] = 8  # random.choice(SPINE_STYLES)

    if params['ytick_label_rotation'] > 45:
        params['font_size'] = 8

    if params['width'] <= 5:
        params['xtick_label_rotation'] = random.uniform(45, 90)

    if params['height'] <= 5:
        params['ytick_label_rotation'] = 0

    return params


#########################################################################################
# Main ---
#########################################################################################


class BasicScatterPlot:
    def __init__(self, cfg, the_example, texture_files=None, debug=False):
        self.cfg = cfg
        self.example = deepcopy(the_example)
        self.params = generate_random_params()
        self.debug = debug
        self.texture_files = texture_files

        # configure style & create figure ---
        self.configure_style()
        self.fig, self.ax = self.get_figure_handles()

        # axis ---
        self.x_axis_type = 'standard'
        self.y_axis_type = 'standard'

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
    # ticks
    # -----------------------------------------------------------------------------------#

    def configure_ticks(self):
        # tick parameters---
        direction = random.choice(['in', 'out', 'inout'])

        x_top = (random.random() >= 0.6) & (self.params['aux_spine'])
        x_labeltop = (random.random() >= 0.8) & (x_top)
        self.x_labeltop = x_labeltop

        y_right = (random.random() >= 0.6) & (self.params['aux_spine'])
        y_labelright = (random.random() >= 0.75) & (y_right)

        # tick formatter
        x_formatter = get_formatter(self.x_values)
        self.ax.xaxis.set_major_formatter(x_formatter)

        y_formatter = get_formatter(self.y_values)
        self.ax.yaxis.set_major_formatter(y_formatter)

        # set ticks
        # Set the y-axis ticks
        y_range = max(self.y_values) - min(self.y_values)
        if (y_range > 1e5) & (self.y_axis_type == 'standard'):
            num_bins = random.randint(4, 6)
            tick_locator = MaxNLocator(nbins=num_bins)
            self.ax.yaxis.set_major_locator(tick_locator)

        x_range = max(self.x_values) - min(self.x_values)
        if (x_range > 1e5) & (self.x_axis_type == 'standard'):
            num_bins = random.randint(4, 6)
            tick_locator = MaxNLocator(nbins=num_bins)
            self.ax.xaxis.set_major_locator(tick_locator)
            self.params['xtick_label_rotation'] = random.uniform(45, 90)

        if self.x_axis_type == 'log':
            tick_locator = LogLocator(base=self.x_log_base)
            self.ax.xaxis.set_major_locator(tick_locator)

        if self.y_axis_type == 'log':
            tick_locator = LogLocator(base=self.y_log_base)
            self.ax.yaxis.set_major_locator(tick_locator)

        # if random.random() > 0.5:
        self.ax.minorticks_on()

        # set tick params
        self.ax.tick_params(
            axis='x',
            which='both',
            rotation=self.params['xtick_label_rotation'],
            direction=direction,
            top=x_top,  # ---
            labeltop=x_labeltop,
        )

        if random.random() > 0.5:
            self.ax.xaxis.set_tick_params(which='minor', bottom=False, top=False)

        self.ax.tick_params(
            axis='y',
            which='both',
            rotation=self.params['ytick_label_rotation'],
            direction=direction,
            right=y_right,  # ---
            labelright=y_labelright,
        )

        if random.random() > 0.5:
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

    # -----------------------------------------------------------------------------------#
    # legend
    # -----------------------------------------------------------------------------------#

    def configure_legend(self):
        # --------------------------------
        font_dict = {
            'family': self.params['font_family'],
            'style': random.choice(['normal', 'italic']),
            'variant': random.choice(['normal', 'small-caps']),
            'weight': self.params['font_weight'],
            'stretch': random.choice(['normal', 'condensed', 'expanded']),
            'size': self.params['font_size'] - random.randint(2, 4),
        }

        if self.params['add_legend']:
            pos = random.choice(
                ['center left']
            )
            legend = self.ax.legend(prop=font_dict, loc=pos, bbox_to_anchor=(1, 0.5))

            # Adjust the legend bounding box
            legend.get_frame().set_facecolor('lightgray')  # set face color
            legend.get_frame().set_edgecolor(random.choice(['r', 'k', 'b']))      # set edge color
            legend.get_frame().set_linewidth(random.uniform(1.0, 2.0))            # set line width
            legend.get_frame().set_linestyle('-')         # set line style
            legend.get_frame().set_alpha(random.uniform(0.8, 1.0))              # set transparency

    # ------------------------------------------------------------------------------------#
    # titles
    # ------------------------------------------------------------------------------------#

    def configure_titles(self):
        if self.params['font_color'] == 'y':  # no yellow font color
            self.params['font_color'] = 'k'

        font_dict = {
            'family': self.params['font_family'],
            'color': self.params['font_color'],
            'weight': self.params['font_weight'],
            'size': self.params['font_size'],
        }

        axes_font_dict = deepcopy(font_dict)
        axes_font_dict['size'] = self.params['font_size'] + random.randint(0, 1)

        x_location = 'center'
        y_location = 'center'

        if random.random() >= 0.7:
            x_location = random.choice(['left', 'right', 'center'])
            y_location = random.choice(['top', 'bottom', 'center'])

        if (self.params['custom_spine']) & (not self.params['aux_spine']):
            x_location = random.choice(['left', 'right'])
            y_location = random.choice(['top', 'bottom'])

        if (random.random() >= 0.25) & (not self.params['custom_spine']):
            self.ax.set_xlabel(
                self.title_dict['x_title'],
                fontdict=axes_font_dict,
                loc=x_location,
                labelpad=random.uniform(4, 10)
            )

            self.ax.set_ylabel(
                self.title_dict['y_title'],
                fontdict=axes_font_dict,
                loc=y_location,
                labelpad=random.uniform(4, 10),
            )

        # figure title
        fig_width_inch = self.fig.get_figwidth()
        fig_dpi = self.fig.dpi
        fig_width_pixel = fig_width_inch * fig_dpi
        # We assume an average character width of 10 pixels
        char_width_pixel = 12
        num_chars = int(fig_width_pixel / char_width_pixel)

        # Set a long title and use textwrap to automatically wrap the text
        title = '\n'.join(textwrap.wrap(self.title_dict['plot_title'][:16], num_chars))
        title_font_dict = deepcopy(font_dict)
        title_font_dict['size'] = self.params['font_size'] + random.randint(-2, 2)

        title_loc = random.choice(['left', 'right', 'center'])

        if (self.params['custom_spine']) & (not self.params['aux_spine']):
            title_loc = random.choice(['left', 'right'])

        if self.x_labeltop:
            title_pad = - 32
        elif self.params['aux_spine']:
            title_pad = 24
        else:
            title_pad = random.randint(6, 24)
            if random.random() >= 0.8:
                title_pad = - title_pad

        if random.random() >= 0.5:
            self.ax.set_title(title, fontdict=title_font_dict, loc=title_loc, pad=title_pad)

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

        # set linewidth ---
        linewidth = random.uniform(1.0, 3.0)
        for spine in active_spines:
            self.ax.spines[spine].set_linewidth(linewidth)

        # set color ---
        color = random.choice(['r', 'g', 'b', 'k'])
        if random.random() >= 0.75:
            color = 'k'
        if self.params['plt_style'] == 'dark_background':
            color = 'w'

        for spine in active_spines:
            self.ax.spines[spine].set_color(color)

        # set spine positions ---
        picked_style = random.choice(SPINE_STYLES)

        if (self.params['custom_spine']) & (not self.params['aux_spine']):
            self.params['x_grid'] = False
            self.params['y_grid'] = False

            if picked_style == 'zero':
                if (np.amin(self.y_values) < 0) & (np.amax(self.y_values) > 0):
                    self.ax.spines['bottom'].set_position('zero')

                if (np.amin(self.x_values) < 0) & (np.amax(self.x_values) > 0):
                    if 'left' in active_spines:
                        self.ax.spines['left'].set_position('zero')

            if picked_style == 'center':
                self.ax.spines[active_spines].set_position('center')

        else:
            if random.random() >= 0.95:
                for spine in active_spines:
                    self.ax.spines[spine].set_position(('outward', 10))

    # -----------------------------------------------------------------------------------#
    # trend lines
    # -----------------------------------------------------------------------------------#

    def add_trend_line(self):
        # Add a trend line
        if random.random() >= 0.5:
            trend = np.polyfit(self.x_values, self.y_values, 1)
            trend_line = np.poly1d(trend)

            # Calculate R-squared value
            y_pred = trend_line(self.x_values)
            y_mean = np.mean(self.y_values)
            ss_total = np.sum((self.y_values - y_mean) ** 2)
            ss_residual = np.sum((self.y_values - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total)

            x_trend = deepcopy(self.x_values)
            if random.random() >= 0.5:
                start, end = random.randint(1, 2), len(x_trend) - random.randint(1, 2)
                if start >= len(x_trend)-2:
                    start = 0
                if end <= start:
                    end = start + 1
                x_trend = x_trend[start:end]

            if r_squared >= 0.85:
                self.ax.plot(
                    x_trend,
                    trend_line(x_trend),
                    linestyle='-',
                    linewidth=random.uniform(0.5, 1.5),
                    color=random.choice(COLOR_OPTIONS),
                    alpha=random.uniform(0.8, 1),
                )

                # Display the equation of the trend line using mathtext/latex
                if random.random() >= 0.75:
                    equation = r"$y = {:.1f}x + {:.1f}$".format(trend[0], trend[1])
                    self.ax.annotate(
                        equation,
                        xy=(random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)),
                        xycoords='axes fraction',
                        fontsize=random.randint(6, 10),
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=random.uniform(0.25, 0.75)),
                    )

    def add_trend_line_v1(self):
        # Add a trend line
        if random.random() >= 0.25:
            try:
                trend = np.polyfit(self.x_values, self.y_values, random.randint(2, 3))
            except Exception as e:
                print(e)
                trend = np.polyfit(self.x_values, self.y_values, 1)

            trend_line = np.poly1d(trend)

            # Calculate R-squared value
            y_pred = trend_line(self.x_values)
            y_mean = np.mean(self.y_values)
            ss_total = np.sum((self.y_values - y_mean) ** 2)
            ss_residual = np.sum((self.y_values - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total)

            x_trend = deepcopy(self.x_values)
            if random.random() >= 0.5:
                start, end = random.randint(1, 2), len(x_trend) - random.randint(1, 2)
                if start >= len(x_trend)-2:
                    start = 0
                if end <= start:
                    end = start + 1
                x_trend = x_trend[start:end]

            if r_squared >= 0.85:
                self.ax.plot(
                    x_trend,
                    trend_line(x_trend),
                    linestyle='-',
                    linewidth=random.uniform(0.5, 1.5),
                    color=random.choice(COLOR_OPTIONS),
                    alpha=random.uniform(0.8, 1),
                )
    # -----------------------------------------------------------------------------------#
    # axis transformation
    # -----------------------------------------------------------------------------------#

    def axis_transformation(self):
        activity = random.random() >= 0.5

        if (activity) & (is_log_scale_feasible(self.y_values)):
            self.y_axis_type = 'log'
            self.y_log_base = random.choice([2, 10])
            self.ax.set_yscale('log', base=self.y_log_base)
            print('log y axis')

        if (activity) & (is_log_scale_feasible(self.x_values)):
            self.x_axis_type = 'log'
            self.x_log_base = random.choice([2, 10])
            self.ax.set_xscale('log', base=self.x_log_base)
            print('log x axis')

        if self.y_axis_type == 'standard':
            if (random.random() >= 0.95) & (np.amin(self.y_values) * np.amax(self.y_values) > 0):
                self.ax.invert_yaxis()
                self.y_axis_type = 'inverted'
                print('inverted y axis')

    # -----------------------------------------------------------------------------------#
    # stats
    # -----------------------------------------------------------------------------------#

    def add_stat_info(self):
        if random.random() >= 0.95:
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
    # plot
    # -----------------------------------------------------------------------------------#

    def basic_plot(self):
        self.ax.scatter(
            self.x_values,
            self.y_values,
            c=self.params['color'],
            marker=self.params['marker'],
            s=self.params['marker_size'],
            label=add_legend_name(self.title_dict)[:random.randint(4, 10)],
            zorder=10,
        )

    def fancy_plot_v1(self):
        # Compute the point density
        xy = np.vstack([self.x_values, self.y_values])
        z = gaussian_kde(xy)(xy)

        scatter = self.ax.scatter(
            self.x_values,
            self.y_values,
            c=z,
            cmap='viridis',
            marker=self.params['marker'],
            s=self.params['marker_size'],
            edgecolor='black',
            label=add_legend_name(self.title_dict)[:random.randint(4, 10)],
            zorder=10,

        )

        # Creating a colorbar to indicate density
        if random.random() >= 0.9:
            cbar = plt.colorbar(scatter)
            cbar.set_label('Density')

    def fancy_plot_v2(self):
        markers = [
            r'$\clubsuit$',
            r'$\diamondsuit$',
            r'$\heartsuit$',
            r'$\spadesuit$',
            r'$\otimes$'
        ]

        marker = random.choice(markers)
        facecolor = random.choice(['none', 'black', 'red'])
        if self.params['plt_style'] == 'dark_background':
            facecolor = 'w'

        self.ax.scatter(
            self.x_values,
            self.y_values,
            c=self.params['color'],
            marker=marker,
            s=self.params['marker_size'],
            facecolor=facecolor,
            label=add_legend_name(self.title_dict)[:random.randint(4, 10)],
            zorder=10,
        )

    def fancy_plot_v3(self):
        colors = [random.choice(['r', 'g']) for _ in range(len(self.x_values))]
        sizes = [random.uniform(6.0, 12.0) for _ in range(len(self.x_values))]
        if self.params['plt_style'] == 'dark_background':
            colors = 'w'

        self.ax.scatter(
            self.x_values,
            self.y_values,
            c=colors,
            marker='o',
            s=sizes,
            label=add_legend_name(self.title_dict)[:random.randint(4, 10)],
            zorder=10,
        )

    def create_plot(self):
        plot_fn = random.choices(
            [self.basic_plot, self.fancy_plot_v1, self.fancy_plot_v2, self.fancy_plot_v3],
            # weights=[0.0, 0.0, 0.0, 1],
            weights=[0.50, 0.10, 0.30, 0.10],
            k=1,
        )[0]
        plot_fn()

    # -----------------------------------------------------------------------------------#
    # annotate point
    # -----------------------------------------------------------------------------------#

    def special_annotate(self):
        if (random.random() >= 0.97):
            n = random.randint(1, 2)
            color = random.choice(COLOR_OPTIONS)
            alpha = random.uniform(0.25, 0.5)
            num_points = len(self.x_values)
            fs = random.randint(4, 8)

            for i in range(n):
                try:
                    idx = random.randint(2, len(self.y_values) - 2)
                    if idx > num_points/2:
                        angle = 90
                    else:
                        angle = 270

                    offset = random.randint(30, 50)
                    xdata, ydata = round(self.x_values[idx], 4), round(self.y_values[idx], 4)

                    bbox = dict(boxstyle="round", fc=color, alpha=alpha)

                    arrowprops = dict(
                        arrowstyle="->",
                        connectionstyle=f"angle,angleA=0,angleB={angle},rad=10",
                        color=color,
                    )

                    self.ax.annotate(
                        f'{random.choice(RANDOM_LABELS)}',
                        (xdata, ydata),
                        xytext=(-2*offset, offset//2), textcoords='offset points',
                        bbox=bbox, arrowprops=arrowprops,
                        fontsize=fs,
                    )

                except ValueError:
                    pass

    def add_highest_lowest(self):
        if random.random() >= 0.95:
            # Find the index of the highest and lowest y value
            idx_max = np.argmax(self.y_values)
            idx_min = np.argmin(self.y_values)

            # Annotation styling
            bbox = dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)

            # Annotate the highest point
            t = random.choice(['Highest point', fake.word()])
            self.ax.annotate(t,
                             (self.x_values[idx_max], self.y_values[idx_max]),
                             textcoords="offset points",
                             xytext=(0, 10),
                             ha='center',
                             fontsize=8,
                             bbox=bbox)

            # Annotate the lowest point
            t = random.choice(['Highest point', fake.word()])
            self.ax.annotate(t,
                             (self.x_values[idx_min], self.y_values[idx_min]),
                             textcoords="offset points",
                             xytext=(0, -15),
                             ha='center',
                             fontsize=8,
                             bbox=bbox)

    # -----------------------------------------------------------------------------------#
    # annotations
    # -----------------------------------------------------------------------------------#

    def annotate_points(self):
        # Delta: Annotate each point with the corresponding category

        if random.random() >= 0.95:
            num_points = len(self.x_values)
            annotate_every = random.randint(1, 4)
            if num_points >= 24:
                annotate_every = 4

            l = random.randint(1, 4)
            cat_var = [generate_annotations(length=l) for i in range(num_points)]
            fs = random.randint(6, 8)

            for i, cat in enumerate(cat_var):
                if i % annotate_every == 0:
                    self.ax.annotate(
                        cat,
                        (self.x_values[i], self.y_values[i]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center',
                        fontsize=fs,
                    )

    def add_random_text(self):
        if random.random() >= 0.98:
            t2 = random.choice(RANDOM_LABELS)
            t3 = random.choice(RANDOM_LABELS)
            text = f'{t2} {t3}'
            text = "\n".join(textwrap.wrap(text, width=16))

            self.ax.text(
                random.uniform(0.1, 0.5), random.uniform(0.6, 0.9), text,
                fontsize=random.randint(10, 12),
                color=random.choice(COLOR_OPTIONS),
                alpha=random.uniform(0.5, 0.95),
                transform=self.ax.transAxes,
            )
    # -----------------------------------------------------------------------------------#
    # inset
    # -----------------------------------------------------------------------------------#

    def add_inset(self):
        if random.random() >= 0.98:
            # create an inset axes for a zoomed in plot
            w = random.uniform(0.1, 0.2)
            a, b = random.uniform(0.1, 0.8), random.uniform(0.1, 0.8)
            axins = self.ax.inset_axes([a, b, w, w])
            axins.plot(self.x_values, self.y_values, color='blue', linewidth=1)
            axins.set_xticklabels('')
            axins.set_yticklabels('')

    # -----------------------------------------------------------------------------------#
    # fill vertical region
    # -----------------------------------------------------------------------------------#

    def fill_vertical_region(self):
        # Define the threshold value
        # print('filling vertical region')
        if random.random() >= 0.98:
            color = random.choice(COLOR_OPTIONS)
            threshold = np.percentile(self.y_values, random.randint(20, 80))

            if random.random() >= 0.9:
                self.ax.axhline(
                    threshold,
                    color=color,
                    lw=random.uniform(0.5, 2.5),
                    alpha=random.uniform(0.5, 0.75)
                )

            # x_ = np.arange(len(self.y_values))
            self.ax.fill_between(
                self.x_values, 0, 1,
                where=self.y_values > threshold,
                color=color,
                alpha=random.uniform(0.25, 0.5),
                transform=self.ax.get_xaxis_transform(),
            )

    # -----------------------------------------------------------------------------------#
    # main api
    # -----------------------------------------------------------------------------------#

    def make_scatter_plot(self, graph_id):
        try:
            the_example = deepcopy(self.example)
            title_dict, data_series = parse_example(the_example)

            self.title_dict = title_dict
            self.x_values = data_series['x_values']
            self.y_values = data_series['y_values']

            if len(self.x_values) >= 24:
                self.params['marker_size'] = random.randint(6, 10)

            # create the plot ---
            self.create_plot()
            # self.axis_transformation() # no log transformation
            self.configure_spine()
            self.configure_ticks()
            self.configure_gridlines()
            self.configure_legend()

            self.configure_titles()

            # special effects ---
            self.add_trend_line()
            # self.add_trend_line_v1()
            self.annotate_points()
            self.special_annotate()
            self.add_random_text()
            self.add_stat_info()
            self.add_highest_lowest()
            self.add_inset()
            self.fill_vertical_region()

            # sns.despine(left=True, bottom=True)
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

                if random.random() <= 0.1:
                    # Save the image as a Grayscale
                    if self.debug:
                        print('converting to grayscale')
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
