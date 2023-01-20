from abc import abstractmethod
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import quantile_transform, minmax_scale
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

FIG_HEIGHT = 1000
FIG_WIDTH = 3000
TIMEZONE = 'Europe/Zurich'


class _BaseIndicator(object):
    """base indicator class to calc some data and save it to ArcticCloud"""

    def __init__(self, sec, debug=False):
        self.debug = debug
        self.sec = sec
        self.data = None
        self.start_year = 2000
        self.floor = True
        self.floor_level = 0.01

    def run(self):
        self._run()
        self.apply_floor()
        self.chart()

    def apply_floor(self):
        """floors all indicators, so it is not perfectly zero, and looks nicer"""
        if self.floor:
            self.data = self.data.clip(lower=self.floor_level)

    @abstractmethod
    def _run(self):
        raise Exception('not implemented error')

    def chart(self, window=750):
        """charts the resulting series"""
        chrt = ChartPresets()
        fig = chrt.size_presets('16:9')
        ax = fig.add_subplot(111)
        self.data.tail(window).plot(ax=ax, lw=1)
        chrt.despine_default(ax)
        plt.tight_layout()


def resample_daily(data, type_agg='last', prune=True, dropna=True):
    """resamples to daily data
    :param data: DataFrame or Series
    :param type_agg: last, sum
    :param prune: if True, prune NaN"""
    obj = data.tz_convert(TIMEZONE).resample('B')
    out = getattr(obj, type_agg)()
    if prune:
        out = out.replace(0, np.NaN)
        if dropna:
            out = out.dropna()
    return out


def resample_daily(data, type_agg='last', prune=True, dropna=True):
    """resamples to daily data
    :param data: DataFrame or Series
    :param type_agg: last, sum
    :param prune: if True, prune NaN"""
    obj = data.tz_convert(TIMEZONE).resample('B')
    out = getattr(obj, type_agg)()
    if prune:
        out = out.replace(0, np.NaN)
        if dropna:
            out = out.dropna()
    return out


def quantile_transformation(data, smooth=10):
    data = DataFrame(data)
    q = quantile_transform(data)
    q = DataFrame(q, index=data.index, columns=data.columns)
    # clip and rescale
    f = 0.95
    data = q.clip(upper=f) / f
    # smooth
    return data.ewm(span=smooth).mean()


def rescale(data, preserve_zero=True, smooth=10):
    data = DataFrame(data)
    if preserve_zero:
        a = data[data < 0]
        a = DataFrame(minmax_scale(a, (0.01, 0.5)), index=data.index, columns=data.columns)
        b = data[data >= 0]
        b = DataFrame(minmax_scale(b, (0.5, 0.99)), index=data.index, columns=data.columns)
        out = pd.concat([a, b], axis=0).sort_index()
    else:
        out = minmax_scale(data)
        out = DataFrame(out, index=data.index, columns=data.columns)
    # smooth
    return out.ewm(span=smooth).mean()


class ChartPresets(object):
    """ default values for plotting as well as default styles and elements used charts with a consistent look & feel """

    def __init__(self, preset='print'):
        self.cmap = None
        self.cmap_r = None
        self.cmap_red = None
        self.cmap_red_r = None
        self.cmap_violet = None
        self.cmap_violet_r = None
        self.cmap_brown = None
        self.cmap_brown_r = None
        self.cmap_div = None
        self.cmap_div_r = None
        self.cmap_div_violet = None
        self.cmap_div_violet_r = None
        self.cmap_primary = None
        self.cmap_primary_r = None
        self.preset = preset
        self.scaling_factor = 1.18
        self.colors = None  # all colors
        self._primary = None
        self._secondary = None
        self.colors_aionite = None  # specific Aionite Capital Systematic Macro
        self.colors_bm = None
        self.colors_bm_multiple = None
        self._rc_params()
        self._set_color()
        # self.db = ArcticWrapper()

    def _rc_params(self):
        """ preset
        use print/presentation """
        if self.preset.upper() == 'PRINT':
            font_size_title = 12
            font_size_subtitle = 11
            font_size_main = 10
            self.scaling_factor = 1

        elif self.preset.upper() == 'PRESENTATION':
            font_size_title = 16
            font_size_subtitle = 14
            font_size_main = 12
            self.factor = 1.18

        elif self.preset.upper() == 'FACTSHEET':
            font_size_title = 10
            font_size_subtitle = 8
            font_size_main = 8
            self.scaling_factor = 1

        elif self.preset.upper() == 'TWO_PAGER':
            font_size_title = 8
            font_size_subtitle = 8
            font_size_main = 8
            self.scaling_factor = 1

        elif self.preset.upper() in ['SLIDEUMENT', 'PITCHBOOK']:
            font_size_title = 12
            font_size_subtitle = 10
            font_size_main = 10
            self.scaling_factor = 1.18
        else:
            raise ValueError('preset not known')
        dic = {'font.family': 'Montserrat',
               'font.serif': 'Montserrat',
               'font.monospace': 'Inconsolata',
               'font.weight': 'regular',
               'font.size': font_size_main,
               'axes.titlesize': font_size_subtitle,
               'axes.titleweight': 'regular',
               'axes.labelsize': font_size_main,
               'axes.labelweight': 'regular',
               'axes.grid': False,
               'axes.labelcolor': 'k',
               # 'axes.prop_cycle': cycler('color', ['r', 'grey', 'k']),
               'axes.linewidth': 0.5,
               'axes.facecolor': 'none',
               'axes.edgecolor': 'k',
               'xtick.major.width': 0,
               'ytick.major.width': 0,
               'ytick.labelsize': font_size_main,
               'xtick.labelsize': font_size_main,
               'legend.fontsize': font_size_main,
               'figure.titlesize': font_size_title,
               'figure.edgecolor': 'w',
               'figure.facecolor': 'w',
               'figure.dpi': 100,
               'savefig.dpi': 300,
               'lines.linewidth': 1.5,
               'legend.frameon': False
               }
        plt.rcParams.update(dic)

    @staticmethod
    def set_suptitle(title, offset=0.93):
        """ somehow uses different defaults """
        plt.suptitle(title,
                     size=plt.rcParams['axes.titlesize'],
                     weight=plt.rcParams['axes.titleweight'],
                     name=plt.rcParams['font.family'])
        plt.tight_layout()
        plt.subplots_adjust(top=offset)

    def _set_color(self):
        """ define colors for all charts
          o color list
          o main color for benchmark data
          o color map
        """
        primary = [(18, 71, 71), (217, 198, 185), (243, 198, 153), (147, 99, 38)]
        # red, violet, dark_gray, light gray
        secondary = [(255, 78, 0), (114, 0, 25), (53, 53, 53), (193, 191, 180)]
        self._secondary = secondary
        self._primary = primary
        # convert all to 0 to 1
        primary = [self._normalize_rgb(c) for c in primary]
        secondary = [self._normalize_rgb(c) for c in secondary]

        # cmaps: here we need RGB->A to add opacy
        self.cmap, self.cmap_r = self.make_cmap_from_rgb(primary[0], add_white=True)
        self.cmap_red, self.cmap_red_r = self.make_cmap_from_rgb(secondary[0])
        self.cmap_violet, self.cmap_violet_r = self.make_cmap_from_rgb(secondary[1])
        self.cmap_brown, self.cmap_brown_r = self.make_cmap_from_rgb(primary[3])
        self.cmap_div, self.cmap_div_r = self.make_cmap_from_rgb(primary[0], primary[3])
        self.cmap_div_violet, self.cmap_div_violet_r = self.make_cmap_from_rgb(primary[0], secondary[1])
        self.cmap_primary, self.cmap_primary_r = self.make_cmap_from_rgb(primary[0], primary[1])

        main_colors = [primary[0],  # green
                       secondary[0],  # orange
                       secondary[1],  # violet
                       secondary[2],  # dark gray
                       primary[-1],  # brown
                       secondary[3]]  # light gray

        sns.set_palette(main_colors)
        self.colors = main_colors
        self.colors_bm = [secondary[0]]
        self.colors_bm_multiple = secondary + [primary[-1]]
        self.colors_aionite = [primary[0], primary[3], primary[1], primary[2]]

    @staticmethod
    def _normalize_rgb(rgb):
        """ normalizes RGB values to 0 and 1 """
        assert isinstance(rgb, tuple), 'provide RGB tuple'
        rgb_new = [c_ / 255. for c_ in rgb[:3]]
        # if RGBA, add opacy
        if len(rgb) > 3:
            rgb_new.append(rgb[-1])
        return tuple(rgb_new)

    @staticmethod
    def _generate_cmap(color_list):
        """ useful helper for experimenting """
        cmap = ListedColormap(color_list)
        cmap_r = ListedColormap(color_list[::-1])
        return cmap, cmap_r

    def make_cmap_from_rgb(self, c1, c2=None, steps=5, add_white=False):
        """ makes a cmap from two colors (needs rgb as input)
        :type c1: tuple
        :type c2: tuple | None
        :type steps: int
        :type add_white: bool
        """
        # divide by 255. to lie between 0 and 1
        # c1 = self._normalize_rgb(c1_)
        lst_a = [c1 + (f,) for f in np.linspace(0.2, 1, steps)]
        if add_white:
            lst_a.append((1, 1, 1))  # normalized values! 255 / 255 = 1

        if c2 is not None:
            # c2 = self._normalize_rgb(c2_)
            lst_b = [c2 + (f,) for f in np.linspace(0.2, 1, steps)]
            lst = lst_a[::-1] + lst_b
        else:
            lst = lst_a
        return self._generate_cmap(lst)

    def _dummy_plot(self):
        """ generate dummy data and plot chart for quick check of settings """
        df = DataFrame(np.random.normal(0.03 / 250., 0.055 / 16., [100, 3]))
        df.columns = ['A', 'B', 'C']
        df.index = pd.date_range('2016-01-01', freq='B', periods=len(df))
        self.df = df

        fig, ax = plt.subplots(1, 1)
        self.df.cumsum().plot(ax=ax)
        self.despine_default(ax)
        self.pretty_legend(ax)
        plt.tight_layout(pad=0.5)

    @staticmethod
    def despine_default(ax=None):
        """ apply default styling to (new) ax """
        if ax is None:
            ax = plt.gca()

        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True, offset=10)

    @staticmethod
    def add_text(ax, txt, x, y, box=None, alpha=None):
        """ add text to ax
        :param ax: Axis
        :param x: coordinates
        :param y: coordinates
        :param txt: Text to print
        :param box: use True to plot box around text
        :param alpha: only used with box. sets the transparency
        :type ax: mpl.axes._subplots.Axes
        """
        if box is None:
            ax.text(x, y, txt, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
        else:
            if alpha is None:
                alpha = 0.5
            ax.text(x, y, txt, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, bbox=dict(facecolor='grey', alpha=alpha))

    @staticmethod
    def format_yticks_pct(ax):
        """ formats yticks to percentage
        :type ax: mpl.axes._subplots.Axes
        """
        x = ax.get_yticks() * 100
        lbl = ['%.2f%%' % a for a in x]
        ax.set_yticklabels(lbl)

    @staticmethod
    def add_line(ax, x, y, s, horizont, is_title=False):
        """ add a new text line to a chart (ax)
        :param ax:
        :param x: list of floats
        :param y: scalar
        :param s: list of string, same dimension als x
        :param horizont: list of strings with alignment-settings
        :type x: list[float]
        :type y: float
        :type s: list[str]
        :type horizont: list[str]
        :type is_title: bool
        """
        if y < 0:
            raise ValueError('visible range between 0/1. y value negative')
        assert len(x) == len(s), 'Dimension of <x> and <s> has to be the same'

        for i, s_ in enumerate(s):
            if is_title:
                size = plt.rcParams['axes.titlesize']
                weight = plt.rcParams['font.weight']
            else:
                size = plt.rcParams['font.size']
                weight = plt.rcParams['font.weight']
            ax.text(x[i], y, s_, horizontalalignment=horizont[i], verticalalignment='top',
                    size=size, weight=weight)

    @staticmethod
    def make_semi_transparent_legend(ax, color='white', alpha=0.85):
        """ makes the legend transparent with white """
        legend = ax.get_legend()
        legend.set_frame_on(True)
        frame = legend.get_frame()
        frame.set_facecolor(color)
        frame.set_edgecolor(color)
        frame.set_alpha(alpha)

    def size_presets(self, preset, orientation='landscape', custom_size=None):
        """
        A4    210x297mm, 2480x3508 pixel @ 300dpi
        A5    148x210mm, 1748x2480 pixel @ 300dpi
        16:9             1350x2400 pixel @ 300dpi
        slide_wide_01  widest chart that uses the full page
        slide_wide_02  2:1 split of the page, bigger piece
        slide_wide_03  2:1 split of the page, smaller piece
        slide_wide_04  3:1 split in 3 thirds
        orientation   portrait/landscape
        """
        h = 1209  # hight

        if custom_size is not None:
            a, b = custom_size
        else:
            if preset.upper() == 'A4':
                a = 2480.
                b = 3508.
            elif preset.upper() == 'A5':
                a = 1748.
                b = 2480
            elif preset.upper() == '16:9':
                a = 1350
                b = 2400
            elif preset.lower() == 'widescreen':
                # used in monitoring. fills the screen
                a = 1440 * 2
                b = 2560 * 2
            elif preset == 'slide_wide':  # used for 1 wide chart on a slide
                a = h
                b = 3100  # 2921
            elif preset == 'slide_2to1_big':  # wide chart using 2:1 space, leaving space for a comment
                a = h
                b = 1880
            elif preset == 'slide_2to1_small':
                a = h
                b = 1160
            elif preset == 'slide_3x':  # used for 3 charts on a slide
                a = h
                b = 990
            elif preset == 'slide_split':  # two equal charts on one page
                a = h
                b = 1520
            else:
                raise ValueError('preset unknown: %s' % preset)
        a *= self.scaling_factor
        b *= self.scaling_factor
        fig = plt.figure()
        dpi = float(plt.rcParams['savefig.dpi'])
        if orientation == 'portrait':
            fig.set_size_inches(a / dpi, b / dpi, forward=True)
        elif orientation == 'landscape':
            fig.set_size_inches(b / dpi, a / dpi, forward=True)
        else:
            raise ValueError('orientation unknown')
        return fig

    def get_chart(self, height_factor=1):
        """"returns a chart with preset size, used mostly in notebooks. aspect ratio is 3:1"""
        fig = self.size_presets(None, custom_size=(FIG_HEIGHT * height_factor, FIG_WIDTH))
        ax = fig.add_subplot(111)
        return fig, ax


    @staticmethod
    def set_log_axis(ax):
        """sets the yaxis to log scale"""
        # ymin, ymax = ax.get_ylim()
        yticks = ax.get_yticks()
        ax.set_yscale('log')
        # use same ticks as before
        ax.set_yticks(yticks)
        # use normal number formatting
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # make sure minor ticks are off
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        # keep min/max as before
        # ax.set_ylim((ymin, ymax))
        ax.autoscale(axis='y')

    @staticmethod
    def append_yticks_log(ax, data, base=100):
        """adds 100 as baseline. also, if negative numbers are present, add those """
        # add 100 to yticks
        tks = sorted(filter(lambda x: x > 0, ax.get_yticks()))
        if base not in tks:
            tks.append(base)
        # remove all below 100
        tks = sorted(filter(lambda x: x >= base, tks))
        # check for negative
        v = DataFrame(data).min().min()
        assert v > 0, 'wrong input data! minimum value is below zero: %0.2f' % v
        for threshold in [0.75, 0.5]:
            if v < threshold * base:
                tks.append(threshold * base)

        tks = sorted(tks)
        ax.set_yticks(tks)

    @staticmethod
    def set_invisible(ax):
        """ hide all chart elements.
        useful for text/table output """
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

    @staticmethod
    def pos_vertical(lst_ax, offset, top, bottom, left, right):
        """ sets the position for each ax in lst_ax
        offset  spacing between charts """
        n = float(len(lst_ax))
        height = (top - bottom - offset * (n + 1)) / n
        if height < 0:
            raise ValueError('height below 0. check input')
        y = top
        for ax in lst_ax:
            y += -height - offset
            ax.set_position([left, y, 1 - left - right, height])
            if y < 0:
                raise ValueError('y-Coordinate is below 0. check input')
        plt.draw()

    @staticmethod
    def chart_label(ax, title=None, xlabel=None, ylabel=None, legend_loc=None, show_zero=False):
        """set labels and title in one go and apply defaults """
        if show_zero:
            ax.axhline(linestyle=':', color=plt.rcParams['axes.edgecolor'], linewidth=plt.rcParams['axes.linewidth'])

        if legend_loc is not None:
            ax.legend(loc=legend_loc)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel, rotation=0)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    @staticmethod
    def bar_add_label(ax, fmt=None, inside=True, color='k', h=None, size=None):
        """add description label on top of each bar
        :param ax: axes
        :param fmt: format to be used. i.e. 0.4f,
        :param inside: if True, plots the number inside the bar
        :param color: set color here
        :param h: custom height offset
        :param size: font size
        """

        if fmt is None:
            fmt = '%d'

        if h is None:
            if inside:
                h = -0.03
            else:
                h = 0.03

        rects = ax.patches
        for rect in rects:
            height = rect.get_height()
            if height < 0:
                h_ = abs(h)
            else:
                h_ = h
            ax.text(rect.get_x() + rect.get_width() / 2., height + h_, fmt % height, ha='center', va='bottom',
                    color=color, size=size)

    @staticmethod
    def barh_add_label(ax, fmt=None, inside=True, color='k', w=None, size=None):
        """add description label next to each bar
        :param ax: axes
        :param fmt: format to be used. i.e. 0.4f,
        :param inside: if True, plots the number inside the bar
        :param color: set color here
        :param w: custom width offset
        :param size: font size
        """

        if fmt is None:
            fmt = '%d'

        if w is None:
            if inside:
                w = -0.03
            else:
                w = 0.03

        rects = ax.patches
        for i, rect in enumerate(rects):
            width = rect.get_width()
            height = i
            if width < 0:
                alignment = 'left'
                if inside:
                    w_ = abs(w)
                else:
                    w_ = -abs(w)
            else:
                alignment = 'right'
                if inside:
                    w_ = -abs(w)
                else:
                    w_ = abs(w)

            ax.text(width + w_, height, fmt % width, ha=alignment, va='center',
                    color=color, size=size)

    @staticmethod
    def pretty_legend(ax, loc=None, handlelength=2, linewidth=3, **kwargs):
        """format the legend: same color of text as lines
        https://stackoverflow.com/questions/13828246/matplotlib-color-coded-text-in-legend-instead-of-a-line
        """
        if loc is None:
            loc = 'best'

        leg = ax.legend(loc=loc, handlelength=handlelength, **kwargs)
        for l in leg.legendHandles:
            l.set_linewidth(linewidth)

        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())

    @staticmethod
    def set_rotation(ax, rotation):
        plt.sca(ax)
        plt.xticks(rotation=rotation)



def lighten_color(color, amount=0.5):
    """ mix with white
    :param color: normalized color
    :param amount: 1 -> color, 0 -> white """
    white = 1  # using normalize value
    return tuple([c * amount + (1 - amount) * white for c in color])

