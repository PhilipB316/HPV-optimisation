# Copyright (C) 2026 Philip Brand
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import matplotlib.pyplot as plt

SERIES_META = {
    'calc_speeds': dict(axis='speed', unit='km/h', label='Simulated Speed'),
    'speeds':      dict(axis='speed', unit='km/h', label='Measured Speed'),
    'watts':       dict(axis='power', unit='W', label='Power'),
}

DEFAULT_SERIES = ['calc_speeds', 'speeds', 'watts']

DATASET_STYLES = [
    {'calc_speeds': dict(color='orange', linestyle='-'),
     'speeds':      dict(color='red', linestyle='--', alpha=0.9),
     'watts':       dict(color='purple', linestyle='-')},
    {'calc_speeds': dict(color='green', linestyle='-'),
     'speeds':      dict(color='pink', linestyle='--', alpha=0.9),
     'watts':       dict(color='blue', linestyle='-')},
]

def paper_style():
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
        "axes.spines.top": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 27,
    })


def _plot_series(ax1, ax2, run, series, style, labels, suffix):
    """Plot the requested series from one run. Returns the Line2D handles."""
    times = run.times
    lines = []

    for key in series:
        values = getattr(run, key, None)
        if values is None:
            continue
        if key not in SERIES_META:
            raise ValueError(f"Unknown series '{key}'. Known series: {list(SERIES_META)}")

        meta = SERIES_META[key]

        if key == 'watts':
            values = list(values)
            values[0] = 0.0

        x = times[:len(values)]
        y = values

        ax = ax1 if meta['axis'] == 'speed' else ax2
        label_text = labels.get(key) or meta['label'] + suffix
        label = f"{label_text} (Max: {max(y):.2f} {meta['unit']})"
        line, = ax.plot(x, y, label=label, **style.get(key, {}))
        lines.append(line)

    return lines


def plot(*runs, series=None, labels=None, show=True, savefig=None):
    """Plot one or more runs on shared axes, speed on the left and power on the right.

    Args:
        *runs: Run objects to plot.
        series: keys to plot from every run, or a list of lists, one per run.
            Keys are "watts", "speeds" and "calc_speeds". Defaults to all
            that each run has.
        labels: list of dicts, one per run, mapping series key to legend text.
        show: call plt.show().
        savefig: file path. A .pdf and a .png are both written.

    Returns:
        The left matplotlib Axes.

    Example:
        hpvopt.plot(optimised, measured,
                    series=[["watts"], ["watts", "calc_speeds"]],
                    labels=[{"watts": "Optimised Power"}, {"watts": "Measured Power"}])
    """

    series = list(DEFAULT_SERIES) if series is None else series
    if isinstance(series[0], str):
        series = [series] * len(runs)
    labels = labels or [{}] * len(runs)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Speed (km/h)')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Power (W)')

    lines = []
    for i, run in enumerate(runs):
        suffix = f" ({run.label})" if len(runs) > 1 else ""
        lines += _plot_series(ax1, ax2, run, series[i],
                              DATASET_STYLES[i % len(DATASET_STYLES)], labels[i], suffix)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    ax1.legend(lines, [line.get_label() for line in lines],
               framealpha=.81, loc="lower right")
    ax1.grid(True, which='both', axis='both')
    ax1.minorticks_on()

    fig.tight_layout()
    if savefig:
        root = os.path.splitext(savefig)[0]
        plt.savefig(root + '.pdf')
        plt.savefig(root + '.png')

    if show:
        plt.show()
    return ax1

