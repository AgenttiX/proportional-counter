import os.path
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from devices.oscilloscope import MeasOsc
from meas import MeasCal

FIG_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report", "fig", "python")


def double_x_axis(ax: plt.Axes, tick_locs: np.ndarray, tick_label_func: callable) -> plt.Axes:
    """
    Add a secondary x axis to an existing Matplotlib [sub]figure

    Based on
    https://stackoverflow.com/questions/31803817/how-to-add-second-x-axis-at-the-bottom-of-the-first-one-in-matplotlib
    """
    ax2: plt.Axes = ax.twiny()
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.15))
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.spines["bottom"].set_visible(True)
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels(tick_label_func(tick_locs))
    return ax2


def legend_multi(ax: plt.Axes, lines):
    """Create a legend for a multi-axis plot"""
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)


def plot_failed_cals(cal_data: tp.List[MeasCal]):
    """Detection and plotting of failed calibration measurements"""
    failed_meas: tp.List[MeasCal] = []
    for meas in cal_data:
        if meas.peak_height[1] > 0.01:
            failed_meas.append(meas)
    if failed_meas:
        fig: plt.Figure = plt.figure()
        fig.suptitle("These calibration measurements have bad traces. Please fix!")
        for i, meas in enumerate(failed_meas):
            ax = fig.add_subplot(len(failed_meas), 1, i+1)
            meas.plot_traces(ax)
            ax.set_title(f"V = {meas.voltage:.2f} V")


def plot_osc(meas: MeasOsc, ax: plt.Axes):
    """Plot an oscilloscope measurement"""
    x_mult = 1e6
    y_mult = 1e3
    ax.plot(meas.t*x_mult, meas.voltage*y_mult, label="data")
    peak_ind = np.argmax(meas.voltage)
    ax.hlines(
        meas.zero_level*y_mult,
        meas.t[0]*x_mult, meas.t[peak_ind]*x_mult,
        label="zero level", colors=["r"], zorder=10)
    ax.scatter(meas.t[peak_ind]*x_mult, meas.voltage[peak_ind]*y_mult, label="peak", c="r", zorder=10)

    args = meas.exp_decay_fit
    exp_fit = args[0]*np.exp(args[1]*meas.t[peak_ind:])
    ax.plot(meas.t[peak_ind:]*x_mult, exp_fit*y_mult, label="exp fit")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend()


def save_fig(fig: plt.Figure, name: str):
    """Save a figure in multiple formats"""
    fig.savefig(os.path.join(FIG_FOLDER, f"{name}.eps"))
    fig.savefig(os.path.join(FIG_FOLDER, f"{name}.png"))
    fig.savefig(os.path.join(FIG_FOLDER, f"{name}.svg"))
