import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from devices.oscilloscope import MeasOsc
from meas import MeasCal


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
    ax.plot(meas.t, meas.voltage, label="data")
    peak_ind = np.argmax(meas.voltage)
    ax.hlines(meas.zero_level, meas.t[0], meas.t[peak_ind], label="zero level", colors=["r"])
    ax.scatter(meas.t[peak_ind], meas.voltage[peak_ind], label="peak", c="r")

    args = meas.exp_decay_fit
    exp_fit = args[0]*np.exp(args[1]*meas.t[peak_ind:])
    ax.plot(meas.t[peak_ind:], exp_fit, label="exp fit")
    ax.legend()


def legend_multi(ax: plt.Axes, lines):
    """Create a legend for a multi-axis plot"""
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)
