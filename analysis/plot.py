import matplotlib.pyplot as plt
import numpy as np

from devices.oscilloscope import MeasOsc


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
