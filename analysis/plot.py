import matplotlib.pyplot as plt
import numpy as np

from devices.oscilloscope import MeasOsc


def plot_osc(meas: MeasOsc, ax: plt.Axes):
    ax.plot(meas.t, meas.voltage)
    peak_ind = np.argmax(meas.voltage)
    ax.hlines(meas.zero_level, meas.t[0], meas.t[peak_ind], label="zero level")
    ax.scatter(meas.t[peak_ind], meas.voltage[peak_ind], label="peak")

    args = meas.exp_decay_fit
    exp_fit = np.exp(args[0]*meas.t[peak_ind:]) + args[1]
    ax.plot(meas.t[peak_ind:], exp_fit[0]*np.exp(exp_fit[1]*meas.t[peak_ind:]), label="exp fit")
    ax.legend()
