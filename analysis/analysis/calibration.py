import typing as tp

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import plot
from meas import Meas, MeasCal


def calibration(
        cal_data: tp.List[MeasCal],
        # coarse_gain: float = 10,
        # fine_gain: float = 10,
        preamp_capacitance: float = 1e-12) -> np.ndarray:
    """Analyze calibration data"""

    # Oscilloscope example
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    meas = cal_data[0]
    osc = meas.traces[0]
    plot.plot_osc(osc, ax)
    plot.save_fig(fig, "calibration_trace")

    set_voltages = np.array([meas.voltage for meas in cal_data])
    peak_data = np.array([meas.peak_height for meas in cal_data])
    peak_heights = peak_data[:, 0]
    peak_stds = peak_data[:, 1]
    # gain = coarse_gain*fine_gain
    charges = preamp_capacitance * peak_heights
    charges_std = preamp_capacitance * peak_stds

    #####
    # Pulser calibration
    #####
    fig2: plt.Figure = plt.figure()
    # fig2.suptitle("Pulser calibration")
    ax: plt.Axes = fig2.add_subplot()

    ax.errorbar(set_voltages, peak_heights, yerr=peak_stds, fmt=".", capsize=3, label="data")
    # This transforms the scale to a more reasonable one for the fitting algorithm and therefore
    # reduces errors.
    fit = curve_fit(
        lambda x, a, b: a*x + b,
        xdata=set_voltages,
        ydata=peak_heights,
        sigma=peak_stds,
    )
    coeff = np.array([fit[0][0], fit[0][1]])
    coeff_stds = np.array([fit[1][0, 0], fit[1][1, 1]])
    ax.plot(
        set_voltages,
        np.polyval(coeff, set_voltages),
        label=f"fit (y = {coeff[0]:.3e}±{coeff_stds[1]:.3e}x + {coeff[1]:.3e}±{coeff_stds[1]:.3e})"
    )
    ax.set_xlabel("Pulser voltage setting (V)")
    ax.set_ylabel("Pulse height (V)")
    ax.legend()
    plot.save_fig(fig2, "pulser_calibration")

    plot.plot_failed_cals(cal_data)

    #####
    # MCA calibration
    #####
    mca_peak_inds = np.array([np.argmax(cal.mca.data) for cal in cal_data])

    fig3: plt.Figure = plt.figure()
    # fig3.suptitle("MCA calibration")
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax3: plt.Axes = fig3.add_subplot(gs[0])
    ax4: plt.Axes = fig3.add_subplot(gs[1])

    y_mult = 1e12
    ax3.errorbar(
        mca_peak_inds,
        charges*y_mult,
        yerr=charges_std*y_mult,
        fmt=".", capsize=3, label="data", color="black"
    )
    fit = curve_fit(
        lambda x, a, b: a*x + b,
        xdata=mca_peak_inds,
        ydata=charges,
        sigma=charges_std,
    )
    coeff = np.array([fit[0][0], fit[0][1]])
    coeff_stds = np.array([fit[1][0, 0], fit[1][1, 1]])
    ax3.plot(
        mca_peak_inds,
        np.polyval(coeff, mca_peak_inds) * y_mult,
        label=f"fit (y = {coeff[0]*y_mult:.3e}±{coeff_stds[1]*y_mult:.3e}x + {coeff[1]*y_mult:.3e}±{coeff_stds[1]*y_mult:.3e})",
        color="tab:blue"
    )
    # ax3.set_xlabel("MCA channel")
    ax3.set_ylabel("Collected charge (pC)")
    ax3.legend()

    for meas in cal_data:
        ax4.plot(meas.mca.channels, meas.mca.data, color="tab:blue")

    for ax in (ax3, ax4):
        ax.set_xlim(0, cal_data[0].mca.channels[-1])

    ax4.set_xlabel("MCA channel")
    ax4.set_ylabel("Counts")
    plot.save_fig(fig3, "mca_calibration")

    return coeff
