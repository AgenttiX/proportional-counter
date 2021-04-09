import typing as tp

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.odr
from scipy.optimize import curve_fit

import plot
from meas import MeasCal
import utils


def calibration(
        cal_data: tp.List[MeasCal],
        mca_diff_nonlin: float,
        pulser_voltage_rel_std: float,
        # coarse_gain: float = 10,
        # fine_gain: float = 10,
        preamp_capacitance: float = 1e-12,
        fig_titles: bool = True
        ) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Analyze calibration data"""
    utils.print_title("Calibration")

    # Oscilloscope example
    fig: plt.Figure = plt.figure()
    if fig_titles:
        fig.suptitle("Oscilloscope trace")
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
    if fig_titles:
        fig2.suptitle("Pulser calibration")
    ax: plt.Axes = fig2.add_subplot()

    ax.errorbar(
        set_voltages,
        peak_heights,
        xerr=set_voltages * pulser_voltage_rel_std,
        yerr=peak_stds,
        fmt=".", capsize=3, label="data"
    )
    fit = curve_fit(
        lambda x, a, b: a*x + b,
        xdata=set_voltages,
        ydata=peak_heights,
        sigma=peak_stds,
    )
    print("Pulser calibration fit: least squares")
    print(fit[0])
    print(fit[1])
    model = scipy.odr.Model(lambda coeff, x: coeff[0]*x + coeff[1])
    data = scipy.odr.RealData(
        x=set_voltages,
        y=peak_heights,
        sx=set_voltages * pulser_voltage_rel_std,
        sy=peak_stds
    )
    odr = scipy.odr.ODR(data, model, beta0=fit[0])
    out = odr.run()
    print("Pulser calibration fit: ODR")
    print(out.pprint())
    # ODR results in a numerical error, so its results are not used
    # coeff = out.beta
    # coeff_covar = out.cov_beta
    # coeff_stds = out.sd_beta
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
    # Prevent accidental use of old variables
    # del fit, coeff, coeff_stds, data, out

    #####
    # MCA calibration
    #####
    mca_peak_inds = np.array([np.argmax(cal.mca.counts) for cal in cal_data])

    fig3: plt.Figure = plt.figure()
    if fig_titles:
        fig3.suptitle("MCA calibration")
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax3: plt.Axes = fig3.add_subplot(gs[0])
    ax4: plt.Axes = fig3.add_subplot(gs[1])

    # The y axis unit is pC = 1e-12 C
    y_mult = 1e12
    ax3.errorbar(
        mca_peak_inds,
        charges*y_mult,
        xerr=mca_diff_nonlin * mca_peak_inds,
        yerr=charges_std*y_mult,
        fmt=".", capsize=3, label="data", color="black"
    )
    fit = curve_fit(
        lambda x, a, b: a*x + b,
        xdata=mca_peak_inds,
        ydata=charges,
        sigma=charges_std,
    )
    print("MCA calibration fit: least squares")
    print(fit[0])
    print(fit[1])
    model = scipy.odr.Model(lambda coeff, x: coeff[0]*x + coeff[1])
    data = scipy.odr.RealData(
        x=mca_peak_inds,
        y=charges,
        sx=mca_peak_inds * mca_diff_nonlin,
        sy=charges_std
    )
    odr = scipy.odr.ODR(data, model, beta0=fit[0])
    out = odr.run()
    print("MCA calibration fit: ODR")
    print(out.pprint())
    coeff = out.beta
    coeff_covar = out.cov_beta
    coeff_stds = out.sd_beta
    # coeff = np.array([fit[0][0], fit[0][1]])
    # coeff_covar = fit[1]
    # coeff_stds = np.array([coeff_covar[0, 0], coeff_covar[1, 1]])
    ax3.plot(
        mca_peak_inds,
        np.polyval(coeff, mca_peak_inds) * y_mult,
        label=f"fit (y = {coeff[0]*y_mult:.2e}±{coeff_stds[1]*y_mult:.2e}x + {coeff[1]*y_mult:.2e}±{coeff_stds[1]*y_mult:.2e})",
        color="tab:blue"
    )
    # ax3.set_xlabel("MCA channel")
    ax3.set_ylabel("Collected charge (pC)")
    ax3.legend(prop={"size": 9})

    for meas in cal_data:
        ax4.plot(meas.mca.channels, meas.mca.counts, color="tab:blue")

    for ax in (ax3, ax4):
        ax.set_xlim(0, cal_data[0].mca.channels[-1])

    ax4.set_xlabel("MCA channel")
    ax4.set_ylabel("Counts")
    plot.save_fig(fig3, "mca_calibration")

    print("Calibration coefficients:")
    print(coeff)
    print("Calibration covariances:")
    print(coeff_covar)
    print()
    return coeff, coeff_covar
