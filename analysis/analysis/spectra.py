import matplotlib.pyplot as plt
import numpy as np
import scipy.odr
from scipy.optimize import curve_fit

from devices.mca import MeasMCA
import fitting
import plot
import utils


def spectra(
        am_path: str,
        fe_path: str,
        noise_path: str,
        gain: float,
        voltage: float,
        voltage_std: float,
        mca_diff_nonlin: float = None,
        mca_int_nonlin: float = None,
        fig_titles: bool = True,
        name: str = None,
        vlines: bool = True,
        sec_fits: bool = True) -> None:
    """Analyze spectral measurement"""
    title = "Spectra"
    if name:
        title += f" ({name})"
    utils.print_title(title)
    am = MeasMCA(am_path, diff_nonlin=mca_diff_nonlin, int_nonlin=mca_int_nonlin)
    fe = MeasMCA(fe_path, diff_nonlin=mca_diff_nonlin, int_nonlin=mca_int_nonlin)
    noise = MeasMCA(noise_path, diff_nonlin=mca_diff_nonlin, int_nonlin=mca_int_nonlin)

    fig: plt.Figure = plt.figure()
    if fig_titles:
        title = "Spectral measurements"
        if name:
            title += f" ({name})"
        fig.suptitle(title)
    fig.subplots_adjust(bottom=0.2, right=0.85)
    ax1: plt.Axes = fig.add_subplot()
    ax2: plt.Axes = ax1.twinx()
    ax1.set_xlim(0, am.channels[-1])
    ax2.set_xlim(0, am.channels[-1])

    am_subtracted = am.counts - noise.counts * noise.real_length / am.real_length
    fe_subtracted = fe.counts - noise.counts * noise.real_length / fe.real_length
    y_mult = 1.1
    ax1.set_ylim(0, np.max(am_subtracted)*y_mult)
    ax2.set_ylim(0, np.max(fe_subtracted)*y_mult)

    line_am = ax1.plot(
        am.channels,
        am_subtracted,
        label="$^{241}$Am", color="darkgrey")[0]
    line_fe = ax2.plot(
        fe.channels,
        fe_subtracted,
        label="$^{55}$Fe", color="peru")[0]
    # line_noise = ax1.plot(noise.data, label="noise", color="orange")[0]

    try:
        fit_am = fitting.fit_am(am, ax1, subtracted=am_subtracted, vlines=vlines)
    except RuntimeError:
        print("WARNING! Am fit failed.")
        fit_am = None
    try:
        fit_fe = fitting.fit_fe(fe, ax2, subtracted=fe_subtracted, vlines=vlines)
    except (RuntimeError, IndexError):
        print("WARNING! Fe fit failed.")
        fit_fe = None

    if fit_fe is not None and fit_am is not None:
        ind_am_peak = fit_am[0][1]
        ind_fe_peak = fit_fe[0][0][1]
        ind_fe_escape_peak = fit_fe[1][0][1]
        print("Am peak index:", ind_am_peak)
        print("Fe peak index:", ind_fe_peak)
        print("Fe escape peak index:", ind_fe_escape_peak)
        am_peak = 59.5409e3
        fe_peak = 5.90e3
        fe_escape_peak = 3.19e3
        cal_x = np.array([ind_am_peak, ind_fe_peak, ind_fe_escape_peak])
        cal_y = np.array([am_peak, fe_peak, fe_escape_peak])
        cal_xerr = np.sqrt([fit_am[1][1, 1], fit_fe[0][1][1, 1], fit_fe[1][1][1, 1]])
        print("Energy calibration x errors:")
        print(cal_xerr)
        fit = fitting.fit_odr(fitting.poly1, cal_x, cal_y, std_x=cal_xerr)

        a = fit[0][0]
        b = fit[0][1]

        cal_fig: plt.Figure = plt.figure()
        cal_ax: plt.Axes = cal_fig.add_subplot()
        cal_fit_x = np.array([0, 1024])
        cal_ax.errorbar(
            cal_x, cal_y / 1000,
            xerr=cal_xerr / 1000,
            fmt=".", capsize=3,
            label="centroids of measured peaks"
        )
        cal_ax.plot(
            cal_fit_x,
            fitting.poly1(cal_fit_x, a, b) / 1000,
            label=f"linear fit (y = {fit[0][0]:.2e}±{np.sqrt(fit[1][0, 0]):.2e}x + "
                  f"{fit[0][1]:.2e}±{np.sqrt(fit[1][1, 1]):.2e})"
        )
        if fig_titles:
            cal_suptitle = "Spectral calibration"
            if name:
                cal_suptitle += f" ({name})"
            cal_fig.suptitle(cal_suptitle)
        cal_ax.set_xlabel("MCA channel")
        cal_ax.set_ylabel("Energy (keV)")
        cal_ax.legend(fontsize=8)
        cal_filename = "spectral_calibration"
        if name:
            cal_filename += f"_{name}"
        plot.save_fig(cal_fig, cal_filename)
    else:
        a = 0
        b = 0

    def ind_conv_func(channel: int):
        """Convert MCA index to energy (keV)"""
        return (a * channel + b) / 1000

    def ind_label_func(channels: np.ndarray):
        """Convert energy value to text label"""
        return [f"{ind_conv_func(x):.2f}" for x in channels]

    if a != 0:
        # The correspondence of the values is dependent on the manual limits
        ax3 = plot.double_x_axis(ax1, tick_locs=np.arange(0, am.channels[-1], 200), tick_label_func=ind_label_func)
        ax3.set_xlabel("Energy (keV)")

    if sec_fits:
        try:
            am_sec_fits = [
                fitting.fit_manual(am, ax1, 250, 300, subtracted=am_subtracted, vlines=vlines),
                fitting.fit_manual(am, ax1, 300, 360, subtracted=am_subtracted, vlines=vlines),
                fitting.fit_manual(am, ax1, 380, 440, subtracted=am_subtracted, vlines=vlines),
                fitting.fit_manual(am, ax1, 710, 830, subtracted=am_subtracted, vlines=vlines)
            ]
        except RuntimeError:
            print("WARNING! Secondary fits failed.")
            am_sec_fits = None

        if fit_fe is not None and fit_am is not None and am_sec_fits is not None:
            for i_fit, fit in enumerate(am_sec_fits):
                ch = fit[0][1]
                print(f"Am secondary peak {i_fit+1}: ch {ch}, {ind_conv_func(ch)} keV")

    ax1.set_xlabel("MCA channel")
    ax1.set_ylabel(r"Count ($^{241}Am$)")
    ax2.set_ylabel(r"Count ($^{55}Fe$)")
    # plot.legend_multi(ax1, [line_am, line_fe, line_noise])
    plot.legend_multi(ax1, [line_am, line_fe])

    filename = "spectra"
    if name:
        filename += f"_{name}"
    plot.save_fig(fig, filename)
