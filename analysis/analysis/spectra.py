import matplotlib.pyplot as plt
import numpy as np

from devices.mca import MeasMCA
import fitting
import plot
import utils


def spectra(am_path, fe_path, noise_path, gain, voltage, voltage_std, fig_titles: bool = True):
    """Analyze spectral measurement"""
    utils.print_title("Spectra")
    am = MeasMCA(am_path)
    fe = MeasMCA(fe_path)
    noise = MeasMCA(noise_path)

    fig: plt.Figure = plt.figure()
    if fig_titles:
        fig.suptitle("Spectral measurements")
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
    fit_am = fitting.fit_am(am, ax1, subtracted=am_subtracted)
    fit_fe = fitting.fit_fe(fe, ax2, subtracted=fe_subtracted)

    am_sec_fits = [
        fitting.fit_manual(am, ax1, 250, 300, subtracted=am_subtracted),
        fitting.fit_manual(am, ax1, 300, 360, subtracted=am_subtracted),
        fitting.fit_manual(am, ax1, 380, 440, subtracted=am_subtracted),
        fitting.fit_manual(am, ax1, 710, 830, subtracted=am_subtracted)
    ]

    ind_am_peak = fit_am[0][1]
    ind_fe_peak = fit_fe[0][0][1]
    print("Am peak index:", ind_am_peak)
    print("Fe peak index:", ind_fe_peak)
    print("Fe escape peak index:", fit_fe[0][1][1])
    am_peak = 59.5409e3
    fe_peak = 5.90e3
    a = (am_peak - fe_peak)/(ind_am_peak - ind_fe_peak)
    b = am_peak - a*ind_am_peak

    def ind_conv_func(channel: int):
        return (a*channel + b) / 1000

    def ind_label_func(channels):
        return [f"{ind_conv_func(x):.2f}" for x in channels]

    for i_fit, fit in enumerate(am_sec_fits):
        ch = fit[0][1]
        print(f"Am secondary peak {i_fit+1}: ch {ch}, {ind_conv_func(ch)} keV")

    # The correspondence of the values is dependent on the manual limits
    ax3 = plot.double_x_axis(ax1, tick_locs=np.arange(0, am.channels[-1], 200), tick_label_func=ind_label_func)
    ax3.set_xlabel("Energy (keV)")

    ax1.set_xlabel("MCA channel")
    ax1.set_ylabel(r"Count ($^{241}Am$)")
    ax2.set_ylabel(r"Count ($^{55}Fe$)")
    # plot.legend_multi(ax1, [line_am, line_fe, line_noise])
    plot.legend_multi(ax1, [line_am, line_fe])
    plot.save_fig(fig, "spectra")
