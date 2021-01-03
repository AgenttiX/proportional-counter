import typing as tp

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from devices.mca import MeasMCA
import plot
import stats


def fit_fe(
        mcas: tp.List[MeasMCA],
        threshold_level: float = 0.5,
        cut_width_mult: float = 2):
    """Create fits for Fe-55 MCA measurements"""
    fig: plt.Figure
    # axes: tp.List[plt.Axes]
    grid_aspect_ratio = 1
    num_plots_x = int(np.sqrt(len(mcas))*grid_aspect_ratio)
    num_plots_y = int(np.ceil(len(mcas) / num_plots_x))
    fig, axes = plt.subplots(num_plots_y, num_plots_x)
    axes_flat: tp.List[plt.Axes] = [item for sublist in axes for item in sublist]

    # Remove unnecessary axes
    for i in range(len(mcas), len(axes_flat)):
        fig.delaxes(axes_flat[i])

    fig.suptitle("Fe fits")
    # fig.tight_layout()

    peak_channels = np.zeros(len(mcas))
    fit_stds = np.zeros_like(peak_channels)

    max_peak_height = np.max([np.max(mca.data) for mca in mcas])
    max_ch = np.max([mca.channels[-1] for mca in mcas])
    y_adjust_step = 50
    max_peak_height_round = y_adjust_step * np.ceil(max_peak_height/y_adjust_step)

    for i, mca in enumerate(mcas):
        ax = axes_flat[i]
        ax.plot(mca.data)
        if i % num_plots_x == 0:
            ax.set_ylabel("Count")
        else:
            ax.yaxis.set_ticklabels([])

        if len(mcas) - i <= num_plots_x:
            ax.set_xlabel("MCA ch.")
        else:
            ax.xaxis.set_ticklabels([])

        ax.set_xlim(0, max_ch)
        ax.set_ylim(0, max_peak_height_round)

        peak_ind = np.argmax(mca.data)
        peak = mca.data[peak_ind]
        threshold_inds = np.where(mca.data > peak*threshold_level)[0]
        threshold_width = threshold_inds[-1] - threshold_inds[0]
        cut_ind_min = max(0, peak_ind - cut_width_mult*(peak_ind-threshold_inds[0]))
        cut_ind_max = min(mca.data.size, peak_ind + cut_width_mult*(threshold_inds[-1]-peak_ind) + 1)
        cut_inds = np.arange(cut_ind_min, cut_ind_max)

        # Vertical lines according to the cuts
        ax.vlines((cut_ind_min, cut_ind_max), ymin=0, ymax=peak, label="fit cut", colors="r", linestyles=":")

        # Double Gaussian fitting is too error-prone
        # fit = curve_fit(
        #     stats.double_gaussian,
        #     cut_inds,
        #     mca.data[cut_inds],
        #     # p0=2*(peak/2, peak_ind, threshold_width)
        # )
        # if fit[0][0] > fit[0][3]:
        #     better_fit = fit[0][:3]
        # else:
        #     better_fit = fit[0][3:]
        # ax.plot(mca.channels, better_fit[0]*stats.gaussian(mca.channels, *better_fit[1:]))

        fit = curve_fit(
            stats.gaussian_scaled,
            cut_inds,
            mca.data[cut_inds],
            p0=(peak, peak_ind, threshold_width)
        )
        ax.plot(mca.channels, fit[0][0]*stats.gaussian(mca.channels, *fit[0][1:]))
        peak_channels[i] = fit[0][1]
        fit_stds[i] = fit[0][2]

    plot.save_fig(fig, "fe_scan_fits")

    return peak_channels, fit_stds
