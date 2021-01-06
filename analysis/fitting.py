import typing as tp

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from devices.mca import MeasMCA
import plot
import stats
import type_hints

THRESHOLD_LEVEL = 0.5
CUT_WIDTH_MULT = 2


def create_subplot_grid(
        num_plots: int,
        grid_aspect_ratio: float = 1,
        xlabel: str = None,
        ylabel: str = None) -> tp.Tuple[plt.Figure, tp.List[plt.Axes], int, int]:
    fig: plt.Figure
    num_plots_x = int(np.sqrt(num_plots)*grid_aspect_ratio)
    num_plots_y = int(np.ceil(num_plots / num_plots_x))
    fig, axes = plt.subplots(num_plots_y, num_plots_x)
    axes_flat: tp.List[plt.Axes] = [item for sublist in axes for item in sublist]

    # Remove unnecessary axes
    for i in range(num_plots, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    for i, ax in enumerate(axes_flat):
        if xlabel is not None:
            if num_plots - i <= num_plots_x:
                ax.set_xlabel(xlabel)
            else:
                ax.xaxis.set_ticklabels([])

            if i % num_plots_x == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.yaxis.set_ticklabels([])

    return fig, axes_flat, num_plots_x, num_plots_y


def get_cut(
        data: np.ndarray,
        threshold_level: float = THRESHOLD_LEVEL,
        cut_width_mult: float = CUT_WIDTH_MULT) -> tp.Tuple[np.ndarray, int, int, float]:
    """Get cut parameters for fitting to a peak"""
    peak_ind: int = np.argmax(data)
    peak = data[peak_ind]
    threshold_inds = np.where(data > peak * threshold_level)[0]
    threshold_width = threshold_inds[-1] - threshold_inds[0]
    cut_ind_min = max(0, peak_ind - cut_width_mult * (peak_ind - threshold_inds[0]))
    cut_ind_max = min(data.size, peak_ind + cut_width_mult * (threshold_inds[-1] - peak_ind) + 1)
    cut_inds = np.arange(cut_ind_min, cut_ind_max)

    return cut_inds, threshold_width, peak_ind, peak


def fit_am(
        mca: MeasMCA,
        ax: plt.Axes,
        threshold_level: float = THRESHOLD_LEVEL,
        cut_width_mult: float = CUT_WIDTH_MULT):
    """Fit the peaks of an Am-241 spectrum"""
    peak = np.max(mca.counts)
    above_threshold = np.where(mca.counts > threshold_level * peak)[0]
    half_ind = (above_threshold[0] + above_threshold[-1]) // 2
    filtered = mca.counts.copy()
    filtered[:half_ind] = 0
    # ax.plot(filtered)

    cut_inds, threshold_width, peak_ind, peak = get_cut(filtered, threshold_level, cut_width_mult)

    # Vertical lines according to the cuts
    ax.vlines((cut_inds[0], cut_inds[-1]), ymin=0, ymax=peak, label="fit cut", colors="r", linestyles=":")

    fit = curve_fit(
        stats.gaussian_scaled,
        cut_inds,
        mca.counts[cut_inds],
        p0=(peak, peak_ind, threshold_width)
    )
    ax.plot(
        mca.channels,
        fit[0][0] * stats.gaussian(mca.channels, *fit[0][1:]),
        linestyle="--",
        label="Fe-55 fit"
    )
    return fit


def fit_am_hv_scan(mcas: tp.List[MeasMCA]):
    """Create fits for the Am-241 HV scan measurements"""
    fig, axes, num_plots_x, num_plots_y = create_subplot_grid(len(mcas), xlabel="MCA ch.", ylabel="Count")
    # fig.suptitle("Am fits")

    max_peak_height = np.max([np.max(mca.counts) for mca in mcas])
    max_ch = np.max([mca.channels[-1] for mca in mcas])
    y_adjust_step = 50
    max_peak_height_round = y_adjust_step * np.ceil(max_peak_height/y_adjust_step)

    fits = []
    for i, mca in enumerate(mcas):
        ax = axes[i]
        ax.plot(mca.counts)
        ax.set_xlim(0, max_ch)
        ax.set_ylim(0, max_peak_height_round)

        fits.append(fit_am(mca, ax))

    plot.save_fig(fig, "am_scan_fits")
    return fits


def fit_fe(
        mca: MeasMCA,
        ax: plt.Axes,
        threshold_level: float = THRESHOLD_LEVEL,
        cut_width_mult: float = CUT_WIDTH_MULT,
        secondary: bool = True):
    """Fit the peaks of an Fe-55 spectrum

    TODO: this could be combined to the HV scan fitting function
    """
    cut_inds, threshold_width, peak_ind, peak = get_cut(mca.counts, threshold_level, cut_width_mult)

    # Vertical lines according to the cuts
    ax.vlines((cut_inds[0], cut_inds[-1]), ymin=0, ymax=peak, label="fit cut", colors="r", linestyles=":")

    fit = curve_fit(
        stats.gaussian_scaled,
        cut_inds,
        mca.counts[cut_inds],
        p0=(peak, peak_ind, threshold_width)
    )
    if not secondary:
        ax.plot(
            mca.channels,
            fit[0][0] * stats.gaussian(mca.channels, *fit[0][1:]),
            linestyle="--",
            label="Fe-55 fit"
        )
        return fit

    # The secondary peak is the Argon escape peak and therefore not a property of the Fe-55 source itself
    cut_inds2, threshold_width2, peak_ind2, peak2 = get_cut(mca.counts[:cut_inds[0]], threshold_level, cut_width_mult)
    ax.vlines((cut_inds2[0], cut_inds2[-1]), ymin=0, ymax=peak, label="fit cut", colors="r", linestyles=":")
    fit2 = curve_fit(
        stats.gaussian_scaled,
        cut_inds2,
        mca.counts[cut_inds2],
        p0=(peak2, peak_ind, threshold_width)
    )
    fit1_data = fit[0][0] * stats.gaussian(mca.channels, *fit[0][1:])
    fit2_data = fit2[0][0] * stats.gaussian(mca.channels, *fit2[0][1:])
    ax.plot(
        mca.channels,
        fit1_data + fit2_data,
        linestyle="--",
        label="Fe-55 fit",
    )
    return fit, fit2


def fit_fe_hv_scan(
        mcas: tp.List[MeasMCA],
        threshold_level: float = THRESHOLD_LEVEL,
        cut_width_mult: float = CUT_WIDTH_MULT) -> tp.List[type_hints.CURVE_FIT]:
    """Create fits for Fe-55 HV scan measurements"""

    fig, axes, num_plots_x, num_plots_y = create_subplot_grid(len(mcas), xlabel="MCA ch.", ylabel="Count")
    # fig.suptitle("Fe fits")
    # fig.tight_layout()

    # peak_channels = np.zeros(len(mcas))
    # fit_stds = np.zeros_like(peak_channels)

    max_peak_height = np.max([np.max(mca.counts) for mca in mcas])
    max_ch = np.max([mca.channels[-1] for mca in mcas])
    y_adjust_step = 50
    max_peak_height_round = y_adjust_step * np.ceil(max_peak_height/y_adjust_step)

    fits = []
    for i, mca in enumerate(mcas):
        ax = axes[i]
        ax.plot(mca.counts)

        ax.set_xlim(0, max_ch)
        ax.set_ylim(0, max_peak_height_round)

        fit = fit_fe(mca, ax, threshold_level, cut_width_mult, secondary=False)

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

        # peak_channels[i] = fit[0][1]
        # fit_stds[i] = fit[0][2]
        fits.append(fit)

    plot.save_fig(fig, "fe_scan_fits")

    return fits
