import functools
import glob
import os.path
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import uncertainties as unc
import uncertainties.unumpy as unp

import const
from devices.mca import MeasMCA
import fitting
import plot
import type_hints
import utils


def get_total_charges(mcas: tp.List[MeasMCA], gains: np.ndarray, cal_coeff: np.ndarray, cal_gain: float):
    charges = np.zeros(len(mcas))
    for i, mca in enumerate(mcas):
        charge_mults = np.polyval(cal_coeff, mca.channels)
        charges[i] = np.sum(mca.counts * charge_mults) * cal_gain / gains[i]
    return charges


def get_peak_charges(
        fits: tp.List[type_hints.CURVE_FIT],
        gains: np.ndarray,
        cal_coeff: np.ndarray,
        cal_coeff_covar: np.ndarray,
        cal_gain: float,
        gain_rel_std: float):
    """Get the charge corresponding to an MCA peak

    In the article:
    "The centroids of the peaks were converted to total charge or the number readout
    electrons using the measured electronics calibration function."
    Since the centroid of a Gaussian refers to its mean value, I presume that this
    means that we should multiply the peak height with the value of the calibration function at that channel.
    """
    if cal_coeff.size != 2:
        raise NotImplementedError
    counts = np.array([fit[0][0] for fit in fits])
    peak_channels = np.array([fit[0][1] for fit in fits])
    charges = counts * np.polyval(cal_coeff, peak_channels) * cal_gain / gains
    charges_std = np.empty_like(charges)
    # Partial derivatives
    A_N = (cal_coeff[0] * counts + cal_coeff[1]) * cal_gain / gains
    A_M = counts * cal_coeff[0] * cal_gain / gains
    A_g = counts * peak_channels * cal_gain / gains
    A_h = counts * cal_gain / gains
    A_gc = counts * (cal_coeff[0] * peak_channels + cal_coeff[1]) / gains
    A_gm = - counts * (cal_coeff[0] * peak_channels + cal_coeff[1]) * cal_gain / gains**2
    # This could be done without a loop but it's easier to understand this way
    for i, fit in enumerate(fits):
        # Order: N, M, g, h, gc, gm
        A = np.array([A_N[i], A_M[i], A_g[i], A_h[i], A_gc[i], A_gm[i]])
        V = scipy.linalg.block_diag(fit[1][:2, :2], cal_coeff_covar, gain_rel_std**2 * cal_gain, gain_rel_std**2 * gains[i])
        # print(A.shape)
        # print(V.shape)
        # print(fit[1].shape)
        # print(cal_coeff_covar.shape)
        # print(V)
        U = A @ V @ A.T
        charges_std[i] = U

    return charges, charges_std


def hv_scan(
        folder: str,
        prefix: str,
        diff_nonlin: float,
        int_nonlin: float,
        voltage_std: float,
        gain_rel_std: float = 0,
        ) -> tp.Tuple[np.array, np.array, tp.List[MeasMCA]]:
    """Analyze HV scan data"""
    print(f"{prefix} HV scan")
    gains, voltages, mcas = read_hv_scan(folder, prefix, diff_nonlin=diff_nonlin, int_nonlin=int_nonlin)
    print("Gains:")
    print(gains)
    print("Voltages:")
    print(voltages)
    print(f"Voltage range: {min(voltages)} - {max(voltages)} V")

    # gains = unp.uarray(gains, gains*gain_rel_std)
    # voltages = unp.uarray(voltages, voltage_std)
    # gains_std = gains * gain_rel_std
    return gains, voltages, mcas


def hv_scans(
        folder: str,
        cal_coeff: np.ndarray,
        cal_coeff_covar: np.ndarray,
        cal_gain: float,
        diff_nonlin: float,
        int_nonlin: float,
        voltage_std: float,
        gain_rel_std: float):
    utils.print_title("HV scans")
    am_gains, am_voltages, am_mcas = hv_scan(
        folder, "Am",
        diff_nonlin=diff_nonlin,
        int_nonlin=int_nonlin,
        voltage_std=voltage_std,
        gain_rel_std=gain_rel_std
    )
    fe_gains, fe_voltages, fe_mcas = hv_scan(
        folder, "Fe",
        diff_nonlin=diff_nonlin,
        int_nonlin=int_nonlin,
        voltage_std=voltage_std,
        gain_rel_std=gain_rel_std
    )

    am_fits = fitting.fit_am_hv_scan(am_mcas)
    fe_fits = fitting.fit_fe_hv_scan(fe_mcas)

    # am_charges = get_charges(am_mcas, am_gains, cal_coeff, cal_gain)
    # fe_charges = get_charges(fe_mcas, fe_gains, cal_coeff, cal_gain)

    # TODO: find where the fixing factor comes from
    # fix = 1e-4
    fix = 1
    am_charges, am_charges_std = get_peak_charges(am_fits, am_gains, cal_coeff, cal_coeff_covar, cal_gain, gain_rel_std)
    fe_charges, fe_charges_std = get_peak_charges(fe_fits, fe_gains, cal_coeff, cal_coeff_covar, cal_gain, gain_rel_std)
    am_charges *= fix
    am_charges_std *= fix
    fe_charges *= fix
    fe_charges_std *= fix

    ###
    # Measured charges
    ###
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    ax.errorbar(
        am_voltages,
        am_charges / const.ELEMENTARY_CHARGE,
        yerr=am_charges_std,
        fmt=".", capsize=3,
        label=r"$\gamma$ (59.5 keV) of $^{241}$Am"
    )
    ax.errorbar(
        fe_voltages,
        fe_charges / const.ELEMENTARY_CHARGE,
        yerr=fe_charges_std,
        fmt=".", capsize=3,
        label=r"$\gamma$ (5.9 keV) of $^{55}$Fe"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Number of electrons")
    ax.set_title("TODO fix y-scale")
    ax.legend()
    plot.save_fig(fig, "hv_scans")

    ###
    # Gas multiplication factors
    ###
    fig2: plt.Figure = plt.figure()
    ax2: plt.Axes = fig.add_subplot()
    # theor_gas_mult = utils.diethorn(
    #     V=voltages
    # )

    ###
    # Resolution
    ###
    fig3: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()

    print()


def read_hv_scan(
        folder: str,
        prefix: str,
        diff_nonlin: float,
        int_nonlin: float) -> tp.Tuple[np.ndarray, np.ndarray, tp.List[MeasMCA]]:
    """Read HV scan data from files"""
    # glob returns random order so the data has to be sorted later for convenient display
    paths = glob.glob(os.path.join(folder, f"{prefix}_*.mca"))
    gains = []
    voltages = []
    mcas = []
    for path in paths:
        file_name = os.path.basename(path)
        if "FAIL" in file_name:
            continue
        name = file_name.split(".")[0]
        parts = name.split("_")
        gains.append(int(parts[1]))
        voltages.append(int(parts[2]))
        mcas.append(MeasMCA(path, diff_nonlin=diff_nonlin, int_nonlin=int_nonlin))
    data = list(zip(gains, voltages, mcas))

    # In the measurement order the voltage is increased and the gain is decreased
    def compare(data1, data2):
        if data1[0] != data2[0]:
            return -1 + 2*int(data1[0] < data2[0])
        elif data1[1] != data2[1]:
            return -1 + 2*int(data1[1] > data2[1])
        raise ValueError("Measurements have same settings:", data1, data2)

    data.sort(key=functools.cmp_to_key(compare))
    gains_sorted = np.array([val[0] for val in data])
    voltages_sorted = np.array([val[1] for val in data])
    mcas_sorted = [val[2] for val in data]

    return gains_sorted, voltages_sorted, mcas_sorted
