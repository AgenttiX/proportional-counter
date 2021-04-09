import functools
import glob
import os.path
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from scipy.optimize import curve_fit
import sympy as sp
# import uncertainties as unc
# import uncertainties.unumpy as unp

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
    """Get the charge of an individual event corresponding to an MCA peak location

    In the article:
    "The centroids of the peaks were converted to total charge or the number readout
    electrons using the measured electronics calibration function."
    Since the centroid of a Gaussian refers to its mean value, I presume that this
    means that we should multiply the peak height with the value of the calibration function at that channel.
    """
    if cal_coeff.size != 2:
        raise NotImplementedError
    # counts = np.array([fit[0][0] for fit in fits])
    peak_channels = np.array([fit[0][1] for fit in fits])
    # charges = np.polyval(cal_coeff, peak_channels) * cal_gain / gains

    ch, c0, c1, gc, g = sp.symbols("ch c0 c1 gc g")
    syms = [ch, c0, c1, gc, g]
    func = (c0*ch + c1)*gc/g

    charges = np.empty_like(peak_channels)
    charges_std = np.empty_like(charges)
    for i, fit in enumerate(fits):
        charge, charge_std = utils.error_propagation(
            func,
            syms=syms,
            vals=np.array([peak_channels[i], cal_coeff[0], cal_coeff[1], cal_gain, gains[i]]),
            covar=scipy.linalg.block_diag(fit[1][1, 1], cal_coeff_covar, gain_rel_std**2 * cal_gain, gain_rel_std**2 * gains[i])
        )
        charges[i] = charge
        charges_std[i] = charge_std

    # # Manual error analysis
    # charges_std = np.empty_like(charges)
    # # Partial derivatives
    # A_N = (cal_coeff[0] * counts + cal_coeff[1]) * cal_gain / gains
    # A_M = counts * cal_coeff[0] * cal_gain / gains
    # A_g = counts * peak_channels * cal_gain / gains
    # A_h = counts * cal_gain / gains
    # A_gc = counts * (cal_coeff[0] * peak_channels + cal_coeff[1]) / gains
    # A_gm = - counts * (cal_coeff[0] * peak_channels + cal_coeff[1]) * cal_gain / gains**2
    # # This could be done without a loop but it's easier to understand this way
    # for i, fit in enumerate(fits):
    #     # Order: N, M, g, h, gc, gm
    #     A = np.array([A_N[i], A_M[i], A_g[i], A_h[i], A_gc[i], A_gm[i]])
    #     V = scipy.linalg.block_diag(fit[1][:2, :2], cal_coeff_covar, gain_rel_std**2 * cal_gain, gain_rel_std**2 * gains[i])
    #     # print(A.shape)
    #     # print(V.shape)
    #     # print(fit[1].shape)
    #     # print(cal_coeff_covar.shape)
    #     # print(V)
    #     U = A @ V @ A.T
    #     charges_std[i] = U

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
        gain_rel_std: float,
        can_diameter: np.ndarray,
        wire_diameter: np.ndarray,
        pressure: float,
        pressure_std: float,
        fig_titles: bool = True):
    utils.print_title("HV scans")
    am_gains, am_voltages, am_mcas = hv_scan(
        folder, "Am",
        diff_nonlin=diff_nonlin,
        int_nonlin=int_nonlin,
        voltage_std=voltage_std,
        gain_rel_std=gain_rel_std,
    )
    fe_gains, fe_voltages, fe_mcas = hv_scan(
        folder, "Fe",
        diff_nonlin=diff_nonlin,
        int_nonlin=int_nonlin,
        voltage_std=voltage_std,
        gain_rel_std=gain_rel_std,
    )

    am_fits = fitting.fit_am_hv_scan(am_mcas, fig_titles=fig_titles)
    fe_fits = fitting.fit_fe_hv_scan(fe_mcas, fig_titles=fig_titles)

    am_charges, am_charges_std = get_peak_charges(am_fits, am_gains, cal_coeff, cal_coeff_covar, cal_gain, gain_rel_std)
    fe_charges, fe_charges_std = get_peak_charges(fe_fits, fe_gains, cal_coeff, cal_coeff_covar, cal_gain, gain_rel_std)

    ###
    # Measured charges
    ###
    am_text = r"$\gamma$ (59.5 keV) of $^{241}$Am"
    fe_text = r"$\gamma$ (5.9 keV) of $^{55}$Fe"

    fig: plt.Figure = plt.figure()
    if fig_titles:
        fig.suptitle("Measured charges")
    ax: plt.Axes = fig.add_subplot()
    ax.errorbar(
        am_voltages,
        am_charges / const.ELEMENTARY_CHARGE,
        yerr=am_charges_std,
        fmt=".", capsize=3,
        label=am_text
    )
    ax.errorbar(
        fe_voltages,
        fe_charges / const.ELEMENTARY_CHARGE,
        yerr=fe_charges_std,
        fmt=".", capsize=3,
        label=fe_text
    )
    ax.set_yscale("log")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Number of electrons")
    ax.legend()
    plot.save_fig(fig, "hv_scans")

    fig12: plt.Figure = plt.figure()
    if fig_titles:
        fig12.suptitle("Measured charges (with units)")
    ax12: plt.Axes = fig12.add_subplot()
    ax12.errorbar(
        am_voltages,
        am_charges,
        yerr=am_charges_std,
        fmt=".", capsize=3,
        label=am_text
    )
    ax12.errorbar(
        fe_voltages,
        fe_charges,
        yerr=fe_charges_std,
        fmt=".", capsize=3,
        label=fe_text
    )
    ax12.set_yscale("log")
    ax12.set_xlabel("Voltage (V)")
    ax12.set_ylabel("Collected charge (C)")
    ax12.legend()
    plot.save_fig(fig, "hv_scans_units")

    ###
    # Gas multiplication factors
    ###
    fig2: plt.Figure = plt.figure()
    if fig_titles:
        fig2.suptitle("Gas multiplication factors")
    ax2: plt.Axes = fig2.add_subplot()

    # Theoretical
    volt_range = np.linspace(1100, 2400)
    wire_radius = wire_diameter / 2
    can_radius = can_diameter / 2
    theor_gas_mult_log = np.array([
        utils.log_gas_mult_factor_p10(
            V=volt, a=wire_radius.mean(), b=can_radius.mean(), p=pressure,
            std_V=0, std_a=wire_radius.std(), std_b=can_radius.std(), std_p=pressure_std)
        for volt in volt_range
    ]).T
    print(theor_gas_mult_log[0])
    print(theor_gas_mult_log[1])
    ax2.plot(volt_range, np.exp(theor_gas_mult_log[0]), label="theoretical prediction", color="g")
    ax2.plot(volt_range, np.exp(theor_gas_mult_log[0] + theor_gas_mult_log[1]), linestyle=":", color="g", label=r"$1\sigma$ limits")
    ax2.plot(volt_range, np.exp(theor_gas_mult_log[0] - theor_gas_mult_log[1]), linestyle=":", color="g")

    # Observed
    # For Argon
    E_pair = 26  # eV
    E_fe = 5.9e3  # eV
    E_am = 59.54e3  # eV
    mult_am = utils.gas_mult_factor(am_charges, E_rad=E_am, E_ion_pair=E_pair)
    mult_fe = utils.gas_mult_factor(fe_charges, E_rad=E_fe, E_ion_pair=E_pair)
    # The equation is proportional to the charge, so the error propagation works directly like this
    mult_am_std = utils.gas_mult_factor(am_charges_std, E_rad=E_am, E_ion_pair=E_pair)
    mult_fe_std = utils.gas_mult_factor(fe_charges_std, E_rad=E_fe, E_ion_pair=E_pair)
    ax2.errorbar(
        am_voltages,
        mult_am,
        yerr=mult_am_std,
        fmt=".", capsize=3,
        label=am_text
    )
    ax2.errorbar(
        fe_voltages,
        mult_fe,
        yerr=mult_fe_std,
        fmt=".", capsize=3,
        label=fe_text,
    )

    ax2.set_yscale("log")
    ax2.set_ylabel("M")
    ax2.set_xlabel("Voltage (V)")
    ax2.legend()
    plot.save_fig(fig2, "gas_mult")

    ###
    # Resolution
    ###
    fig3: plt.Figure = plt.figure()
    if fig_titles:
        fig3.suptitle("Resolution")
    ax3: plt.Axes = fig3.add_subplot()
    hv_scan_resolution(ax3, am_voltages, am_fits, am_text)
    hv_scan_resolution(ax3, fe_voltages, fe_fits, fe_text)

    ax3.set_xlabel("Voltage (V)")
    ax3.set_ylabel("Peak width (FWHM) / peak channel")
    ax3.legend()
    plot.save_fig(fig3, "resolution")

    print()


def hv_scan_resolution(ax: plt.Axes, voltages: np.ndarray, fits: tp.List[type_hints.CURVE_FIT], label: str):
    peak_locs = np.array([fit[0][1] for fit in fits])
    peak_stds = np.array([fit[0][2] for fit in fits])
    rel_fwhms = peak_stds * const.STD_TO_FWHM / peak_locs
    fit = curve_fit(
        fitting.poly2,
        voltages,
        rel_fwhms,
    )
    fit_x = np.linspace(np.min(voltages), np.max(voltages), 1000)
    fit_eval = fitting.poly2(fit_x, *fit[0])
    ax.errorbar(
        voltages,
        rel_fwhms,
        fmt=".", capsize=3,
        label=label
    )
    ax.plot(fit_x, fit_eval)
    print("Resolution fit:", label)
    print(fitting.poly2_fit_text(fit))


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
        if data1[1] != data2[1]:
            return -1 + 2*int(data1[1] > data2[1])
        raise ValueError("Measurements have same settings:", data1, data2)

    data.sort(key=functools.cmp_to_key(compare))
    gains_sorted = np.array([val[0] for val in data])
    voltages_sorted = np.array([val[1] for val in data])
    mcas_sorted = [val[2] for val in data]

    return gains_sorted, voltages_sorted, mcas_sorted
