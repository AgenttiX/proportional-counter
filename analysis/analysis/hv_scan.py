import functools
import glob
import os.path
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

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
        charges[i] = np.sum(mca.data * charge_mults) * cal_gain / gains[i]
    return charges


def get_peak_charges(fits: tp.List[type_hints.CURVE_FIT], gains: np.ndarray, cal_coeff: np.ndarray, cal_gain: float):
    """Get the charge corresponding to an MCA peak

    In the article:
    "The centroids of the peaks were converted nto total charge or the number readout
    electrons using the measured electronics calibration function."
    Since the centroid of a Gaussian refers to its mean value, I presume that this
    means that we should multiply the peak height with the value of the calibration function at that channel.
    """
    peaks = np.array([fit[0][0] for fit in fits])
    peak_channels = np.array([fit[0][1] for fit in fits])
    charges = peaks * np.polyval(cal_coeff, peak_channels) * cal_gain / gains
    return charges


def hv_scan(folder: str, prefix: str):
    """Analyze HV scan data"""
    print(f"{prefix} HV scan")
    gains, voltages, mcas = read_hv_scan(folder, prefix)
    print("Gains:")
    print(gains)
    print("Voltages:")
    print(voltages)
    print(f"Voltage range: {min(voltages)} - {max(voltages)} V")

    return gains, voltages, mcas


def hv_scans(folder: str, cal_coeff: np.ndarray = None, cal_gain: float = None):
    utils.print_title("HV scans")
    am_gains, am_voltages, am_mcas = hv_scan(folder, "Am")
    fe_gains, fe_voltages, fe_mcas = hv_scan(folder, "Fe")

    am_fits = fitting.fit_am_hv_scan(am_mcas)
    fe_fits = fitting.fit_fe_hv_scan(fe_mcas)

    if cal_coeff is None or cal_gain is None:
        return
    # am_charges = get_charges(am_mcas, am_gains, cal_coeff, cal_gain)
    # fe_charges = get_charges(fe_mcas, fe_gains, cal_coeff, cal_gain)

    # TODO: find where the fixing 100 comes from
    fix = 1e-4
    am_charges = get_peak_charges(am_fits, am_gains, cal_coeff, cal_gain) * fix
    fe_charges = get_peak_charges(fe_fits, fe_gains, cal_coeff, cal_gain) * fix

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    ax.scatter(am_voltages, am_charges / const.ELEMENTARY_CHARGE, label=r"$\gamma$ (59.5 keV) of $^{241}$Am")
    ax.scatter(fe_voltages, fe_charges / const.ELEMENTARY_CHARGE, label=r"$\gamma$ (5.9 keV) of $^{55}$Fe")
    ax.set_yscale("log")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Number of electrons")
    ax.set_title("TODO fix y-scale")
    ax.legend()
    plot.save_fig(fig, "hv_scans")


def read_hv_scan(folder: str, prefix: str) -> tp.Tuple[np.ndarray, np.ndarray, tp.List[MeasMCA]]:
    """Read HV scan data from files"""
    # glob returns random order
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
        mcas.append(MeasMCA(path))
    data = list(zip(gains, voltages, mcas))

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
