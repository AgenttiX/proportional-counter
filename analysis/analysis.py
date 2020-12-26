import glob
import os.path
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from devices.mca import MeasMCA
from meas import Meas, MeasCal
import fitting
import plot


def analyze_sizes(sizes: tp.Dict[str, tp.List[float]]):
    print("Sizes")
    for name, data in sizes.items():
        print(name)
        print(f"µ = {np.mean(data)}")
        print(f"σ = {np.std(data)}")
    print()


def read_hv_scan(folder: str, prefix: str) -> tp.Tuple[np.ndarray, np.ndarray, tp.List[MeasMCA]]:
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
    return np.array(gains), np.array(voltages), mcas


def hv_scan(folder: str, prefix: str):
    gains, voltages, mcas = read_hv_scan(folder, prefix)
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    true_volts = voltages / gains
    ax.plot(mcas[0].data)

    if prefix == "Fe":
        fitting.fit_fe(mcas)


def spectra(am_path, fe_path, noise_path, gain, voltage):
    am = MeasMCA(am_path)
    fe = MeasMCA(fe_path)
    noise = MeasMCA(noise_path)

    fig: plt.Figure = plt.figure()
    ax1: plt.Axes = fig.add_subplot()
    ax2: plt.Axes = ax1.twinx()
    line_am = ax1.plot(am.data, label="Am-241")[0]
    line_fe = ax2.plot(fe.data, label="Fe-55")[0]
    line_noise = ax1.plot(noise.data, label="noise")[0]

    ax1.set_xlabel("MCA channel")
    ax1.set_ylabel("count")
    plot.legend_multi(ax1, [line_am, line_fe, line_noise])


def calibration(
        cal_data: tp.List[MeasCal],
        coarse_gain: float = 10,
        fine_gain: float = 10,
        preamp_capacitance: float = 1e-12) -> np.ndarray:
    # meas = cal_data[0]
    # osc = meas.traces[0]
    fig: plt.Figure = plt.figure()
    fig.suptitle("Calibration")
    ax: plt.Axes = fig.add_subplot()
    # plot.plot_osc(osc, ax)

    peak_data = np.array([meas.peak_height for meas in cal_data])
    peak_heights = peak_data[:, 0]
    peak_stds = peak_data[:, 1]
    gain = coarse_gain*fine_gain
    charges = preamp_capacitance*np.array([meas.voltage for meas in cal_data]) / gain

    ax.errorbar(charges, peak_heights, yerr=peak_stds, fmt=".", capsize=3, label="data")
    # This transforms the scale to a more reasonable one for the fitting algorithm and therefore
    # reduces errors.
    fit_accuracy_fixer = 1e14
    fit = curve_fit(
        lambda x, a, b: a*x + b,
        xdata=charges*fit_accuracy_fixer,
        ydata=peak_heights,
    )
    coeff = np.array([fit[0][0]*fit_accuracy_fixer, fit[0][1]])
    ax.plot(charges, np.polyval(coeff, charges), label=f"fit (y = {coeff[0]:.3e}x + {coeff[1]:.3e}")
    ax.set_xlabel("Collected charge (C)")
    ax.set_ylabel("Pulse height (V)")
    ax.legend()

    # Special analysis for failed measurements
    failed_meas = []
    for meas in cal_data:
        if meas.peak_height[1] > 0.01:
            failed_meas.append(meas)
    if failed_meas:
        fig2: plt.Figure = plt.figure()
        fig2.suptitle("These calibration measurements have bad traces. Please fix!")
        for i, meas in enumerate(failed_meas):
            ax = fig2.add_subplot(len(failed_meas), 1, i+1)
            meas.plot_traces(ax)
            ax.set_title(f"V = {meas.voltage:.2f} V")

    return coeff


def analyze(cal_data: tp.List[MeasCal]):
    calibration(cal_data)
    # plt.show()
    # return

    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    hv_scan_folder = os.path.join(data_folder, "hv_scan")
    hv_scan(hv_scan_folder, "Am")
    hv_scan(hv_scan_folder, "Fe")

    spectra(
        os.path.join(data_folder, "spectra", "Am_10_1810_long_meas.mca"),
        os.path.join(data_folder, "spectra", "Fe_10_1810_long_meas.mca"),
        os.path.join(data_folder, "spectra", "Noise_1810.mca"),
        gain=10,
        voltage=1810
    )

    plt.show()
