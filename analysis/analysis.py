import glob
import os.path
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from devices.mca import MeasMCA
from devices.oscilloscope import MeasOsc
import plot


class MeasCal:
    def __init__(
            self,
            voltage: float, traces: tp.Tuple[int, int],
            file: str):
        self.voltage = voltage
        if not os.path.isabs(file):
            if file.startswith("cal"):
                folder = "calibration"
            elif file.startswith(("Am", "Fe")):
                folder = "hv_scan"
            else:
                raise ValueError(
                    "Cannot deduce file location from the file name. "
                    "Please use and absolute path.")
            file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data",
                folder,
                f"{file}.mca"
            )
        self.mca = MeasMCA(file)

        if isinstance(traces, tuple) and len(traces) == 2:
            inds = range(traces[0], traces[1]+1)
        elif isinstance(traces, list):
            inds = traces
        else:
            raise NotImplementedError("Unknown trace settings")
        folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "calibration"
        )
        self.traces = [
            MeasOsc(
                os.path.join(folder, f"C1Trace{str(ind).zfill(5)}.txt")
            )
            for ind in inds
        ]
        self.trace_inds = np.array(inds)

    @property
    def peak_height(self):
        peak_heights = np.array([meas_osc.peak_height for meas_osc in self.traces])
        return np.mean(peak_heights), np.std(peak_heights)

    def plot_traces(self, ax: plt.Axes):
        for trace, trace_ind in zip(self.traces, self.trace_inds):
            ax.plot(trace.t, trace.voltage, label=trace_ind)
            ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")


class Meas:
    def __init__(self, path: str, gain: int, voltage: int):
        self.gain = gain
        self.voltage = voltage
        self.mca = MeasMCA(path)


def read_hv_scan(folder: str, prefix: str):
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
        preamp_capacitance: float = 1e-12):
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

    ax.errorbar(charges, peak_heights, yerr=peak_stds, fmt=".", capsize=3)
    fit_accuracy_fixer = 1e14
    fit = curve_fit(lambda x, a, b: a*x + b, xdata=charges*fit_accuracy_fixer, ydata=peak_heights)
    coeff = np.array([fit[0][0]*fit_accuracy_fixer, fit[0][1]])
    ax.plot(charges, np.polyval(coeff, charges), label="linear fit")
    ax.set_xlabel("Collected charge (C)")
    ax.set_ylabel("Pulse height (V)")

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


def analyze(cal_data: tp.List[MeasCal]):
    calibration(cal_data)
    plt.show()
    return

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
