"""
Measurement classes
"""

import os.path
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from devices.mca import MeasMCA
from devices.oscilloscope import MeasOsc


class MeasCal:
    """Calibration measurement"""
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
        """Peak heights and stds of the oscilloscope traces"""
        peak_heights = np.array([meas_osc.peak_height for meas_osc in self.traces])
        return np.mean(peak_heights), np.std(peak_heights)

    def plot_traces(self, ax: plt.Axes):
        for trace, trace_ind in zip(self.traces, self.trace_inds):
            ax.plot(trace.t, trace.voltage, label=trace_ind)
            ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")


class Meas:
    """Spectrum measurement"""
    def __init__(self, path: str, gain: int, voltage: int):
        self.gain = gain
        self.voltage = voltage
        self.mca = MeasMCA(path)
