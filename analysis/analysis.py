import os.path
import typing as tp

import matplotlib.pyplot as plt

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
        else:
            raise NotImplementedError("Unknown trace settings")


def analyze(calibration: tp.List[MeasCal]):
    meas = calibration[0]
    osc = meas.traces[0]
    fig = plt.figure()
    ax = fig.add_subplot()
    plot.plot_osc(osc, ax)
    plt.show()
