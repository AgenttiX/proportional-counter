import os.path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class MeasOsc:
    """Wavesurfer 3024z oscilloscope measurement files"""
    def __init__(self, path: str):
        self.data = pd.read_csv(
            path,
            header=None, skiprows=6,
            names=["0", "1", "2", "t", "V"],
            engine="c"
        )
        self.t = self.data["t"].to_numpy()
        self.voltage = self.data["V"].to_numpy()

    @property
    def zero_level(self) -> float:
        return self.voltage[:self.voltage.size//3].mean()

    @property
    def peak_height(self) -> float:
        return self.voltage.max() - self.zero_level

    @property
    def rise_time(self):
        peak_ind = np.argmax(self.voltage)
        noise_max = np.max(self.voltage[:self.voltage.size//3])
        below_noise_max = np.argwhere(self.voltage[:peak_ind] < noise_max)
        return self.t[peak_ind] - self.t[below_noise_max[-1]]

    @property
    def exp_decay_fit(self):
        peak_ind = np.argmax(self.voltage)
        fit = curve_fit(lambda t, a, b: a*np.exp(b*t), self.t[peak_ind:], self.voltage[peak_ind:])
        return fit[0]

    @property
    def decay_time(self):
        return self.exp_decay_fit[0]


if __name__ == "__main__":
    __meas = MeasOsc(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data/calibration/C1Trace00000.txt"))
    print(__meas.data)
