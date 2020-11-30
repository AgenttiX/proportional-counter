import pandas as pd


class MeasOsc:
    """Wavesurfer 3024z oscilloscope measurement files"""
    def __init__(self, path: str):
        self.data = pd.read_csv(
            path,
            header=None, skiprows=6,
            names=["0", "1", "2", "t", "V"],
            engine="c"
        )
        print(self.data)


if __name__ == "__main__":
    MeasOsc("../data/calibration/C1Trace00000.txt")
