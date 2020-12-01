import os.path
import typing as tp

import numpy as np
# import pandas as pd


class MeasMCA:
    def __init__(self, path: str):
        roi_start_line = None
        data_start_line = None
        data_end_line = None
        conf_start_line = None
        conf_end_line = None
        status_start_line = None
        status_end_line = None
        data = []
        self.meta = {}
        self.roi = []
        self.conf = {}
        self.conf_comments = {}
        self.status = {}
        with open(path, encoding="cp1252") as file:
            for i, line in enumerate(file):
                if i == 0 and line != "<<PMCA SPECTRUM>>\n":
                    raise ValueError("Invalid header")
                if line == "<<DATA>>\n":
                    data_start_line = i
                elif line == "<<END>>\n":
                    data_end_line = i
                elif line == "<<DP5 CONFIGURATION>>\n":
                    conf_start_line = i
                elif line == "<<DP5 CONFIGURATION END>>\n":
                    conf_end_line = i
                elif line == "<<DPP STATUS>>\n":
                    status_start_line = i
                elif line == "<<DPP STATUS END>>\n":
                    status_end_line = i
                elif not roi_start_line and not data_start_line:
                    row = [cell.strip() for cell in line.split(" - ")]
                    if len(row) >= 2:
                        self.meta[row[0]] = row[1]
                elif roi_start_line and not data_start_line:
                    self.roi.append(line.split(" "))
                elif data_start_line and not data_end_line:
                    data.append(int(line))
                elif conf_start_line is not None and conf_end_line is None:
                    row = [cell.strip() for cell in line.split(";")]
                    pair = row[0].split("=")
                    self.conf[pair[0]] = self.convert_value(pair[1])
                    if len(row) > 1:
                        self.conf_comments[pair[0]] = row[1]
                elif status_start_line is not None and status_end_line is None:
                    row = [cell.strip() for cell in line.split(":", maxsplit=1)]
                    self.status[row[0]] = self.convert_value(row[1])

        self.data = np.array(data, dtype=np.int_)

        # self.data = pd.read_csv(
        #     path,
        #     skiprows=data_start_line+1, header=None,
        #     nrows=data_end_line - data_start_line-1,
        #     encoding="cp1252",
        #     engine="c"
        # )

    @staticmethod
    def convert_value(value: str) -> tp.Union[int, float, str]:
        if value == "OFF":
            return False
        if value == "ON":
            return True
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value


class TableMCA:
    def __init__(self, path: str):
        pass


if __name__ == "__main__":
    __meas = MeasMCA(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data/calibration/cal_10_5.mca"))
    print(__meas.data)
    print(__meas.conf)
    print(__meas.status)
