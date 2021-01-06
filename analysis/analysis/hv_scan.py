import functools
import glob
import os.path
import typing as tp

import numpy as np

from devices.mca import MeasMCA
import fitting


def read_hv_scan(folder: str, prefix: str) -> tp.Tuple[np.ndarray, np.ndarray, tp.List[MeasMCA]]:
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


def hv_scan(folder: str, prefix: str):
    """Analyze HV scan data"""
    print(f"{prefix} HV scan")
    gains, voltages, mcas = read_hv_scan(folder, prefix)
    print("Gains:")
    print(gains)
    print("Voltages:")
    print(voltages)
    print(f"Voltage range: {min(voltages)} - {max(voltages)} V")

    # fig: plt.Figure = plt.figure()
    # fig.suptitle(f"{prefix} HV scan")
    # ax: plt.Axes = fig.add_subplot()
    # true_volts = voltages / gains
    # ax.plot(mcas[0].data)

    if prefix == "Fe":
        fitting.fit_fe_hv_scan(mcas)
    if prefix == "Am":
        fitting.fit_am_hv_scan(mcas)
