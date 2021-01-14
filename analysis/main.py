"""
This is the main script for running the analysis workflow.
The analysis is run as one large workflow, since some phases need results from the others.
For example the high voltage scan is based on the charge calibration of the MCA.
"""

import os.path

import matplotlib.pyplot as plt

import analysis
from meas import MeasCal

###
# Size measurements
###
# From the file "size_measurements.txt"
can_diam_outer = [66.01, 65.74, 65.81, 65.70, 65.89]
can_top_diam_inner = [47.50, 47.78, 47.58, 47.76, 47.47]
can_top_diam_outer = [53.84, 53.88, 53.98, 53.89, 53.82]
can_bottom_diam_inner = [45.51, 44.93, 45.40, 45.43, 45.47]
can_thickness_top = [240, 250, 240, 250, 280]
can_thickness_opened = [102, 103, 102, 100, 102]
long_brass_tube_length = [29.25, 29.26, 29.27, 29.27, 29.25]
short_brass_tube_length = [10.05, 9.98, 9.99, 10.04]
brass_tube_diameter = [992, 990, 991, 990, 987]
brass_tube_with_connector = [27.30, 27.15, 27.32, 27.59, 27.23]
sizes = {
    "Can outer diameter (mm)": can_diam_outer,
    "Can top end inner diameter (mm)": can_top_diam_inner,
    "Can top end outer diameter (mm)": can_top_diam_outer,
    "Can bottom end inner diameter (mm)": can_bottom_diam_inner,
    "Can thickness (µm, from top section)": can_thickness_top,
    "Can thickness (µm, from opened and straightened can)": can_thickness_opened,
    "Long brass tube length (mm)": long_brass_tube_length,
    "Short brass tube length (mm)": short_brass_tube_length,
    "Brass tube diameter (µm)": brass_tube_diameter,
    "Brass tube with connector (mm)": brass_tube_with_connector,
}

###
# Electronics calibration
###
cal_data = [
    MeasCal(
        voltage=0.067,
        traces=(51, 55),
        file="cal_10_1.8"
    ),
    MeasCal(
        voltage=0,
        traces=(56, 60),
        file="cal_10_5"
    ),
    MeasCal(
        voltage=0.127,
        traces=(61, 65),
        file="cal_10_10"
    ),
    MeasCal(
        voltage=0.285,
        traces=(66, 70),
        file="cal_10_17"
    ),
    MeasCal(
        voltage=0.69,
        traces=(71, 75),
        file="cal_10_34"
    ),
    MeasCal(
        voltage=1.07,
        traces=(76, 80),
        file="cal_10_51"
    ),
    MeasCal(
        voltage=1.49,
        traces=(84, 85),    # Trace 81, 82 and 83 don't have a signal
        file="cal_10_68"
    ),
    MeasCal(
        voltage=1.895,
        traces=(86, 89),    # Trace 90 does not have a signal
        file="cal_10_85"
    ),
    MeasCal(
        voltage=2.27,
        traces=(91, 95),
        file="cal_10_102"
    ),
    MeasCal(
        voltage=2.695,
        traces=(96, 100),
        file="cal_10_119"   # Or should this be 121?
    ),
    MeasCal(
        voltage=3.24,
        traces=(101, 105),
        file="cal_10_136"
    ),
    MeasCal(
        voltage=3.27,
        traces=(106, 109),
        file="cal_10_145"
    )
]
pulser_voltage_rel_std = 0.05

###
# HV scan
###
cal_gain = 10
# Adjustment has a higher uncertainty than a single measurement
hv_adjustment_std = 5  # volts

# Differential nonlinearity: relative x axis error
mca_diff_nonlin = 0.6e-2
# Integral nonlinearity: relative y axis error
mca_int_nonlin = 0.02e-2

preamp_int_nonlin = 0.05e-2
spec_amp_int_nonlin = 0.05e-2

###
# Spectral
###
hv_std = 1  # volts


def main():
    """
    The main analysis function.
    Comment out sections as needed.
    """
    analysis.analyze_sizes(sizes)

    cal_coeff, cal_coeff_covar = analysis.calibration(
        cal_data,
        mca_diff_nonlin=mca_diff_nonlin,
        pulser_voltage_rel_std=pulser_voltage_rel_std
    )
    # plt.show()
    # return

    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    hv_scan_folder = os.path.join(data_folder, "hv_scan")
    analysis.hv_scans(
        hv_scan_folder,
        cal_coeff=cal_coeff,
        cal_coeff_covar=cal_coeff_covar,
        cal_gain=cal_gain,
        diff_nonlin=mca_diff_nonlin,
        int_nonlin=mca_int_nonlin,
        voltage_std=hv_adjustment_std,
        gain_rel_std=spec_amp_int_nonlin
    )

    analysis.spectra(
        os.path.join(data_folder, "spectra", "Am_10_1810_long_meas.mca"),
        os.path.join(data_folder, "spectra", "Fe_10_1810_long_meas.mca"),
        os.path.join(data_folder, "spectra", "Noise_1810.mca"),
        gain=10,
        voltage=1810,
        voltage_std=hv_std
    )

    plt.show()


if __name__ == "__main__":
    main()
