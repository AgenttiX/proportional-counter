from analysis import analyze, MeasCal


calibration = [
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
        traces=(81, 85),
        file="cal_10_68"
    ),
    MeasCal(
        voltage=1.895,
        traces=(86, 90),
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


if __name__ == "__main__":
    analyze(calibration)
