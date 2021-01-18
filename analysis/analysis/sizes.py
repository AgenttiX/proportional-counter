import typing as tp

import numpy as np

import utils


def analyze_sizes(sizes: tp.Dict[str, tp.Union[np.ndarray, tp.List[float]]]):
    utils.print_title("Sizes")
    for name, data in sizes.items():
        print(name)
        print(f"µ = {np.mean(data)}")
        print(f"σ = {np.std(data)}")
    print()
