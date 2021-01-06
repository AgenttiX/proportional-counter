import typing as tp

import numpy as np


def analyze_sizes(sizes: tp.Dict[str, tp.List[float]]):
    print("Sizes")
    for name, data in sizes.items():
        print(name)
        print(f"µ = {np.mean(data)}")
        print(f"σ = {np.std(data)}")
    print()
