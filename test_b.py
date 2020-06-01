"""Random pymfe MFETS tests."""
import sys

import numpy as np
import pandas as pd

import pymfe.tsmfe


def load_data(data_id: int, max_obs_num: int = 512) -> np.ndarray:
    data = pd.read_csv("data/comp-engine-export-sample.20200503.csv",
                       header=0,
                       index_col=0,
                       nrows=1,
                       skiprows=np.arange(1, data_id + 1),
                       squeeze=True,
                       low_memory=True)

    ts = np.asarray(data.values[0].split(","), dtype=float)[-max_obs_num:]

    return ts


def _test() -> None:
    if len(sys.argv) <= 3:
        print("usage:", sys.argv[0], "<data_id> <random_seed> <precomp 0/1>")
        sys.exit(1)

    data_id = int(sys.argv[1])
    random_state = int(sys.argv[2])
    precomp = bool(int(sys.argv[3]))

    if not 0 <= data_id < 20:
        print(f"Require 0 <= data_id < 20 (got {data_id}).")
        sys.exit(2)

    print("Chosen id:", data_id)
    print("Random_state:", random_state)

    ts = load_data(data_id)

    extractor = pymfe.tsmfe.TSMFE(random_state=random_state)
    extractor.fit(ts=ts, precomp_groups="all" if precomp else None)
    res = extractor.extract()

    for name, val in zip(*res):
        print(f"{name:<40}: {val:.4f}")


if __name__ == "__main__":
    _test()
