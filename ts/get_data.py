import numpy as np
import pandas as pd

filepath = (
    ("data/0/CalIt2.data", 2),
    ("data/1/sales-of-shampoo-over-a-three-ye.csv", 1),
    ("data/2/monthly-sunspots.csv", 1),
)


def load_data(data_id: int = 1) -> np.ndarray:
    fp, ts_ind = filepath[data_id]
    data = pd.read_csv(fp)
    return data.iloc[:, ts_ind].values
