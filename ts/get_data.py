import numpy as np
import pandas as pd

filepath = (
    ("data/0/CalIt2.data", 2, 1),
    ("data/1/sales-of-shampoo-over-a-three-ye.csv", 1, 0),
    ("data/2/monthly-sunspots.csv", 1, 0),
)


def load_data(data_id: int = 1) -> np.ndarray:
    fp, ts_ind, time_ind = filepath[data_id]
    data = pd.read_csv(fp,
                       header=0,
                       parse_dates=[time_ind],
                       index_col=time_ind,
                       squeeze=True)
    return data
