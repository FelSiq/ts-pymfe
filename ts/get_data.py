import numpy as np
import pandas as pd


def load_data() -> np.ndarray:
    data = pd.read_csv("data/1/sales-of-shampoo-over-a-three-ye.csv")
    return data.iloc[:, 1].values
