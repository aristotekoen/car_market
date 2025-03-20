import numpy as np
import pandas as pd


def reliability_score(q1, q2, q3, c, shape):
    uncertainty = (q3 - q1) / q2
    return 1 - (1 / (1 + np.exp(-shape * (uncertainty - c))))
