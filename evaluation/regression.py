import numpy as np

def mean_squared_error(target, predicted):
    if len(target) != len(predicted):
        raise ValueError("Both arrays should be of equal length")
    return np.average((target-predicted)**2)