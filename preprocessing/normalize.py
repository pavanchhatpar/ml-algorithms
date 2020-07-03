import numpy as np
def shiftNScale(X):
    X = np.array(X)
    if len(X.shape) != 2:
        raise ValueError("X needs to be 2-dimensional")
    norms = (X.min(axis=0), X.max(axis=0) - X.min(axis=0))
    if (norms[1] == 0).any():
        return X, norms
    return (X - norms[0])/norms[1], norms

def zeroMeanUnitVariance(X):
    X = np.array(X)
    if len(X.shape) != 2:
        raise ValueError("X needs to be 2-dimensional")
    norms = (X.mean(axis=0), X.std(axis=0))
    if (norms[1] == 0).any():
        return X, norms
    return (X - norms[0])/norms[1], norms

def zeroMean(X):
    X = np.array(X)
    if len(X.shape) != 2:
        raise ValueError("X needs to be 2-dimensional")
    norms = (X.mean(axis=0), np.ones(X.shape[1]))
    if (norms[1] == 0).any():
        return X, norms
    return (X - norms[0])/norms[1], norms