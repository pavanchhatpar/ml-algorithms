import numpy as np
import time

class LinearRegression:

    def __init__(self, l2_regularization=0):
        self.__l2=l2_regularization

    def fit(self, X, y):
        s = time.time()
        X = np.matrix(X)
        y = np.matrix(y).T
        l = self.__l2*np.identity(X.shape[1])
        # l[0][0]=0
        self.W = np.linalg.pinv(X.T*X+l)*X.T*y
        wo = y.mean()
        self.W = np.concatenate(([[wo]],self.W))
        return time.time() - s

    def predict(self, X):
        X = np.matrix(X)
        try:
            self.W
        except AttributeError:
            raise ValueError("No model has been fitted yet")
        if self.W.shape[0] != X.shape[1]:
            raise ValueError("X does not have same features as fitted")
        return np.array(X*self.W).flatten()