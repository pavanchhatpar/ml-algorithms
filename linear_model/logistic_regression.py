import numpy as np
import time

from .base_linear import BaseLinear

class LogisticRegression(BaseLinear):

    def _h(self, X, W):
        return self.__sigmoid(X*W)

    def __sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def _costFunction(self, X, y, W):
        m = y.shape[0]
        hx = self._h(X, W)
        hx[hx == 1] = 1-(1e-15)
        hx[hx == 0] = (1e-15)
        if self.l2 == 0:
            return -1/m*(y.T*np.log(hx) + (1-y).T*np.log((1-hx)))
        else:
            return -1/m*(y.T*np.log(hx) + (1-y).T*np.log((1-hx))) + 1/m*self.l2*W[1:,:].T*W[1:,:]

    def _gradCostFunction(self, X, y, W):
        m = y.shape[0]
        err = self._h(X, W) - y
        if self.l2 == 0:
            return X.T*err
        reg = self.l2*W
        reg[0][0] = 0
        return 1/m*(X.T*err) + 1/m*reg

    def _hessian(self, X, y, W):
        m = y.shape[0]
        hx = self._h(X, W)
        regH = self.l2*np.identity(X.shape[1])
        regH[0][0] = 0
        S = np.matrix(np.diag(np.array(np.multiply(hx, 1-hx)).flatten()))
        H = X.T*S*X + regH
        return H