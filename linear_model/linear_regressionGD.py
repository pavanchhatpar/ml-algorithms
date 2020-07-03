import numpy as np
import time

from .base_linear import BaseLinear

class LinearRegressionGD(BaseLinear):
    
    def _h(self, X, W):
        return X*W

    def _costFunction(self, X, y, W):
        m = y.shape[0]
        err = self._h(X, W) - y
        if self.l2 == 0:
            return err.T*(err / (2*m))
        return err.T*(err / (2*m)) + (self.l2*W[1:,:].T)*(W[1:,:] / (2*m))

    def _gradCostFunction(self, X, y, W):
        m = y.shape[0]
        err = self._h(X, W) - y
        if self.l2 == 0:
            return X.T*(err / m)
        reg = self.l2*W
        reg[0][0] = 0
        return X.T*(err / m) + reg / m

    def _hessian(self, X, y, W):
        pass

