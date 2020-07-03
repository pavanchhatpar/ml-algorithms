import numpy as np

from .base_linear import BaseLinear

class PerceptronClassifier(BaseLinear):

    # def __init__(self, threshold, optimizer, num_iters=None, min_reduction=0, 
    # l2_regularization=0, batch_size=None):
    #     super().__init__(optimizer, num_iters=num_iters, min_reduction=min_reduction, 
    #     l2_regularization=l2_regularization, batch_size=batch_size)
    #     self.__threshold = threshold


    def _h(self, X, W):
        return X*W

    def _costFunction(self, X, y, W):
        Xc = X.copy()
        Xc[np.array(y==-1).flatten()] = -Xc[np.array(y==-1).flatten()]
        hx = self._h(Xc, W)
        return -hx[hx[:,0]<=0].sum()

    def _gradCostFunction(self, X, y, W):
        Xc = X.copy()
        Xc[np.array(y==-1).flatten()] = -Xc[np.array(y==-1).flatten()]
        hx = self._h(Xc, W)
        return -Xc[np.array(hx[:,0]<=0).flatten()].sum(axis=0).T

    def _hessian(self, X, y, W):
        pass