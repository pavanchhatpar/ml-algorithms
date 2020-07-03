from .em import EM

import numpy as np
from math import factorial

class BernoulliEM(EM):
    def _m(self, X):
        yk = X.sum(axis=1)
        d = X.shape[1]
        for k in range(self._n_curves):
            self._mixture_coeff[k] = np.average(self._zik[k], axis=1)
            self.__qk[k] = (self._zik[k]*yk)/(d*np.sum(self._zik[k], axis=1))
    
    def _initialize(self, n_samples, n_features):
        self._mixture_coeff = np.random.rand(self._n_curves)
        self._mixture_coeff /= self._mixture_coeff.sum()
        self._zik = np.matrix(np.random.rand(self._n_curves, n_samples))
        self._zik /= self._zik.sum(axis=0)
        self.__qk = np.random.rand(self._n_curves)

    def __comb(self, n, k):
        ans = n
        i = n-1
        while i > n-k:
            ans *= i
            i -= 1
        ans /= factorial(k)
        return ans

    def _poxgparams(self, X, k):
        d = X.shape[1]
        ans = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            yi = X[i].sum()
            ans[i] = self.__comb(d, yi)*(self.__qk[k]**yi)*((1-self.__qk[k])**(d-yi))
        return ans.T
