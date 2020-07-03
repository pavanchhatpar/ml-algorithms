import numpy as np
from .em import EM

class GaussianEM(EM):
    def _initialize(self, n_samples, n_features):
        self.__mu = np.matrix(np.random.rand(self._n_curves, n_features))
        self.__covariance = np.random.rand(self._n_curves, n_features, n_features)
        for k in range(self._n_curves):
            self.__covariance[k] += (np.identity(n_features)*0.5)
        self._mixture_coeff = np.random.rand(self._n_curves)
        self._mixture_coeff /= self._mixture_coeff.sum()
        self._zik = np.matrix(np.random.rand(self._n_curves, n_samples))
        self._zik /= self._zik.sum(axis=0)
        # print(self.__mu)
        # print(self.__covariance)
        # print(self._mixture_coeff)

    def _poxgparams(self, X, k):
        mu = self.__mu[k]
        covariance = self.__covariance[k]
        covariance = np.matrix(covariance)
        return np.exp(-0.5*np.multiply((X-mu)*np.linalg.pinv(covariance), (X-mu)).sum(axis=1))/np.sqrt(2*np.pi*np.linalg.det(covariance))

    def _m(self, X):
        for k in range(self._n_curves):
            num = np.zeros((X.shape[1], X.shape[1]))
            for i in range(X.shape[0]):
                num += self._zik[k,i]*((X[i]-self.__mu[k]).T*(X[i]-self.__mu[k]))
            den = self._zik[k].sum(axis=1)
            self.__covariance[k] = num/den
            self.__mu[k] = (self._zik[k]*X)/np.sum(self._zik[k], axis=1)
            self._mixture_coeff[k] = np.average(self._zik[k], axis=1)
