import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import cm
from mpl_toolkits.mplot3d import Axes3D
import abc

class EM(abc.ABC):
    def __init__(self, n_curves=2, min_change=0):
        self._n_curves=n_curves
        self._min_change = min_change

    @abc.abstractmethod
    def _initialize(self, n_samples, n_features):
        pass

    @abc.abstractmethod
    def _poxgparams(self, X, k):
        pass

    def __e(self, X):
        for k in range(self._n_curves):
            self._zik[k] = self._poxgparams(X, k).T*self._mixture_coeff[k]
        self._zik /= np.sum(self._zik, axis=0)

    @abc.abstractmethod
    def _m(self, X):
        pass
            
    def fit(self, X, max_iters=100):
        X = np.matrix(X)
        self._initialize(X.shape[0], X.shape[1])
        # self.__m(X)
        zik = np.copy(self._zik)
        for i in range(max_iters):
            self.__e(X)
            if (np.abs(zik-self._zik) < self._min_change).all():
                break
            self._m(X)
            zik = np.copy(self._zik)
            # self.__plot(X)
        print('Training done in', i, 'steps')
        # print("Mean", self.__mu, sep='\n')
        # print("Covariance", self.__covariance, sep='\n')
        # print("Mixture coeff", self._mixture_coeff, sep='\n')
    
    def plot(self, X):
        if X.shape[1] > 2:
            raise ValueError("Dimensionality too high to visualize")
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        x, y = np.mgrid[X_min[0]:X_max[0]:50j, X_min[1]:X_max[1]:50j]
        xy = np.column_stack([x.flat, y.flat])
        z = np.zeros((self._n_curves, 50, 50))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for k in range(self._n_curves):
            Z = self._poxgparams(xy, k)
            z[k] = Z.reshape(x.shape)
        z = np.amax(z,axis=0)
        surf=ax.plot_wireframe(x,y,z, color='#0000005f', rstride=2, cstride=2)
        surf.set_facecolor((0,0,0,0.05))
        zik = np.argmax(self._zik,axis=0).T
        for k in range(self._n_curves):
            xd = X[np.where(zik==k)[0]]
            plt.scatter(xd[:,0], xd[:,1], s=1)
        plt.pause(0.01)



