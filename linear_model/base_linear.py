import numpy as np
import time
import abc
import sys
import os

sys.path.append(os.path.abspath('../optimizer'))
from optimizer import GradientDescent, Newtons

class BaseLinear(abc.ABC):
    def __init__(self, optimizer, printAfter=500, printer=None, num_iters=None, min_reduction=0, 
    l2_regularization=0, batch_size=None):
        self.l2=l2_regularization
        self.__num_iters = num_iters
        self.__min_reduction = min_reduction
        self.__batch_size = batch_size
        self.__optimizer = optimizer
        self.__printAfter = printAfter
        self.__printer = printer

    @abc.abstractmethod
    def _h(self, X, W):
        pass

    @abc.abstractmethod
    def _costFunction(self, X, y, W):
        pass

    @abc.abstractmethod
    def _gradCostFunction(self, X, y, W):
        pass

    @abc.abstractmethod
    def _hessian(self, X, y, W):
        pass

    def __gradientDescent(self, X, y, W, batch_size):
        totalBatches=0
        epoch=0
        batchI = 0
        while True:
            X_batch = X[batch_size*batchI: batch_size*(batchI+1), :]
            y_batch = y[batch_size*batchI: batch_size*(batchI+1), :]
            if len(X_batch) == 0:
                batchI = 0
                epoch += 1
                continue
            cost = self._costFunction(X_batch, y_batch, W)
            W = self._updateWeights(X_batch, y_batch, W)
            deltaCost = cost - self._costFunction(X_batch, y_batch, W)
            batchI += 1
            totalBatches += 1
            if (totalBatches%500==0):
                print("Epoch:", epoch, ", Batches:", totalBatches, ", Cost:", cost-deltaCost)
            if self.__num_iters != None and epoch >= self.__num_iters:
                self.W = W
                break
            if np.abs(deltaCost) <= self.__min_reduction:
                self.W = W
                break

    def fit(self, X, y, warm_start=False):
        s = time.time()
        X=np.matrix(X).astype(np.float_)
        y=np.matrix(y).astype(np.float_).T
        if warm_start:
            try:
                initW = self.W
            except AttributeError:
                initW = np.matrix(np.random.rand(X.shape[1])).astype(np.float_).T#/50#*y.mean()/X.mean(axis=0).T
        else:
            initW = np.matrix(np.random.rand(X.shape[1])).astype(np.float_).T#/50#*y.mean()/X.mean(axis=0).T
        if self.__printer == None:
            self.__printer = lambda epochs, W, X, y, loss: print("Epochs:", epochs, ", Cost:", loss(X, y, W))
        if type(self.__optimizer) is GradientDescent:
            self.W = self.__optimizer.minimize(self._costFunction, self._gradCostFunction, 
            X, y, initW, self.__min_reduction, self.__num_iters, self.__printAfter, self.__printer, batch_size=self.__batch_size)
        elif type(self.__optimizer) is Newtons:
            self.W = self.__optimizer.minimize(self._costFunction, self._gradCostFunction, self._hessian,
            X, y, initW, self.__min_reduction, self.__num_iters, self.__printAfter, self.__printer)
        return self._costFunction(X, y, self.W), time.time() - s


    def setPrinter(self, printer):
        self.__printer=printer

    def predict(self, X, W=None):
        try:
            if W==None:
                w=self.W
        except AttributeError:
            raise ValueError("No model has been fitted yet")
        except ValueError:
            w=W
        if w.shape[0] != X.shape[1]:
            raise ValueError("X does not have same features as fitted")
        X=np.matrix(X).astype(np.float_)
        return np.array(self._h(X, w)).flatten()