import numpy as np
import abc

class NaiveBayes(abc.ABC):
    @abc.abstractmethod
    def fit(self, X, y):
        pass

    def _poxjgy(self, X, j, yi):
        pass 

    def predict(self, X):
        ans = np.zeros((self.labels.shape[0], X.shape[0]))
        for i in range(self.labels.shape[0]):
            for j in range(X.shape[1]):
                ans[i] += np.log(self._poxjgy(X, j, i))
            ans[i] += np.log(self.poy[i])
        
        # ans /= ans.sum(axis=0)
        # ans[np.where(ans<1e-15)] = 1e-15
        # ans[np.where(ans>(1-1e-15))] = 1-1e-15
        return ans

                
