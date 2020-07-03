import numpy as np

from .naive_bayes import NaiveBayes

class GaussianNB(NaiveBayes):
    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon

    def fit(self, X, y):
        X = np.matrix(X)
        y = np.matrix(y).T
        self.labels = np.unique(np.array(y))
        self.poy = np.zeros(self.labels.shape)
        for l in range(len(self.labels)):
            self.poy[l] = len(y[np.where(y==self.labels[l])[0]])/len(y)
        self.mu = np.zeros((X.shape[1],len(self.labels)))
        self.variance = np.zeros((X.shape[1],len(self.labels)))
        for i in range(len(self.labels)):
            x = X[np.where(y==self.labels[i])[0]]
            self.mu[:, i] = x.mean(axis=0)
            self.variance[:, i] = np.maximum((np.square(x-self.mu[:,i])).mean(axis=0), self.epsilon)
    
    def _poxjgy(self, X, j, yi):
        ans = np.exp(-np.square(X[:, j]-self.mu[j, yi])/(2*self.variance[j, yi])) / np.sqrt(2*np.pi*self.variance[j, yi])
        ans[np.where(ans==0)] = 1e-10
        return ans

