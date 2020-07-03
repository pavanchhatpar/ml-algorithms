import numpy as np

from .naive_bayes import NaiveBayes

class BernoulliNB(NaiveBayes):
    def fit(self, X, y):
        X = np.matrix(X)
        y = np.matrix(y).T
        self.labels = np.unique(np.array(y))
        self.poy = np.zeros(self.labels.shape)
        for l in range(len(self.labels)):
            self.poy[l] = len(y[np.where(y==self.labels[l])[0]])/len(y)
        self.mu = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            self.mu[i] = np.nanmean(X[:,i])
            nan = np.where(np.isnan(X[:,i]))[0]
            X[:,i] = (X[:,i] > self.mu[i]).astype(int)
            X[nan,i] = np.nan
        self.phij = np.zeros((len(self.labels), X.shape[1], 2))
        for i in range(len(self.labels)):
            for j in range(X.shape[1]):
                yi = np.where((y==self.labels[i]))[0]
                x = X[yi]
                self.phij[i][j][0] = (len(x[np.where(x[:,j] == 0)[0]])+1)/(len(x)+2)
                self.phij[i][j][1] = (len(x[np.where(x[:,j] == 1)[0]])+1)/(len(x)+2)

    def _poxjgy(self, X, j, yi):
        nonNan = np.where(~np.isnan(X[:,j]))[0]
        X[nonNan, j] = (X[nonNan,j] > self.mu[j]).astype(int)
        X = X[:, j]
        ret = []
        for Xi in X:
            if np.isnan(Xi):
                ret.append(1)
            else:
                ret.append(self.phij[yi][j][int(Xi)])
        return np.array(ret)#self.phij[yi][j][X]