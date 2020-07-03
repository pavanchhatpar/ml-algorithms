import numpy as np

from .naive_bayes import NaiveBayes

class MultinomialNB(NaiveBayes):
    def __init__(self, bins=4):
        self.bin_count = bins
    
    def __binData(self, X, y):
        self.mins = np.min(X, axis=0).T
        self.maxs = np.max(X, axis=0).T
        m = np.mean(X, axis=0).T
        m1 = np.mean(X[np.where(y==0)[0]], axis=0).T
        m2 = np.mean(X[np.where(y==1)[0]], axis=0).T
        self.boundaries = np.column_stack((self.mins, m1, m, m2, self.maxs))
        self.boundaries.sort()
        for j in range(X.shape[1]):
            X[:,j] = np.digitize(X[:,j], bins=np.array(self.boundaries[j]).flatten()) - 1
        return X

    def __binAny(self, X, y):
        self.boundaries = np.zeros((X.shape[1],self.bin_count+1))
        for j in range(X.shape[1]):
            _, self.boundaries[j] = np.histogram(np.array(X[:,j]).flatten(), bins=self.bin_count)
            X[:,j] = np.digitize(X[:, j], bins=np.array(self.boundaries[j]).flatten()) - 1
        return X

    def fit(self, X, y):
        X = np.matrix(X)
        y = np.matrix(y).T
        self.labels = np.unique(np.array(y))
        self.poy = np.zeros(self.labels.shape)
        for l in range(len(self.labels)):
            self.poy[l] = len(y[np.where(y==self.labels[l])[0]])/len(y)
        if self.bin_count == 4:
            X = self.__binData(X, y)
        else:
            X = self.__binAny(X, y)
        self.phij = np.zeros((len(self.labels), X.shape[1], self.bin_count))
        for i in range(len(self.labels)):
            for j in range(X.shape[1]):
                yi = np.where((y==self.labels[i]))[0]
                x = X[yi]
                for k in range(self.bin_count):
                    self.phij[i][j][k] = (len(x[np.where(x[:,j] == k)[0]])+1)/(len(x)+self.bin_count)

    def _poxjgy(self, X, j, yi):
        X = np.digitize(X[:,j], bins=np.array(self.boundaries[j]).flatten()) - 1
        X[np.where(X==self.bin_count)] = self.bin_count-1
        return self.phij[yi][j][X]