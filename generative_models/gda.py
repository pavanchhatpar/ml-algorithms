import numpy as np
class GDA:
    def __init__(self, single_covariance=True):
        self.__sigma = None
        self.__poy = None
        self.__single_covariance = single_covariance
    
    def __poxgy(self, X, yi):
        mui = self.__mu[yi]
        Xmui = X-mui
        d = X.shape[1]
        if self.__single_covariance:
            sigma = np.matrix(self.__sigma)
            
        else:
            sigma = np.matrix(self.__sigma[yi])
        return np.exp(-0.5*np.multiply((Xmui)*np.linalg.pinv(sigma), (Xmui)).sum(axis=1))# / np.sqrt(((2*np.pi)**d)*np.linalg.det(sigma))

    def fit(self, X, y):
        y = np.matrix(y).T
        X = np.matrix(X)
        self.__labels = np.unique(np.array(y))
        self.__poy = np.array([y[np.where(y==li)].shape[1]/len(y) for li in self.__labels])
        self.__mu = np.zeros((self.__poy.shape[0], X.shape[1]))
        for l in range(len(self.__labels)):
            self.__mu[l] = np.average(X[np.where(y==self.__labels[l])[0]], axis=0)
        if self.__single_covariance:
            muyi = np.matrix([self.__mu[np.where(self.__labels==yi[0,0])[0][0]] for yi in y]) # n x d
            Xmuyi = X - muyi
            self.__sigma = (Xmuyi).T*(Xmuyi) / X.shape[0]
        else:
            self.__sigma = np.zeros((len(self.__labels), X.shape[1], X.shape[1]))
            for l in range(len(self.__labels)):
                x = X[np.where(y==l)[0]]
                Xmu = x - self.__mu[l]
                self.__sigma[l] = Xmu.T*Xmu / len(x)

    def predict(self, X):
        try:
            self.__mu
        except AttributeError:
            raise ValueError("No model has been fitted yet")
        X = np.matrix(X)
        probs = np.matrix([np.array(self.__poxgy(X, yi)).flatten()*self.__poy[yi] for yi in range(len(self.__labels))]).T

        return np.array(probs.argmax(axis=1)).flatten()

