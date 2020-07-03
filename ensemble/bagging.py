import numpy as np
import logging

class Bagging:
    def __init__(self, clf, **clf_args):
        self.__clf = clf
        self.__clf_args = clf_args
        self.model = None
        self.__logger = logging.getLogger("Bagging")
        self.__logger.setLevel(logging.DEBUG)
        if not self.__logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(name)s:%(message)s'))
            self.__logger.addHandler(ch)

    def __sample(self, X, y):
        n = X.shape[0]
        indices = np.random.randint(n,size=n)
        return X[indices], y[indices]

    def fit(self, X, y, iterations):
        X = np.array(X)
        y = np.array(y)
        learners = []
        for i in range(iterations):
            self.__logger.debug("Fitting learner #%d", i)
            X_sampled, y_sampled = self.__sample(X, y)
            learner = self.__clf(**self.__clf_args)
            learner.fit(X_sampled, y_sampled)
            learners.append(learner)
        self.model = {
            "learners": learners
        }

    def predict(self, X):
        if self.model == None:
            raise ValueError("No model has been fit yet")
        predictions = np.zeros(X.shape[0])
        for learner in self.model['learners']:
            predictions += learner.predict(X)
        predictions = predictions/len(self.model['learners'])
        return predictions
