import numpy as np
import logging

class GradientBoosting:
    def __init__(self, clf, **clf_args):
        self.model = None
        self.__clf = clf
        self.__clf_args = clf_args
        self.__logger = logging.getLogger("Gradient boosting")
        self.__logger.setLevel(logging.DEBUG)
        if not self.__logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(name)s:%(message)s'))
            self.__logger.addHandler(ch)

    def fit(self, X, y, iterations):
        X = np.array(X)
        y = np.array(y)
        learners = []
        for i in range(iterations):
            self.__logger.debug("Fitting learner #%d", i)
            learner = self.__clf(**self.__clf_args)
            learner.fit(X, y)
            y -= learner.predict(X)
            if (y==0).all():
                self.__logger.debug("Converged")
            learners.append(learner)
        self.model = {
            "learners": learners
        }

    def predict(self, X):
        if self.model == None:
            raise ValueError("No model has been fit yet")

        prediction = np.zeros(X.shape[0])
        for learner in self.model['learners']:
            prediction += learner.predict(X)
        
        return prediction
