import numpy as np

import logging
import time

class Adaboost:
    def __format(self, t):
        em = (int)(t / 60)
        t %= 60
        eh = (int)(em / 60)
        em %= 60
        return '{:02}:{:02}:{:02}'.format(eh, em, t)

    def __init__(self, weakLearner, **args):
        self.__weakLearner = weakLearner
        self.__args = args
        self.model = None
        self.__logger = logging.getLogger("Adaboost")
        self.__logger.setLevel(logging.DEBUG)
        if not self.__logger.hasHandlers():
            ch = logging.FileHandler('ada.log')
            ch.setFormatter(logging.Formatter('%(name)s:%(message)s'))
            self.__logger.addHandler(ch)
    
    def fit(self, X, y, iterations, **fit_args):
        s = time.time()
        d = np.ones(X.shape[0])
        d /= X.shape[0]
        learners = []
        alphas = []
        errs = []
        for i in range(iterations):
            learner = self.__weakLearner(**self.__args)
            # self.__logger.debug("Fitting weak learner #%d", i)
            learner.fit(X, y, sample_weight=d, **fit_args)
            y_pred = learner.predict(X)
            notsame = (y!=y_pred).astype(int)
            err = np.dot(notsame, d)
            if err == 0:
                err = 1e-12
            if err == 1:
                err = 1-1e-12
            errs.append(err)
            alpha = 0.5*np.log((1-err)/err)
            notsame = notsame*2 - 1
            d_new = np.zeros(d.shape)
            for j in range(d.shape[0]):
                d_new[j] = d[j]*np.exp(notsame[j]*alpha)
            d_new = np.true_divide(d_new, np.sum(d_new))
            if (d == d_new).all():
                self.__logger.warning("STUCK at " + str(i))
                continue
            d = d_new
            learners.append(learner)
            alphas.append(alpha)
            # print(err)

        self.model = {
            "learners": learners,
            "alphas": alphas,
            "round_errors": errs
        }
        timeTaken = time.time() - s
        self.__logger.debug("Time taken- " + self.__format(timeTaken))
    
    def predict(self, X, model=None):
        if model == None:
            model = self.model
        if model == None:
            raise ValueError("No model has been fit yet")
        prediction = np.zeros(X.shape[0])
        for i in range(len(model["learners"])):
            prediction += model["alphas"][i]*model["learners"][i].predict(X)
        return prediction


