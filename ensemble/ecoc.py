import numpy as np
import logging

class ECOC:
    def __init__(self, n_ecoc, clf, **clf_args):
        self.__clf = clf
        self.__clf_args = clf_args
        self.__n_ecoc = n_ecoc
        self.model = None
        self.__logger = logging.getLogger("ECOC")
        self.__logger.setLevel(logging.DEBUG)
        if not self.__logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(name)s:%(message)s'))
            self.__logger.addHandler(ch)

    def __randomECOC(self, n_labels):
        codes = np.zeros((n_labels, self.__n_ecoc))
        i = 0
        while i < n_labels:
            code = np.random.choice([0,1], self.__n_ecoc)
            if not (code==codes).all(axis=1).any():
                codes[i] = code
                i += 1
        return codes

    def fit(self, X, y, **fit_args):
        n_labels = len(np.unique(y))
        ecoc = self.__randomECOC(n_labels)
        y = ecoc[y]
        learners = []
        for i in range(self.__n_ecoc):
            yc = y[:,i]
            learner = self.__clf(**self.__clf_args)
            self.__logger.debug("Fitting learner for bit #%d ", i)
            learner.fit(X, yc, **fit_args)
            learners.append(learner)
        self.model = {
            "learners": learners,
            "ecoc": ecoc
        }

    def __hamming_distance(self, code1, code2):
        return (code1 != code2).astype(int).sum()

    def predict(self, X, model=None):
        if model == None:
            model = self.model
        if model == None:
            raise ValueError("No model has been fit yet")
        code = np.zeros((X.shape[0], self.__n_ecoc))
        for i, learner in enumerate(model['learners']):
            bits = (learner.predict(X) > 0).astype(int)
            code[:,i] = bits
        predictions = np.zeros(X.shape[0])
        for i, c in enumerate(code):
            hamming_distances = []
            for ecoc in model['ecoc']:
                hamming_distances.append(self.__hamming_distance(ecoc, c))
            predictions[i]=np.argmin(hamming_distances)
        return predictions
