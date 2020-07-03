import numpy as np

class DecisionStump:
    def __init__(self, mode='optimal'):
        #mode can be 'random' or 'optimal'
        if mode not in ['random', 'optimal']:
            raise ValueError("Unsupported mode")
        self.__mode=mode
        self.model = None

    def fit(self, X, y, sample_weight, sorted_indices=None, thresholds=None):
        # y = np.array(y)
        # X = np.array(X)
        # sample_weight = np.array(sample_weight)
        errs_perfeat = []
        thresholds_perfeat = []
        m = X.shape[0]
        if self.__mode == 'random':
            j = np.random.randint(X.shape[1])
            # Xj = np.copy(X[:,j])
            # yc = np.copy(y).flatten()
            # wc = np.copy(sample_weight)
            ind = np.random.randint(X.shape[0])
            # preds = (X[:, j]>X[ind, j]).astype(int)
            # notsame = yc!=preds
            self.model = {
                "feature": j,
                "threshold": X[ind, j],
                # "err": np.abs(0.5-np.dot(notsame, wc))
            }
            return
        for j in range(X.shape[1]): 
            Xj = X[:,j]
            # yc = np.copy(y)
            # wc = np.copy(sample_weight)
            # sorted_indices = np.argsort(Xj)
            Xj = Xj[sorted_indices[:,j]]
            yc = y[sorted_indices[:,j]]
            wc = sample_weight[sorted_indices[:,j]]
            # thresholds = list(np.unique(Xj))
            # thresholds.insert(0, Xj[0]-1)
            errs = []
            err = wc[np.where(yc==0)].sum()
            errs.append(np.abs(0.5-err))
            i=0
            for thres in thresholds[j][1:]:
                while i < m and Xj[i]==thres:
                    if yc[i] != 0:
                        err += wc[i]
                    else:
                        err -= wc[i]
                    i += 1
                errs.append(np.abs(0.5-err))
            ind = np.argmax(errs)
            errs_perfeat.append(errs[ind])
            thresholds_perfeat.append(thresholds[j][ind])
        ind = np.argmax(errs_perfeat)
        self.model = {
            "feature": ind,
            "threshold": thresholds_perfeat[ind],
            "err": errs_perfeat[ind]
        }

    def predict(self, X):
        if self.model == None:
            raise ValueError("No model has been fit yet")
        return (X[:,self.model["feature"]] > self.model["threshold"]).astype(int)