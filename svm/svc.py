import numpy as np
from sklearn.metrics import roc_curve

class SVC:
    def __init__(self, optimizer, **args):
        self.model = {
            'optimizer':[],
            'classes':[]
        }
        self.__optimizer = optimizer
        self.__optimizer_args = args
    
    def fit(self, X, y, maxUnchangedPasses, maxEpochs):
        classes = np.unique(y)
        # print(classes)
        self.model['optimizer'] = []
        self.model['classes'] = []
        self.model['classNames'] = classes
        self.model['thresholds'] = []
        IND = []
        for c in classes:
            IND.append(list(np.where(y==c)[0]))
        for i in range(len(classes)):
            for j in range(i):
                indices = []
                indices.extend(IND[i])
                indices.extend(IND[j])
                self.model['optimizer'].append(self.__optimizer(**self.__optimizer_args))
                self.model['classes'].append((i, j))
                yovo = list(np.ones(len(IND[i])))
                yovo.extend(list(np.zeros(len(IND[j]))))
                self.model['optimizer'][-1].optimize(X[indices], yovo, maxUnchangedPasses, maxEpochs)
                pred = self.model['optimizer'][-1].f(X[indices], useCache=False)
                # print("\nclasses", i, j, classes[i], classes[j])
                # print(np.unique(yovo, return_counts=True))
                fpr, tpr, thresholds = roc_curve(yovo, pred)
                self.model['thresholds'].append(thresholds[np.argmax(tpr-fpr)])
    
    def predict(self, X):
        X = np.matrix(X)
        ys = (np.asarray(self.model['optimizer'][0].f(X, useCache=False)) >= self.model['thresholds'][0]).astype(int).reshape((X.shape[0],1))
        for i, optimizer in enumerate(self.model['optimizer']):
            if i == 0:
                continue
            preds = (np.asarray(optimizer.f(X, useCache=False)) >= self.model['thresholds'][i]).astype(int).reshape((X.shape[0],1))
            ys=np.hstack((ys,preds))
        y = np.zeros(X.shape[0])
        for I in range(X.shape[0]):
            scores = {}
            for i in self.model['classNames']:
                for j in self.model['classNames']:
                    if i==j:
                        continue
                    elif i > j:
                        scores[i] = scores.get(i,0) + ys[I, self.model['classes'].index((i,j))]
                    else:
                        scores[i] = scores.get(i,0) + 1 - ys[I, self.model['classes'].index((j,i))]
            y[I] = list(scores.keys())[np.argmax(list(scores.values()))]
        return y

# from optimizer import SMO

# X = np.array([[0,0],[1,0],[0,1],[1,1]])
# y = np.array([0,0,0,1])
# svc = SVC(SMO)
# svc.fit(X, y, 3, 100)
# print(svc.predict(X))
