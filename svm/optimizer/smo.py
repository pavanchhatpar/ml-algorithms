import numpy as np

class SMO:
    def __init__(self, C=1, tol=0.001, eps=0.001, kernelFunc=np.dot):
        self.__tol = tol
        self.__C = C
        self.model = None
        self.__kernelFunc = kernelFunc
        self.__eps = eps
        
    
    def __init(self, y):
        self.model = {}
        self.model['alphas'] = np.matrix(np.zeros(y.shape))#np.matrix(np.random.rand(y.shape[0], y.shape[1])*self.__C)
        # self.model['alphas'][:,np.where(y==-1)[1]] /= self.model['alphas'][:,np.where(y==-1)[1]].sum()
        # self.model['alphas'][:,np.where(y==1)[1]] /= self.model['alphas'][:,np.where(y==1)[1]].sum()
        # diff = self.model['alphas']*y.T
        # self.model['alphas'][:,np.where(y==1)[1][0]] -= diff

    def f(self, i, useCache=True):
        if useCache:
            ay = np.multiply(self.model['alphas'],self.model['y'])
            kern = self.__multiKernel(i, np.arange(len(self.model['X'])), useCache)
            return np.dot(kern, ay.T) + self.model['b']
        else:
            ay = np.multiply(self.model['alphas'],self.model['y'])
            kern = self.__multiKernel(i, self.model['X'], useCache)
            return np.dot(kern, ay.T) + self.model['b']

    def __E(self, i, y):
        return self.f(i) - y

    def __selectSecond(self, X, y, i):
        Ei = self.__E(i, y[:,i])
        # maxE = np.abs(Ei - self.__E(i, y[:,0]))
        ret = 0
        while True:
            j = np.random.randint(X.shape[0])
            # for j in range(X.shape[0]):
            Ej = self.__E(j, y[:,j])
            if np.abs(Ei - Ej) > 0:
                return j

    def __kernel(self, i, j):
        # print("HERE")
        # if not np.isnan(self.__kernelCache[i, j]):
        return self.__kernelCache[i, j]
        # elif not np.isnan(self.__kernelCache[j, i]):
        #     return self.__kernelCache[j, i]
        # else:
        #     self.__kernelCache[i, j] = self.__kernelFunc(self.__X[i], self.__X[j].T)
        #     return self.__kernelCache[i, j]

    def __multiKernel(self, Is, Js, useCache=True):
        try:
            len(Is)
        except:
            Is = [Is]
        try:
            len(Js)
        except:
            Is = [Js]
        if useCache:
            ret = np.zeros((len(Is), len(Js)))
            for i, I in enumerate(Is):
                for j, J in enumerate(Js):
                    ret[i,j] = self.__kernel(I, J)
            return ret
        else:
            ret = np.zeros((len(Is), len(Js)))
            for i, Xi in enumerate(Is):
                for j, Xj in enumerate(Js):
                    ret[i,j] = self.__kernelFunc(Xi, Xj.T)
            return ret

    def __takeStep(self, X, y, i, j):
        if i==j:
            return False
        s = y[:,i]*y[:,j]
        if s == -1:
            L = max(0, self.model['alphas'][:,j] - self.model['alphas'][:,i])
            H = min(self.__C, self.model['alphas'][:,j] - self.model['alphas'][:,i] + self.__C)
        elif s == 1:
            L = max(0, self.model['alphas'][:,j] + self.model['alphas'][:,i] - self.__C)
            H = min(self.__C, self.model['alphas'][:,j] + self.model['alphas'][:,i])
        if L == H:
            return False
        eta = 2*self.__kernel(i, j) - self.__kernel(i, i) - self.__kernel(j, j)
        if eta >= 0:
            return False
        aj = self.model['alphas'][:,j] - y[:,j]*(self.__E(i, y[:,i])-self.__E(j, y[:,j]))/eta
        aj = max(L, aj)
        aj = min(H, aj)
        if np.abs(aj - self.model['alphas'][:,j]) < self.__eps:
            return False
        ai = self.model['alphas'][:,i] + s*(self.model['alphas'][:,j] - aj)
        self.__biasUpdate(X, y, i, j, ai, aj)
        self.model['alphas'][:,i]=ai
        self.model['alphas'][:,j]=aj
        return True

    def __biasUpdate(self, X, y, i, j, ai, aj):
        bi = self.model['b'] - self.__E(i, y[:,i]) - (self.model['alphas'][:,i]-ai)*y[:,i]*self.__kernel(i, i) - (self.model['alphas'][:,j] - aj)*y[:,j]*self.__kernel(j, i)
        bj = self.model['b'] - self.__E(j, y[:,j]) - (self.model['alphas'][:,i]-ai)*y[:,i]*self.__kernel(i, j) - (self.model['alphas'][:,j] - aj)*y[:,j]*self.__kernel(j, j)
        if self.model['alphas'][:,i] > 0 and self.model['alphas'][:,i] < self.__C and self.model['alphas'][:,j] > 0 and self.model['alphas'][:,j] < self.__C:
            if np.random.rand() > 0.5:
                self.model['b'] = bj
            else:
                self.model['b'] = bi
        elif self.model['alphas'][:,i] > 0 and self.model['alphas'][:,i] < self.__C:
            self.model['b'] = bi
        elif self.model['alphas'][:,j] > 0 and self.model['alphas'][:,j] < self.__C:
            self.model['b'] = bj
        else:
            self.model['b'] = (bi+bj)/2

    def __examine(self, X, y, i):
        ri = self.__E(i, y[:,i])*y[:,i]
        if (ri < -self.__tol and self.model['alphas'][:,i] < self.__C) or (ri > self.__tol and self.model['alphas'][:,i] > 0):
            # non_bounded = np.where(np.logical_and(self.model['alphas'] > 0, self.model['alphas'] < self.__C))[1]
            # if len(non_bounded) > 1:
            #     j = self.__selectSecond(X, y, i)
            #     if self.__takeStep(X, y, i, j):
            #         return 1
            # copy_non_bounded = np.copy(non_bounded)
            # np.random.shuffle(copy_non_bounded)
            j = i
            while j == i:
                j = np.random.randint(X.shape[0])
            if self.__takeStep(X, y, i, j):
                return 1
        return 0

    # def optimize(self, X, y, maxUnchangedPasses):
    #     X = np.matrix(X)
    #     y = np.matrix(y)
    #     self.__X = np.matrix(np.copy(X))
    #     self.__y = np.matrix(np.copy(y))
    #     self.__kernelCache = X*X.T
    #     y = y*2 - 1
    #     self.__init(y)
    #     self.model['b'] = 0
    #     self.model['X'] = X
    #     self.model['y'] = y
    #     passes = 0
    #     while passes < maxUnchangedPasses:
    #         numChanged = 0
    #         for i in range(X.shape[0]):
    #             Ei = self.__E(i, y[:,i])

    def optimize(self, X, y, maxUnchangedPasses, maxEpochs):
        X = np.matrix(X)
        y = np.matrix(y)
        self.__X = np.matrix(np.copy(X))
        self.__y = np.matrix(np.copy(y))
        self.__kernelCache = self.__kernelFunc(X, X.T)
        # self.__kernelCache[:] = np.nan
        y = y*2 - 1
        # y = y.reshape((y.shape[0],1))
        self.__init(y)
        # non_bounded = np.where(np.logical_and(self.model['alphas'] > 0, self.model['alphas'] < self.__C))
        
        # ay = np.multiply(self.model['alphas'],y)
        # kerns = self.__multiKernel(non_bounded[1], np.arange(X.shape[0]))
        self.model['b'] = 0#(1/y[non_bounded] - np.dot(kerns,ay.T)).mean()
        self.model['X'] = X
        self.model['y'] = y
        numChanged = 0
        examineAll = True
        I=0
        passes = 0
        
        while passes < maxUnchangedPasses:
            numChanged = 0
            if examineAll:
                for i in range(X.shape[0]):
                    numChanged += self.__examine(X, y, i)
                examineAll = False
            else:
                non_bounded = np.where(np.logical_and(self.model['alphas'] > 0, self.model['alphas'] < self.__C))[1]
                # maximum = np.where(self.model['alphas'] == self.__C)[1][:int(X.shape[0]*0.1)]
                for i in non_bounded:
                    numChanged += self.__examine(X, y, i)
                # if numChanged == 0:
                #     for i in maximum:
                #         numChanged += self.__examine(X, y, i)
                # if i % 100 == 0:
                #     print(i, "done")
            if numChanged == 0:
                passes += 1
            else:
                passes = 0
            I += 1
            if I % 1 == 0:
                print(numChanged, "changed")
                print(I, "rounds done")
            if I > maxEpochs:
                break
        keepI = []
        for i in range(self.model['alphas'].shape[1]):
            if self.model['alphas'][:,i] > 1e-10:
                keepI.append(i)
        self.model['alphas'] = self.model['alphas'][:,keepI]
        self.model['X'] = self.model['X'][keepI]
        self.model['y'] = self.model['y'][:,keepI]