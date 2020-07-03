import numpy as np

class GradientDescent:
    def __init__(self, learning_rate, dampen=True):
        self.__learning_rate = learning_rate
        self.__dampen = dampen

    def minimize(self, loss, grad_loss, X, y, initW, min_reduction, 
    maxIters, printAfterEpochs, printer, batch_size=None):
        if batch_size == None:
            batch_size = len(X)
        epochs = 0
        batchI = 0
        # delChange = np.Infinity
        W = initW
        printed=-1
        dampenAfter = maxIters // 4
        nDampens = 1
        lr = self.__learning_rate
        while epochs < maxIters:
            X_batch = X[batch_size*batchI: batch_size*(batchI+1), :]
            y_batch = y[batch_size*batchI: batch_size*(batchI+1), :]
            if len(X_batch) == 0:
                batchI = 0
                epochs += 1
                if self.__dampen:
                    if epochs >= nDampens*dampenAfter:
                        lr /= 10
                        nDampens += 1
                continue
            # cost = loss(X_batch, y_batch, W)
            gradient = grad_loss(X_batch, y_batch, W)
            W -= lr*gradient
            # nCost = loss(X_batch, y_batch, W)
            # delChange = np.abs(cost - nCost)
            batchI += 1
            if (epochs%printAfterEpochs==0 and printed < epochs):
                printed = epochs
                printer(epochs, W, X, y, loss)
            # if nCost == 0:
            #     break
        return W