import numpy as np

class Newtons:
    def minimize(self, loss, gradient, hessian, X, y, initW, min_reduction, 
    maxIters, printAfterEpochs, printer):
        epochs = 0
        delChange = np.Infinity
        printed=-1
        W = initW
        while epochs < maxIters and np.abs(delChange) > min_reduction:
            cost = loss(X, y, W)
            W = W - np.linalg.pinv(hessian(X, y, W))*gradient(X, y, W)
            delChange = loss(X, y, W) - cost
            epochs+=1
            if epochs%printAfterEpochs == 0 and printed < epochs:
                printed = epochs
                print("Iteration:", epochs, ", Cost:", cost + delChange)
        return W