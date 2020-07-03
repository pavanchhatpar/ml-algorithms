import numpy as np
class Sigmoid:
    def compute(self, z):
        ret = 1/(1+np.exp(-z))
        ret[ret == 0] = 1e-15
        ret[ret == 1] = 1 - 1e-15
        return ret
    
    def gradient(self, goz):
        return np.multiply(goz,(1-goz))
