import numpy as np

class Softmax:
    def compute(self, z):
        ret = np.exp(z)
        ret /= ret.sum(axis=1)
        return ret
    
    def gradient(self, goz, y):
        # Pc(Yk-Ok)
        return np.multiply(goz[np.where(y==1)].T,(y-goz))
        
