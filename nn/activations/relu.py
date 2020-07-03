import numpy as np
class ReLU:
    def compute(self, z):
        ret = z.copy()
        ret[ret <= 0] = 0
        return ret
    
    def gradient(self, goz):
        ret = goz.copy()
        ret[ret > 0] = 1
        return ret