import numpy as np

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def compute(self, err_j, output_i, step):
        # err_j -> 1 x Dj
        # output_i = np.average(output_i, axis=0) # 1 x Di
        return self.learning_rate/(10**(0))*output_i.T*err_j # Di x Dj
