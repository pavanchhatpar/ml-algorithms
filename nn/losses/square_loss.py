import numpy as np

class MeanSquareLoss:
    def compute(self, target, predicted):
        ret = (1/2)*np.average(np.square(target - predicted))
        return ret

    def gradientOutput(self, target, predicted, activation_fn):
        """
        target -> m x output_neurons
        predicted -> m x output_neurons
        return -> m x output_neurons
        """
        return np.multiply(activation_fn.gradient(predicted),(target-predicted))

    def gradientHidden(self, vkj, output_err, hidden_layer, activation_fn):
        """
        vkj -> hidden_neurons x output_neurons
        output_err -> m x output_neurons
        hidden_layer -> m x hidden_neurons
        return m x hidden_neurons
        """
        return np.multiply(activation_fn.gradient(hidden_layer), 
            (vkj*output_err.T).T)