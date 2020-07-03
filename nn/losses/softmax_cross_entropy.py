import numpy as np

class SoftmaxCrossEntropy:
    def compute(self, target, predicted):
        # log(Pc)
        return np.average(np.log(np.multiply(target, predicted).sum(axis=1)))

    def gradientOutput(self, target, predicted, activation_fn):
        """
        target -> m x output_neurons
        predicted -> m x output_neurons
        return -> m x output_neurons
        """
        # (1/Pc)(Pc(yk-Pk))
        Pc = np.multiply(target, predicted).sum(axis=1) # m x 1
        return target-predicted#np.multiply((1/Pc),activation_fn.gradient(predicted, target))

    def gradientHidden(self, vkj, output_err, hidden_layer, activation_fn):
        """
        vkj -> hidden_neurons x output_neurons
        output_err -> m x output_neurons
        hidden_layer -> m x hidden_neurons
        return m x hidden_neurons
        """
        sum_err = (vkj*output_err.T).T # m x hidden_neurons
        return np.multiply(activation_fn.gradient(hidden_layer), sum_err)