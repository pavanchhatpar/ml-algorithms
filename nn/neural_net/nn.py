import numpy as np
import sys
import os

sys.path.append(os.path.abspath('../utils'))
from utils import one_hot

class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, 
    optimizer, activation_fn, loss, output_activation=None, dropout=0):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.loss = loss
        self.activation_fn = activation_fn
        self.optimizer = optimizer
        if output_activation == None:
            self.output_activation = activation_fn
        else:
            self.output_activation = output_activation
        self.dropout = dropout


    def feed_forward(self, X):
        # X -> m x input_neurons
        hidden_layer = X*self.weights["hidden"] + self.bias["hidden"]
        hidden_layer = self.activation_fn.compute(hidden_layer) # m x hidden_neurons
        output_layer = hidden_layer*self.weights["output"] + self.bias["output"] # m x output_neurons
        output_layer = self.output_activation.compute(output_layer)
        return hidden_layer, output_layer
    
    def backpropagate(self, X, y, output_layer, hidden_layer, step):
        # output_layer = self.one_hot(np.argmax(output_layer, axis=1))
        output_error = self.loss.gradientOutput(y, output_layer, self.output_activation) # m x output_neurons np.average(y-output_layer,axis=0)
        hidden_error = self.loss.gradientHidden(self.weights["output"], output_error, 
        hidden_layer, self.activation_fn) # m x hidden_neurons
        delH = self.optimizer.compute(hidden_error, X, step) # input_neurons x hidden_neurons
        delO = self.optimizer.compute(output_error, hidden_layer, step) # hidden_neurons x output_neurons
        delHb = self.optimizer.compute(hidden_error, np.matrix(np.ones((X.shape[0], 1))), step)
        delOb = self.optimizer.compute(output_error, np.matrix(np.ones((hidden_layer.shape[0], 1))), step)
        self.weights["hidden"] += delH
        self.weights["output"] += delO
        self.bias["hidden"] += delHb
        self.bias["output"] += delOb
    
    def fit(self, X, y, max_epochs, batch_size=None, print_after=100):
        X = np.matrix(X)
        y = np.matrix(y)
        self.weights = {
            "hidden": np.matrix(np.random.standard_normal((self.input_neurons, self.hidden_neurons))),
            "output": np.matrix(np.random.standard_normal((self.hidden_neurons, self.output_neurons)))
        }
        self.bias = {
            "hidden": np.matrix(np.random.standard_normal((1, self.hidden_neurons))),
            "output": np.matrix(np.random.standard_normal((1, self.output_neurons)))
        }
        if batch_size==None:
            batch_size = len(X)
        n_batch = len(X) // batch_size
        for i in range(0, max_epochs):
            for j in range(n_batch):
                if j == n_batch-1:
                    X_batch = X[batch_size*j:]
                    y_batch = y[batch_size*j:]
                else:
                    X_batch = X[batch_size*j:batch_size*(j+1)]
                    y_batch = y[batch_size*j:batch_size*(j+1)]
                hidden_layer, output_layer = self.feed_forward(X_batch)
                self.backpropagate(X_batch, y_batch, output_layer, hidden_layer, i+1)
                _, predicted = self.feed_forward(X)
                # predicted = self.one_hot(np.argmax(predicted, axis=1))
                loss = self.loss.compute(y, predicted)
            if i%print_after==0:
                print("Epoch:", i, ", Loss:", loss)
        print("Epoch:", i, ", Loss:", loss)

    def predict(self, X):
        _, output_layer = self.feed_forward(X)
        return one_hot(np.argmax(output_layer, axis=1), self.output_neurons)
