import tensorflow as tf

class TFNeuralNet:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, 
    optimizer=tf.train.AdamOptimizer(), activation_fn=tf.nn.sigmoid, 
    loss=tf.losses.mean_squared_error, output_activation=None):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.X = tf.placeholder("float32", [None, input_neurons], name="features")
        self.y = tf.placeholder("float32", [None, output_neurons], name="labels")
        self.weights = {
            "hidden": tf.Variable(tf.random_normal([input_neurons, hidden_neurons]), name="hidden_w"),
            "output": tf.Variable(tf.random_normal([hidden_neurons, output_neurons]), name="output_w")
        }
        self.bias = {
            "hidden": tf.Variable(tf.random_normal([hidden_neurons]), name="hidden_b"),
            "output": tf.Variable(tf.random_normal([output_neurons]), name="output_b")
        }
        if output_activation == None:
            output_activation = activation_fn
        self.hidden_layer = tf.add(tf.matmul(self.X, self.weights["hidden"]), self.bias["hidden"])
        self.hidden_layer = activation_fn(self.hidden_layer, name="hidden_layer")
        self.output_layer = tf.add(tf.matmul(self.hidden_layer, self.weights["output"]), self.bias["output"], name="output_layer")
        if not output_activation == tf.nn.softmax or not loss == tf.losses.softmax_cross_entropy:
            self.output_layer = output_activation(self.output_layer, name="output_layer")
        self.loss = loss(self.y, self.output_layer)
        self.optimizer = optimizer.minimize(self.loss)
        

    def fit(self, X, y, max_epochs, batch_size=None, print_after=100):
        init = tf.initialize_all_variables()
        sess = tf.Session()
        self.sess = sess
        sess.run(init)
        if batch_size==None:
            batch_size = len(X)
        n_batch = len(X) // batch_size
        for i in range(1, max_epochs+1):
            for j in range(n_batch):
                if j == n_batch-1:
                    X_batch = X[batch_size*j:]
                    y_batch = y[batch_size*j:]
                else:
                    X_batch = X[batch_size*j:batch_size*(j+1)]
                    y_batch = y[batch_size*j:batch_size*(j+1)]
                _ = sess.run(self.optimizer, feed_dict={self.X:X_batch, self.y:y_batch})
                loss = sess.run(self.loss, feed_dict={self.X:X, self.y:y})
                # if loss == 0:
                #     return
            if i%print_after==0:
                print("Epoch:", i, ", Loss:", loss)

    def predict(self, X):
        sess = self.sess or tf.get_default_session()
        return sess.run(tf.one_hot(tf.argmax(self.output_layer, 1), depth=self.output_neurons), 
        feed_dict={self.X: X})
                

