import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg

        self.layers = (
          FullyConnectedLayer(n_input, hidden_layer_size),
          ReLULayer(),
          FullyConnectedLayer(hidden_layer_size, n_output),
        )

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for param in self.params().values():
          param.grad.fill(0)
        
        for layer in self.layers:
          X = layer.forward(X)
        loss, d_loss = softmax_with_cross_entropy(X, y)
        for layer in reversed(self.layers):
          d_loss = layer.backward(d_loss)

        l2_loss = 0.0
        for param in self.params().values():
          l, d = l2_regularization(param.value, self.reg)
          l2_loss += l
          param.grad += d

        return loss + l2_loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        for layer in self.layers:
          X = layer.forward(X)
        return np.argmax(X, axis=-1)

    def params(self):
        return {f"layers[{i}].{name}": p for i, layer in enumerate(self.layers) for name, p in layer.params().items()}
