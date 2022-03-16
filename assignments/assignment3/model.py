import numpy as np

from layers import (
    FullyConnectedLayer,
    ReLULayer,
    ConvolutionalLayer,
    MaxPoolingLayer,
    Flattener,
    softmax_with_cross_entropy,
    l2_regularization,
)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        w, h, c = input_shape

        self.layers = (
            ConvolutionalLayer(c, conv1_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),

            ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),

            Flattener(),
            FullyConnectedLayer(h * w * conv2_channels // 4 ** 4, n_output_classes)
        )

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for param in self.params().values():
            param.grad.fill(0)

        for layer in self.layers:
            X = layer.forward(X)
        loss, d_loss = softmax_with_cross_entropy(X, y)
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss)

        return loss

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
