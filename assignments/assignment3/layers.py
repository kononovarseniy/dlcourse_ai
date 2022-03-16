import numpy as np


def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    """
    exps = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
    denom = np.sum(exps, axis=-1, keepdims=True)
    return exps / denom


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """

    loss = reg_strength * np.sum(np.square(W))
    return loss, reg_strength * 2 * W


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    indices = (
        target_index
        if predictions.ndim == 1
        else (np.arange(predictions.shape[0]).reshape(target_index.shape), target_index)
    )

    probs = softmax(predictions)
    loss = np.mean(-np.log(probs[indices]))

    d_preds = probs.copy()
    d_preds[indices] -= 1
    d_preds /= probs.shape[0]

    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return (self.X > 0).astype(np.float64) * d_out

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return X @ self.W.value + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """

        self.W.grad += self.X.T @ d_out
        self.B.grad += np.sum(d_out, axis=0)

        return d_out @ self.W.value.T

    def params(self):
        return {"W": self.W, "B": self.B}


def iter_window_slices(stride, size, out_height, out_width):
    for y in range(out_height):
        y_pos = y * stride
        y_slice = slice(y_pos, y_pos + size)
        for x in range(out_width):
            x_pos = x * stride
            x_slice = slice(x_pos, x_pos + size)

            yield y, x, (slice(None), y_slice, x_slice, slice(None))


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, filter_size, padding):
        """
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        """

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(np.random.randn(filter_size, filter_size, in_channels, out_channels))

        self.B = Param(np.zeros(out_channels))

        self.X = None

        self.padding = padding

    def forward(self, X):
        batch_size, in_height, in_width, channels = X.shape
        assert channels == self.in_channels

        out_height = in_height + 2 * self.padding - self.filter_size + 1
        out_width = in_width + 2 * self.padding - self.filter_size + 1

        self.X = np.pad(X, ((0,), (self.padding,), (self.padding,), (0,)))

        result = np.zeros((batch_size, out_height, out_width, self.out_channels))

        n_flat_features = self.filter_size**2 * self.in_channels
        for y, x, wnd_indices in iter_window_slices(1, self.filter_size, out_height, out_width):
            wnd_flat = self.X[wnd_indices].reshape(batch_size, n_flat_features)
            W_flat = self.W.value.reshape(n_flat_features, self.out_channels)
            B_flat = self.B.value

            result[:, y, x, :] = wnd_flat @ W_flat + B_flat
        return result

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        result = np.zeros_like(self.X)

        n_flat_features = self.filter_size**2 * self.in_channels
        for y, x, wnd_indices in iter_window_slices(1, self.filter_size, out_height, out_width):
            wnd_flat = self.X[wnd_indices].reshape(batch_size, n_flat_features)
            W_flat = self.W.value.reshape(n_flat_features, self.out_channels)
            B_flat = self.B.value
            d_out_flat = d_out[:, y, x, :]

            self.W.grad += (wnd_flat.T @ d_out_flat).reshape(self.W.grad.shape)
            self.B.grad += np.sum(d_out_flat, axis=0)

            result[wnd_indices] += (d_out_flat @ W_flat.T).reshape(
                batch_size, self.filter_size, self.filter_size, self.in_channels
            )
        return result[
            :,
            self.padding : self.X.shape[1] - self.padding,
            self.padding : self.X.shape[2] - self.padding,
            :,
        ]

    def params(self):
        return {"W": self.W, "B": self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        """
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        """
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height // self.stride
        out_width = width // self.stride

        self.X = X

        result = np.zeros((batch_size, out_height, out_width, channels))

        for y, x, wnd_indices in iter_window_slices(self.stride, self.pool_size, out_height, out_width):
            wnd_flat = self.X[wnd_indices]
            result[:, y, x, :] = np.amax(wnd_flat, (1, 2))
        return result

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape

        out_height = height // self.stride
        out_width = width // self.stride

        result = np.zeros((batch_size, height, width, channels))

        for y, x, wnd_indices in iter_window_slices(self.stride, self.pool_size, out_height, out_width):
            wnd_flat = self.X[wnd_indices].reshape(batch_size, -1, channels)
            y_inds, x_inds = np.unravel_index(np.argmax(wnd_flat, 1), (self.pool_size, self.pool_size))
            result[
                np.arange(batch_size).reshape(-1, 1),
                y_inds + wnd_indices[1].start,
                x_inds + wnd_indices[2].start,
                np.arange(channels).reshape(1, -1),
            ] += d_out[:, y, x, :]
        return result

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X_shape = X.shape

        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
