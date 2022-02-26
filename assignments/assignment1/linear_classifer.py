import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    exps = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
    denom = np.sum(exps, axis=-1, keepdims=True)
    return exps / denom


def predicted_probability_indices_by_ground_truth(probs, target_index):
    '''
    Returns indices to be used for selecting specific probability in each sample.
    '''
    return target_index if probs.ndim == 1 else (np.arange(probs.shape[0]).reshape(target_index.shape), target_index)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size, 1) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''

    indices = predicted_probability_indices_by_ground_truth(probs, target_index)
    return np.mean(-np.log(probs[indices]))


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # L(probabilities, true_classes)
    #   = - sum_i sum_j {j=true_class[i]}*ln(probabilities[i, j])
    #   = - sum_i ln(probabilities[i, true_classes[i]])
    # probabilities[i, j] = softmax(predictions)[i, j]
    #   = exp(predictions[i, j]) / sum_n(exp(predictions[i, n]))
    # L(predictions, true_classes)
    #   = - sum_i ( predictions[i, true_classes[i]] - ln(sum_n exp(predictions[i, n])) )
    # dL/dy[k, l]
    #   = - sum_i ( {k=i}{l=true_classes[i]} - sum_n({k=i}{l=n}exp(predictions[i, n]))/ sum_n(exp(predictions[i, n])) )
    #   = - sum_i ( {k=i}{l=true_classes[i]} - {k=i}exp(predictions[i, l]) / sum_n(exp(predictions[i, n])) )
    #   = - sum_i ( {k=i}{l=true_classes[i]} - {k=i}softmax(predictions)[i, l] )
    #   = - {l=true_classes[k]} + softmax(predictions)[k, l]
    probs = softmax(predictions)
    indices = predicted_probability_indices_by_ground_truth(probs, target_index)
    grad = probs.copy()
    grad[indices] -= 1

    return cross_entropy_loss(probs, target_index), grad / probs.shape[0]


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = reg_strength * np.sum(np.square(W))

    return loss, reg_strength * 2 * W
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = X @ W

    loss, loss_gradient = softmax_with_cross_entropy(predictions, target_index)
    
    return loss, X.T @ loss_gradient


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            for batch_indices in batches_indices:
              batch = X[batch_indices]
              true_classes = y[batch_indices]
              loss, loss_grad = linear_softmax(batch, self.W, true_classes)
              reg_loss, reg_loss_grad = l2_regularization(self.W, reg)

              self.W -= learning_rate * (loss_grad + reg_loss_grad)
              loss = loss + reg_loss

            loss_history.append(loss)
            # print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.argmax(X @ self.W, axis=-1)

        return y_pred



                
                                                          

            

                
