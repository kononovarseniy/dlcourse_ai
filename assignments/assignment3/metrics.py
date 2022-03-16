import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    """
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    fp = np.count_nonzero((prediction == True) & (ground_truth == False))
    tp = np.count_nonzero((prediction == True) & (ground_truth == True))
    fn = np.count_nonzero((prediction == False) & (ground_truth == True))
    tn = np.count_nonzero((prediction == False) & (ground_truth == False))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (fp + tp + fn + tn)
    f1 = 2 / (1 / recall + 1 / precision)

    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    return np.count_nonzero(prediction == ground_truth) / prediction.shape[0]
