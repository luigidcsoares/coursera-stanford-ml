""" This module contain functions related to linear regression,
such as for computing cost and doing gradient descent.
"""

import numpy as np


def compute_cost(X, y, theta):
    """ Computes the cost of using theta as the parameter
    for linear regression to fit the data points in X and y.

    :param ndarray X: a matrix where each row is a training
    example and each column refers to a feature.

    :param ndarray y: a column vector where each row refers
    to the output of each training example.

    :param ndarray theta: a column vector used as parameter.
    for linear regression.

    :return: cost J.
    :rtype: float.
    """

    # Number of training examples.
    m = X.shape[0]

    return np.sum(np.square(X @ theta) - y) / (2 * m)
