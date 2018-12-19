""" This module contain functions related to linear regression,
such as for computing cost and doing gradient descent.
"""

import numpy as np


def cost(X, y, theta):
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

    return np.sum(np.square(X @ theta - y)) / (2 * m)


def gradient_descent(X, y, theta, alpha, num_iters):
    """ Computes the cost of using theta as the parameter
    for linear regression to fit the data points in X and y.

    :param ndarray X: a matrix where each row is a training
    example and each column refers to a feature.

    :param ndarray y: a column vector where each row refers
    to the output of each training example.

    :param ndarray theta: a column vector used as parameter.
    for linear regression.

    :param float alpha: learning rate.

    :param int num_iters: number of iterations.

    :return: values of theta learned by the algorithm.
    :rtype: ndarray.
    """

    # Number of training examples.
    m = X.shape[0]

    # Do not override theta
    t = theta.copy()

    for _ in range(num_iters):
        t = t - (alpha / m) * (np.transpose(X) @ (X @ t - y))

    return t
