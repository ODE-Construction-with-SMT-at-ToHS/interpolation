"""
This module contains methods to do linear regression. Currently, the least squares method for 1D data is implemented
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def least_squares_1d(x_train, y_train):
    """
    This function does linear regression for 1D values using the least squares method.

    Args:
        x_train ((n,)-array): x-values for the regression
        y_train ((n,)-array): y-values for the regression

    Returns:
        (LinearRegression() instance): Model on which an ordinary least squares regression was performed.
    """
    if x_train.ndim != 1 or y_train.ndim != 1:
        print("Error: input data shape shape must be 1-dimesional")
        return 1
    x_train = np.expand_dims(x_train, axis=1)
    y_train = np.expand_dims(y_train, axis=1)
    x_train.T
    y_train.T
    reg = LinearRegression().fit(x_train, y_train)
    print("Function found! f(x) = ", reg.coef_[0][0], "x +", reg.intercept_[0])
    return reg


def least_squares_2d(x_train, y_train):
    """
    This function does linear regression for 2D values using the least squares method.

    Args:
        x_train ((n,2)-array): x-values for the regression
        y_train ((n,2)-array): y-values for the regression

    Returns:
        (LinearRegression() instance): Model on which an ordinary least squares regression was performed.
    """
    if x_train.ndim != 2 or y_train.ndim != 2 or x_train.shape[1] != 2 or y_train.shape[1] != 2:
        print("Error: input data shape shape must be (n,2)")
        return 1
    reg = LinearRegression().fit(x_train, y_train)
    print("Function found! f(x) = ")
    print(reg.coef_, "x +", reg.intercept_)
    return reg
