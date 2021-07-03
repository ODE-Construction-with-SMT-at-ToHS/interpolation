"""
This module is the top-level module to call functions from other modules.
"""
import matplotlib.pyplot as plt

from regression.sampling import *
from regression.functions import *
from regression.linear import *
from regression.plot3d import *

if __name__ == '__main__':
    # take samples
    # sample_1d(LinearA, 1000+1, [-10, 10], 0.2)

    func_class = LinearA2D

    # load samples
    x_samples, y_samples, x_train, x_test, y_train, y_test = load_samples(func_class)

    # do regression and plot error
    if func_class == LinearA2D:
        # do regression
        reg = least_squares_2d(x_train, y_train)
        # predict y values
        y_prediction = [numpy.matmul(reg.coef_, x)+ reg.intercept_ for x in x_test]
        y_prediction = np.array(y_prediction)
        # plot error
        plot_dist_map(x_test, y_test, y_prediction, func_class.name)
    else:
        # do regression
        reg = least_squares_1d(x_train, y_train)
        # plot samples and prediction
        plt.plot(x_samples, y_samples, 'r', x_test, reg.coef_[0][0]*x_test+reg.intercept_[0], '')
        plt.show()
