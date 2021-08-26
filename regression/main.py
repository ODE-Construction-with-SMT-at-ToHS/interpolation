"""
This module is the top-level module to call functions from other modules.
"""
from sklearn.linear_model import LinearRegression

from regression.sampling import *
from regression.functions import *
import time

if __name__ == '__main__':
    # create singleton
    func_class = LinearB2D()

    # sample points
    sample_nd(func_class, [[-4, 4], [-4, 4]], [100+1, 100+1], 0)
    # sample_nd(func_class, [[-10, 10]], [1000+1], 0)

    start_time = time.time()

    # load samples
    x_samples, y_samples, x_train, x_test, y_train, y_test = load_samples(func_class)

    if False:
        # do polyfit
        # reshape from (n,1) to (n,)
        x_train = x_train.reshape((len(x_train,)))
        y_train = y_train.reshape((len(y_train,)))
        test = np.polynomial.polynomial.polyfit(x_train, y_train, func_class.degree())
        print(test)
    else:
        # do linear regression
        reg = LinearRegression().fit(x_train, y_train)
        print(reg.coef_)

    end_time = time.time()
    print('Overall time:', end_time-start_time, 'seconds')
