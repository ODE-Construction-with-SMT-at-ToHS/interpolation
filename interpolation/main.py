"""
This module is the top-level module to call functions frm other modules.
"""

from interpolation.sampling import *
from interpolation.functions import *

if __name__ == '__main__':
    # take samples
    # sample_1d(LinearA, 1000+1, [-10, 10], 0.2)

    # load samples
    x_samples, y_samples, x_train, x_test, y_train, y_test = load_samples(LinearA)

    print(x_samples)
