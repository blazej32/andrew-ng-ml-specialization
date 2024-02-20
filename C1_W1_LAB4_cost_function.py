import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_stationary, plt_update_onclick
from lab_utils_uni import soup_bowl
# plt.style.use('./deeplearning.mplstyle')


def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): Target values
        w,b (scalar)    : model parameters

    Returns:
        total_cost (float): The cost of using w, b as the paramneters for
            linear regression to fit the data points in x and y

    """
    # number of training examples
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[1]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

fig, ax, dyn_items = plt_stationary(x_train, y_train)
plt.close('all')
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
soup_bowl()
