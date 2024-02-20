import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('./deeplearning.mplstyle')


def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples
      w,b (scalar)    : model parameters
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
m = x_train.shape[0]

w = 100
b = 100
tmp_f_wb = compute_model_output(x_train, w, b)
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title('Housing Prices')
plt.ylabel('Price in 1000s of dollars')
plt.xlabel('Size 1000 sqft')
plt.show()
