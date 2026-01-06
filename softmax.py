import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.array(x)
    axis = None if len(x.shape) == 1 else 1
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)

    return e_x / e_x.sum(axis=axis, keepdims=True)
