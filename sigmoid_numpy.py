import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    arr = np.array(x)
    n_arr = -arr
    return 1 / (1 + np.exp(n_arr))
