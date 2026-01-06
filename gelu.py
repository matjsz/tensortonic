import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: scalar, list, or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    x = np.array(x)
    return x/2 * (1 + np.vectorize(math.erf)(x/math.sqrt(2)))
