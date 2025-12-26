"""
Implement Sigmoid in NumPy

Examples
Input: x = [0, 2, -2]

Output: [0.5, 0.88079708, 0.11920292]

Input: x = 0

Output: 0.5

Input: x = [[-1, 0], [1, 2]]

Output: [[0.26894142, 0.5], [0.73105858, 0.88079708]]
"""

import numpy as np


def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    arr = np.array(x)
    n_arr = -arr
    return 1 / (1 + np.exp(n_arr))
