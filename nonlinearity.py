"""
Housing non linear activation functions used by the network.
"""

import numpy as np

def Softmax(x):
    numerator = np.exp(x-np.max(x))
    return numerator/ numerator.sum(axis=0)

def Sigmoid(x):
    return 1/(np.exp(x)+1)

# reLU for later
def _ReLU(x):
    return max(0, x)
