"""
File for the base Recurrent-Neural-Net model. NO IMPORTS ALLOWED!
"""

import numpy as np
from loss_functions import MSE_loss
from nonlinearity import Softmax

HIDDEN_LAYER_SIZE = 100
RANDOMIZER_CONSTANT = 1250
LEARNING_RATE = 0.001

class Base_RNN:
    def __init__(self, input_dims, output_dims, hidden_layer_size=HIDDEN_LAYER_SIZE):
        """
        Initializes the RNN, given input, output dims and hidden layer size.

        Weights are initialized to random values, and bias vectors are all just 1s.
        """
        # RANDOMIZER_CONSTANT to make sure initial weights aren't too large

        self.U = np.randn(hidden_layer_size, input_dims) / RANDOMIZER_CONSTANT

        # intermediary 
        self.W = np.randn(hidden_layer_size, hidden_layer_size) / RANDOMIZER_CONSTANT
        
        # final
        self.V = np.randn(output_dims, hidden_layer_size)/ RANDOMIZER_CONSTANT

        #biases
        self.b1 = np.zeros((hidden_layer_size, 1))
        self.b2 = np.zeros((output_dims, 1))

        self._hidden_layer_size = hidden_layer_size
        self._input_dims = input_dims

        self.hidden_state = np.zeros((self._hidden_layer_size, 1))

    
    def forward_pass(self, input):
        """
        Forward pass for the RNN

        each input is a word from the sequence, and so it is in vector form,
        
        """
        self.hidden_state = np.tanh(self.U @ input + self.W @ self.hidden_state + self.b1)

        z = self.V @ self.hidden_state + self.b2

        return z, self.hidden_state
        

    def backward(self, y, t):
        L = MSE_loss(y, t)
        # TODO
        dU = None
        dW = None
        dV = None
        db1 = None
        db2 = None


        # only have gradient updates below

        self.U  = self.U - LEARNING_RATE * dU
        self.W = self.W  - LEARNING_RATE * dW 
        self.V = self.V - LEARNING_RATE * dV 
        self.b1 = self.b1 - LEARNING_RATE * db1 
        self.b2 = self.b2 - LEARNING_RATE * db2