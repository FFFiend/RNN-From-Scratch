"""
File for the base Recurrent-Neural-Net model. NO IMPORTS ALLOWED!
"""

import numpy as np

HIDDEN_LAYER_SIZE = 100
RANDOMIZER_CONSTANT = 1250
LEARNING_RATE = 0.001

class Base_RNN:
    def __init__(self, input_dims, output_dims, hidden_layer_size=HIDDEN_LAYER_SIZE):
        # first mult, Ux
        
        self.U = np.randn(hidden_layer_size, input_dims) / RANDOMIZER_CONSTANT

        # intermediary 
        self.W = np.randn(hidden_layer_size, hidden_layer_size) / RANDOMIZER_CONSTANT
        
        # final
        self.V = np.randn(output_dims, hidden_layer_size)/ RANDOMIZER_CONSTANT

        #biases, V(W(Ux)) + b1 + b2
        self.b1 = np.zeros((hidden_layer_size, 1))
        self.b2 = np.zeroes((output_dims, 1))

        self._hidden_layer_size = hidden_layer_size

    
    def forward_pass(self, input):
        """
        Forward pass for the RNN
        """
        h = np.zeros((self._hidden_layer_size, 1))
        prev_timestep_h = {0:h}

        for i, x in enumerate(input):
            h = np.tanh(self.W @ x + self.U @ h + self.b1)
            prev_timestep_h[i+1] = h
        
        z = self.V @ h + self.b2

        return z, h

    def backprop(self):
        

        dU = None
        dW = None
        dV = None
        db1 = None
        db2 = None
        # TODO

        # only have gradient updates below

        self.U  = self.U - LEARNING_RATE * dU
        self.W = self.W  - LEARNING_RATE * dW 
        self.V = self.V - LEARNING_RATE * dV 
        self.b1 = self.b1 - LEARNING_RATE * db1 
        self.b2 = self.b2 - LEARNING_RATE * db2





    





