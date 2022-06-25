""" This module is a lite implementation of neutal network. 
"""

import numpy as np

def get_weights_size(layer_sizes):
    """Calculate number of weight parameters (including biases) of a neural network from layer sizes

    Parameters
    ----------
    layer_sizes : list
        1d array of the size of each neural network layer

    Returns
    -------
    weights_size : int
        number of weights
    """
    weights_size = 0
    for i in range(len(layer_sizes) - 1):
        weights_size += (layer_sizes[i] + 1) * layer_sizes[i + 1]
    return weights_size

class NN():
    """
    A lite implementation of a fully connected neural network with bias on each layer.

    """


    def __init__(self, layer_sizes):
        """Initialize `NN` object

        Parameters
        ----------
        layer_sizes : list
            1d array of the size of each neural network layer
        """
        self.layer_sizes = layer_sizes
        self.weights_size = get_weights_size(layer_sizes)
        # tanh activation
        self.activation = lambda x: 2 / (1 + np.exp(-x)) - 1
    def set_flatten_weights(self, flatten_weights):
        """Set weight parameterss of the neural network

        Parameters
        ----------
        flatten_weights : numpy array
            1d array of every weight parameters 
        """
        self.weights = []
        weight_i = 0
        for i in range(len(self.layer_sizes) - 1):
            next_weight_i = weight_i + (self.layer_sizes[i] + 1) * self.layer_sizes[i + 1]
            self.weights.append(
                flatten_weights[weight_i:next_weight_i].reshape(self.layer_sizes[i + 1], self.layer_sizes[i] + 1))
            weight_i = next_weight_i        
    def predict(self, input):
        """Predict on input

        Parameters
        ----------
        input : numpy array
            1d array of input to neural network


        Returns
        -------
        prediction : numpy array
            1d array of prediction
        """
        cloned_input = np.copy(input)
        for i in range(len(self.weights)):
            cloned_input = np.dot(self.weights[i][:,1:], cloned_input) + self.weights[i][:,0]
            cloned_input = self.activation(cloned_input)            
        return cloned_input
