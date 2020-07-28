import tensorflow as tf
import numpy as np
from ffm import *

# pylint: disable=no-value-for-parameter, unexpected-keyword-arg, arguments-differ
# pylint: disable=attribute-defined-outside-init

class FourierMLP(tf.keras.Model):

    def __init__(self, num_layers : int, num_units_FFM : int, num_units : int, num_units_final : int, gaussian : bool = None, staddev : float = None):

        """
        Creates the Fourier MLP based on the arguments specified

        Arguments:
            gaussian : To specify if the first layer is a BasicFFM or GaussianFFM layer
            staddev  : Standard deviation for the GaussianFFM
            num_layers : Number of layers for the MLP
            num_units_FFM : Number of units for the GaussianFFM layer
            num_units : Number of units for the MLP layers
            num_units_final : Number of units for the final output layer
        """

        super().__init__()

        MLP_layers = list()

        # Add GaussianFFM for the gaussian MLP
        if gaussian is not None and staddev is not None:
            MLP_layers.append(GaussianFFM(num_units = num_units_FFM, std_dev = staddev))
        else:
            MLP_layers.append(BasicFFM())

        # Adding num_layers - 1 Dense layers
        for i in range(num_layers - 1):
            MLP_layers.append(tf.keras.layers.Dense(num_units, use_bias = False, activation = 'relu'))

        # Adding the final output layer
        final_layer = tf.keras.layers.Dense(num_units_final, use_bias = False, activation = 'sigmoid')
        MLP_layers.append(final_layer)

        # Keras Sequential Model
        self.MLP_network = tf.keras.Sequential(MLP_layers)

    def call(self, MLP_inputs):

        # Passing the inputs through the network
        output = self.MLP_network(MLP_inputs)
        return output

