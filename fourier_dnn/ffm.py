import tensorflow as tf
import numpy as np

class BasicFFM(tf.keras.layers.Layer):
    def __init__(self):
        
        super().__init__()

    def build(self, shape_input):

        # Taking the number of inputs
        num_units = shape_input[-1]

        # Defining the basic FFM layer (Fully connected) with non-trainable weights
        self.FFM_kernel = tf.keras.layers.Dense(num_units, trainable = False, kernel_initializer = 'identity')

    def call(self, points_input):

        # Inputs = 2 * pi * inputs
        inputs = 2 * np.pi * points_input

        # Applying the FFM_kernel to inputs
        kernel_output = self.FFM_kernel(inputs)

        # Taking sin & cos of inputs
        sin_output = tf.sin(kernel_output)
        cos_output = tf.cos(kernel_output)

        # Concatenating the outputs for final result
        final_output = tf.concat([sin_output, cos_output], axis = -1)
        return final_output

class GaussianFFM(tf.keras.layers.Layer):

    def __init__(self, std_dev : float, num_units : int):

        super().__init__()
        self.std_dev = float(std_dev)
        self.num_units = num_units

    def build(self, shape_input):

        # Defining the weights initializer
        layer_initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev = self.std_dev, seed = 0)

        # Defining the Gaussian FFM layer
        self.FFM_kernel = tf.keras.layers.Dense(self.num_units, trainable = False, kernel_initializer = layer_initializer)

    def call(self, points_input):

        # Inputs = 2 * pi * inputs
        inputs = 2 * np.pi * points_input

        # Applying the Gaussian FFM_kernel to inputs
        kernel_output = self.FFM_kernel(inputs)

        # Taking sin & cos of inputs
        sin_output = tf.sin(kernel_output)
        cos_output = tf.cos(kernel_output)

        # Concatenating the outputs for final result
        final_output = tf.concat([sin_output, cos_output], axis = -1)
        return final_output



