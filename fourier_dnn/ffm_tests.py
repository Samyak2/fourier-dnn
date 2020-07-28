import numpy as np
import pytest
import tensorflow as tf
from ffm import *


def test_basic_FFM():

    # Initializing random input
    tf.random.set_seed(1)
    test_input = tf.random.uniform(shape = [500, 2])

    # Creating the layer and appling FFM
    basic_FFM_kernel = BasicFFM()
    output = basic_FFM_kernel(test_input)

    assert output.shape == (500, 4)

def test_gaussian_FFM():

    # Initializing random input
    tf.random.set_seed(1)
    test_input = tf.random.uniform(shape = [500, 2])

    # Creating the layer and appling FFM
    gaussian_FFM_kernel = GaussianFFM(num_units = 126, std_dev = 2.3)
    output = gaussian_FFM_kernel(test_input)

    assert output.shape == (500, 252)


if __name__ == '__main__':

    # Using pytest to test the FFM layers
    pytest.main([__file__])