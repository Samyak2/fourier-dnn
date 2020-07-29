import sys

import tensorflow as tf
import matplotlib.pyplot as plt

from fourier_dnn.ffm_mlp import FourierMLP
from fourier_dnn.metrics import PSNR
from image_regression_data import train_dataset

def get_model(num_layers=10, num_units=128, num_units_final=3,
              gaussian=False, staddev=10, num_units_FFM=128,
              learning_rate=1e-3):
    """Constructs a Fourier MLP model for 2D image regression
    with default arguments
    """

    model = FourierMLP(num_layers, num_units, num_units_final,
                       gaussian=gaussian, staddev=staddev, num_units_FFM=num_units_FFM)

    loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                  loss=loss_fn,
                  metrics=['accuracy', PSNR()])

    return model


def model_test(model, image_index, show_output=True, save_output=True):
    d = train_dataset.skip(image_index).take(1)
    d = next(iter(d))
    o = model(d[0])

    if show_output:
        plt.imshow(o)
        if save_output:
            plt.savefig(f"output_{image_index}.png")
            print(f"Image saved as output_{image_index}.png")
        else:
            plt.show()
    
    psnr = tf.image.psnr(d[1], o, 1.0)
    print(f"PSNR: {psnr}")
    return o, psnr

def train_model(model, image_index, epochs):
    """Trains the model on 2D image dataset, but only on image with index
    image_index and for given number of epochs. The output is shown if show_output is True.
    """
    d = train_dataset.skip(image_index).take(1)
    d = next(iter(d))

    model.fit(d[0], d[1], epochs=epochs, verbose=2)

if __name__ == "__main__":
    i_model = get_model()
    index = 0
    epochs = 1000
    if len(sys.argv) > 1:
        index = int(sys.argv[1])
        if len(sys.argv) > 2:
            epochs = int(sys.argv[2])
    train_model(i_model, index, epochs)
    model_test(i_model, index)
