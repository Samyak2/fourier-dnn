import tensorflow as tf
import matplotlib.pyplot as plt

from fourier_dnn.ffm_mlp import FourierMLP
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
                  metrics=['accuracy'])

    return model

def train_model(model, image_index, epochs, show_output=True):
    """Trains the model on 2D image dataset, but only on image with index
    image_index and for given number of epochs. The output is shown if show_output is True.
    """
    d = train_dataset.skip(image_index).take(1)
    d = next(iter(d))

    model.fit(d[0], d[1], epochs=epochs)

    o = model(d[0])

    if show_output:
        plt.imshow(o)
        plt.show()

if __name__ == "__main__":
    i_model = get_model()
    train_model(i_model, 0, 1000)
