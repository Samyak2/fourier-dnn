import tensorflow as tf
import matplotlib.pyplot as plt

from fourier_dnn import ffm
from fourier_dnn.ffm_mlp import FourierMLP
from image_regression_data import train_dataset, test_dataset

EPOCHS = 100

model = FourierMLP(4, 256, 256, 3, gaussian=False)

loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
              loss=loss_fn,
              metrics=['accuracy'])

d = next(iter(train_dataset))

model.fit(d[0], d[1], epochs=100)

o = model(d[0])

plt.imshow(o)
plt.show()

