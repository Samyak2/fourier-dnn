import tensorflow as tf
from ffm_mlp import FourierMLP
from image_regression_data import train_dataset, test_dataset
import matplotlib.pyplot as plt
import os

EPOCHS = 1000

model = FourierMLP(10, 128, 256, 3)

loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
              loss=loss_fn,
              metrics=['accuracy'])

d = next(iter(train_dataset))

model.fit(d[0], d[1], epochs=EPOCHS)

o = model(d[0])

plt.imshow(o)
plt.show()

