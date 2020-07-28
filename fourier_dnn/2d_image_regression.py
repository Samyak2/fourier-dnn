import tensorflow as tf
from fourier_dnn import ffm
from image_regression_data import train_dataset, test_dataset
import matplotlib.pyplot as plt

EPOCHS = 100

model = tf.keras.models.Sequential([
    ffm.BasicFFM(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

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

