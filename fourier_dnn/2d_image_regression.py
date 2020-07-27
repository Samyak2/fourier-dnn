import tensorflow as tf
import ffm

model = tf.keras.models.Sequential([
    ffm.BasicFFM(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(3, activation="sigmoid")
])

loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
              loss=loss_fn,
              metrics=['accuracy'])

