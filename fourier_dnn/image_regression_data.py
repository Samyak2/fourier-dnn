import tensorflow as tf
import numpy as np

filename = 'data/data_div2k.npz'
npz_data = np.load(filename)

train_images = npz_data['train_data'] / 255
test_images = npz_data['test_data'] / 255

RES = 512

x1 = np.linspace(0, 1, RES//2+1)[:-1]
x_train = np.stack(np.meshgrid(x1, x1), axis=-1)
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)

x1_t = np.linspace(0, 1, RES+1)[:-1]
x_test = np.stack(np.meshgrid(x1_t, x1_t), axis=-1)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

x_train_t = tf.stack([x_train for _ in range(train_images.shape[0])], axis=0)
y_train_t = tf.convert_to_tensor(train_images[:, ::2, ::2, :], dtype=tf.float32)

x_test_t = tf.stack([x_test for _ in range(train_images.shape[0])], axis=0)
y_test_t = tf.convert_to_tensor(train_images[:, 1::2, 1::2, :], dtype=tf.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_t, y_train_t))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_t, y_test_t))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.imshow(train_images[0, :, :, :])
    plt.show()
