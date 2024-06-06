import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import base


dataset = base.dataset_gen('./', 0.01, 0.01, 0.01, 'UFPR04')

train, test, validation = dataset

img_size = 32

x_train = tf.data.Dataset.from_generator(
            base.dataset_generator,
            args=[train, img_size, True, True],
            output_signature=(tf.TensorSpec(shape=(img_size,img_size,3), dtype=tf.float32),
                tf.TensorSpec(shape=(img_size,img_size,3), dtype=tf.float32)))

x_test = tf.data.Dataset.from_generator(
            base.dataset_generator,
            args=[test, img_size, True, True],
            output_signature=(tf.TensorSpec(shape=(img_size,img_size,3), dtype=tf.float32),
                tf.TensorSpec(shape=(img_size,img_size,3), dtype=tf.float32)))

x_validation = tf.data.Dataset.from_generator(
            base.dataset_generator,
            args=[validation, img_size, True, True],
            output_signature=(tf.TensorSpec(shape=(img_size,img_size,3), dtype=tf.float32),
                tf.TensorSpec(shape=(img_size,img_size,3), dtype=tf.float32)))

class Autoencoder(Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(32,32,3)),
      layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
      layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),padding='valid'),
      layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
      layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
      layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
      layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
      layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

shape = (32, 32, 3)
latent_dim = 64

print(f'SHAPE IS: {shape}')

autoencoder = Autoencoder(latent_dim, shape)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train.batch(32).prefetch(4),
                epochs=10,
                validation_data=x_test.batch(32).prefetch(4))

img = iter(x_test.take(10))
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    render_img = next(img)[0]
    encoded_imgs = autoencoder.encoder(render_img.numpy())
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(render_img)
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs)
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
