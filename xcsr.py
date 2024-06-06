from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf

def model(latent_dim, shape):
    class Autoencoder(Model):
      def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
          keras.layers.Rescaling(1./255),
          keras.layers.Input(shape=(192,192,3)),
          keras.layers.Conv2D(192, kernel_size=(3,3), strides=(1,1), activation
          keras.layers.Conv2D(32,kernel_size=
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
    return Autoencoder(latent_dim, shape)
