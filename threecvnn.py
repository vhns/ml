from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf

class ThreeCvnn(keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        self.block_1 = keras.layers.Input(shape=(32,32,3))
        self.block_2 = keras.layers.Rescaling(1./255),
        self.block_3 = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
        self.block_4 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),padding='valid'),
        self.block_5 = keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
        self.block_6 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        self.block_7 = keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
        self.block_8 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        self.block_9 = keras.layers.Flatten(),
        self.block_10 = keras.layers.Dense(64,activation='relu'),
        self.block_11 = keras.layers.Dense(2,activation='softmax')
    def call(self, inputs, training=False):
        x = self.block_1(inputs)
        x = self.block_2(x, training=training)
        x = self.block_3(x, training=training)
        x = self.block_4(x, training=training)
        x = self.block_5(x, training=training)
        x = self.block_6(x, training=training)
        x = self.block_7(x, training=training)
        x = self.block_8(x, training=training)
        x = self.block_9(x, training=training)
        x = self.block_10(x, training=training)
        x = self.block_11(x, training=training)



































def model():
    model = keras.Sequential(
        layers=[
            keras.layers.Input(shape=(32,32,3)),
            keras.layers.Rescaling(1./255),
            keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),padding='valid'),
            keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Flatten(),
            keras.layers.Dense(64,activation='relu'),
            keras.layers.Dense(2,activation='softmax')
        ],
        name='threecvnn')

    model.compile(optimizer='Adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def encoder():
    class Autoencoder(Model):
        def __init__(self):
            super(Autoencoder, self).__init__()
            self.encoder = keras.Sequential([
                keras.layers.Input(shape=(64,64,3)),

                keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),padding='valid'),
                keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
                keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
                keras.layers.Flatten(),
                keras.layers.Dense(64,activation='relu')])

            self.decoder = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(64,)),
                keras.layers.Reshape((4, 4, 4)),
                keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
                keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
                keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
                keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='sigmoid'),

            ])



            self.output_signature=(tf.TensorSpec(shape=(64,64,3), dtype=tf.uint8),tf.TensorSpec(shape=(64,64,3), dtype=tf.uint8))


        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    model = Autoencoder()

    return model
