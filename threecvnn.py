import tensorflow as tf
import base

def model_3cvnn():
    model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(32,32,3)),
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), \
                                activation='relu', padding='same'),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), \
                                padding='valid'),
      tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), \
                                padding='valid'),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), \
                                padding='valid'),
      tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), \
                                padding='valid'),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), \
                                padding='valid'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64,activation='relu'),
      tf.keras.layers.Dense(2,activation='softmax')
    ])

    model.compile(optimizer='Adam', \
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
                    metrics=['accuracy'])

    return model
