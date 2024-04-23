import tensorflow as tf
import base

def mdl_mbnetv3lg_ntp_static():
    base_model = tf.keras.applications.MobileNetV3Large(
                    input_shape=(128,128,3),
                    include_top=False)
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(2,activation='softmax')
    ])

    model.compile(optimizer='Adam', \
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
                    metrics=['accuracy'])

    return model

def mdl_mbnetv3lg_ntp_dynamic():
    base_model = tf.keras.applications.MobileNetV3Large(
                    input_shape=(128,128,3),
                    include_top=False)
    base_model.trainable = True
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(2,activation='softmax')
    ])

    model.compile(optimizer='Adam', \
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
                    metrics=['accuracy'])

    return model
