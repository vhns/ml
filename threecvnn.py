from tensorflow import keras

def model():
    model = keras.Sequential(
        layers=[
            keras.layers.Rescaling(1./255),
            keras.layers.Input(shape=(32,32,3)),
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
