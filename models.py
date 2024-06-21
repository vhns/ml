from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from PIL import Image
import base
import matplotlib.pyplot as plt
import io
import json
# import pdb


class ThreeCvnn(keras.Model):
    def __init__(self, num_classes=2, random=True, dataset=None, model_path=None, epochs=None, logger=None):
        super().__init__()

        self.logger = logger

        self.logger.info("Instanciando modelo ThreeCvnn com os parametros:")
        self.logger.info(f'num_classes: {num_classes}')
        self.logger.info(f'random: {random}')
        json.dump(dataset, open(f'{model_path}/dataset.json', 'w'), indent=4)
        self.logger.info(f'dataset: salvo em {model_path}/dataset.json')
        self.logger.info(f'model_path: {model_path}')
        self.logger.info(f'epochs: {epochs}')

        self.random = random
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs

        logger.info(
            f'Quantidade de dias de teste: {len(self.dataset["test"])}')
        logger.info(
            f'Quantidade de dias de treino: {len(self.dataset["train"])}')
        logger.info(
            f'Quantidade de dias de validation: {len(self.dataset["validation"])}')

        self.callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}/best.weights.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=f'{self.model_path}/training.log',
                separator=',',
                append=False)
        ]

        self.output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
                                 tf.TensorSpec(shape=(), dtype=tf.uint8))
        self.train_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['train']), 32, True],
            output_signature=self.output_signature)
        self.test_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['test']), 32, True],
            output_signature=self.output_signature)
        self.validation_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['validation']), 32, True],
            output_signature=self.output_signature)

        self.block_0 = keras.layers.Conv2D(32, kernel_size=(
            3, 3), strides=(1, 1), activation='relu', padding='same')
        self.block_1 = keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.block_2 = keras.layers.Conv2D(
            64, kernel_size=(3, 3), strides=(1, 1), padding='valid')
        self.block_3 = keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.block_4 = keras.layers.Conv2D(
            64, kernel_size=(3, 3), strides=(1, 1), padding='valid')
        self.block_5 = keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.block_6 = keras.layers.Flatten()
        self.block_7 = keras.layers.Dense(64, activation='relu')
        self.block_8 = keras.layers.Dense(2, activation='softmax')

        self.logger.info("Concluida instanciação do modelo ThreeCvnn")

    def call(self, inputs, training=False):
        x = self.block_0(inputs, training=training)
        x = self.block_1(x, training=training)
        x = self.block_2(x, training=training)
        x = self.block_3(x, training=training)
        x = self.block_4(x, training=training)
        x = self.block_5(x, training=training)
        x = self.block_6(x, training=training)
        x = self.block_7(x, training=training)
        return self.block_8(x, training=training)

    def generator(self, data, img_size, random):

        idx = np.arange(len(data))

        if random:
            np.random.shuffle(idx)

        for i in idx:
            label = base.convert_label(data[i][1].decode("utf-8"))
            img = tf.keras.utils.load_img(data[i][0])

            img = img.resize((img_size, img_size))
            #img = tf.keras.utils.img_to_array(img)/255.0
            img = tf.keras.utils.img_to_array(img)

            yield img, label

    def train(self):

        self.logger.info("Iniciando treino do modelo ThreeCvnn")

        self.logger.info("Compilando o modelo")
        self.compile(optimizer='Adam',
                     loss=keras.losses.SparseCategoricalCrossentropy(
                         from_logits=True),
                     metrics=['accuracy'])
        self.logger.info("Modelo compilado")

        self.logger.info("Iniciando model.fit do ThreeCvnn")
        history = self.fit(x=self.train_dataset.batch(32).prefetch(4), epochs=self.epochs,
                           validation_data=self.validation_dataset.batch(
                               32).prefetch(4),
                           callbacks=self.callback)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/loss.png')
        plt.close()

        self.logger.info("Concluido model.fit")

        self.logger.info("Salvando o modelo")
        self.save(f'{self.model_path}/final_model.keras')
        self.logger.info(
            f'Modelo salvo em {self.model_path}/final_model.keras')

    def visualization(self):
        self.load_weights(f'{self.model_path}/final_model.keras')
        data = iter(self.test_dataset.take(36))
        fig = plt.figure(figsize=(32, 32))
        rows = 6
        columns = 6
        for i in range(36):
            image, label = next(data)
            image_with_batch = np.expand_dims(image, axis=0)
            predicted_label = self.predict(image_with_batch)[0].tolist()

            prediction = None
            for j in predicted_label:
                if prediction == None or j > prediction:
                    prediction = j
                    prediction_pos = predicted_label.index(j)
                    prediction_pos = base.convert_label(prediction_pos)

            fig.add_subplot(rows, columns, i+1)
            plt.imshow(image)
            plt.axis("off")
            plt.title(f"Predicted: {prediction_pos} Truth: {label}")
        plt.savefig(f'{self.model_path}/prediction.png')
        plt.close()

class Whale(keras.Model):
    def __init__(self, num_classes=2, random=True, dataset=None, model_path=None, epochs=None, logger=None):
        super().__init__()

        self.logger = logger

        self.logger.info(
            "Instanciando modelo ThreeCvnn_Encoder com os parametros:")
        self.logger.info(f'num_classes: {num_classes}')
        self.logger.info(f'random: {random}')
        json.dump(dataset, open(f'{model_path}/dataset.json', 'w'), indent=4)
        self.logger.info(f'dataset: salvo em {model_path}/dataset.txt')
        self.logger.info(f'model_path: {model_path}')
        self.logger.info(f'epochs: {epochs}')

        self.random = random
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs

        logger.info(
            f'Quantidade de dias de teste: {len(self.dataset["test"])}')
        logger.info(
            f'Quantidade de dias de treino: {len(self.dataset["train"])}')
        logger.info(
            f'Quantidade de dias de validation: {len(self.dataset["validation"])}')

        self.callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}/best.weights.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=f'{self.model_path}/training.log',
                separator=',',
                append=False)
        ]

        self.output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
                                 tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32))

        self.train_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['train']), 32, True],
            output_signature=self.output_signature)
        self.test_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['test']), 32, True],
            output_signature=self.output_signature)
        self.validation_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['validation']), 32, True],
            output_signature=self.output_signature)

        self.encoder = tf.keras.Sequential([
            keras.layers.Conv2D(192, kernel_size=(3, 3), strides=(
                1, 1), activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(
                2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Conv2D(64, kernel_size=(
                3, 3), strides=(1, 1), padding='valid', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(
                2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Conv2D(64, kernel_size=(
                3, 3), strides=(1, 1), padding='valid', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(
                2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu')
        ],
            name='encoder'
        )
        self.decoder = tf.keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            # Shape to match the output of the last MaxPool2D layer in encoder
            keras.layers.Reshape((4, 4, 4)),
            keras.layers.Conv2DTranspose(64, kernel_size=(
                3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(64, kernel_size=(
                3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(32, kernel_size=(
                3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=(
                1, 1), padding='same', activation='sigmoid')
        ],
            name='decoder'
        )

        self.logger.info("Concluida instanciação do modelo ThreeCvnn_encoder")

    def call(self, inputs, training=False):
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def generator(self, data, img_size, random):

        idx = np.arange(len(data))

        if random:
            np.random.shuffle(idx)

        for i in idx:
            img = tf.keras.utils.load_img(data[i][0])

            img = img.resize((img_size, img_size))
            #img = tf.keras.utils.img_to_array(img)/255.0
            img = tf.keras.utils.img_to_array(img)

            yield img, img

    def train(self):
        self.logger.info("Iniciando treino do modelo ThreeCvnn_encoder")

        self.logger.info("Compilando o modelo")
        self.compile(optimizer='Adam',
                     loss=keras.losses.MeanAbsoluteError(),
                     metrics=['accuracy'])
        self.logger.info("Modelo compilado")

        self.logger.info("Iniciando model.fit do ThreeCvnn_encoder")
        history = self.fit(x=self.train_dataset.batch(32).prefetch(4), epochs=self.epochs,
                           validation_data=self.validation_dataset.batch(
                               32).prefetch(4),
                           callbacks=self.callback)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/loss.png')
        plt.close()
        self.logger.info("Concluido model.fit")

        self.logger.info("Salvando o modelo")
        self.save_weights(f'{self.model_path}/final_model.weights.h5')
        self.logger.info(
            f'Modelo salvo em {self.model_path}/final_model.weights.h5')

    def visualization(self):
        self.load_weights(f'{self.model_path}/final_model.weights.h5')
        images = iter(self.test_dataset.take(9))
        fig = plt.figure(figsize=(32, 32))
        rows = 2
        columns = 9
        for i in range(9):
            image, _ = next(images)
            image_with_batch = np.expand_dims(image, axis=0)
            image_generated = tf.keras.utils.array_to_img(
                self(image_with_batch)[0])
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(image_generated)
            plt.axis("off")
            plt.title("Generated")
            fig.add_subplot(rows, columns, i+10)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title("Original")
        plt.savefig(f'{self.model_path}/prediction.png')
        plt.close()


class ThreeCvnn_Encoder(keras.Model):
    def __init__(self, num_classes=2, random=True, dataset=None, model_path=None, epochs=None, logger=None):
        super().__init__()

        self.logger = logger

        self.logger.info(
            "Instanciando modelo ThreeCvnn_Encoder com os parametros:")
        self.logger.info(f'num_classes: {num_classes}')
        self.logger.info(f'random: {random}')
        json.dump(dataset, open(f'{model_path}/dataset.json', 'w'), indent=4)
        self.logger.info(f'dataset: salvo em {model_path}/dataset.txt')
        self.logger.info(f'model_path: {model_path}')
        self.logger.info(f'epochs: {epochs}')

        self.random = random
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs

        logger.info(
            f'Quantidade de dias de teste: {len(self.dataset["test"])}')
        logger.info(
            f'Quantidade de dias de treino: {len(self.dataset["train"])}')
        logger.info(
            f'Quantidade de dias de validation: {len(self.dataset["validation"])}')

        self.callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}/best.weights.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=f'{self.model_path}/training.log',
                separator=',',
                append=False)
        ]

        self.output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
                                 tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32))

        self.train_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['train']), 32, True],
            output_signature=self.output_signature)
        self.test_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['test']), 32, True],
            output_signature=self.output_signature)
        self.validation_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['validation']), 32, True],
            output_signature=self.output_signature)

        self.encoder = tf.keras.Sequential([
            keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(
                1, 1), activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(
                2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Conv2D(64, kernel_size=(
                3, 3), strides=(1, 1), padding='valid', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(
                2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Conv2D(64, kernel_size=(
                3, 3), strides=(1, 1), padding='valid', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(
                2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu')
        ],
            name='encoder'
        )
        self.decoder = tf.keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            # Shape to match the output of the last MaxPool2D layer in encoder
            keras.layers.Reshape((4, 4, 4)),
            keras.layers.Conv2DTranspose(64, kernel_size=(
                3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(64, kernel_size=(
                3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(32, kernel_size=(
                3, 3), strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=(
                1, 1), padding='same', activation='sigmoid')
        ],
            name='decoder'
        )

        self.logger.info("Concluida instanciação do modelo ThreeCvnn_encoder")

    def call(self, inputs, training=False):
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def generator(self, data, img_size, random):

        idx = np.arange(len(data))

        if random:
            np.random.shuffle(idx)

        for i in idx:
            img = tf.keras.utils.load_img(data[i][0])

            img = img.resize((img_size, img_size))
            #img = tf.keras.utils.img_to_array(img)/255.0
            img = tf.keras.utils.img_to_array(img)

            yield img, img

    def train(self):
        self.logger.info("Iniciando treino do modelo ThreeCvnn_encoder")

        self.logger.info("Compilando o modelo")
        self.compile(optimizer='Adam',
                     loss=keras.losses.MeanAbsoluteError(),
                     metrics=['accuracy'])
        self.logger.info("Modelo compilado")

        self.logger.info("Iniciando model.fit do ThreeCvnn_encoder")
        history = self.fit(x=self.train_dataset.batch(32).prefetch(4), epochs=self.epochs,
                           validation_data=self.validation_dataset.batch(
                               32).prefetch(4),
                           callbacks=self.callback)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/loss.png')
        plt.close()
        self.logger.info("Concluido model.fit")

        self.logger.info("Salvando o modelo")
        self.save_weights(f'{self.model_path}/final_model.weights.h5')
        self.logger.info(
            f'Modelo salvo em {self.model_path}/final_model.weights.h5')

    def visualization(self):
        self.load_weights(f'{self.model_path}/final_model.weights.h5')
        images = iter(self.test_dataset.take(9))
        fig = plt.figure(figsize=(32, 32))
        rows = 2
        columns = 9
        for i in range(9):
            image, _ = next(images)
            image_with_batch = np.expand_dims(image, axis=0)
            image_generated = tf.keras.utils.array_to_img(
                self(image_with_batch)[0])
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(image_generated)
            plt.axis("off")
            plt.title("Generated")
            fig.add_subplot(rows, columns, i+10)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title("Original")
        plt.savefig(f'{self.model_path}/prediction.png')
        plt.close()


class ThreeCvnnClassifier(keras.Model):
    def __init__(self, num_classes=2, random=True, dataset=None, model_path=None, epochs=None, pretrained_weights=None, logger=None):
        super().__init__()

        self.logger = logger

        self.logger.info(
            "Instanciando modelo ThreeCvnnClassifier com os parametros:")
        self.logger.info(f'num_classes: {num_classes}')
        self.logger.info(f'random: {random}')
        json.dump(dataset, open(f'{model_path}/dataset.json', 'w'), indent=4)
        self.logger.info(f'dataset: salvo em {model_path}/dataset.txt')
        self.logger.info(f'model_path: {model_path}')
        self.logger.info(f'epochs: {epochs}')

        self.random = random
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs
        self.pretrained_weights = pretrained_weights

        logger.info(
            f'Quantidade de dias de teste: {len(self.dataset["test"])}')
        logger.info(
            f'Quantidade de dias de treino: {len(self.dataset["train"])}')
        logger.info(
            f'Quantidade de dias de validation: {len(self.dataset["validation"])}')

        self.callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}/best.weights.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=f'{self.model_path}/training.log',
                separator=',',
                append=False)
        ]

        self.output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
                                 tf.TensorSpec(shape=(), dtype=tf.uint8))

        self.train_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['train']), 32, True],
            output_signature=self.output_signature)
        self.test_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['test']), 32, True],
            output_signature=self.output_signature)
        self.validation_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['validation']), 32, True],
            output_signature=self.output_signature)

        self.base_model = ThreeCvnn_Encoder(
            num_classes=num_classes, random=random, dataset=dataset, model_path=model_path, epochs=epochs, logger=logger)
        self.base_model.load_weights(pretrained_weights)
        self.base_model.trainable = True
        self.base_model.layers.pop()

        self.dense = tf.keras.Sequential(
            layers=[
                keras.layers.Flatten(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation='relu'),                
                keras.layers.Dense(2, activation='softmax')
            ],
            name='dense'
        )

        self.logger.info(
            "Concluida instanciação do modelo ThreeCvnnClassifier")

    def call(self, x, training=False):
        x = self.base_model(x, training=training)
        x = self.dense(x, training=training)
        return x

    def generator(self, data, img_size, random):

        idx = np.arange(len(data))

        if random:
            np.random.shuffle(idx)

        for i in idx:
            label = base.convert_label(data[i][1].decode("utf-8"))
            img = tf.keras.utils.load_img(data[i][0])

            img = img.resize((img_size, img_size))
            # img = tf.keras.utils.img_to_array(img)/255.0
            img = tf.keras.utils.img_to_array(img)

            yield img, label

    def train(self):
        self.logger.info("Iniciando treino do modelo ThreeCvnnClassifier")

        self.logger.info("Compilando o modelo")
        self.compile(optimizer='Adam',
                     loss=keras.losses.SparseCategoricalCrossentropy(
                         from_logits=True),
                     metrics=['accuracy'])
        self.logger.info("Modelo compilado")

        self.logger.info("Iniciando model.fit do ThreeCvnnClassifier")
        history = self.fit(x=self.train_dataset.batch(32).prefetch(4), epochs=self.epochs,
                           validation_data=self.validation_dataset.batch(
                               32).prefetch(4),
                           callbacks=self.callback)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{self.model_path}/accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{self.model_path}/loss.png')
        plt.close()
        self.logger.info("Concluido model.fit")

        self.logger.info("Salvando o modelo")
        self.save_weights(f'{self.model_path}/final_model.weights.h5')
        self.logger.info(
            f'Modelo salvo em {self.model_path}/final_model.weights.h5')

    def visualization(self):
        self.load_weights(f'{self.model_path}/final_model.weights.h5')
        data = iter(self.test_dataset.take(36))
        fig = plt.figure(figsize=(32, 32))
        rows = 6
        columns = 6
        for i in range(36):
            image, label = next(data)
            image_with_batch = np.expand_dims(image, axis=0)
            predicted_label = self.predict(image_with_batch)[0].tolist()
            #predicted_label = self.predict(image_with_batch)[0]

            prediction = None
            for j in predicted_label:
                if prediction == None or j > prediction:
                    prediction = j
                    prediction_pos = predicted_label.index(j)
                    prediction_pos = base.convert_label(prediction_pos)

            fig.add_subplot(rows, columns, i+1)
            #plt.imshow(image)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title(f"Predicted: {prediction_pos} Truth: {label}")
        plt.savefig(f'{self.model_path}/prediction.png')
        plt.close()


class Simple(keras.Model):
    def __init__(self, num_classes=2, random=True, dataset=None, model_path=None, epochs=None, logger=None):
        super().__init__()

        self.logger = logger

        self.logger.info("Instanciando modelo Simple com os parametros:")
        self.logger.info(f'num_classes: {num_classes}')
        self.logger.info(f'random: {random}')
        json.dump(dataset, open(f'{model_path}/dataset.json', 'w'), indent=4)
        self.logger.info(f'dataset: salvo em {model_path}/dataset.txt')
        self.logger.info(f'model_path: {model_path}')
        self.logger.info(f'epochs: {epochs}')

        self.random = random
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs

        logger.info(
            f'Quantidade de dias de teste: {len(self.dataset["test"])}')
        logger.info(
            f'Quantidade de dias de treino: {len(self.dataset["train"])}')
        logger.info(
            f'Quantidade de dias de validation: {len(self.dataset["validation"])}')

        self.callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}/best.weights.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=str(f'{self.model_path}/training.log'),
                separator=',',
                append=False)
        ]

        self.output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
                                 tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32))
        self.train_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['train']), 32, True],
            output_signature=self.output_signature)
        self.test_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['test']), 32, True],
            output_signature=self.output_signature)
        self.validation_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[list(self.dataset['validation']), 32, True],
            output_signature=self.output_signature)

        self.latent_dim = 8
        self.shape = (32, 32, 3)

        self.encoder = tf.keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(self.latent_dim, activation='relu'),
        ],
            name='encoder')
        self.decoder = tf.keras.Sequential([
            keras.layers.Dense(tf.math.reduce_prod(
                self.shape).numpy(), activation='sigmoid'),
            keras.layers.Reshape(self.shape)
        ],
            name='decoder')

        self.logger.info("Concluida instanciação do modelo Simple")

    def call(self, inputs, training=False):
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def generator(self, data, img_size, random):

        idx = np.arange(len(data))

        if random:
            np.random.shuffle(idx)

        for i in idx:
            img = tf.keras.utils.load_img(data[i][0])

            img = img.resize((img_size, img_size))
            # img = tf.keras.utils.img_to_array(img)/255.0
            img = tf.keras.utils.img_to_array(img)

            yield img, img

    def train(self):
        self.logger.info("Iniciando treino do modelo Simple")

        self.logger.info("Compilando o modelo")
        self.compile(optimizer='Adam',
                     loss=keras.losses.MeanAbsoluteError(),
                     metrics=['accuracy'])
        self.logger.info("Modelo compilado")

        self.logger.info("Iniciando model.fit do Simple")
        history = self.fit(x=self.train_dataset.batch(32).prefetch(4), epochs=self.epochs,
                           validation_data=self.validation_dataset.batch(
                               32).prefetch(4),
                           callbacks=self.callback)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{self.model_path}/accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{self.model_path}/loss.png')
        plt.close()
        self.logger.info("Concluido model.fit")

        self.logger.info("Salvando o modelo")
        self.save(f'{self.model_path}/final_model.keras')
        self.logger.info(f'Modelo salvo em {self.model_path}/final_model/')

    def visualization(self):
        self.load_weights(f'{self.model_path}/final_model.keras')
        images = iter(self.test_dataset.take(9))
        fig = plt.figure(figsize=(32, 32))
        rows = 2
        columns = 9
        for i in range(9):
            image, _ = next(images)
            image_with_batch = np.expand_dims(image, axis=0)
            image_generated = tf.keras.utils.array_to_img(
                self(image_with_batch)[0])
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(image_generated)
            plt.axis("off")
            plt.title("Generated")
            fig.add_subplot(rows, columns, i+10)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title("Original")
        plt.savefig(f'{self.model_path}/prediction.png')
        plt.close()


class SiameseTraining(keras.Model):
    def __init__(self, num_classes=2, random=True, dataset=None, model_path=None, epochs=None, logger=None):
        super().__init__()

        self.logger = logger

        self.logger.info(
            "Instanciando modelo SiameseTraining com os parametros:")
        self.logger.info(f'num_classes: {num_classes}')
        self.logger.info(f'random: {random}')
        json.dump(dataset, open(f'{model_path}/dataset.json', 'w'), indent=4)
        self.logger.info(f'dataset: salvo em {model_path}/dataset.json')
        self.logger.info(f'model_path: {model_path}')
        self.logger.info(f'epochs: {epochs}')

        self.random = random
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs

        logger.info(
            f'Quantidade de dias de teste: {len(self.dataset["test"])}')
        logger.info(
            f'Quantidade de dias de treino: {len(self.dataset["train"])}')
        logger.info(
            f'Quantidade de dias de validation: {len(self.dataset["validation"])}')

        self.output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
                                 tf.TensorSpec(shape=(), dtype=tf.uint8))

        self.reference_pairs_train = self.generate_pairs(
            'train', 32, self.random)
        self.reference_pairs_test = self.generate_pairs(
            'test', 32, self.random)
        self.reference_pairs_validation = self.generate_pairs(
            'validation', 32, self.random)

        self.train_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[self.dataset['train'], 32, True,
                  self.reference_pairs_train, False],
            output_signature=self.output_signature)
        self.test_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[self.dataset['test'], 32, True,
                  self.reference_pairs_test, False],
            output_signature=self.output_signature)
        self.validation_dataset = tf.data.Dataset.from_generator(
            self.generator,
            args=[self.dataset['validation'], 32, True,
                  self.reference_pairs_test, False],
            output_signature=self.output_signature)

        self.mobilenet = keras.applications.MobileNetV2(weights='imagenet',
                                                        include_top=False,
                                                        input_shape=(
                                                            32, 32, 3),
                                                        pooling='avg')

        for layer in self.mobilenet.layers:
            layer.trainable = False

        self.left = keras.Sequential([
            self.mobilenet,
            keras.layers.Flatten(),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu',
                               kernel_regularizer=keras.regularizers.L2(0.001))
        ])
        self.right = keras.Sequential([
            self.mobilenet,
            keras.layers.Flatten(),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu',
                               kernel_regularizer=keras.regularizers.L2(0.001))
        ])

    def call(self, x, training=False):
        x_right = self.right(x)
        x_left = self.left(x)
        x = keras.layers.Lambda(self.euclidean_distance,
                                output_shape=(1,))(x_right, x_left)
        return keras.layers.BatchNormalization()(x)

    def euclidean_distance(x, y):
        sum_square = tf.keras.ops.sum(
            keras.ops.square(x - y), axis=1, keepdims=True)
        return keras.ops.sqrt(keras.ops.maximum(sum_square, keras.backend.epsilon()))

    def generate_pairs(self, datatype, img_size, random):

        output = []
        data = self.dataset[datatype]

        idx = np.arange(len(data))
        np.random.shuffle(idx)
        while len(output) < 5:
            index, idx = idx[-1], idx[:-1]
            label = data[index][1]
            if label == 'Empty':
                img = data[index][0]
                output.append([img, label])
                data.pop(index)

        idx = np.arange(len(data))
        np.random.shuffle(idx)
        while len(output) < 10:
            index, idx = idx[-1], idx[:-1]
            label = data[index][1]
            if label == 'Occupied':
                img = data[index][0]
                output.append([img, label])
                data.pop(index)

        self.dataset[datatype] = data

        return output

    def generator(self, data, img_size, random, pairs, reference):

        data_augmentation = tf.keras.Sequential([
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.4),
        ])

        if reference:
            idx = np.arange(len(data))

            n = 0
            for i in idx:
                for n in pairs:
                    pair_image = pairs[n][0]
                    pair_label = pairs[n][1]
                    pair_image = tf.keras.utils.load_img(pair_image[i][0])
                    pair_image = pair_image.resize((img_size, img_size))
                    pair_image = data_augmentation(
                        tf.keras.utils.img_to_array(pair_image)/255.0)

                    image_label = data[i][1]
                    label = int(image_label == pair_label)

                    yield (pair_image, label)

        else:
            idx = np.arange(len(data))

            if random:
                np.random.shuffle(idx)

            n = 0
            for i in idx:

                if n > 9:
                    n = 0
                pair_label = pairs[n][1]

                image_label = data[i][1]
                image = tf.keras.utils.load_img(data[i][0])
                image = image.resize((img_size, img_size))
                image = tf.keras.utils.img_to_array(image)/255.0
                label = (image_label == pair_label)

                n += 1
                yield (image, np.array(label))

    def train(self):

        self.callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_path}/best.weights.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=f'{self.model_path}/training_inputs.log',
                separator=',',
                append=False)
        ]

        self.compile(optimizer='Adam',
                     loss=keras.losses.MeanAbsoluteError(),
                     metrics=['accuracy'])

        history = self.fit(x=self.train_dataset.batch(32).prefetch(4), epochs=self.epochs,
                           validation_data=self.validation_dataset.batch(
            32).prefetch(4),
            callbacks=self.callback)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{self.model_path}/loss.png')
        plt.close()

        self.save(f'{self.model_path}/final_model/siamese_inputs.keras')

    def visualization(self):
        self.load_weights(
            f'{self.model_path}/final_model/siamese_inputs.keras')
        images = iter(self.test_dataset.take(9))
        fig = plt.figure(figsize=(32, 32))
        rows = 2
        columns = 9
        for i in range(9):
            image, _ = next(images)
            image_with_batch = np.expand_dims(image, axis=0)
            image_generated = tf.keras.utils.array_to_img(
                self(image_with_batch)[0])
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(image_generated)
            plt.axis("off")
            plt.title("Generated")
            fig.add_subplot(rows, columns, i+10)
            plt.imshow(tf.keras.utils.array_to_img(image))
            plt.axis("off")
            plt.title("Original")
        plt.savefig(f'{self.model_path}/prediction.png')
        plt.close()

    def visualization_pairs(self):
        pairs = [self.reference_pairs_test,
                 self.reference_pairs_train,
                 self.reference_pairs_validation]

        for pair in pairs:
            fig = plt.figure(figsize=(32, 32))
            rows = 2
            columns = 5
            for i in range(10):
                image = pair[i][0]
                image = tf.keras.utils.load_img(image)
                image = image.resize((32, 32))
                image = tf.keras.utils.img_to_array(image)/255.0
                label = pair[i][1]
                fig.add_subplot(rows, columns, i+1)
                plt.imshow(image)
                plt.axis("off")
                plt.title(f'{label}')
            plt.savefig(f'{self.model_path}/pairs.png')
            plt.close()


class Siamese(keras.Model):
    def __init__(self):
        super().__init__()

        # self.latent_dim = 8
        # self.shape = (32, 32, 3)

        # self.encoder = tf.keras.Sequential([
        #    keras.layers.Flatten(),
        #    keras.layers.Dense(self.latent_dim, activation='relu'),
        # ])
        # self.decoder = tf.keras.Sequential([
        #    keras.layers.Dense(tf.math.reduce_prod(self.shape).numpy(), activation='sigmoid'),
        #    keras.layers.Reshape(self.shape)
        # ])
        self.block_0 = keras.layers.Conv2D(32, kernel_size=(
            3, 3), strides=(1, 1), activation='relu', padding='same')
        self.block_1 = keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.block_2 = keras.layers.Conv2D(
            64, kernel_size=(3, 3), strides=(1, 1), padding='valid')
        self.block_3 = keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.block_4 = keras.layers.Conv2D(
            64, kernel_size=(3, 3), strides=(1, 1), padding='valid')
        self.block_5 = keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.block_6 = keras.layers.Flatten()
        self.block_7 = keras.layers.Dense(64, activation='relu')
        self.block_8 = keras.layers.Dense(2, activation='softmax')

    def call(self, inputs, training=False):
        x = self.block_0(inputs, training=training)
        x = self.block_1(x, training=training)
        x = self.block_2(x, training=training)
        x = self.block_3(x, training=training)
        x = self.block_4(x, training=training)
        x = self.block_5(x, training=training)
        x = self.block_6(x, training=training)
        x = self.block_7(x, training=training)
        return self.block_8(x, training=training)

    # def call(self, inputs):
    #    encoded = self.encoder(inputs)
    #    decoded = self.decoder(encoded)
    #    return decoded
