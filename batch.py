import os
import time
import tensorflow as tf
import base
import threecvnn
import mobilenetv3large
import efficientnet
import resnet50

def train_model(model,train_dataset,validation_dataset,test_dataset,checkpoint_filepath,img_size):

    train_ds = tf.data.Dataset.from_generator(
                base.shuffle_generator,
                args=[train_dataset, img_size],
                output_signature=(
                    tf.TensorSpec(shape=(img_size,img_size,3), dtype=tf.uint8),
                    tf.TensorSpec(shape=(), dtype=tf.uint8)))

    validation_ds = tf.data.Dataset.from_generator(
                base.shuffle_generator,
                args=[validation_dataset, img_size],
                output_signature=(
                    tf.TensorSpec(shape=(img_size,img_size,3), dtype=tf.uint8),
                    tf.TensorSpec(shape=(), dtype=tf.uint8)))

    test_ds = tf.data.Dataset.from_generator(
                base.shuffle_generator,
                args=[test_dataset, img_size],
                output_signature=(
                    tf.TensorSpec(shape=(img_size,img_size,3), dtype=tf.uint8),
                    tf.TensorSpec(shape=(), dtype=tf.uint8)))

    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model.fit(x=train_ds.batch(32).prefetch(4),epochs=1, \
            validation_data=validation_ds.batch(32).prefetch(4),
              callbacks=[callback],steps_per_epoch=(len(train_dataset)/32))

    model.evaluate(x=test_ds.batch(32).prefetch(4))

    return model


def batchdata(path,train,test,validation,university):

    if path == None:
        path = './'

    if train == None:
        train = 0.4

    if test == None:
        test = 0.5

    if validation == None:
        validation = 0.1

    universities = ['PUC', 'UFPR04', 'UFPR05']

    dataset = base.dataset_gen(path,train,test,validation,university)

    train_dataset, test_dataset, validation_dataset = dataset

    models = [threecvnn.model_3cvnn,
              mobilenetv3large.mdl_mbnetv3lg_ntp_static,
              mobilenetv3large.mdl_mbnetv3lg_ntp_dynamic,
              efficientnet.mdl_efficientnet_ntp_static,
              efficientnet.mdl_efficientnet_ntp_dynamic,
              resnet50.mdl_rsnt50_ntp_static,
              resnet50.mdl_rsnt50_ntp_dynamic]


    for model in models:

        modelname = model.__name__

        img_size = 128

        if modelname == 'model_3cvnn':
            img_size = 32

        for university in universities:

            university = university

            model_run = model()

            os.makedirs(name=f'./tests/{modelname}/{university}/ending',
                        exist_ok=True)

            checkpoint_filepath = f'./tests/{modelname}/{university}/'

            model_run = train_model(model_run,train_dataset,validation_dataset,test_dataset, checkpoint_filepath, img_size)

            model_run.save(f'./tests/{modelname}/{university}/ending/model.keras')

            accuracy_file = open(f'./tests/{modelname}/{university}/ending/accuracy.txt', 'w')
            accuracy_file.write(f'Accuracy: {model_run.get_metrics_result()}')
