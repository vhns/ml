import PIL
import numpy as np
import tensorflow as tf
from pathlib import PurePath
import argparse
import re
import os
import math
import random
import sys
import visualization
import fnmatch

#This function generates an array of arrays (a matrix?) which contains the following:
#[0] = an array of image path(s) from the specified path and respective university
#[1] = an array of label(s) corresponding to each image
#
#The way we do so is by walking through the directory tree and trying to match the proper
#parameters specified in the function. Mind: we do a reverse match, that is, we match
#from the lowest item in the tree and then work our way up to make sure we are adding the
#proper file and label to the array. This could be better worked upon as to not iterate
#through useless paths as well as include some sort of multi threading.
#
# Here's some ASCII art of how we do so:
#
# pklot=[]
# matches = []
# requirements = ['PKLotSegmented', 'PUC']
#
# 2012-09-21_06_10_10#001.jpg (lowest item in the tree, and is a match to our jpg regex)
# |
# |                                       pklot=[]
# |                                       matches=['2012-09-21_06_10_10#001.jpg']
# |
# |
# |___
#      Empty
#      |
#      |_______
#              2012-09-21
#              |
#              |____________
#                           Rainy
#                           |
#                           |________
#                                    PUC (match, requirements[1])
#                                    |
#                                    |____________
#                                                 PKLotSegmented(match, requirements[0])
#
#                                                 pklot=[['PUC','Rainy','2012-09-21', \
#                                                         'Empty', \
#                                                         'PATHTOTHEJPG']]
#
#
def pathgen(path: str, university: str)->[[]]:

    dataset = []
    requirements = ['PKLotSegmented']
    pklot = []

    if university != None:
        requirements.append(university)

    for root, dirs, files in os.walk(top=path,followlinks=True):

        root_path = PurePath(root).parts

        if len(files) > 0:
            if all(requirement in root_path for requirement in requirements):
                for file in files:
                    if fnmatch.fnmatch(file, '*.jpg'):

                        university = root_path[-4]
                        weather = root_path[-3]
                        date = root_path[-2]
                        state = root_path[-1]
                        file_path = os.path.abspath(os.path.join(root,file))

                        pklot.append([university, weather, date, file_path])

    return pklot


def dataset_gen(datasetpath: str, percentagetrain: float, percentagetest: float, percentagevalidation: float, university: str, overlap_days: bool)->[[]]:

    dataset = pathgen(datasetpath, university)
    days = list(set([day[2] for day in dataset]))
    x = 0
    y = 0
    z = 0
    train = []
    test = []
    validation = []
    train_files = []
    test_files = []
    validation_files = []

    if overlap_days:

        x = int(math.floor(len(dataset)*percentagetrain))
        y = int(math.ceil(len(dataset)*percentagetest))
        z = int(math.ceil(len(dataset)*percentagevalidation))

        for i in range(0,x):
            l = random.randrange(0,len(dataset))
            train.append(dataset[l])
            dataset.pop(l)
        for i in range(0,y):
            l = random.randrange(0,len(dataset))
            test.append(dataset[l])
            dataset.pop(l)
        for i in range(0,z):
            l = random.randrange(0,len(dataset))
            validation.append(dataset[l])
            dataset.pop(l)

        train_files = train
        test_files = test
        validation_files = validation


    else:

        x = int(math.ceil(len(days)*percentagetrain))
        y = int(math.ceil(len(days)*percentagetest))
        z = int(math.ceil(len(days)*percentagevalidation))

        for i in range(0,x):
            l = random.randrange(0,len(days))
            train.append(days[l])
            days.pop(l)
        for i in range(0,y):
            l = random.randrange(0,len(days))
            test.append(days[l])
            days.pop(l)
        for i in range(0,z):
            l = random.randrange(0,len(days))
            validation.append(days[l])
            days.pop(l)



        for i in train:
                for j in dataset:
                        if j[2] == i:
                                train_files.append([j[-1],j[-2]])


        for i in test:
                for j in dataset:
                        if j[2] == i:
                                test_files.append([j[-1],j[-2]])


        for i in validation:
                for j in dataset:
                    if j[2] == i:
                        validation_files.append([j[-1],j[-2]])

        print(f"DIAS DE TESTE: {test}")
        print(f"DIAS DE TREINO: {train}")
        print(f"DIAS DE VALIDACAO: {validation}")


    print(f"QUANTIDADE TESTE: {len(test_files)}")
    print(f"QUANTIDADE TREINO: {len(train_files)}")
    print(f"QUANTIDADE VALIDACAO: {len(validation_files)}")

    return train_files, test_files, validation_files


#Converts the dataset labels from their string counterparts to ints.
def convert_label(i):
    match i:
        case 'Empty':
            return int(0)
        case 'Occupied':
            return int(1)
        case _:
            sys.exit("Error: Invalid label conversion, input: {label}")


def dataset_generator(data, img_size, random):
    idx = np.arange(len(data))

    if random:
        np.random.shuffle(idx)

    for i in idx:
        #label = str(data[i][1],encoding='utf-8')
        #label = convert_label(label)
        img = data[i][-1]
        #print(img)
        img = tf.keras.utils.load_img(img)
        img = img.resize((img_size,img_size))
        img = tf.keras.utils.img_to_array(img)
        #print(f'IMG SIZE 1: {img.shape}')
        #print(f'IMG SIZE 2: {img.shape}')
        yield img, img


def train_model(model,train_dataset,validation_dataset,test_dataset,checkpoint_filepath,img_size, model_path):

    train_ds = tf.data.Dataset.from_generator(
                dataset_generator,
                args=[train_dataset, img_size, True],
                output_signature=model.output_signature)

    validation_ds = tf.data.Dataset.from_generator(
                dataset_generator,
                args=[validation_dataset, img_size, True],
                output_signature=model.output_signature)

    test_ds = tf.data.Dataset.from_generator(
                dataset_generator,
                args=[test_dataset, img_size, True],
                output_signature=model.output_signature)

    callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.CSVLogger(
                filename=f'{model_path}/training.log',
                separator=',',
                append=False)
            ]

    visualization.show_sample(train_ds)

    visualization.show_sample(test_ds)

    visualization.show_sample(validation_ds)


    model.fit(x=train_ds.batch(32).prefetch(4),epochs=5, \
            validation_data=validation_ds.batch(32).prefetch(4),
              callbacks=callback)

    model.evaluate(x=test_ds.batch(32).prefetch(4))

    visualization.show_sample_with_results(model, test_ds)
