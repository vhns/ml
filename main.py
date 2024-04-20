import PIL
#from kera import layers
#from time import gtime, strftime
#import csv
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import pdb
from pathlib import PurePath
import argparse
import re
import os
import math
import random
import sys

def search(pattern: type[re.Pattern], files: []):

    matches = []

    for i in files:
        if pattern.match(i) != None:
            matches.append(i)

    return matches


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

    pattern = re.compile("^.*\.jpg$")
    pklot = []
    requirements = ['PKLotSegmented']

    if university:
        requirements.append(university)

    for root, dirs, files in os.walk(top=path,followlinks=True):

        rootpath = PurePath(root).parts
        file = search(pattern, files)

        if all(i in rootpath for i in requirements):
                if len(file) != 0:
                    for i in file:
                        pklot.append([rootpath[-4], \
                        rootpath[-3], \
                        rootpath[-2], \
                        rootpath[-1], \
                        os.path.abspath(os.path.join(root,i))])

    return pklot


def dataset_gen(datasetpath, percentagetrain, percentagetest, university):

    dataset = pathgen(datasetpath, university)
    days = []
    x = 0
    y = 0
    reduction = 0
    train = []
    test = []
    train_files = []
    test_files = []

    for i in dataset:
        if i[2] not in days:
                days.append(i[2])


    if percentagetrain < percentagetest:
        x = percentagetrain
        reduction = x
        x = int(math.floor(len(days)*reduction))
        y = int(math.ceil(x*(1-reduction))/reduction)

        for i in range(0,x):
                l = random.randrange(0,len(days))
                train.append(days[l])
                days.pop(l)
        test = days


    else:
        x = percentagetest
        reduction = x
        x = int(math.floor(len(days)*reduction))
        y = int(math.ceil(x*(1-reduction))/reduction)

        for i in range(0,x):
                l = random.randrange(0,len(days))
                test.append(days[l])
                days.pop(l)
        train = days


    for i in train:
            for j in dataset:
                    if j[2] == i:
                            train_files.append([j[-1],j[-2]])


    for i in test:
            for j in dataset:
                    if j[2] == i:
                            test_files.append([j[-1],j[-2]])

    return train_files, test_files

def convert_label(i):
    match i:
        case 'Empty':
            return int(0)
        case 'Occupied':
            return int(1)
        case _:
            sys.exit("Error!")

def shuffle_generator(data, seed):
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    for i in idx:
        label = str(data[i][1],encoding='utf-8')
        label = convert_label(label)
        img = data[i][0]
        img = tf.keras.utils.load_img(img)
        img = img.resize((32,32))
        img = tf.keras.utils.img_to_array(img)
        yield img, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default='./', required=False,\
                            type=str, help='Specifies the path in which the dataset\
                                            lives in')
    parser.add_argument('--train', '-tr', default=0.4, required=False, type=float)
    parser.add_argument('--test', '-tt', default=0.6, required=False, type=float)
    parser.add_argument('--university', '-uni', default=None, required=False, type=str)
    dataset = dataset_gen(parser.parse_args().path,parser.parse_args().train,parser.parse_args().test,parser.parse_args().university)


    train_dataset, test_dataset = dataset

    train_ds = tf.data.Dataset.from_generator(
                shuffle_generator,
                args=[train_dataset, 42],
                output_signature=(
                    tf.TensorSpec(shape=(32,32,3), dtype=tf.uint8),
                    tf.TensorSpec(shape=(), dtype=tf.uint8)))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,32,3)),
        tf.keras.layers.Conv2D(64, 16, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_ds.batch(32).prefetch(4),epochs=50)
