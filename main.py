import PIL
#from kera import layers
#from time import gtime, strftime
#import csv
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pdb
from pathlib import PurePath
import argparse
import re
import os
import math
import random

def pathgen(path: str, university: str)->[[]]:
        pattern = re.compile("^.*\.jpg$")
        if university != None:
                requirements = ['PKLotSegmented', university]
        else:
                requirements = 'PKLotSegmented'
        pklot = []
        def search(pattern: type[re.Pattern], files: []):
                matches = []
                for i in files:
                        if pattern.match(i) != None:
                                matches.append(i)               
                return matches
        for root, dirs, files in os.walk(top=path,followlinks=True):
                rootpath = PurePath(root).parts
                file = search(pattern, files)
                if university != None:  
                        if requirements[0] in rootpath and requirements[1] in rootpath:
                                #print(f'Root: {root}\n', \
                                #       f'Dirs: {dirs}\n', \
                                #       F'Files: {file}\n')
                                if len(file) != 0:
                                        for i in file:
                                                pklot.append([rootpath[-4], \
                                                                rootpath[-3], \
                                                                rootpath[-2], \
                                                                rootpath[-1], \
                                                                os.path.abspath(os.path.join(root,i))])
                else:
                        if requirements in rootpath:
                                #print(f'Root: {root}\n', \
                                #       f'Dirs: {dirs}\n', \
                                #       F'Files: {file}\n')
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
        class ds_gen(object):
                def __init__(self, files):
                        self.files = files
                        self.num = 0
                def __iter__(self):
                        return self
                def __next__(self):
                        return self.next()
                def next(self):
                        if self.num < len(self.files):
                                cur, self.num = self.num, self.num+1
                                image = tf.keras.utils.load_img(path=self.files[cur][0])
                                image = image.resize((32,32))
                                image = tf.keras.utils.img_to_array(image)
                                match self.files[cur][1]:
                                        case 'Empty':
                                                label = 0
                                        case 'Occupied':
                                                label = 1
                                        case _:
                                                sys.exit("Corrupt dataset label")
                                #print(image, label)
                                yield image, label
                        else:
                                raise StopIteration()

        return ds_gen(train_files), ds_gen(test_files)

#def model_run(dataset):
        
                                

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--path', '-p', default='./', required=False,\
                                type=str, help='Specifies the path in which the dataset\
                                                lives in')
        parser.add_argument('--train', '-tr', default=0.4, required=False, type=float)
        parser.add_argument('--test', '-tt', default=0.6, required=False, type=float)
        parser.add_argument('--university', '-uni', default=None, required=False, type=str)
        dataset = dataset_gen(parser.parse_args().path,parser.parse_args().train,parser.parse_args().test,parser.parse_args().university)
        tr_ds, _ = dataset
        oficial_ds = tf.data.Dataset.from_generator(tr_ds.next, output_types=(tf.float32, tf.int32))
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
        model.fit(oficial_ds,epochs=50)
        #model_run(dataset)
