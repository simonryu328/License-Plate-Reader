import math
import numpy as np
import re

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image
from time import sleep
from sklearn.metrics import confusion_matrix
import sys
import cv2
import random

import os
import glob

from keras import layers
from keras import models
from keras import optimizers

from keras.utils import plot_model
from keras import backend

def files_in_folder(folder_path):
    files = glob.glob(folder_path)
    return files

def char_to_int(char):
    ascii_int_offset = 48
    ascii_alpha_offset = 65
    onehot_int_offset = 26

    numeric = ord(char)
    if numeric < ascii_alpha_offset:
        return int(numeric - ascii_int_offset + onehot_int_offset)
    return int(numeric - ascii_alpha_offset)

def int_to_char(numeric):
    ascii_int_offset = 48
    ascii_alpha_offset = 65
    onehot_int_offset = 26

    if numeric < onehot_int_offset:
        return chr(numeric + ascii_alpha_offset)
    return chr(numeric - onehot_int_offset + ascii_int_offset)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def train_plate_detector_cnn():
    T_path = "./training_data/*.png"
    V_path = "./validation_data/*.png"

    # collect list of file names in training and validation datasets
    T_files= files_in_folder(T_path)
    V_files = files_in_folder(V_path)

    # shuffle datasets
    random.shuffle(T_files)
    random.shuffle(V_files)

    # read in datasets, and convert alpha to numeric
    T_dataset_orig = np.array(
                              np.array([[np.array(cv2.cvtColor(
                              cv2.imread(file), cv2.COLOR_BGR2RGB)),char_to_int(file.split("/")[-1].split("_")[0])]
                              for file in T_files]))

    V_dataset_orig = np.array(
                              np.array([[np.array(cv2.cvtColor(
                              cv2.imread(file), cv2.COLOR_BGR2RGB)),char_to_int(file.split("/")[-1].split("_")[0])]
                              for file in V_files]))

    cv2.imshow("first training",T_dataset_orig[0,0])
    cv2.imshow("first validation",V_dataset_orig[0,0])

    print("first label T", T_dataset_orig[0,1])
    print("first label V", V_dataset_orig[0,1])
    cv2.waitKey(0)

    print("T_dataset_orig size: {}".format(len(T_dataset_orig)))
    print("V_dataset_orig size: {}".format(len(V_dataset_orig)))

    # Split X and Y datasets
    XT_dataset_orig = np.array([data[0] for data in T_dataset_orig])
    XV_dataset_orig = np.array([data[0] for data in V_dataset_orig])
    # It is required that Y is a dimensional array for onehot encoding
    YT_dataset_orig = np.array([[data[1]] for data in T_dataset_orig]).T
    YV_dataset_orig = np.array([[data[1]] for data in T_dataset_orig]).T

    # One-hot Encoding
    NUMBER_OF_LABELS = 36
    CONFIDENCE_THRESHOLD = 0.01

    # Normalize X (images) dataset
    XT_dataset = XT_dataset_orig/255.
    XV_dataset = XV_dataset_orig/255.

    # Convert Y dataset to one-hot encoding
    YT_dataset = convert_to_one_hot(YT_dataset_orig, NUMBER_OF_LABELS).T
    YV_dataset = convert_to_one_hot(YT_dataset_orig, NUMBER_OF_LABELS).T

    print("XT shape: " + str(XT_dataset.shape))
    print("YT shape: " + str(YT_dataset.shape))
    print("XV shape: " + str(XV_dataset.shape))
    print("YV shape: " + str(YV_dataset.shape))

    # train CNN
    conv_model = models.Sequential()
    conv_model.add(layers.Conv2D(8, (5,5), activation='relu',
                             input_shape=(27, 37, 1)))
    conv_model.add(layers.MaxPooling2D((2,2)))
    conv_model.add(layers.Conv2D(16, (5,5), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2,2)))
    conv_model.add(layers.Conv2D(32, (5,5), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2,2)))
    conv_model.add(layers.Flatten())

    # how should I decide the size of these fully connected layer?
    conv_model.add(layers.Dense(300, activation='relu'))
    conv_model.add(layers.Dense(200, activation='relu'))
    conv_model.add(layers.Dense(NUMBER_OF_LABELS, activation='softmax'))

    conv_model.summary()
    LEARNING_RATE = 1e-4
    conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics = ['categorical_accuracy'])

    # history_conv = conv_model.fit(XT_dataset, YT_dataset,
    #                           epochs=4,
    #                           batch_size=16,
    #                           validation_data=(XV_dataset, YV_dataset))
if __name__ == '__main__':
    train_plate_detector_cnn()
