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
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

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


def custom_preprocessing(img):
    # "zoom out" then "zoom in"
    final_dim = np.array([img.shape[1], img.shape[0]])
    m = random.uniform(0,0.5)
    intermediate_dim = np.rint(final_dim*(0.9 + m)).astype(int)

    # zoom out
    img = cv2.resize(img, tuple(intermediate_dim), interpolation =cv2.INTER_AREA)
    # zoom back in
    img = cv2.resize(img, tuple(final_dim), interpolation =cv2.INTER_AREA)
    # Gaussian blur
    img = cv2.GaussianBlur(img,(5,5),0.7)
    # restore binary
    threshold = 0.6
    _, img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    # restore dimensions necessary for keras
    img = np.expand_dims(img,axis=2)
    # print("final img shape: {}".format(img.shape))
    # cv2.imshow("final",img)
    # cv2.waitKey(0)
    return img

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
                              np.array([[np.array(
                              cv2.imread(file, cv2.IMREAD_GRAYSCALE)),char_to_int(file.split("/")[-1].
                                  split("_")[0])]
                              for file in T_files]))

    V_dataset_orig = np.array(
                              np.array([[np.array(
                              cv2.imread(file, cv2.IMREAD_GRAYSCALE)),char_to_int(file.split("/")[-1].
                                  split("_")[0])]
                              for file in V_files]))


    # Split X and Y datasets
    XT_dataset_orig = np.array([data[0] for data in T_dataset_orig])
    XV_dataset_orig = np.array([data[0] for data in V_dataset_orig])
    # It is required that Y is a dimensional array for onehot encoding
    YT_dataset_orig = np.array([[data[1]] for data in T_dataset_orig]).T
    YV_dataset_orig = np.array([[data[1]] for data in V_dataset_orig]).T

    # One-hot Encoding
    NUMBER_OF_LABELS = 36
    CONFIDENCE_THRESHOLD = 0.01

    # Normalize X (images) dataset
    XT_dataset = XT_dataset_orig/255.
    XV_dataset = XV_dataset_orig/255.
    # Reshape the dataset because grayscale and keras 3 dimensional input
    # XT_dataset = XT_dataset.reshape(XT_dataset.shape[0], XT_dataset.shape[1],
    #                                 XT_dataset.shape[2], 1)
    # XV_dataset = XV_dataset.reshape(XV_dataset.shape[0], XV_dataset.shape[1],
    #                                 XV_dataset.shape[2], 1)
    XT_dataset = np.expand_dims(XT_dataset,axis=3)
    XV_dataset = np.expand_dims(XV_dataset,axis=3)
    print("example 4th dim: {}".format(XT_dataset[1,0:2,:,:]))
    # Convert Y dataset to one-hot encoding
    YT_dataset = convert_to_one_hot(YT_dataset_orig, NUMBER_OF_LABELS).T
    YV_dataset = convert_to_one_hot(YV_dataset_orig, NUMBER_OF_LABELS).T

    print("XT shape: " + str(XT_dataset.shape))
    print("YT shape: " + str(YT_dataset.shape))
    print("XV shape: " + str(XV_dataset.shape))
    print("YV shape: " + str(YV_dataset.shape))

    # augment data
    datagen = ImageDataGenerator(
                                 preprocessing_function=custom_preprocessing,
                                 rotation_range=2,
                                 width_shift_range=0.03,
                                 height_shift_range=0.03,
                                 zoom_range=[0.99,1.01],
                                 shear_range=2
                                 )
    # example purely for viewing
    it = datagen.flow(XT_dataset, YT_dataset, batch_size=1)
    # generate samples and plot
    fig = plt.figure(figsize=(20,20))
    for i in range(9):
        # generate batch of images
        batch = it.next()
        print("batch shape 0: {} \n batch shape 1: {}".format(batch[0].shape,
            batch[1].shape))
        ax = fig.add_subplot(330+1+i)
        ax.title.set_text(int_to_char(np.where(batch[1].ravel()==1)[0]))
        # plot raw pixel data

        ax.imshow(batch[0][0].squeeze(axis=2), cmap='gray', vmin=0, vmax=1)
    # show the figure
    plt.show()

    # real iterator
    it = datagen.flow(XT_dataset, YT_dataset, batch_size=30)

    # train CNN
    conv_model = models.Sequential()
    conv_model.add(layers.Conv2D(2, (5,5), activation='relu',
                             input_shape=(27, 37, 1)))
    conv_model.add(layers.MaxPooling2D((2,2)))
    conv_model.add(layers.Conv2D(4, (5,5), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2,2)))
    conv_model.add(layers.Flatten())

    # how should I decide the size of these fully connected layer?
    conv_model.add(layers.Dense(500, activation='relu'))
    conv_model.add(layers.Dense(200, activation='relu'))
    conv_model.add(layers.Dense(NUMBER_OF_LABELS, activation='softmax'))

    conv_model.summary()
    LEARNING_RATE = 1e-4
    conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics = ['categorical_accuracy'])

    history_conv = conv_model.fit(it,
                              epochs=20,
                              validation_data=(XV_dataset, YV_dataset))

    plt.plot(history_conv.history['loss'])
    plt.plot(history_conv.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper left')
    plt.show()

    plt.plot(history_conv.history['categorical_accuracy'])
    plt.plot(history_conv.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
    plt.show()

    # display an example image
    index = 100
    img = XV_dataset[index]
    img_aug = np.expand_dims(img, axis=0)
    y_predict = conv_model.predict(img_aug)
    print("img shape {}".format(img.shape))
    plt.imshow(img.squeeze(axis=2), cmap='gray', vmin=0, vmax=1)
    caption = ("One hot \n GND truth: {}".format(YV_dataset[index]) +
                " Predicted: {}".format(y_predict))
    plt.text(0.6, 0.6, caption,
           color='orange', fontsize = 10,
           horizontalalignment='left', verticalalignment='bottom')
    plt.show()
if __name__ == '__main__':
    train_plate_detector_cnn()
