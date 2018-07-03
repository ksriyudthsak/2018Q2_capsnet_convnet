# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import random
from generate_parameters import *
import numpy
from tensorflow.contrib.learn.python.learn.datasets import base
from input_data_cifar import DataSet

def input_image(csv_name, num_batch_size, num_label):
    # load mushroom data
    fname_queue = tf.train.string_input_producer([csv_name])
    reader = tf.TextLineReader()
    key, val = reader.read(fname_queue)
    fname, label = tf.decode_csv(val, [["aa"], [1]])

    # decode and resize images
    jpeg_r = tf.read_file(fname)
    image = tf.image.decode_jpeg(jpeg_r, channels=colour_mode)
    image = tf.cast(image, tf.float32)
    resized_image = tf.image.resize_images(image, [set_size,set_size])
    label = tf.one_hot(indices=label, depth=num_label)
    # create tensorflow batch
    image_batch, label_batch = tf.train.batch([resized_image, label], batch_size=num_batch_size)
    return image_batch, label_batch


def image_augmentation(x_train, y_train, num_batch_size):
    train_datagen_augmented = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=.1,
            horizontal_flip=True,
            vertical_flip=True)
    train_datagen_augmented.fit(x_train)
    x_train = train_datagen_augmented.flow(x_train, y_train, batch_size=num_batch_size)
    return x_train, y_train


def image_augmentation_test(x_test, y_test, num_batch_size):
    train_datagen_augmented = ImageDataGenerator(
            rotation_range=10.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=.1,
            horizontal_flip=True,
            vertical_flip=True)
    train_datagen_augmented.fit(x_test)
    x_train = train_datagen_augmented.flow(x_test, y_test, batch_size=num_batch_size)
    return x_train, y_test


def augmentation_plot(X_batch, num_batch_size, iteration, filename):
    plt.figure()
    n_samples = num_batch_size
    for index in range(n_samples):
        plt.subplot(2, n_samples/2, index + 1)
        plt.title(filename+"_:" + str(iteration))
        plt.imshow(X_batch[index]/255, interpolation='nearest')
        plt.axis("off")
    plt.savefig("output/"+filename+"_"+str(iteration) + ".jpg")

