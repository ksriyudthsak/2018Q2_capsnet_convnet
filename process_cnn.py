# -*- coding: utf-8 -*-

# modified from https://qiita.com/fujin/items/960b6854700d1bd50043

# To support both Python 2 and Python 3:
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import random
from generate_parameters import *


# 重み変数
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# バイアス変数
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 畳み込み
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# pooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def cnn_module(num_label):
    # # create placeholders for images (X) and labels (y)
    X = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name="X")
    y = tf.placeholder(tf.float32, [None, num_label])

    # Convolution layer #1
    W_conv1 = weight_variable([5, 5, 3, 32])
    print ("show_W_conv1",W_conv1)
    b_conv1 = bias_variable([32])
    print ("show_b_conv1",b_conv1)
    x_image = tf.reshape(X, [-1, 32, 32, 3])
    print ("show_x_image",x_image)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    print ("show_h_conv1",h_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    print ("show_h_pool1",h_pool1)

    # Convolution layer #2
    W_conv2 = weight_variable([5, 5, 32, 64])
    print ("show_W_conv2",W_conv2)
    b_conv2 = bias_variable([64])
    print("show_b_conv2", b_conv2)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    print("show_h_conv2", h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print("show_h_pool2", h_pool2)

    # Full connected layer
    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    print("show_W_fc1", W_fc1)
    b_fc1 = bias_variable([1024])
    print("show_b_fc1", b_fc1)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    print("show_h_pool2_flat", h_pool2_flat)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    print("show_h_fc1", h_fc1)

    # Dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print ("show_keep_prob", keep_prob)
    print("show_h_fc1_drop", h_fc1_drop)

    # output layer:  linear layer(WX + b)
    W_fc2 = weight_variable([1024, num_label])
    print("show_W_fc2", W_fc2)
    biases = bias_variable([num_label])
    print("show_biases", biases)
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + biases

    print ("show_y", y)
    print ("show_y_conv", y_conv)

    # Loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))

    # Training Operations
    training_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return X, y, keep_prob, training_op, accuracy, cross_entropy
