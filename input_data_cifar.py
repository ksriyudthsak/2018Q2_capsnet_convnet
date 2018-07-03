# -*- coding: UTF-8 -*-
# modified from https://qiita.com/fujin/items/960b6854700d1bd50043


import pickle

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import numpy as np
import numpy
import tensorflow as tf


# DateSet
class DataSet(object):

    def __init__(self, images, labels, dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

        # データ件数セット
        self._num_examples = images.shape[0]

        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def images_2d_array(self):
        images = np.reshape(self._images, [-1, self._images.shape[1] * self._images.shape[2] * self._images.shape[3]])
        return images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    #
    # batch_size のデータを取得
    #   最後まで到達した場合、データをシャップルして最初からデータを取り出す
    #
    def next_batch(self, batch_size, array_2d=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        images = self._images[start:end]

        # データを 2D へ整形
        if array_2d:
            images = np.reshape(images,
                [-1, self.images.shape[1] * self.images.shape[2] * self.images.shape[3]])

        return images, self._labels[start:end]


# One-hot 変換
def dense_to_one_hot(labels_dense: numpy.ndarray, num_classes: int) -> numpy.ndarray:
    """
    Scalars から One-hot へコンバートする
    :param labels_dense: ラベル配列
    :param num_classes:  分類クラス数
    :return: ラベル配列（One-hot）
    """
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# get input data
def get_input_data(dir: str, filename: str, rows: int, cols: int, channel: int, class_numm: int) \
        -> (numpy.ndarray, numpy.ndarray):

    # データ読み込み（CIFAR-10）
    filepath = dir + filename
    dict = unpickle(filepath)

    # データ取得
    data = dict[b'data']
    labels = np.array(dict[b'labels'])

    # サイズ取得
    if labels.shape[0] != data.shape[0]:
        raise Exception('Error: Different length')
    num_images = labels.shape[0]

    # 入力データ整形（４次元配列）
    data = data.reshape(num_images, channel, rows, cols) # 2017.7.7 修正
    # [n, 3, 32, 32] -> [n, 32, 32, 3] へ変換
    data = data.transpose([0, 2, 3, 1]) # 2017.7.7 追加

    # ラベルデータ整形（One-Hot）
    labels = dense_to_one_hot(labels, class_numm)

    return data, labels


# load pickle
def unpickle(file: str) -> dict:
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="bytes")
    return dict


# create dataset - single set
def create_dataset(images: numpy.ndarray, labels: numpy.ndarray) -> DataSet:
    dataset = DataSet(images, labels, dtype=dtypes.float32)
    return dataset


# create dataset - combine train validation test
def create_datasets(train: DataSet, validation: DataSet, test: DataSet) -> base.Datasets:
    return base.Datasets(train=train, validation=validation, test=test)


# input data
def _input_data(DATASET_DIR, VALIDATION_SIZE) -> base.Datasets:
    image_size = 32
    images1, labels1 = get_input_data(DATASET_DIR, "data_batch_1", image_size, image_size, 3, 10)
    images2, labels2 = get_input_data(DATASET_DIR, "data_batch_2", image_size, image_size, 3, 10)
    images3, labels3 = get_input_data(DATASET_DIR, "data_batch_3", image_size, image_size, 3, 10)
    images4, labels4 = get_input_data(DATASET_DIR, "data_batch_4", image_size, image_size, 3, 10)
    images5, labels5 = get_input_data(DATASET_DIR, "data_batch_5", image_size, image_size, 3, 10)
    test_images, test_labels = get_input_data(DATASET_DIR, "test_batch", image_size, image_size, 3, 10)

    # combine data from various files
    images = np.concatenate((images1, images2, images3, images4, images5), axis=0)
    labels = np.concatenate((labels1, labels2, labels3, labels4, labels5), axis=0)

    # seperate training data
    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]
    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]

    # generate dataset
    train_dataset = create_dataset(train_images, train_labels)
    validation_dataset = create_dataset(validation_images, validation_labels)
    test_dataset = create_dataset(test_images, test_labels)

    inputdatas = create_datasets(train_dataset, validation_dataset, test_dataset)
    return inputdatas
