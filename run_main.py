from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from input_data_cifar import _input_data
from input_data_cifar import *
from input_data_mushroom import input_image, image_augmentation, image_augmentation_test, augmentation_plot
from generate_parameters import *


import tensorflow as tf
from process_cnn import cnn_module
from process_capsnets import capsnet_module
import argparse


# 入力データパス
DATASET_DIR = os.getcwd()+"/dataset/cifar-10-batches-py/"
# バリデーションサイズ
VALIDATION_SIZE = 7000



def main(args):
    foutput = open("output/"+str(args.sample)+"_"+str(args.network)+".txt", 'w')
    # call model
    if args.sample == "cifar10":
        num_label = num_cifar10_label
    elif args.sample == "mushroom":
        num_label = num_mushroom_label
    print ("-----network",args.network)
    if args.network == "cnn":
        X, y, keep_prob, train_step, accuracy, loss = cnn_module(num_label)
    elif args.network == "capsnet":
        X, y, loss, train_step, accuracy, mask_with_labels = capsnet_module(args.sample, num_label)

    # ## Initializer and Saver
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    checkpoint_path = "./my_network"

    # get data
    print("-----sample", args.sample)
    # if args.sample == "cifar10":
    input_data = _input_data(DATASET_DIR, VALIDATION_SIZE)
    best_train_accuracy = 0.0

    with tf.Session() as sess:
        if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
            saver.restore(sess, checkpoint_path)
        else:
            init.run()

        # initialisation
        init = tf.global_variables_initializer()
        sess.run(init)
        print ("done initialisation")
        train_image_batch, train_label_batch = input_image('dataset/mushroom/picture_name_train.csv',
                                                           num_batch_size, num_label)

        tf.train.start_queue_runners()
        image_batch = train_image_batch.eval()
        label_batch = train_label_batch.eval()

        # training (for creating model)
        for i in range(n_epochs):
            # read training data for num_batch_size per one training epoch
            # if no more data left, the data will be shuffled and be read from beginning again
            if args.sample == "cifar10":
                batch_x, batch_y = input_data.train.next_batch(num_batch_size, array_2d=True)
            elif args.sample == "mushroom":

                # # augment image for every iteration
                # image_batch = image_batch / 255
                # image_batch1, label_batch = image_augmentation(image_batch, label_batch, num_batch_size)
                # if random.uniform(0, 1) < 0.99:
                #     image_batch = next(image_batch1)
                #     image_batch = image_batch[0]
                # else:
                #     image_batch = image_batch
                # image_batch = image_batch * 255

                # display augmentation image
                if (n_epochs == 1):
                    augmentation_plot(image_batch, num_batch_size, 0, "augment_train")

                # generate dataset
                generated_dataset = create_dataset(image_batch, label_batch)
                batch_x, batch_y = generated_dataset.next_batch(num_batch_size, array_2d=True)


            # training model
            if args.network == "cnn":
                # if i % 100 == 0:
                    # check every 100 loop
                train_accuracy = accuracy.eval(feed_dict={X: batch_x.reshape([-1, 32, 32, 3]),
                                                          y: batch_y,
                                                          keep_prob: 1.0})
                print("step %d, training accuracy %f" % (i, train_accuracy))
                foutput.write("\rtrain_accuracy\t{}\t{:.4}%".format(i, train_accuracy*100))

                train_step.run(feed_dict={X: batch_x.reshape([-1, 32, 32, 3]),
                                          y: batch_y, keep_prob: 0.5})
            elif args.network == "capsnet":
                # Run the training operation and measure the loss:
                # if i % 100 == 0:
                    # check every 100 loop
                train_accuracy = accuracy.eval(feed_dict={X: batch_x.reshape([-1, 32, 32, 3]),
                                                          y: batch_y,
                                                          mask_with_labels: True})
                print("step %d, training accuracy %f" % (i, train_accuracy))
                foutput.write("\rtrain_accuracy\t{}\t{:.4}%".format(i, train_accuracy*100))

                _, loss_train = sess.run(
                    [train_step, loss],
                    feed_dict={X: batch_x.reshape([-1, 32, 32, 3]),
                               y: batch_y,
                               mask_with_labels: True})

            # save the model if it improved:
            if train_accuracy < best_train_accuracy:
            # if loss_val < best_loss_val:
            #     save_path = saver.save(sess, checkpoint_path)
                best_train_accuracy = train_accuracy

        # testing (for evaluating model)
        for i in range(100):
            if args.sample == "cifar10":
                test_x, test_y = input_data.test.next_batch(num_batch_size, array_2d=True)
            elif args.sample == "mushroom":
                test_image_batch, test_label_batch = input_image('dataset/mushroom/picture_name_test.csv', num_batch_size, num_label)
                print ("test")

                # # restore
                # saver.restore(sess, checkpoint_path)
                # tf.train.start_queue_runners()  # restore sess so do not need to pass sess

                # # initialisation
                init = tf.global_variables_initializer()
                sess.run(init)
                tf.train.start_queue_runners()

                X_test = test_image_batch.eval()
                y_test = test_label_batch.eval()

                # augment image for every iteration
                X_test = X_test / 255
                X_test1, y_batch = image_augmentation_test(X_test, y_test, num_batch_size)
                if random.uniform(0, 1) < 0.50:
                    X_test = next(X_test1)
                    X_test = X_test[0]
                else:
                    X_test = X_test
                    X_test = X_test * 255

                if (n_epochs == 1):
                    augmentation_plot(X_test, num_test_set_mushroom, 0, "augment_test_mushroom")
                    augmentation_plot(X_test, num_test_set_mushroom, 0, "augment_test_mushroom")

                # generate dataset
                test_dateset = create_dataset(X_test, y_test)
                test_x, test_y = test_dateset.next_batch(num_batch_size, array_2d=True)

            if args.network == "cnn":
                test_accuracy = accuracy.eval(feed_dict={X: test_x.reshape([-1, 32, 32, 3]),
                                                         y: test_y, keep_prob: 1.0})
            else:
                test_accuracy = accuracy.eval(feed_dict={X: test_x.reshape([-1, 32, 32, 3]),
                                                         y: test_y, mask_with_labels: True})
            print("test accuracy %f" % test_accuracy)
            foutput.write("\rtest accuracy\t{:.4}f%".format(test_accuracy*100))

    foutput.close()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="process image")
    parser.add_argument('--sample', help='select sample - i.e., "cifar10" or "mushroom" ', type=str)
    parser.add_argument('--network', help='select network - i.e., "cnn" or "capsnet" ', type=str)

    args = parser.parse_args()
    print ("show_argument", args)
    main(args)