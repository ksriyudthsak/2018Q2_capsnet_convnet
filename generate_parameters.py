import numpy as np
import tensorflow as tf

# Set the random seeds so that the calculation always produces the same output
np.random.seed(42)
tf.set_random_seed(42)

# set up parameters
restore_checkpoint = False  # True
n_epochs = 1000  # 10000
num_test_set_cifar10 = 40
num_test_set_mushroom = 20

num_train_samples = 1200
num_validate_samples = 600
num_test_samples = 600

num_batch_size = 4
num_cifar10_label = 10
num_mushroom_label = 4

# set image size
set_size = 32
set_kernel_size = 11
colour_mode = 3  # channels=1 for grayscale, channels=3 for RGB
