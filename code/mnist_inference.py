
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.stats import truncnorm

import numpy as np
import math

import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def init_weights(size):
    # we truncate the normal distribution at two times the standard deviation (which is 2)
    # to account for a smaller variance (but the same mean), we multiply the resulting matrix with he desired std
    return np.float32(truncnorm.rvs(-2, 2, size=size)*1.0/math.sqrt(float(size[0])))


def inference(images, Hidden1, Hidden2):

  with tf.name_scope('hidden1'):

    weights = tf.Variable(init_weights([IMAGE_PIXELS, Hidden1]), name='weights', dtype=tf.float32)
    biases = tf.Variable(np.zeros([Hidden1]),name='biases',dtype=tf.float32)
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

  with tf.name_scope('hidden2'):

    weights = tf.Variable(init_weights([Hidden1, Hidden2]),name='weights',dtype=tf.float32)
    biases = tf.Variable(np.zeros([Hidden2]),name='biases',dtype=tf.float32)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  with tf.name_scope('out'):

    weights = tf.Variable(init_weights([Hidden2, NUM_CLASSES]), name='weights',dtype=tf.float32)
    biases = tf.Variable(np.zeros([NUM_CLASSES]), name='biases',dtype=tf.float32)
    logits = tf.matmul(hidden2, weights) + biases

  return logits

def inference_no_bias(images, Hidden1, Hidden2):

  with tf.name_scope('hidden1'):

    weights = tf.Variable(init_weights([IMAGE_PIXELS, Hidden1]), name='weights', dtype=tf.float32)
    hidden1 = tf.nn.relu(tf.matmul(images, weights))

  with tf.name_scope('hidden2'):

    weights = tf.Variable(init_weights([Hidden1, Hidden2]),name='weights',dtype=tf.float32)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights))

  with tf.name_scope('out'):

    weights = tf.Variable(init_weights([Hidden2, NUM_CLASSES]), name='weights',dtype=tf.float32)
    logits = tf.matmul(hidden2, weights)

  return logits


def loss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):

  # Add a scalar summary for the snapshot loss.
  # 将计算图中的标量数据写入tf的日志文件，以便为将来的tf可视化做准备
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):

  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def placeholder_inputs(batch_size):
    
    images_placeholder = tf.placeholder(tf.float32, shape=(None,IMAGE_PIXELS), name='images_placeholder')
    labels_placeholder = tf.placeholder(tf.int32, shape=(None), name='labels_placeholder')
    return images_placeholder, labels_placeholder


def mnist_cnn_model(batch_size):

    # - placeholder for the input Data (in our case MNIST), depends on the batch size specified in C
    data_placeholder, labels_placeholder = placeholder_inputs(batch_size)

    # Input Layer
    input_layer = tf.reshape(data_placeholder, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels_placeholder, logits=logits)

    eval_correct = evaluation(logits, labels_placeholder)

    global_step = tf.Variable(0, dtype=tf.float32, trainable=False, name='global_step')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=global_step)

    return train_op, eval_correct, loss, data_placeholder, labels_placeholder


def mnist_fully_connected_model(batch_size, hidden1, hidden2):
    # - placeholder for the input Data (in our case MNIST), depends on the batch size specified in C
    data_placeholder, labels_placeholder = placeholder_inputs(batch_size)

    # - logits : output of the fully connected neural network when fed with images. The NN's architecture is
    #           specified in '
    logits = inference_no_bias(data_placeholder, hidden1, hidden2)

    # - loss : when comparing logits to the true labels.
    Loss = loss(logits, labels_placeholder)

    # - eval_correct: When run, returns the amount of labels that were predicted correctly.
    eval_correct = evaluation(logits, labels_placeholder)


    # - global_step :          A Variable, which tracks the amount of steps taken by the clients:
    global_step = tf.Variable(0, dtype=tf.float32, trainable=False, name='global_step')

    # - learning_rate : A tensorflow learning rate, dependent on the global_step variable.
    learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step,
                                                                           decay_steps=27000, decay_rate=0.1,
                                                                           staircase=False, name='learning_rate')


    train_op = training(Loss, learning_rate)

    return train_op, eval_correct, Loss, data_placeholder, labels_placeholder