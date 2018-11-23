import pickle as pkl
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.loat32, (None, real_dim), name = 'input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name = 'input_z')

    return inputs_real, inputs_z

def lrelu(x, a):
    return tf.maximum(x*a, x)

def tanh(x):
    return tf.tanh(x)

def generator(z, out_dim, n_units = 128, reuse = False, al = 0.01):
    with tf.varaible_scope('generator', reuse = reuse)
        h1 = tf.layers.dense(z, n_units, activation = None)

        #leaky relu
        h1 = lrelu(h1, al)

        logits = tf.layers.dense(h1, out_dim, activation = None)
        out = tanh(h1)

        return out


