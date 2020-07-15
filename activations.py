import tensorflow as tf


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)