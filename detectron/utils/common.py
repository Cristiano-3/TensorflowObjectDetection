# coding: utf-8

import tensorflow as tf
import math
from configs import cfgs


def _bn(inputs, is_training):
    bn = tf.layers.batch_normalization(
        inputs=inputs,
        axis=3 if cfgs.data_format == 'channels_last' else 1,
        training=True#is_training
    )
    return bn

def conv_bn_actibation(inputs, filters, ksize, strides,
                        activation=tf.nn.relu, is_training=True):
    # conv
    conv = tf.layers.conv2d(inputs, filters, ksize, strides,
                            padding='same',
                            data_format=cfgs.data_format,
                            kernel_initializer=tf.variance_scaling_initializer()
                            )
    # bn
    bn = _bn(conv, is_training)
    # activation
    if activation is not None:
        bn = activation(bn)

    return bn

def bn_activation_conv(inputs, filters, ksize, strides,
                        activation=tf.nn.relu, pi_init=False, is_training=True):
    # bn
    bn = _bn(inputs, is_training)
    # activation
    if activation is not None:
        bn = activation(bn)
    # conv
    if not pi_init:
        conv = tf.layers.conv2d(bn, filters, ksize, strides,
                                padding='same',
                                data_format=cfgs.data_format,
                                kernel_initializer=tf.variance_scaling_initializer()
                                )
    else:
        conv = tf.layers.conv2d(bn, filters, ksize, strides,
                                padding='same',
                                data_format=cfgs.data_format,
                                kernel_initializer=tf.variance_scaling_initializer(),
                                bias_initializer=tf.constant_initializer(-math.log((1 - cfgs.pi) / cfgs.pi))
                                )
        # kernel initializer:
        # tf.truncated_normal_initializer(stddev=0.01)
        # tf.glorot_normal_initializer # xavier
    return conv

def max_pooling(bottom, pool_size, strides, name):
    return tf.layers.max_pooling2d(
        inputs=bottom,
        pool_size=pool_size,
        strides=strides,
        padding='same',
        data_format=cfgs.data_format,
        name=name
    )

def avg_pooling(bottom, pool_size, strides, name):
    return tf.layers.average_pooling2d(
        inputs=bottom,
        pool_size=pool_size,
        strides=strides,
        padding='same',
        data_format=cfgs.data_format,
        name=name
    )

def dropout(bottom, prob, is_training, name):
    return tf.layers.dropout(
        inputs=bottom,
        rate=prob,
        training=is_training,
        name=name
    )
