# coding: utf-8

import tensorflow as tf
import math


def batch_normalize(bottom, data_format, is_training):
    return tf.layers.batch_normalization(
        inputs=bottom,
        axis=3 if data_format == 'channel_last' else 1,
        training=is_training
    )


def conv_bn_activation(bottom, filters, kernel_size,
                       strides, data_format, activation=tf.nn.relu, is_training=False):
    conv = tf.layers.conv2d(
        inputs=bottom,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        data_format=data_format,
        kernel_initializer=tf.variance_scaling_initializer()
    )
    bn = batch_normalize(conv, data_format, is_training)
    if activation is not None:
        bn = activation(bn)

    return bn


def bn_activation_conv(bottom, filters, kernel_size,
                       strides, data_format, activation=tf.nn.relu, pi=None, is_training=False):
    bn = batch_normalize(bottom, data_format, is_training)
    if activation is not None:
        bn = activation(bn)

    if pi is None:
        conv = tf.layers.conv2d(
            inputs=bn,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=data_format,
            kernel_initializer=tf.variance_scaling_initializer()
        )
    else:
        conv = tf.layers.conv2d(
            inputs=bn,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=data_format,
            kernel_initializer=tf.variance_scaling_initializer(),
            bias_initializer=tf.constant_initializer(-math.log((1 - pi) / pi))
        )

    return conv


def max_pooling(bottom, pool_size, strides, data_format, name):
    return tf.layers.max_pooling2d(
        inputs=bottom,
        pool_size=pool_size,
        strides=strides,
        padding='same',
        data_format=data_format,
        name=name
    )


def avg_pooling(bottom, pool_size, strides, data_format, name):
    return tf.layers.average_pooling2d(
        inputs=bottom,
        pool_size=pool_size,
        strides=strides,
        padding='same',
        data_format=data_format,
        name=name
    )


def dropout(bottom, prob, is_training, name):
    return tf.layers.dropout(
        inputs=bottom,
        rate=prob,
        training=is_training,
        name=name
    )

