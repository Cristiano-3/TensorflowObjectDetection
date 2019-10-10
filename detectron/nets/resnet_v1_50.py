# coding: utf-8

import tensorflow as tf
import math
from detectron.utils.common import *
from configs import cfgs


class ResNet():
    def __init__(self, inputs, is_training=True, scope='resnet50'):
        # 传入初始化参数
        self.inputs = inputs
        self.is_training = is_training
        self.scope = scope

        endpoints = []
        # build graph and get endpoints
        with tf.variable_scope(scope):
            conv1 = self._conv_bn_actibation(self.inputs, 16, 7, 2)
            pool1 = self._max_pooling()

        self.endpoints = endpoints

    def _bn(self, inputs):
        bn = tf.layers.batch_normalization(
            inputs=inputs,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _conv_bn_actibation(self, inputs, filters, ksize, strides,
                            activation=tf.nn.relu):
        conv = tf.layers.conv2d(inputs, filters, ksize, strides,
                                padding='same',
                                data_format=self.data_format,
                                kernel_initializer=tf.variance_scaling_initializer()
                               )
        bn = self._bn(conv)
        if activation is not None:
            return activation(bn)
        else:
            return bn

    def _bn_activation_conv(self, inputs, filters, ksize, strides,
                            activation=tf.nn.relu, pi_init=False):
        # bn
        bn = self._bn(inputs)
        # activation
        if activation is not None:
            bn = activation(bn)
        # conv
        if not pi_init:
            conv = tf.layers.conv2d(bn, filters, ksize, strides,
                                    padding='same',
                                    data_format=self.data_format,
                                    kernel_initializer=tf.variance_scaling_initializer()
                                    )
        else:
            conv = tf.layers.conv2d(bn, filters, ksize, strides,
                                    padding='same',
                                    data_format=self.data_format,
                                    kernel_initializer=tf.variance_scaling_initializer(),
                                    bias_initializer=tf.constant_initializer(-math.log((1 - self.pi) / self.pi))
                                    )
            # kernel initializer:
            # tf.truncated_normal_initializer(stddev=0.01)
            # tf.glorot_normal_initializer # xavier
        return conv

    def _residual_block(self, inputs, filters, strides, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('residual'):
                conv = self._bn_activation_conv(inputs, filters, 3, strides)  # maybe down sample
                conv = self._bn_activation_conv(conv, filters, 3, 1)  # no down sample

            with tf.variable_scope('identity'):
                if strides != 1:
                    shortcut = self._bn_activation_conv(inputs, filters, 3, strides)
                else:
                    shortcut = inputs  # self._bn_activation_conv(inputs, filters, 1, 1)

        return conv + shortcut

    def _residual_bottleneck(self, inputs, filters, strides, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('residual'):
                conv = self._bn_activation_conv(inputs, filters, 1, 1)      # ?strides
                conv = self._bn_activation_conv(conv, filters, 3, strides)  # ?1
                conv = self._bn_activation_conv(conv, 4*filters, 1, 1)

            with tf.variable_scope('identity'):
                shortcut = self._bn_activation_conv(inputs, 4*filters, 3, strides)  # ? 1, 1

            return conv + shortcut
