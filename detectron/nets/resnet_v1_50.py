# coding: utf-8

import tensorflow as tf
import math
from detectron.utils import common
from configs import cfgs


class ResNet():
    def __init__(self, inputs, is_training=True, scope='resnet50'):
        # 传入初始化参数
        self.inputs = inputs
        self.is_training = is_training
        self.scope = scope
        
        self.data_format = cfgs.data_format
        self.is_bottleneck = cfgs.is_bottleneck

        endpoints = []
        # build graph and get endpoints
        with tf.variable_scope(self.scope):
            # 224x224x3 -> 112x112x64
            with tf.variable_scope('conv1'):
                conv1 = common.conv_bn_actibation(self.inputs, 64, 7, 2, is_training=self.is_training)
                # conv1 = tf.layers.conv2d(self.inputs, 64, 7, 2, 'conv1')

            # 112x112x64 -> 56x56x64
            pool1 = common.max_pooling(conv1, 3, 2, name='pool1')

            # select residual unit
            if self.is_bottleneck:
                residual_unit_fn = self._residual_bottleneck
            else:
                residual_unit_fn = self._residual_block

            with tf.variable_scope('block1'):
                # block1
                # 56x56x64
                residual_block = pool1
                for i in range(3):
                    residual_block = residual_unit_fn(residual_block, 64, 1, 'block1_unit'+str(i+1))

                endpoints.append(residual_block)

            with tf.variable_scope('block2'):
                # block2
                for i in range(4):
                    if i == 0:
                        # downsample by strides=2, at first unit
                        residual_block = residual_unit_fn(residual_block, 128, 2, 'block2_unit1')
                    else:
                        residual_block = residual_unit_fn(residual_block, 128, 1, 'block2_unit'+str(i+1))

                endpoints.append(residual_block)

            with tf.variable_scope('block3'):
                # block3
                for i in range(6):
                    if i == 0:
                        residual_block = residual_unit_fn(residual_block, 256, 2, 'block3_unit1')
                    else:
                        residual_block = residual_unit_fn(residual_block, 256, 1, 'block3_unit'+str(i+1))

                endpoints.append(residual_block)

            with tf.variable_scope('block4'):
                # block4
                for i in range(3):
                    if i == 0:
                        residual_block = residual_unit_fn(residual_block, 512, 2, 'block4_unit1')
                    else:
                        residual_block = residual_unit_fn(residual_block, 512, 1, 'block4_unit'+str(i+1))

                endpoints.append(residual_block)

        self.endpoints = endpoints

    def _residual_block(self, inputs, filters, strides, scope):
        with tf.variable_scope(scope):
            # residual-branch
            with tf.variable_scope('residual'):
                conv = common.bn_activation_conv(inputs, filters, 3, strides, is_training=self.is_training)  # maybe down sample
                conv = common.bn_activation_conv(conv, filters, 3, 1, is_training=self.is_training)  # no down sample

            # identity-branch
            with tf.variable_scope('identity'):
                if strides != 1:
                    shortcut = common.bn_activation_conv(inputs, filters, 3, strides, is_training=self.is_training)
                else:
                    shortcut = inputs  # common.bn_activation_conv(inputs, filters, 1, 1)

        return conv + shortcut

    def _residual_bottleneck(self, inputs, filters, strides, scope):
        with tf.variable_scope(scope):
            # residual-branch
            with tf.variable_scope('residual'):
                conv = common.bn_activation_conv(inputs, filters, 1, 1, is_training=self.is_training)      # ?strides
                conv = common.bn_activation_conv(conv, filters, 3, strides, is_training=self.is_training)  # ?1
                conv = common.bn_activation_conv(conv, 4*filters, 1, 1, is_training=self.is_training)

            # identity-branch
            with tf.variable_scope('identity'):
                shortcut = common.bn_activation_conv(inputs, 4*filters, 3, strides, is_training=self.is_training)  # ? 1, 1

            return conv + shortcut
