# coding: utf-8
import tensorflow as tf
import numpy as np
from configs import cfgs
from utils.common import *


class RetinaNet():
    def __init__(self, mode, trainset):
        # check cfgs
        assert mode in ['train', 'test']
        assert cfgs.data_format in ['channel_first', 'channel_last']

        # get cfgs
        self.mode = mode

        if mode == 'train':
            self.is_training = True
            self.train_generator = trainset['train_generator']
            self.train_initializer, self.train_iterator = self.train_generator
        else:
            self.is_training = False

        # build network architecture
        self._define_inputs()
        self._build_detection_architecture()

        #
        self._init_session()

    def _init_session(self):
        self.sess = tf.Session()

        # init global variables
        self.sess.run(tf.global_variables_initializer())

        # init data iterator
        if self.mode == 'train':
            if self.train_initializer is not None:
                self.sess.run(self.train_initializer)
        

    def _define_inputs(self):
        """
        shape/keep_aspect_ratio_resizer or fixed_shape_resizer
        mean order, where to do minus mean, PIXEL_STD?
        """
        shape = [self.batch_size, None, None, 3]
        mean = tf.convert_to_tensor([123.68, 116.779, 103.979], dtype=tf.float32)

        if cfgs.data_format == 'channel_last':
            mean = tf.reshape(mean, [1, 1, 1, 3])
        else:
            mean = tf.reshape(mean, [1, 3, 1, 1])

        # train mode
        if self.mode == 'train':
            self.images, self.ground_truth = self.train_iterator.get_next()
            self.images.set_shape(shape)
            self.images = self.images - mean
        
        # test mode
        else:
            self.images = tf.placeholder(tf.float32, shape, name='images')
            self.images = self.images -mean
            self.ground_truth = tf.placeholder(tf.float32, [self.batch_size, None, 5], name='labels')
        
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')


    def _build_detection_architecture(self):
        pass

    def train_one_epoch(self):
        sess.run(self.train_initializer)
        while True:
            try:
                global_step, cls_loss, reg_loss, total_loss \
                = self.sess.run(train_op)
            except tf.errors.OutOfRangeError:
                break
            
