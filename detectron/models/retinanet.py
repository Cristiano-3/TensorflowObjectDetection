# coding: utf-8
import tensorflow as tf
import numpy as np
from configs import cfgs
from nets.resnet_v1_50 import ResNet
from utils import common


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
        with tf.variable_scope('feature_pyramid'):
            # backbone
            resnet = ResNet(self.images, is_training=self.is_training)
            feat1, feat2, feat3 = resnet.endpoints[-3:]
            p5 = self._get_pyramid(feat3, 256)
            p4, top_down = self._get_pyramid(feat2, 256, p5)
            p3, _ = self._get_pyramid(feat1, 256, top_down)  # biggest resolution
            p6 = common.bn_activation_conv(p5, 256, 3, 2)
            p7 = common.bn_activation_conv(p6, 256, 3, 2)

        with tf.variable_scope('subnets')
            # cls and reg subnets
            p3_cls = self._classification_subnet(p3, 256)
            p3_reg = self._regression_subnet(p3, 256)
            p4_cls = self._classification_subnet(p3, 256)
            p4_reg = self._regression_subnet(p3, 256)
            p5_cls = self._classification_subnet(p3, 256)
            p5_reg = self._regression_subnet(p3, 256)
            p6_cls = self._classification_subnet(p3, 256)
            p6_reg = self._regression_subnet(p3, 256)
            p7_cls = self._classification_subnet(p3, 256)
            p7_reg = self._regression_subnet(p3, 256)

            # if NCHW
            if cfgs.data_format == 'channels_first':
                p3_cls = tf.transpose(p3_cls, [0, 2, 3, 1])
                p3_reg = tf.transpose(p3_reg, [0, 2, 3, 1])
                p4_cls = tf.transpose(p4_cls, [0, 2, 3, 1])
                p4_reg = tf.transpose(p4_reg, [0, 2, 3, 1])
                p5_cls = tf.transpose(p5_cls, [0, 2, 3, 1])
                p5_reg = tf.transpose(p5_reg, [0, 2, 3, 1])
                p6_cls = tf.transpose(p6_cls, [0, 2, 3, 1])
                p6_reg = tf.transpose(p6_reg, [0, 2, 3, 1])
                p7_cls = tf.transpose(p7_cls, [0, 2, 3, 1])
                p7_reg = tf.transpose(p7_reg, [0, 2, 3, 1])

            # get preds' shape
            p3shape = tf.shape(p3_cls)
            p4shape = tf.shape(p4_cls)
            p5shape = tf.shape(p5_cls)
            p6shape = tf.shape(p6_cls)
            p7shape = tf.shape(p7_cls)

        with tf.variable_scope(''):

    def train_one_epoch(self):
        sess.run(self.train_initializer)
        while True:
            try:
                global_step, cls_loss, reg_loss, total_loss \
                = self.sess.run(train_op)
            except tf.errors.OutOfRangeError:
                break
            
    def _classification_subnet(self, featmap, filters):
        conv1 = common.bn_activation_conv(featmap, filters, 3, 1)
        conv2 = common.bn_activation_conv(conv1, filters, 3, 1)
        conv3 = common.bn_activation_conv(conv2, filters, 3, 1)
        conv4 = common.bn_activation_conv(conv3, filters, 3, 1)
        pred = common.bn_activation_conv(conv4, cfgs.num_anchors*cfgs.num_classes, 3, 1, pi_init=True)
        return pred

    def _regression_subnet(self, featmap, filters):
        conv1 = common.bn_activation_conv(featmap, filters, 3, 1)
        conv2 = common.bn_activation_conv(conv1, filters, 3, 1)
        conv3 = common.bn_activation_conv(conv2, filters, 3, 1)
        conv4 = common.bn_activation_conv(conv3, filters, 3, 1)
        pred = common.bn_activation_conv(conv4, cfgs.num_anchors*4, 3, 1)
        return pred

    def _get_pyramid(self, featmap, filters, top_feat=None):
        if top_feat is None:
            return common.bn_activation_conv(featmap, filters, 3, 1)

        else:
            if cfgs.data_format == 'channels_last':
                feat = common.bn_activation_conv(featmap, filters, 1, 1)
                top_feat = tf.image.resize_bilinear(top_feat, [tf.shape(feat)[1], tf.shape(feat)[2]])
                total_feat = feat + top_feat

                return common.bn_activation_conv(total_feat, filters, 3, 1), total_feat
            else:
                feat = common.bn_activation_conv(featmap, filters, 1, 1)
                feat = tf.transpose(feat, [0, 2, 3, 1])  # NCHW->NHWC
                top_feat = tf.transpose(top_feat, [0, 2, 3, 1])
                top_feat = tf.image.resize_bilinear(top_feat, [tf.shape(feat)[1], tf.shape(feat)[2]])
                total_feat = feat + top_feat
                total_feat = tf.transpose(total_feat, [0, 3, 1, 2])  # NHWC->NCHW

                return common.bn_activation_conv(total_feat, filters, 3, 1), total_feat
