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
        self.is_training = (mode == 'train')

        if self.is_training:
            self.train_generator = trainset['train_generator']
            self.train_initializer, self.train_iterator = self.train_generator

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
            # cls and reg subnets: NxHxWxAxclass, NxHxWxAx4
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

            # if NCHW transpose to NHWC
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

        with tf.variable_scope('inference'):
            # cls & reg -> bbox:  NxHWAx2, NxHWAx2, NxHWAxclass
            # diff H, W for each pyramid-level
            p3bbox_yx, p3bbox_hw, p3bbox_conf = self._get_pbbox(p3_cls, p3_reg)
            p4bbox_yx, p4bbox_hw, p4bbox_conf = self._get_pbbox(p4_cls, p4_reg)
            p5bbox_yx, p5bbox_hw, p5bbox_conf = self._get_pbbox(p5_cls, p5_reg)
            p6bbox_yx, p6bbox_hw, p6bbox_conf = self._get_pbbox(p6_cls, p6_reg)
            p7bbox_yx, p7bbox_hw, p7bbox_conf = self._get_pbbox(p7_cls, p7_reg)

            # anchor bbox: HWAx2
            a3bbox_y1x1, a3bbox_y2x2, a3bbox_yx, a3bbox_hw = self._get_abbox(self.anchors[0], p3shape)
            a4bbox_y1x1, a4bbox_y2x2, a4bbox_yx, a4bbox_hw = self._get_abbox(self.anchors[1], p4shape)
            a5bbox_y1x1, a5bbox_y2x2, a5bbox_yx, a5bbox_hw = self._get_abbox(self.anchors[2], p5shape)
            a6bbox_y1x1, a6bbox_y2x2, a6bbox_yx, a6bbox_hw = self._get_abbox(self.anchors[3], p6shape)
            a7bbox_y1x1, a7bbox_y2x2, a7bbox_yx, a7bbox_hw = self._get_abbox(self.anchors[4], p7shape)

            # merge predictions of all pyramid-level
            pbbox_yx = tf.concat([p3bbox_yx, p4bbox_yx, p5bbox_yx, p6bbox_yx, p7bbox_yx], axis=1)
            pbbox_hw = tf.concat([p3bbox_hw, p4bbox_hw, p5bbox_hw, p6bbox_hw, p7bbox_hw], axis=1)
            pbbox_conf = tf.concat([p3bbox_conf, p4bbox_conf, p5bbox_conf, p6bbox_conf, p7bbox_conf], axis=1)

            # merge anchors of all pyramid-level
            abbox_y1x1 = tf.concat([a3bbox_y1x1, a4bbox_y1x1, a5bbox_y1x1, a6bbox_y1x1, a7bbox_y1x1], axis=0)
            abbox_y2x2 = tf.concat([a3bbox_y2x2, a4bbox_y2x2, a5bbox_y2x2, a6bbox_y2x2, a7bbox_y2x2], axis=0)
            abbox_yx = tf.concat([a3bbox_yx, a4bbox_yx, a5bbox_yx, a6bbox_yx, a7bbox_yx], axis=0)
            abbox_hw = tf.concat([a3bbox_hw, a4bbox_hw, a5bbox_hw, a6bbox_hw, a7bbox_hw], axis=0)

            if self.mode == 'train':
                cond = lambda loss, i: tf.less(i, tf.cast(cfgs.batch_size, tf.float32))
                body = lambda loss, i: (
                    tf.add(loss, self._compute_one_image_loss(
                        tf.squeeze(tf.gather(pbbox_yx, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(pbbox_hw, tf.cast(i, tf.int32))),
                        abbox_y1x1,
                        abbox_y2x2,
                        abbox_yx,
                        abbox_hw,
                        tf.squeeze(tf.gather(pconf, tf.cast(i, tf.int32))),
                        tf.squeeze(tf.gather(self.ground_truth, tf.cast(i, tf.int32))),
                    )),
                    tf.add(i, 1.)
                )
                i = 0.
                loss = 0.
                init_state = (loss, i)
                state = tf.while_loop(cond, body, init_state)
                
                total_loss, _ = state
                total_loss = total_loss / cfgs.batch_size
                fpn_l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables('feature_pyramid')])
                sbn_l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables('subnets')])
                self.loss = total_loss + cfgs.weight_decay * (fpn_l2_loss + sbn_l2_loss)

                optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=.9)
                train_op = optimizer.minimize(self.loss, global_step=self.global_step)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.train_op = tf.group([update_ops, train_op])
            else:
                # 

    def _compute_one_image_loss(self, pbbox_yx, pbbox_hw, pconf, 
                                abbox_y1x1, abbox_y2x2, abbox_yx, abbox_hw, 
                                ground_truth):
        # ground_truth (ymin, ymax, xmin, xmax, id) ? (centerx, centery, h, w, id)
        slice_index = tf.argmin(ground_truth, axis=0)[0]
        ground_truth = tf.gather(ground_truth, tf.range(0, slice_index, dtype=tf.int64))
        gbbox_yx = ground_truth[..., 0:2]
        gbbox_hw = ground_truth[..., 2:4]
        class_id = tf.cast(ground_truth[..., 4], dtype=tf.int32)
        labels = class_id
        gbbox_y1x1 = gbbox_yx - gbbox_hw / 2.
        gbbox_y2x2 = gbbox_yx + gbbox_hw / 2. 

        abbox_hwti = tf.reshape(abbox_hw, [1, -1, 2])
        abbox_y1x1ti = tf.reshape(abbox_y1x1, [1, -1, 2])
        abbox_y2x2ti = tf.reshape(abbox_y2x2, [1, -1, 2])
        ashape = tf.shape(abbox_hwti)

        gbbox_hwti = tf.reshape(gbbox_hw, [-1, 1, 2])
        gbbox_y1x1ti = tf.reshape(gbbox_y1x1, [-1, 1, 2])
        gbbox_y2x2ti = tf.reshape(gbbox_y2x2, [-1, 1, 2])
        gshape = tf.shape(gbbox_hwti)

        abbox_hwti = tf.tile(abbox_hwti, [gshape[0], 1, 1])
        abbox_y1x1ti = tf.tile(abbox_y1x1ti, [gshape[0], 1, 1])
        abbox_y2x2ti = tf.tile(abbox_y2x2ti, [gshape[0], 1, 1])
        gbbox_hwti = tf.tile(gbbox_hwti, [1, ashape[1], 1])
        gbbox_y1x1ti = tf.tile(gbbox_y1x1ti, [1, ashape[1], 1])
        gbbox_y2x2ti = tf.tile(gbbox_y2x2ti, [1, ashape[1], 1])

        gaIoU_y1x1ti = tf.maximum(abbox_y1x1ti, gbbox_y1x1ti)
        gaIoU_y2x2ti = tf.minimum(abbox_y2x2ti, gbbox_y2x2ti)
        gaIoU_area = tf.reduce_prod(tf.maximum(gaIoU_y2x2ti - gaIoU_y1x1ti, 0), axis=-1)
        aarea = tf.reduce_prod(abbox_hwti, axis=-1)
        garea = tf.reduce_prod(gbbox_hwti, axis=-1)
        gaIoU = gaIoU_area / (aarea + garea - gaIoU_area)

        tf.argmax(gaIoU, axis=1)
        
    def _get_pbbox(self, predc, predr):
        """
        prediction -> bbox: yx, hw, conf
        """
        pconf = tf.reshape(predc, [cfgs.batch_size, -1, cfgs.num_classes])
        pbbox = tf.reshape(predr, [cfgs.batch_size, -1, 4])
        pbbox_yx = pbbox[..., :2]
        pbbox_hw = pbbox[..., 2:]
        return pbbox_yx, pbbox_hw, pconf

    def _get_abbox(self, size, pshape):
        """
        get all anchors' yx, hw
        size: base size of anchors in this layer
        pshape: is NHWC channel order
        """
        ph = tf.cast(pshape[1], tf.float32)
        pw = tf.cast(pshape[2], tf.float32)

        if cfgs.data_format == 'channels_last':
            input_h = tf.shape(self.images).as_list()[1]
            downsampling_rate = tf.cast(input_h, tf.float32) / ph
        else:
            input_h = tf.shape(self.images).as_list()[1]
            downsampling_rate = tf.cast(input_h, tf.float32) / ph

        # tl_yx, top-left yx for each anchors, YXAS
        tl_y = tf.range(0., ph, dtype=tf.float32)
        tl_x = tf.range(0., pw, dtype=tf.float32)
        tl_y = tf.reshape(tl_y, [-1, 1, 1, 1]) + 0.5  # center
        tl_x = tf.reshape(tl_x, [1, -1, 1, 1]) + 0.5

        # repeat along axis, and scale back
        tl_y = tf.tile(tl_y, [1, pshape[2], 1, 1]) * downsampling_rate
        tl_x = tf.tile(tl_x, [pshape[1], 1, 1, 1]) * downsampling_rate
        tl_yx = tf.concat([tl_y, tl_x], -1)

        # repeat for all anchors: HxWxAx1
        tl_yx = tf.tile(tl_yx, [1, 1, self.num_anchors, 1])

        # get all shapes for each anchor
        priors = []
        for r in cfgs.aspect_ratios:
            for s in cfgs.anchor_size:
                # anchor shapes
                priors.append([s*size*(r**0.5), s*size/(r**0.5)])
        priors = tf.convert_to_tensor(priors, tf.float32)
        priors = tf.reshape(priors, [1, 1, -1, 2])  # 1x1xAx2

        # HxWxAx1 - 1x1xAx2 -> HxWxAx2 -> HWAx2
        abbox_y1x1 = tf.reshape(tl_yx - priors / 2., [-1, 2])
        abbox_y2x2 = tf.reshape(tl_yx + priors / 2., [-1, 2])
        abbox_yx = (abbox_y1x1 + abbox_y2x2) / 2.
        abbox_hw = abbox_y2x2 - abbox_y1x1
        return abbox_y1x1, abbox_y2x2, abbox_yx, abbox_hw

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
