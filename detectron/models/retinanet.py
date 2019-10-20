# coding: utf-8
import tensorflow as tf
import numpy as np
import os, sys, time

from configs import cfgs
from detectron.nets.resnet_v1_50 import ResNet
from detectron.utils import common
from detectron.utils.show_box_in_tensor import draw_boxes_with_categories
from detectron.utils.show_box_in_tensor import draw_boxes_with_categories_and_scores


class RetinaNet():
    def __init__(self, mode, trainset):
        # check cfgs
        assert mode in ['train', 'test']
        assert cfgs.data_format in ['channels_first', 'channels_last']

        # get cfgs
        self.mode = mode
        self.is_training = (mode == 'train')

        if self.is_training:
            # self.train_generator = trainset['train_generator']
            self.train_initializer, self.train_iterator = trainset  # self.train_generator

        self.global_step = tf.get_variable(initializer=tf.constant(0), trainable=False, name='global_step')
        # build network architecture
        self._define_inputs()
        self._build_detection_architecture()

        #
        self._init_session()
        self._create_saver()
        self._create_summary_writer(cfgs.summary_path)
        self._create_summary()

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
        shape = [cfgs.batch_size, None, None, 3]

        # PIX_MEAN
        mean = tf.convert_to_tensor([123.68, 116.779, 103.979], dtype=tf.float32)
        if cfgs.data_format == 'channels_last':
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
            self.images = self.images - mean
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

        with tf.variable_scope('subnets'):
            # cls and reg subnets: NxHxWxAxclass, NxHxWxAx4
            p3_cls = self._classification_subnet(p3, 256)
            p3_reg = self._regression_subnet(p3, 256)
            p4_cls = self._classification_subnet(p4, 256)
            p4_reg = self._regression_subnet(p4, 256)
            p5_cls = self._classification_subnet(p5, 256)
            p5_reg = self._regression_subnet(p5, 256)
            p6_cls = self._classification_subnet(p6, 256)
            p6_reg = self._regression_subnet(p6, 256)
            p7_cls = self._classification_subnet(p7, 256)
            p7_reg = self._regression_subnet(p7, 256)

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
            p3bbox_yx, p3bbox_hw, p3conf = self._get_pbbox(p3_cls, p3_reg)
            p4bbox_yx, p4bbox_hw, p4conf = self._get_pbbox(p4_cls, p4_reg)
            p5bbox_yx, p5bbox_hw, p5conf = self._get_pbbox(p5_cls, p5_reg)
            p6bbox_yx, p6bbox_hw, p6conf = self._get_pbbox(p6_cls, p6_reg)
            p7bbox_yx, p7bbox_hw, p7conf = self._get_pbbox(p7_cls, p7_reg)

            # anchor bbox: HWAx2
            a3bbox_y1x1, a3bbox_y2x2, a3bbox_yx, a3bbox_hw = self._get_abbox(cfgs.anchors[0], p3shape)
            a4bbox_y1x1, a4bbox_y2x2, a4bbox_yx, a4bbox_hw = self._get_abbox(cfgs.anchors[1], p4shape)
            a5bbox_y1x1, a5bbox_y2x2, a5bbox_yx, a5bbox_hw = self._get_abbox(cfgs.anchors[2], p5shape)
            a6bbox_y1x1, a6bbox_y2x2, a6bbox_yx, a6bbox_hw = self._get_abbox(cfgs.anchors[3], p6shape)
            a7bbox_y1x1, a7bbox_y2x2, a7bbox_yx, a7bbox_hw = self._get_abbox(cfgs.anchors[4], p7shape)

            # merge predictions of all pyramid-level
            pbbox_yx = tf.concat([p3bbox_yx, p4bbox_yx, p5bbox_yx, p6bbox_yx, p7bbox_yx], axis=1)
            pbbox_hw = tf.concat([p3bbox_hw, p4bbox_hw, p5bbox_hw, p6bbox_hw, p7bbox_hw], axis=1)
            pconf = tf.concat([p3conf, p4conf, p5conf, p6conf, p7conf], axis=1)

            # merge anchors of all pyramid-level
            abbox_y1x1 = tf.concat([a3bbox_y1x1, a4bbox_y1x1, a5bbox_y1x1, a6bbox_y1x1, a7bbox_y1x1], axis=0)
            abbox_y2x2 = tf.concat([a3bbox_y2x2, a4bbox_y2x2, a5bbox_y2x2, a6bbox_y2x2, a7bbox_y2x2], axis=0)
            abbox_yx = tf.concat([a3bbox_yx, a4bbox_yx, a5bbox_yx, a6bbox_yx, a7bbox_yx], axis=0)
            abbox_hw = tf.concat([a3bbox_hw, a4bbox_hw, a5bbox_hw, a6bbox_hw, a7bbox_hw], axis=0)

            if self.mode == 'train':
                cond = lambda loss, conf_loss, pos_coord_loss, i: tf.less(i, tf.cast(cfgs.batch_size, tf.float32))
                def body(loss, conf_loss, pos_coord_loss, i):
                    losses = self._compute_one_image_loss(
                            tf.squeeze(tf.gather(pbbox_yx, tf.cast(i, tf.int32))),
                            tf.squeeze(tf.gather(pbbox_hw, tf.cast(i, tf.int32))),
                            tf.squeeze(tf.gather(pconf, tf.cast(i, tf.int32))),
                            abbox_y1x1,
                            abbox_y2x2,
                            abbox_yx,
                            abbox_hw,
                            tf.squeeze(tf.gather(self.ground_truth, tf.cast(i, tf.int32)))
                        )
                    tloss, closs, ploss = losses
                    loss = tf.add(loss, tloss)
                    conf_loss = tf.add(conf_loss, closs)
                    pos_coord_loss = tf.add(pos_coord_loss, ploss)
                    i = tf.add(i, 1.)
                    # loss = loss + tloss,
                    # conf_loss = conf_loss + closs,
                    # pos_coord_loss = pos_coord_loss + ploss,
                    # i = i + 1.

                    return loss, conf_loss, pos_coord_loss, i
                
                i = 0.
                loss = 0.
                conf_loss = 0. 
                pos_coord_loss = 0.
                init_state = (loss, conf_loss, pos_coord_loss, i)                
                state = tf.while_loop(cond, body, init_state)
                
                #total_loss, _ = state
                total_loss = state[0] / cfgs.batch_size
                self.cls_loss = state[1] / cfgs.batch_size
                self.reg_loss = state[2] / cfgs.batch_size

                # weight regularization loss
                fpn_l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables('feature_pyramid')])
                sbn_l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables('subnets')])
                self.weight_decay_loss = cfgs.weight_decay * (fpn_l2_loss + sbn_l2_loss)
                self.loss = total_loss + self.weight_decay_loss

                optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=.9)
                train_op = optimizer.minimize(self.loss, global_step=self.global_step)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.train_op = tf.group([update_ops, train_op])
            #else:
                # 
                pbbox_yxt = pbbox_yx[0, ...]
                pbbox_hwt = pbbox_hw[0, ...]
                confidence= tf.nn.softmax(pconf[0, ...])
                class_id = tf.argmax(confidence, axis=-1)
                conf_mask = tf.less(class_id, cfgs.num_classes - 1)

                pbbox_yxt = tf.boolean_mask(pbbox_yxt, conf_mask)
                pbbox_hwt = tf.boolean_mask(pbbox_hwt, conf_mask)
                confidence = tf.boolean_mask(confidence, conf_mask)

                abbox_yxt = tf.boolean_mask(abbox_yx, conf_mask)
                abbox_hwt = tf.boolean_mask(abbox_hw, conf_mask)
                
                dpbbox_yxt = pbbox_yxt * abbox_hwt + abbox_yxt
                dpbbox_hwt = tf.exp(pbbox_hwt) * abbox_hwt
                dpbbox_y1x1 = dpbbox_yxt - dpbbox_hwt / 2.
                dpbbox_y2x2 = dpbbox_yxt + dpbbox_hwt / 2. 
                dpbbox_y1x1y2x2 = tf.concat([dpbbox_y1x1, dpbbox_y2x2], axis=-1)
                
                filter_mask = tf.greater_equal(confidence, cfgs.nms_score_threshold)
                scores = []
                class_id = []
                bbox = []
                for i in range(cfgs.num_classes - 1):
                    scoresi = tf.boolean_mask(confidence[:, i], filter_mask[:, i])
                    bboxi = tf.boolean_mask(dpbbox_y1x1y2x2, filter_mask[:, i])
                    selected_indices = tf.image.non_max_suppression(
                        bboxi, scoresi, cfgs.nms_max_boxes, cfgs.nms_iou_threshold, name='nms'
                    )

                    scores.append(tf.gather(scoresi, selected_indices))
                    bbox.append(tf.gather(bboxi, selected_indices))
                    class_id.append(tf.ones_like(tf.gather(scoresi, selected_indices)))

                bbox = tf.concat(bbox, axis=0)
                scores = tf.concat(scores, axis=0)
                class_id = tf.concat(class_id, axis=0)
                self.detection_pred = [scores, bbox, class_id]

    def _compute_one_image_loss(self, pbbox_yx, pbbox_hw, pconf, 
                                abbox_y1x1, abbox_y2x2, abbox_yx, abbox_hw, 
                                ground_truth):
        # ground_truth (ymin, ymax, xmin, xmax, id) ? (centerx, centery, h, w, id)
        slice_index = tf.argmin(ground_truth, axis=0)[0]
        ground_truth = tf.gather(ground_truth, tf.range(0, slice_index, dtype=tf.int64))
        gbbox_yx = ground_truth[..., 0:2]
        gbbox_hw = ground_truth[..., 2:4]
        class_id = tf.cast(ground_truth[..., 4], dtype=tf.int32)
        label = class_id
        gbbox_y1x1 = gbbox_yx - gbbox_hw / 2.
        gbbox_y2x2 = gbbox_yx + gbbox_hw / 2. 

        # ti? tile
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

        best_raIdx = tf.argmax(gaIoU, axis=1)  # relative anchor index
        best_pbbox_yx = tf.gather(pbbox_yx, best_raIdx)
        best_pbbox_hw = tf.gather(pbbox_hw, best_raIdx)
        best_pconf = tf.gather(pconf, best_raIdx)
        best_abbox_yx = tf.gather(abbox_yx, best_raIdx)
        best_abbox_hw = tf.gather(abbox_hw, best_raIdx)

        bestmask, _ = tf.unique(best_raIdx)
        bestmask = tf.sort(bestmask)
        bestmask = tf.reshape(bestmask, [-1, 1])
        bestmask = tf.SparseTensor(tf.concat([bestmask, tf.zeros_like(bestmask)], axis=-1),
                                          tf.squeeze(tf.ones_like(bestmask)), dense_shape=[ashape[1], 1])
        bestmask = tf.reshape(tf.cast(tf.sparse.to_dense(bestmask), tf.float32), [-1])

        othermask = 1. - bestmask
        othermask = othermask > 0. 
        other_pbbox_yx = tf.boolean_mask(pbbox_yx, othermask)
        other_pbbox_hw = tf.boolean_mask(pbbox_hw, othermask)
        other_pconf = tf.boolean_mask(pconf, othermask)

        other_abbox_yx = tf.boolean_mask(abbox_yx, othermask)
        other_abbox_hw = tf.boolean_mask(abbox_hw, othermask)

        agIoU = tf.transpose(gaIoU)
        other_agIoU = tf.boolean_mask(agIoU, othermask)
        best_agIoU  = tf.reduce_max(other_agIoU, axis=1)
        pos_agIoU_mask = best_agIoU > 0.5
        neg_agIoU_mask = best_agIoU < 0.4
        rgIdx = tf.argmax(other_agIoU, axis=1)
        pos_rgIdx = tf.boolean_mask(rgIdx, pos_agIoU_mask)
        pos_pbbox_yx = tf.boolean_mask(other_pbbox_yx, pos_agIoU_mask)
        pos_pbbox_hw = tf.boolean_mask(other_pbbox_hw, pos_agIoU_mask)
        pos_pconf = tf.boolean_mask(other_pconf, pos_agIoU_mask)
        
        pos_abbox_yx = tf.boolean_mask(other_abbox_yx, pos_agIoU_mask)
        pos_abbox_hw = tf.boolean_mask(other_abbox_hw, pos_agIoU_mask)

        pos_label = tf.gather(label, pos_rgIdx)
        pos_gbbox_yx = tf.gather(gbbox_yx, pos_rgIdx)
        pos_gbbox_hw = tf.gather(gbbox_hw, pos_rgIdx)

        neg_pconf = tf.boolean_mask(other_pconf, neg_agIoU_mask)
        neg_shape = tf.shape(neg_pconf)
        num_neg = neg_shape[0]
        neg_class_id = tf.constant([cfgs.num_classes-1])
        neg_label = tf.tile(neg_class_id, [num_neg])

        pos_pbbox_yx = tf.concat([best_pbbox_yx, pos_pbbox_yx], axis=0)
        pos_pbbox_hw = tf.concat([best_pbbox_hw, pos_pbbox_hw], axis=0)
        pos_pconf = tf.concat([best_pconf, pos_pconf], axis=0)
        pos_label = tf.concat([label, pos_label], axis=0)
        pos_gbbox_yx = tf.concat([gbbox_yx, pos_gbbox_yx], axis=0)
        pos_gbbox_hw = tf.concat([gbbox_hw, pos_gbbox_hw], axis=0)
        pos_abbox_yx = tf.concat([best_abbox_yx, pos_abbox_yx], axis=0)
        pos_abbox_hw = tf.concat([best_abbox_hw, pos_abbox_hw], axis=0)
        conf_loss = self._focal_loss(pos_label, pos_pconf, neg_label, neg_pconf)

        pos_truth_pbbox_yx = (pos_gbbox_yx - pos_abbox_yx) / pos_abbox_hw
        pos_truth_pbbox_hw = tf.log(pos_gbbox_hw / pos_abbox_hw)
        pos_yx_loss = tf.reduce_sum(self._smooth_l1_loss(pos_pbbox_yx - pos_truth_pbbox_yx), axis=-1)
        pos_hw_loss = tf.reduce_sum(self._smooth_l1_loss(pos_pbbox_hw - pos_truth_pbbox_hw), axis=-1)
        pos_coord_loss = tf.reduce_mean(pos_yx_loss + pos_hw_loss)

        total_loss = conf_loss + pos_coord_loss
        return total_loss, conf_loss, pos_coord_loss

    def _smooth_l1_loss(self, x):
        return tf.where(tf.abs(x) < 1., 0.5*x*x, tf.abs(x)-0.5)

    def _focal_loss(self, poslabel, posprob, neglabel, negprob):
        posprob = tf.nn.softmax(posprob)
        negprob = tf.nn.softmax(negprob)
        pos_index = tf.concat([
            tf.expand_dims(tf.range(0, tf.shape(posprob)[0], dtype=tf.int32), axis=-1),
            tf.reshape(poslabel, [-1, 1])
        ], axis=-1)
        neg_index = tf.concat([
            tf.expand_dims(tf.range(0, tf.shape(negprob)[0], dtype=tf.int32), axis=-1),
            tf.reshape(neglabel, [-1, 1])
        ], axis=-1)
        posprob = tf.clip_by_value(tf.gather_nd(posprob, pos_index), 1e-8, 1.)
        negprob = tf.clip_by_value(tf.gather_nd(negprob, neg_index), 1e-8, 1.)
        posloss = - cfgs.alpha * tf.pow(1. - posprob, cfgs.gamma) * tf.log(posprob)
        negloss = - cfgs.alpha * tf.pow(1. - negprob, cfgs.gamma) * tf.log(negprob)
        total_loss = tf.concat([posloss, negloss], axis=0)
        loss = tf.reduce_sum(total_loss) / tf.cast(tf.shape(posloss)[0], tf.float32)
        return loss

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
            input_h = tf.shape(self.images)[1]
            downsampling_rate = tf.cast(input_h, tf.float32) / ph
        else:
            input_h = tf.shape(self.images)[2]
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

        # repeat for all anchors: HxWxAx2
        tl_yx = tf.tile(tl_yx, [1, 1, cfgs.num_anchors, 1])

        # get all shapes for each anchor
        priors = []
        for r in cfgs.aspect_ratios:
            for s in cfgs.anchor_size:
                # anchor shapes
                priors.append([s*size*(r**0.5), s*size/(r**0.5)])
        priors = tf.convert_to_tensor(priors, tf.float32)
        priors = tf.reshape(priors, [1, 1, -1, 2])  # 1x1xAx2

        # HxWxAx2 - 1x1xAx2 -> HxWxAx2 -> HWAx2
        abbox_y1x1 = tf.reshape(tl_yx - priors / 2., [-1, 2])
        abbox_y2x2 = tf.reshape(tl_yx + priors / 2., [-1, 2])
        abbox_yx = (abbox_y1x1 + abbox_y2x2) / 2.
        abbox_hw = abbox_y2x2 - abbox_y1x1
        return abbox_y1x1, abbox_y2x2, abbox_yx, abbox_hw

    def train_one_epoch(self, lr):
        self.is_training = True
        self.sess.run(self.train_initializer)
        mean_loss = []
        step = 0
        while True:
            try:
                # train
                if step==0 or (step+1) % cfgs.show_inter == 0:
                    start = time.time()

                    # train a step
                    _, cls_loss, reg_loss, total_loss, global_step \
                    = self.sess.run([self.train_op, self.cls_loss, self.reg_loss, self.loss, self.global_step], feed_dict={self.lr: lr})
                    
                    end = time.time()

                    # show infos
                    training_time = time.strftime('%Y-%M-%D %H:%M:%S', time.localtime(time.time()))
                    print('{}: step {:d}, cls_loss:{:.4f}, reg_loss:{:.4f}, total_loss:{:.4f}, per_cost_time:{:.4f}s' \
                        .format(training_time, global_step, cls_loss, reg_loss, total_loss, (end - start)))

                else:
                    # train a step
                    _, global_step = self.sess.run([self.train_op, self.global_step], feed_dict={self.lr: lr})

                # save
                if global_step % cfgs.save_inter == 0:
                    self._save_weight(cfgs.checkpoint_path)

                # summary
                if global_step % cfgs.sumr_inter == 0:
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, global_step=global_step)
                    self.summary_writer.flush()

                step += 1
                mean_loss.append(total_loss)

            except tf.errors.OutOfRangeError:
                print('Finish one epoch!')
                break

        sys.stdout.write('\n')
        mean_loss = np.mean(mean_loss)
        return mean_loss

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
                # squeeze channel to size 'filters'
                feat = common.bn_activation_conv(featmap, filters, 1, 1)

                # resize top feat
                top_feat = tf.image.resize_bilinear(top_feat, [tf.shape(feat)[1], tf.shape(feat)[2]])

                # add 
                total_feat = feat + top_feat

                # get final pyra use total feat, and pass total feat to next
                return common.bn_activation_conv(total_feat, filters, 3, 1), total_feat
            else:
                feat = common.bn_activation_conv(featmap, filters, 1, 1)
                feat = tf.transpose(feat, [0, 2, 3, 1])  # NCHW->NHWC

                top_feat = tf.transpose(top_feat, [0, 2, 3, 1])
                top_feat = tf.image.resize_bilinear(top_feat, [tf.shape(feat)[1], tf.shape(feat)[2]])

                total_feat = feat + top_feat
                total_feat = tf.transpose(total_feat, [0, 3, 1, 2])  # NHWC->NCHW

                return common.bn_activation_conv(total_feat, filters, 3, 1), total_feat

    def _create_saver(self):
        self.saver = tf.train.Saver(max_to_keep=5)

    def _save_weight(self, path):
        if not tf.gfile.Exists(os.path.dirname(path)):
            tf.gfile.MakeDirs(os.path.dirname(path))
            print(os.path.dirname(path), 'does not exist, create it done!')

        self.saver.save(self.sess, path, global_step=self.global_step)
        print('save model in:', path)

    def _create_summary_writer(self, summary_path):
        self.summary_writer = tf.summary.FileWriter(summary_path, graph=self.sess.graph)

    def _create_summary(self):
        with tf.variable_scope('RETINA_LOSS'):
            tf.summary.scalar('cls_loss', self.cls_loss)
            tf.summary.scalar('reg_loss', self.reg_loss)

        with tf.variable_scope('LOSS'):
            tf.summary.scalar('weight_decay_loss', self.weight_decay_loss)
            tf.summary.scalar('total_loss', self.loss)

        img_gt = draw_boxes_with_categories(self.images[0:1,...], 
                                            boxes=self.ground_truth[0, :, :-1],
                                            labels=self.ground_truth[0, :, -1])
        img_det = draw_boxes_with_categories_and_scores(self.images[0:1,...], 
                                                        boxes=self.detection_pred[1],
                                                        labels=self.detection_pred[2],
                                                        scores=self.detection_pred[0])

        tf.summary.image('DETECT_CMP/final_detection', img_det)
        tf.summary.image('DETECT_CMP/ground_truth', img_gt)
        self.summary_op = tf.summary.merge_all()
