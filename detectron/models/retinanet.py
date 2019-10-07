# coding: utf-8
import tensorflow as tf
import numpy as np
from configs import cfgs

class RetinaNet():
    def __init__(self, mode, trainset):
        # check cfgs
        assert mode in ['train', 'test']
        assert cfgs.data_format in ['channel_first', 'channel_last']

        # get cfgs
        self.mode = mode

        # build network architecture
        self._define_inputs()
        self._build_detection_architecture()

        #
        self._init_session()

    def _init_session(self):
        pass

    def _define_inputs(self):
        # train mode
        # test mode
        pass

    def _build_detection_architecture(self):
        pass

    def train_one_epoch(self):
        global_step, cls_loss, reg_loss, total_loss \
            = self.sess.run()