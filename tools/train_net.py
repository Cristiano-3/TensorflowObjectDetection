# coding: utf-8
import sys
sys.path.append("../")

from configs import cfgs
from datasets.voc_tfrecord_utils import get_generator
from detectron.models.retinanet import RetinaNet


def train(epochs):
    # prepare dataset
    data = [
        '../../Object-Detection-API-Tensorflow/data/train_00001-of-00010.tfrecord',
        '../../Object-Detection-API-Tensorflow/data/train_00002-of-00010.tfrecord',
        '../../Object-Detection-API-Tensorflow/data/train_00003-of-00010.tfrecord',
        '../../Object-Detection-API-Tensorflow/data/train_00004-of-00010.tfrecord',
        '../../Object-Detection-API-Tensorflow/data/train_00005-of-00010.tfrecord',
        '../../Object-Detection-API-Tensorflow/data/train_00006-of-00010.tfrecord',
        '../../Object-Detection-API-Tensorflow/data/train_00007-of-00010.tfrecord',
        '../../Object-Detection-API-Tensorflow/data/train_00008-of-00010.tfrecord',
        '../../Object-Detection-API-Tensorflow/data/train_00009-of-00010.tfrecord',
        '../../Object-Detection-API-Tensorflow/data/train_00010-of-00010.tfrecord'
        ]

    trainset = get_generator(data)

    # build network
    retinanet = RetinaNet('train', trainset)

    # start training
    for i in range(retinanet.current_epoch, epochs):
        print("-"*25, "epoch", i, "-"*25)

        # train a epoch
        retinanet.train_one_epoch()


if "__main__" == __name__:
    train(cfgs.epochs)
