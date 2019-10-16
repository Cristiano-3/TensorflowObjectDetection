# coding: utf-8
import sys
sys.path.append("../")

from configs import cfgs
from datasets.voc_tfrecord_utils import get_generator
from detectron.models.retinanet import RetinaNet


def train(epochs, lr):
    # prepare dataset
    data = [
        '../../Object-Detection-API-Tensorflow/data/train_00001-of-00010.tfrecord',
        '../../Object-Detection-API-Tensorflow/data/train_00002-of-00010.tfrecord',
        '../../Object-Detection-API-Tensorflow/data/train_00003-of-00010.tfrecord'
        ]

    trainset = get_generator(data)

    # build network
    retinanet = RetinaNet('train', trainset)

    # start training
    for i in range(epochs):
        print("-"*25, "epoch", i, "-"*25)

        # reduce learning rate
        if i % 10 == 0:
            lr /= 10.0
            print("reduce lr, lr=", lr)

        # train a epoch
        retinanet.train_one_epoch(lr)


if "__main__" == __name__:
    train(cfgs.epochs, cfgs.lr)
