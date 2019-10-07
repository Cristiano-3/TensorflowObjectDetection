# coding: utf-8

from configs import cfgs
from datasets.voc_tfrecord_utils import get_generator
from detectron.models.retinanet import RetinaNet

def train(epochs, lr):
    # prepare dataset
    trainset = get_generator()

    # build network
    retinanet = RetinaNet('train', trainset)

    # start training
    for i in range(epochs):
        print("-"*25, "epoch", i, "-"*25)

        # reduce learning rate
        if i % 20 == 0:
            lr /= 10.0
            print("reduce lr, lr=", lr)

        # train a epoch
        retinanet.train_one_epoch(lr)


if "__main__" == __name__:
    train(cfgs.epochs, cfgs.lr)
