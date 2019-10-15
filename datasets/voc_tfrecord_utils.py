# coding: utf-8
import tensorflow as tf
import numpy as np 
import os, sys
from lxml import etree
import warnings
from datasets.image_augmentor import augment
from configs import cfgs

class2id = {
    'VAT_electronic_invoice': 0,
    'VAT_roll_invoice': 1,
    'VAT_general_invoice': 2,
    'VAT_special_invoice': 3,
    'Air_electronic_itinerary': 4,
    'Train_ticket': 5,
    'Taxi_ticket': 6,
    'Passenger_transport_ticket': 7,
    'Road_Bridge_ticket': 8,
    'General_invoice': 9,
    'Quota_invoice': 10,
    'Car_sales_invoice': 11,
    'Others': 12,
}


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def xml_to_example(xmlpath, imgpath):
    xml = etree.parse(xmlpath)
    root = xml.getroot()

    # read image
    imgname = xmlpath.split('/')[-1].replace('.xml', '.jpeg')  # image's ext=jpeg
    imgname = os.path.join(imgpath, imgname)
    image = tf.gfile.GFile(imgname, 'rb').read()

    # read shape
    size = root.find('size')
    height = int(size.find('height').text)
    width = int(size.find('width').text)
    depth = int(size.find('depth').text)
    shape = np.asarray([height, width, depth], np.int32)

    # read gt
    xpath = xml.xpath('//object')
    gt = np.zeros([len(xpath), 5], np.float32)
    for i in range(len(xpath)):
        obj = xpath[i]
        id = class2id[obj.find('name').text]
        bndbox = obj.find('bndbox')
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)
        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        gt[i, :] = np.asarray([ymin, ymax, xmin, xmax, id], np.float32)

    # get all features
    features = {
        'image': bytes_feature(image),
        'shape': bytes_feature(shape.tobytes()),
        'ground_truth': bytes_feature(gt.tobytes())
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def dataset2tfrecord(xml_dir, img_dir, output_dir, name, total_shards=5):
    # check output dir
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
        print(output_dir, 'does not exist, create it done!')
    else:
        if len(tf.gfile.ListDirectory(output_dir)) == 0:
            print(output_dir, 'already exist, need not create new.')
        else:
            warnings.warn(output_dir + 'is not empty!', UserWarning)

    outputfiles = []

    # read xml list
    xmllist = tf.gfile.Glob(os.path.join(xml_dir, '*.xml'))
    num_per_shard = int(math.ceil(len(xmllist)) / float(total_shards))

    for shard_id in range(total_shards):
        # tfrecord file name
        outputname = '%s_%05d-of-%05d.tfrecord' % (name, shard_id+1, total_shards)
        outputname = os.path.join(output_dir, outputname)
        outputfiles.append(outputname)

        with tf.python_io.TFRecordWriter(outputname) as writer:
            # xml index range
            start_ndx = shard_id * num_per_shard
            end_ndx = min( (shard_id+1) * num_per_shard, len(xmllist))

            for i in range(start_ndx, end_ndx):
                # show progress
                sys.stdout.write('\r>> Converting image %d/%d shard %d/%d' % 
                                (i+1, len(xmllist), shard_id+1, total_shards))
                sys.stdout.flush()

                # write an example to tfrecord file
                example = xml_to_example(xmllist[i], img_dir)
                writer.write(example.SerializeToString())

            sys.stdout.write('\n')
            sys.stdout.flush()

    return outputfiles


def parse_fn(data, config):
    features = tf.parse_single_example(data, features={
        'image': tf.FixedLenFeature([], tf.string),
        'shape': tf.FixedLenFeature([], tf.string),
        'ground_truth': tf.FixedLenFeature([], tf.string)
    })

    shape = tf.decode_raw(features['shape'], tf.int32)
    gt = tf.decode_raw(features['ground_truth'], tf.float32)
    image = tf.image.decode_jpeg(features['image'], channels=3)
    shape = tf.reshape(shape, [3])
    gt = tf.reshape(gt, [-1, 5])
    image = tf.cast(tf.reshape(image, shape), tf.float32)
    image, gt = augment(image=image,
                        input_shape=shape,
                        ground_truth=gt,
                        **config)

    return image, gt


def get_generator(tfrecords):  #, batch_size, buffer_size, config):
    """
    :param tfrecords:
    :param batch_size:
    :param buffer_size:
    :return:
    """
    # create dataset from tfrecords
    dataset = tf.data.TFRecordDataset(tfrecords)

    # operations to dataset
    dataset = (dataset.map(lambda x: parse_fn(x, cfgs.augment_config))
        .shuffle(buffer_size=cfgs.buffer_size)
        .batch(cfgs.batch_size, drop_remainder=True)
        .repeat()
    )

    # get iterator of dataset, and corresponding initializer
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    init_op = iterator.make_initializer(dataset)
    return init_op, iterator


# https://cs230-stanford.github.io/tensorflow-input-data.html
# one good order for the different transformations is:

# create the dataset
# shuffle (with a big enough buffer size)
# repeat
# map with the actual work (preprocessing, augmentation…) using multiple parallel calls
# batch
# prefetch