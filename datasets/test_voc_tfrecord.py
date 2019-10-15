# coding: utf-8
import voc_tfrecord_utils as voc_utils
annotations_dir = ''
images_dir = ''
save_dir = './data/'
dataset_name = 'train'
num_shard = 5
tfrecords = voc_utils.dataset2tfrecord(
    annotations_dir, 
    images_dir,
    save_dir,
    dataset_name,
    num_shard)

print(tfrecords)