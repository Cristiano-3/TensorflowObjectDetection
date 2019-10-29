# coding: utf-8
import os

root_path = os.path.abspath('../')
summary_path = root_path + '/output/summaries'
checkpoint_path = root_path + '/output/checkpoints/retina'
num_train_samples = 7412
epochs = 100
lr = 1e-3
batch_size = 2
buffer_size = 100
steps_per_epoch = num_train_samples/batch_size
# ------- data pre-processing cfgs -------
data_format = 'channels_last'
pi = 0.01
is_bottleneck = True 
num_classes = 14
anchors = [32, 64, 128, 256, 512]
aspect_ratios = [1, 1/2, 2, 1/3, 3]
anchor_size = [2**0, 2**(1/3), 2**(2/3)]
num_anchors = len(aspect_ratios) * len(anchor_size)
alpha = 0.25
gamma = 2.0
weight_decay = 1e-4
nms_score_threshold = 0.8
nms_max_boxes = 20
nms_iou_threshold = 0.3  # 0.45

augment_config = {
    'data_format': 'channels_last',
    'output_shape': [500, 500],
    'zoom_size': None,  # [520, 520],
    'crop_method': None,  # 'random',
    'flip_prob': None,  # [0., 0.5],
    'fill_mode': 'BILINEAR',
    'keep_aspect_ratios': False,
    'constant_values': 255.0,  # 0.,
    'color_jitter_prob': None,  # 0.5,  # 抖动
    'rotate': None,  # [0.5, -5., -5.],
    'pad_truth_to': 20,  # 60,
}

show_inter = 20
sumr_inter = 200
save_inter = 10000
