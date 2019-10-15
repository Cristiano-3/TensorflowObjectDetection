# coding: utf-8

epochs = 100
lr = 1e-2
batch_size = 1
buffer_size = 100
# ------- data pre-processing cfgs -------
data_format = 'channels_last'
pi = 0.01
is_bottleneck = True 
num_classes = 13
anchors = [32, 64, 128, 256, 512]
aspect_ratios = [1, 1/2, 2]
anchor_size = [2**0, 2**(1/3), 2**(2/3)]
num_anchors = len(aspect_ratios) * len(anchor_size)
alpha = 0.25
gamma = 2.0
weight_decay = 1e-4
nms_score_threshold = 0.8
nms_max_boxes = 20
nms_iou_threshold = 0.45

augment_config = {
    'data_format': 'channels_last',
    'output_shape': [500, 500],
    'zoom_size': [520, 520],
    'crop_method': 'random',
    'flip_prob': [0., 0.5],
    'fill_mode': 'BILINEAR',
    'keep_aspect_ratios': False,
    'constant_values': 0.,
    'color_jitter_prob': 0.5,        # 抖动
    'rotate': [0.5, -5., -5.],
    'pad_truth_to': 60,
}