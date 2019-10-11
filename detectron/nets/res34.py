# coding: utf-8

import tensorflow as tf


# 定义残差块
def residual_block(x, output_channel):
    input_channel = x.get_shape().as_list()[-1]
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = (2, 2)  # 降采样了
    elif input_channel == output_channel:
        increase_dim = False
        strides = (1, 1)
    else:
        raise Exception("output_channel not match with input_channel")

    # no bottleneck
    conv1 = tf.layers.conv2d(x,
                            output_channel,
                            (3, 3),
                            strides=strides,
                            padding='same',
                            activation=tf.nn.relu,
                            name='conv1')

    conv2 = tf.layers.conv2d(conv1, 
                            output_channel,
                            (3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation=tf.nn.relu,
                            name='conv2')

    # Identity 分支
    if increase_dim:
        # 需要降采样
        # [None,image_width,image_height,channel] -> [,,,channel*2]
        pooled_x = tf.layers.average_pooling2d(x,
                                              (2,2), # pooling 核
                                              (2,2), # strides strides = pooling 不重叠
                                              padding = 'valid' # 这里图像大小是32*32，都能除尽，padding是什么没有关系
                                              )
        
        # average_pooling2d使得图的大小变化了，但是output_channel还是不匹配，下面修改output_channel
        padded_x = tf.pad(pooled_x,
                         [[0,0],
                          [0,0],
                          [0,0],
                          [input_channel // 2,input_channel //2]])
    else:
        padded_x = x
    output_x = conv2 + padded_x
    return output_x


# 定义残差网络
def res_net(x,
            num_residual_blocks,  
            num_filter_base, 
            class_num): 
    """residual network implementation"""
    """
    Args:
    - x: 输入数据
    - num_residual_blocks: 残差链接块数 eg: [3,4,6,3]
    - num_filter_base: 最初的通道数目
    - class_num: 类别数目
    """
    # 需要做多少次降采样
    num_subsampling = len(num_residual_blocks)
    layers = []
    # [None,image_width,image_height,channel] -> [image_width,image_height,channel]
    # kernal size：image_width,image_height
    input_size = x.get_shape().as_list()[1:]
    with tf.variable_scope('conv0'):
        conv0 = tf.layers.conv2d(x,
                                 num_filter_base,
                                 (3,3),
                                 strides = (1,1),
                                 activation = tf.nn.relu,
                                 padding = 'same',
                                 name = 'conv0')
        layers.append(conv0)
        
    # eg: num_subsampling = 4 ，sample_id = [1，2，3，4]   
    for sample_id in range(num_subsampling):
        for i in range(num_residual_blocks[sample_id]):
            with tf.variable_scope("conv%d_%d" % (sample_id, i)):
                conv = residual_block(
                    layers[-1],
                    num_filter_base * (2 ** sample_id)) # 每次翻倍
                layers.append(conv)
    multiplier = 2 ** (num_subsampling - 1)
    assert layers[-1].get_shape().as_list()[1:] \
        == [input_size[0] / multiplier,
            input_size[1] / multiplier,
            num_filter_base * multiplier]
    with tf.variable_scope('fc'):
        # layers[-1].shape : [None, width, height, channel]
        global_pool = tf.reduce_mean(layers[-1], [1, 2]) # pooling
        logits = tf.layers.dense(global_pool, class_num) # 全连接
        layers.append(logits)
    return layers[-1]


# 使用残差网络
x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])

# 将向量变成具有三通道的图片的格式
x_image = tf.reshape(x, [-1,3,32,32])
# 32*32
x_image = tf.transpose(x_image, perm = [0, 2, 3, 1])

y_ = res_net(x_image, [2,3,2], 32, 10)


# 交叉熵
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
# y_-> softmax
# y -> one_hot
# loss = ylogy_

# bool
predict = tf.argmax(y_, 1)
# [1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

"""
ref:
https://github.com/liangyihuai/my_tensorflow/blob/master/com/huai/converlution/resnets/hand_classifier_with_resnet.py
https://juejin.im/post/5c4c1794f265da61285a7590
http://www.voidcn.com/article/p-zqkmsrxp-brp.html
https://www.cnblogs.com/Negan-ZW/p/9538414.html
"""
