# -*- coding: utf-8 -*-
import tensorflow as tf

INPUT_NODE = 784  # 输入层
OUTPUT_NODE = 10  # 输出层

IMAGE_SIZE = 28  # 图像大小
NUM_CHANNELS = 1  # 图像深度
NUM_LABELS = 10  # 图像类别数

# 第一层卷积层参数
CONV1_DEEP = 32  # 深度
CONV1_SIZE = 5  # 尺寸

# 第二层卷积层参数
CONV2_DEEP = 64  # 深度
CONV2_SIZE = 5  # 尺寸

# 全连接层参数
FC_SIZE = 512


def inference(X, regularizer, train=False):
    # 第一层卷积层
    with tf.name_scope('layer_conv1'):
        conv1_weights = tf.Variable(
            tf.truncated_normal([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], stddev=.1, dtype=tf.float32),
            name='weights')
        conv1_biases = tf.Variable(tf.constant(.01, shape=[CONV1_DEEP], dtype=tf.float32), name='biases')

        conv1 = tf.nn.conv2d(X, conv1_weights, strides=[1, 1, 1, 1, ], padding='SAME')
        relu1 = tf.nn.relu(conv1 + conv1_biases)

    # 第一层池化层
    with tf.name_scope('layer_pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第二层卷积层
    with tf.name_scope('layer_conv2'):
        conv2_weights = tf.Variable(
            tf.truncated_normal([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], stddev=.1, dtype=tf.float32),
            name='weights')
        conv2_biases = tf.Variable(tf.constant(.01, shape=[CONV2_DEEP], dtype=tf.float32), name='biases')

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1, ], padding='SAME')
        relu2 = tf.nn.relu(conv2 + conv2_biases)

    # 第二层池化层
    with tf.name_scope('layer_pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第五层全连接层
    pool_shape = pool2.get_shape().as_list()  # 0 batch数量 1 2 3 数据维度
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [-1, nodes])  # 将卷积+池化层的数据拉平
    with tf.name_scope('layer_fc1'):
        fc1_weights = tf.Variable(tf.truncated_normal([nodes, FC_SIZE], stddev=.1, dtype=tf.float32), 'weights')
        fc1_biases = tf.Variable(tf.constant(.1, shape=[FC_SIZE], dtype=tf.float32), 'biases')

        # 只有全连接层需要正则化
        if regularizer is not None:
            tf.add_to_collection('regularization', regularizer(fc1_weights))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        # 只有训练时才加dropout
        if train:
            fc1 = tf.nn.dropout(fc1, .5)

    # 第六层输出层
    with tf.name_scope('layer_output'):
        fc2_weights = tf.Variable(tf.truncated_normal([FC_SIZE, NUM_LABELS], stddev=.1, dtype=tf.float32), 'weights')
        fc2_biases = tf.Variable(tf.constant(.1, shape=[NUM_LABELS], dtype=tf.float32), 'biases')

        if regularizer is not None:
            tf.add_to_collection('regularization', regularizer(fc2_weights))

        Ylogits = tf.matmul(fc1, fc2_weights) + fc2_biases

    return Ylogits
