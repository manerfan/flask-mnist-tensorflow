# -*- coding: utf-8 -*-
import tensorflow as tf

INPUT_NODE = 784  # 输入层
OUTPUT_NODE = 10  # 输出层
# 隐藏层
L = 200
M = 100
N = 60
O = 30


def get_weight_variable(shape, regularizer=None):
    """
    获取某一层的W参数
    :param shape:           weight的shape
    :param regularizer:     正则化生成函数
    :return:                weight
    """
    weights = tf.Variable(tf.truncated_normal(shape, stddev=.1))

    # 如果提供了正则化函数，则一并加入regularization集合
    if regularizer is not None:
        tf.add_to_collection('regularization', regularizer(weights))

    return weights

def inference(X, regularizer):
    """
    定义神经网络前向传播过程
    :param X:               输入 X
    :param regularizer:     正则化函数
    :return:                输出 Y
    """
    with tf.name_scope('layer'):
        W1 = get_weight_variable([INPUT_NODE, L], regularizer)
        b1 = tf.Variable(tf.constant(.1, shape=[L]))
        W2 = get_weight_variable([L, M], regularizer)
        b2 = tf.Variable(tf.constant(.1, shape=[M]))
        W3 = get_weight_variable([M, N], regularizer)
        b3 = tf.Variable(tf.constant(.1, shape=[N]))
        W4 = get_weight_variable([N, O], regularizer)
        b4 = tf.Variable(tf.constant(.1, shape=[O]))
        W5 = get_weight_variable([O, OUTPUT_NODE], regularizer)
        b5 = tf.Variable(tf.constant(.1, shape=[OUTPUT_NODE]))

        Y1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
        Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
        Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)
        Ylogits = tf.matmul(Y4, W5) + b5

        return Ylogits
