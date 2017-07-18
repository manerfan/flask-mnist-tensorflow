# -*- coding: utf-8 -*-
import tensorflow as tf

# Layers
# 无隐藏层
INPUT_NODE = 784  # 输入层
OUTPUT_NODE = 10  # 输出层


def get_weight_variable(shape, regularizer=None):
    """
    获取某一层的W参数
    :param shape:           weight的shape
    :param regularizer:     正则化生成函数
    :return:                weight
    """
    weights = tf.get_variable("weight", initializer=tf.truncated_normal(shape, stddev=.1))

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
    with tf.name_scope('layer_out'):
        W = get_weight_variable([INPUT_NODE, OUTPUT_NODE], regularizer)
        b = tf.get_variable('biases', initializer=tf.constant(.1, shape=[OUTPUT_NODE]))

        Ylogits = tf.matmul(X, W) + b

        return Ylogits
