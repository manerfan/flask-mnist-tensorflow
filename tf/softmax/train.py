# -*- coding: utf-8 -*-
import os
import shutil

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.python.framework import graph_util

from .inference import INPUT_NODE, OUTPUT_NODE, inference

# Learning Rate
LEARNING_RATE_BASE = .8
LEARNING_RATE_DECAY = .99  # 学习率衰减率

# 训练论数
TRAINING_STEPS = 30000
BATCH_SIZE = 100

# 正则化系数
REGULARIZATION_RATE = .0001

# mnist数据存放路径
mnist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# 模型存放路径
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
model_name = 'model.pb'


def train(mnist):
    """
    训练模型
    :param mnist:   mnist数据
    """
    with tf.name_scope('input'):
        # 输入
        X = tf.placeholder(tf.float32, [None, INPUT_NODE], name='X')
        # 正确输出
        Y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='Y_')

    # 正则化函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    with tf.name_scope('output'):
        ## 预测输出
        Ylogits = inference(X, regularizer)
        Y = tf.nn.softmax(Ylogits, name='predict')

    with tf.name_scope('loss'):
        # 交叉熵
        # softmax_cross_entropy_with_logits:
        # Computes softmax cross entropy between `logits` and `labels`.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
        # 交叉熵均值
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        ## 损失函数
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('regularization'))

    # 当前训练轮数
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('train'):
        ## 指数衰减学习率
        # learning_rate, global_step, decay_steps, decay_rate, staircase
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        decayed_learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                           global_step,
                                                           mnist.train.num_examples / BATCH_SIZE,
                                                           LEARNING_RATE_DECAY,
                                                           staircase=True)

        # 使用梯度下降优化参数
        train_step = tf.train.GradientDescentOptimizer(decayed_learning_rate) \
            .minimize(loss, global_step=global_step)

        # 正确率
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.arg_max(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        validate_feed = {X: mnist.validation.images, Y_: mnist.validation.labels}
        test_feed = {X: mnist.test.images, Y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={X: xs, Y_: ys})

            if i % 1000 == 0:
                print(
                    f'step *** {sess.run(global_step):<7} '
                    f'validate accuracy *** {sess.run(accuracy, feed_dict=validate_feed):<0.20f}'
                )

        graph_def = sess.graph.as_graph_def()
        predict_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['output/predict'])
        with tf.gfile.GFile(os.path.join(model_dir, model_name), 'wb') as f:
            f.write(predict_graph_def.SerializeToString())
        print(predict_graph_def.node)

        # saver.save(sess, os.path.join(model_dir, model_name))

        print(
            f'step *** {sess.run(global_step):<7} '
            f'test accuracy *** {sess.run(accuracy, feed_dict=test_feed):<0.20f}'
        )


def main(argv=None):
    print('=== softmax_train ===')
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    mnist = mnist_data.read_data_sets(mnist_dir, one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
