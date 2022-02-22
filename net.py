import tensorflow.contrib.slim as slim
import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm


def model_v9(input,num_classes, keep_pro=0.2):                      ### 11/3 v8延伸
    with tf.variable_scope('model_v9'):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d],
                            padding='SAME'
                            ):
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                # weights_regularizer=slim.l2_regularizer(0.0005),
                                # normalizer_fn=slim.batch_norm,
                                activation_fn=tf.nn.relu,
                                # kernel_size=[3, 3],
                                stride=1,
                                ):
                print('\033[33m ------------------------------------------', '\033[0m')
                # input = input[:,:,:,0:4]
                print('input shape:%s' % input.get_shape())


                # res_att = attention_block(res_gar,res_gar.get_shape()[3],ratio=4)
                # res_att = attention_block(res_gar, 24, ratio=6)

                # conv_1a = slim.conv2d(res_gar, 64, [3, 3], stride=1, scope='conv_1a_3x3')
                # Max_pool_1a = slim.max_pool2d(conv_1a, [2, 2], stride=2, scope='Max_pool_1a')

                # print('conv_1a_3x3_shape:%s' % conv_1a.get_shape())
                # print('Max_pool_1a shape:%s' % Max_pool_1a.get_shape())

                # '''
                conv_1a = slim.conv2d(input, 64, [5, 5],stride = 2, padding = 'SAME',scope='conv_1a_5x5')          #256 ##Note 7*7要改strides!!!
                Max_pool_0a = slim.max_pool2d(conv_1a, [2, 2], stride=2, scope='Max_pool_0a')           #!!!NOTE:懶得改所以論文上改寫成用3*3,stride=2

                # conv_0b = slim.conv2d(Max_pool_0a, 84, [3, 3], scope='conv_0b_3x3')
                # Max_pool_0b = slim.max_pool2d(conv_0b, [2, 2], stride=2, scope='Max_pool_0b')

                conv_2a = slim.conv2d(Max_pool_0a, 96, [3, 3], scope='conv_2a_3x3')
                Max_pool_1a = slim.max_pool2d(conv_2a, [2, 2], stride=2, scope='Max_pool_1a')       #!!!NOTE:懶得改所以論文上改寫成用3*3,stride=2

                print('conv_1a_5x5 shape:%s' % conv_1a.get_shape())
                print('Max_pool_0a shape:%s' % Max_pool_0a.get_shape())

                # print('conv_0b_3x3 shape:%s' % conv_0b.get_shape())
                # print('Max_pool_0b shape:%s' % Max_pool_0b.get_shape())

                print('conv_2a_3x3 shape:%s' % conv_2a.get_shape())
                print('Max_pool_1a shape:%s' % Max_pool_1a.get_shape())
                #
                # '''
                print('\033[33m ------------------------------------------', '\033[0m')
                with tf.variable_scope('Mixed_1b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(Max_pool_1a, 48, [1, 1], scope='Conv2d_0a_1x1')
                        print('Conv2d_0a_1x1 shape:%s' % branch_0.get_shape())

                        branch_0 = slim.conv2d(branch_0, 96, [3, 3], scope='Conv2d_0b_3x3')  # 注意stride
                        print('Conv2d_0b_3x3 shape:%s' % branch_0.get_shape())

                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(Max_pool_1a, 48, [1, 1], scope='Conv2d_0a_1x1')      #64
                        print('Conv2d_0a_1x1 shape:%s' % branch_1.get_shape())

                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                        print('Conv2d_0b_3x3 shape:%s' % branch_1.get_shape())

                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0c_3x3')
                        print('Conv2d_0c_3x3 shape:%s' % branch_1.get_shape())

                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(Max_pool_1a, [3, 3], stride=1, scope='MaxPool_2d_3x3')
                        print('MaxPool_2d_3x3 shape:%s' % branch_2.get_shape())

                        branch_2 = slim.conv2d(branch_2, 96, [1, 1], scope='Conv2d_0b_1x1')
                        print('Conv2d_0b_1x1 shape:%s' % branch_2.get_shape())

                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
                    print('concat shape:%s' % net.get_shape())
                    print('\033[33m ------------------------------------------', '\033[0m')

                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_2a_3x3')
                    print('MaxPool_2a_3x3 shape:%s' % net.get_shape())

                with tf.variable_scope('Mixed_2b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                        print('Conv2d_0a_1x1 shape:%s' % branch_0.get_shape())

                        branch_0 = slim.conv2d(branch_0, 128, [3, 3], scope='Conv2d_0b_3x3')  # 注意stride
                        print('Conv2d_0b_3x3 shape:%s' % branch_0.get_shape())

                        branch_0 = slim.conv2d(branch_0, 128, [3, 3], scope='Conv2d_0c_3x3')  # 注意stride
                        print('Conv2d_0c_3x3 shape:%s' % branch_0.get_shape())

                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                        print('Conv2d_0a_1x1 shape:%s' % branch_1.get_shape())

                        branch_1 = slim.conv2d(branch_1, 72, [3, 3], scope='Conv2d_0b_3x3')
                        print('Conv2d_0b_3x3 shape:%s' % branch_1.get_shape())

                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net, [3, 3], stride=1, scope='MaxPool_2d_3x3')
                        print('MaxPool_2d_3x3 shape:%s' % branch_2.get_shape())

                        branch_2 = slim.conv2d(branch_2, 96, [1, 1], scope='Conv2d_0b_1x1')
                        print('Conv2d_0b_1x1 shape:%s' % branch_2.get_shape())

                    net = tf.concat(axis=3, values=[branch_0, branch_1,branch_2])
                    print('concat shape:%s' % net.get_shape())
                    print('\033[33m ------------------------------------------', '\033[0m')

                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
                    print('MaxPool_3a_3x3 shape:%s' % net.get_shape())

                with tf.variable_scope('Mixed_3b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 224, [1, 1], scope='Conv2d_0a_1x1')
                        print('Conv2d_0a_1x1 shape:%s' % branch_0.get_shape())

                        branch_0 = slim.conv2d(branch_0, 256, [3, 3], scope='Conv2d_0b_3x3')
                        print('Conv2d_0b_3x3 shape:%s' % branch_0.get_shape())

                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.max_pool2d(net, [3, 3], stride=1, scope='MaxPool_2d_3x3')
                        print('MaxPool_2d_3x3 shape:%s' % branch_1.get_shape())

                        branch_1 = slim.conv2d(branch_1, 224, [1, 1], scope='Conv2d_0b_1x1')
                        print('Conv2d_0b_1x1 shape:%s' % branch_1.get_shape())

                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                    print('concat shape:%s' % net.get_shape())
                    print('\033[33m ------------------------------------------', '\033[0m')

                    # net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
                    # print('MaxPool_3a_3x3 shape:%s' % net.get_shape())


                glo_av_pool = tf.reduce_mean(net, axis=[1, 2])  ### 沒有GAP 就要用 flatten

                fc1 = slim.fully_connected(glo_av_pool, 256, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                           # weights_regularizer=slim.l2_regularizer(0.0005),
                                           scope='fc1')
                dpout = slim.dropout(fc1, keep_pro, scope='dropout1')

                out = slim.fully_connected(dpout, num_classes,weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                           # weights_regularizer=slim.l2_regularizer(0.0005),
                                           activation_fn=None, scope='out')

                print('Global avg_pool shape:%s' % glo_av_pool.get_shape())
                print('1_FC shape:%s' % fc1.get_shape())
                print('out shape:%s' % out.get_shape())
                print('\033[33m ------------------------------------------', '\033[0m')

                return out

