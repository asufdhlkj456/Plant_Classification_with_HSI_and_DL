import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


def create_conv_3dlayer(input,
                        filter_width,
                        filter_height,
                        filter_depth,
                        stride=1,
                        num_output_channels=1,
                        relu=True):
    layer = tf.layers.conv3d(inputs=input,
                             filters=num_output_channels,       #num_output_channels-->kernel 的數目 幾個filter就幾個輸出
                             kernel_size=[filter_width, filter_height, filter_depth],
                             strides=(1, 1, stride),
                             #padding='valid',
                             padding='same',
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                             bias_initializer=tf.constant_initializer(0.05),
                             data_format='channels_last')

    if relu:
        layer = tf.nn.relu(layer)

    return layer


def create_conv_2dlayer(input,
                        filter_size,
                        num_output_channel,
                        relu=True,
                        pooling=False,
                        padding='valid',
                        d_format='channels_last'):
    layer = tf.layers.conv2d(inputs=input, filters=num_output_channel,
                             kernel_size=[filter_size, filter_size],
                             padding=padding,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                             bias_initializer=tf.constant_initializer(0.05),
                             data_format=d_format)

    if pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 1, 1, 1],
                               padding='VALID')

    if relu:
        layer = tf.nn.relu(layer)

    return layer


def create_conv_1dlayer(input,
                        filter_size,
                        num_output_channel,
                        stride=1,
                        relu=True,
                        padding='SAME'):
    layer = tf.layers.conv1d(inputs=input, filters=num_output_channel, kernel_size=[filter_size], padding='valid',
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                             bias_initializer=tf.constant_initializer(0.05))

    if relu:
        layer = tf.nn.relu(layer)

    return layer


def fully_connected_layer(input,
                          num_inputs,
                          num_outputs,
                          activation=None):
    weights = tf.get_variable('weights', shape=[num_inputs, num_outputs])
    biases = tf.get_variable('biases', shape=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if activation is not None:
        if activation == 'relu':
            layer = tf.nn.relu(layer)

        elif activation == 'softmax':
            layer = tf.nn.softmax(layer)

    return layer


def flatten_layer(layer):
    layer_shape = layer.get_shape()  # layer = [num_images, img_height, img_width, num_channels]
    num_features = layer_shape[1:].num_elements()  # Total number of elements in the network
    layer_flat = tf.reshape(layer, [-1, num_features])  # -1 means total size of dimension is unchanged

    return layer_flat, num_features


def icp3D(input,  HEIGHT, WIDTH, CHANNELS,NUM_CLASSES):

    input = tf.reshape(input,[-1, HEIGHT, WIDTH, CHANNELS ,1])
    print('\033[33m ------------------------------------------', '\033[0m')
    print('input shape: %s' % input.get_shape())
    with tf.variable_scope('ict3D'):
        with tf.variable_scope('Branch_0'):
            branch_0 = create_conv_3dlayer(input = input,
                                           filter_width= 1,
                                           filter_height= 1,
                                           filter_depth= 1,
                                           stride = 1,
                                           num_output_channels = 64,
                                           relu = True,)
            print('branch_0 shape: %s' % branch_0.get_shape())



        with tf.variable_scope('Branch_1'):
            branch_1 = create_conv_3dlayer(input = input,
                                           filter_width= 1,
                                           filter_height= 1,
                                           filter_depth= 1,
                                           stride = 1,
                                           num_output_channels = 96,
                                           relu = True)
            print('branch_1_1 shape: %s' % branch_1.get_shape())

            branch_1 = create_conv_3dlayer(input=branch_1,
                                           filter_width=3,
                                           filter_height=3,
                                           filter_depth=3,
                                           stride=1,
                                           num_output_channels=128,
                                           relu=True)
            print('branch_1_2 shape: %s' % branch_1.get_shape())

        with tf.variable_scope('Branch_2'):
            branch_2 =  create_conv_3dlayer(input = input,
                                           filter_width= 1,
                                           filter_height= 1,
                                           filter_depth= 1,
                                           stride = 1,
                                           num_output_channels = 16,
                                           relu = True)
            print('branch_2_1 shape: %s' % branch_2.get_shape())

            branch_2 =  create_conv_3dlayer(input = branch_2,
                                           filter_width= 3,
                                           filter_height= 3,
                                           filter_depth= 3,
                                           stride = 1,
                                           num_output_channels = 32,
                                           relu = True)
            print('branch_2_2 shape: %s' % branch_2.get_shape())

        with tf.variable_scope('Branch_3'):
            branch_3 = tf.nn.max_pool3d(input,ksize =[1,3,3,3,1],strides = [1,1,1,1,1],padding = 'SAME')
            print('branch_3_1 shape: %s' % branch_3.get_shape())

            branch_3 = create_conv_3dlayer(input = branch_3,
                                           filter_width= 1,
                                           filter_height= 1,
                                           filter_depth= 1,
                                           stride = 1,
                                           num_output_channels = 32,
                                           relu = True)
            print('branch_3_2 shape: %s' % branch_3.get_shape())
            print('\033[33m ------------------------------------------', '\033[0m')

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        net = slim.flatten(net, scope='flatten')
        net = slim.fully_connected(net, 16, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc3')
        net = slim.dropout(net, 0.5, scope='dropout1')
        net = slim.fully_connected(net, NUM_CLASSES, weights_regularizer=slim.l2_regularizer(0.0005),
                                   activation_fn=None, scope='out')
        """
        net,nb_feature = flatten_layer(net)
        net = fully_connected_layer(input = net,
                                    num_inputs = nb_feature,
                                    num_outputs = 16,
                                    activation = 'relu')
        net = tf.nn.dropout(net,0.5)
        net = fully_connected_layer(input=net,
                                    num_inputs=16,
                                    num_outputs=NUM_CLASSES,
                                    activation=None)
        """


        return net