import tensorflow.contrib.slim as slim
import tensorflow as tf


def inception_moudle_v1(net,scope,filters_num):
    with tf.variable_scope(scope):
        with tf.variable_scope('bh1'):
            bh1 = slim.conv2d(net,filters_num[0],1,scope='bh1_conv1_1x1')
        with tf.variable_scope('bh2'):
            bh2 = slim.conv2d(net,filters_num[1],1,scope='bh2_conv1_1x1')
            bh2 = slim.conv2d(bh2,filters_num[2],3,scope='bh2_conv2_3x3')
        with tf.variable_scope('bh3'):
            bh3 = slim.conv2d(net,filters_num[3],1,scope='bh3_conv1_1x1')
            bh3 = slim.conv2d(bh3,filters_num[4],5,scope='bh3_conv2_5x5')
        with tf.variable_scope('bh4'):
            bh4 = slim.max_pool2d(net,3,scope='bh4_max_3x3')
            bh4 = slim.conv2d(bh4,filters_num[5],1,scope='bh4_conv_1x1')
        net = tf.concat([bh1,bh2,bh3,bh4],axis=3)
    return net


##  https://github.com/shankezh/DL_HotNet_Tensorflow/blob/master/net/GoogLeNet/InceptionV1.py
def V1_slim(inputs,num_cls,is_train = True,keep_prob=0.4,spatital_squeeze=True):
    with tf.name_scope('reshape'):
        net = tf.reshape(inputs, [-1, 224, 224, 60])################################################################################9->3

    with tf.variable_scope('GoogLeNet_V1'):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_regularizer=slim.l2_regularizer(5e-4),
                weights_initializer=slim.xavier_initializer(),
        ):
            with slim.arg_scope(
                [slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                    padding='SAME',
                    stride=1,
            ):
                net = slim.conv2d(net, 64, 7, stride=2, scope='layer1')
                # net = slim.conv2d(inputs,64,7,stride=2,scope='layer1')
                net = slim.max_pool2d(net,3,stride=2,scope='layer2')
                net = tf.nn.lrn(net)
                net = slim.conv2d(net,64,1,scope='layer3')
                net = slim.conv2d(net,192,3,scope='layer4')
                net = tf.nn.lrn(net)
                net = slim.max_pool2d(net,3,stride=2,scope='layer5')
                net = inception_moudle_v1(net,'layer6',[64,96,128,16,32,32])
                net = inception_moudle_v1(net,'layer8',[128,128,192,32,96,64])
                net = slim.max_pool2d(net,3,stride=2,scope='layer10')
                net = inception_moudle_v1(net,'layer11',[192,96,208,16,48,64])
                net = inception_moudle_v1(net,'layer13',[160,112,224,24,64,64])
                net_1 = net
                net = inception_moudle_v1(net,'layer15',[128,128,256,24,64,64])
                net = inception_moudle_v1(net,'layer17',[112,144,288,32,64,64])
                net_2 = net
                net = inception_moudle_v1(net,'lauer19',[256,160,320,32,128,128])
                net = slim.max_pool2d(net,3,stride=2,scope='layer21')
                net = inception_moudle_v1(net,'layer22',[256,160,320,32,128,128])
                net = inception_moudle_v1(net,'layer24',[384,192,384,48,128,128])


                # net = slim.avg_pool2d(net,7,stride=1,padding='VALID',scope='layer26')
                net = tf.reduce_mean(net, axis=[1, 2])
                net = tf.reshape(net, [-1, 1, 1, 1024])

                net = slim.dropout(net,keep_prob=keep_prob,scope='dropout')
                net = slim.conv2d(net,num_cls,1,activation_fn=None, normalizer_fn=None,scope='layer27')     ## shape:(?,1,1,class)


                if spatital_squeeze:
                    net = tf.squeeze(net,[1,2],name='squeeze')          ## shape:(?,class)

                # net = slim.softmax(net,scope='softmax2')
                # print('out shape:%s' % net.get_shape())

                if is_train:
                    net_1 = slim.avg_pool2d(net_1, 5, padding='VALID', stride=3, scope='auxiliary0_avg')
                    net_1 = slim.conv2d(net_1, 128, 1, scope='auxiliary0_conv_1X1')
                    net_1 = slim.flatten(net_1)
                    net_1 = slim.fully_connected(net_1,1024,scope='auxiliary0_fc1')
                    net_1 = slim.dropout(net_1, keep_prob)            ## ori 0.7
                    net_1 = slim.fully_connected(net_1,num_cls,activation_fn=None,scope='auxiliary0_fc2')

                    # net_1 = slim.softmax(net_1, scope='softmax0')

                    net_2 = slim.avg_pool2d(net_2, 5, padding='VALID', stride=3, scope='auxiliary1_avg')
                    net_2 = slim.conv2d(net_2, 128, 1, scope='auxiliary1_conv_1X1')
                    net_2 = slim.flatten(net_2)
                    net_2 = slim.fully_connected(net_2,1024,scope='auxiliary1_fc1')
                    net_2 = slim.dropout(net_2, keep_prob)            ## ori 0.7
                    net_2 = slim.fully_connected(net_2,num_cls,activation_fn=None,scope='auxiliary1_fc2')

                    # net_2 = slim.softmax(net_2, scope='softmax1')

                    # net = net * 1 #+ net_2 * 0.1 + net * 0.1
                    net = net_1 * 0.3 + net_2 * 0.3 + net * 0.4     ## 我們的網路最後一層不要用softmax(會重複)

                    print(net.shape)

    return net
    # return net_2

#######################################################################################################################

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           keep_pro=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=True):          ## False     ##因為拿掉幾層 所以只能選擇True
    """Oxford Net VGG 16-Layers version D Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the outputs. Useful to remove unnecessary dimensions for classification.
      global_pool: Optional boolean flag. If True, the input to the classification layer is avgpooled to size 1x1, for any input size. (This is not part of the original VGG architecture.)

    Returns:
      end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(5e-4),      ###原5e-4
                            weights_initializer=slim.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.repeat(net, 1, slim.conv2d, 512, [3, 3], scope='conv4')          ### 原3 8層(1)極限 8層(2以上)就開始無法收斂
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            # net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # print('pool5 shape:%s' % net.get_shape())

            # Use conv2d instead of fully_connected layers.  [7*7 將最後 7*7的特徵圖變成1*1 等同於flatten!?]
            net = slim.conv2d(net, 512, [7, 7], padding=fc_conv_padding, scope='fc6')          ### ori 4096
            net = slim.dropout(net, keep_pro, is_training=is_training,
                               scope='dropout6')
            net = slim.conv2d(net, 256, [1, 1], scope='fc7')                        ### ori 4096
            print('fc7 shape:%s' % net.get_shape())

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                print('gap shape:%s' % net.get_shape())
                end_points['global_pool'] = net
            if num_classes:
                net = slim.dropout(net, keep_pro, is_training=is_training,
                                   scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='fc8')
                print('fc8 shape:%s' % net.get_shape())
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    print('out shape:%s' % net.get_shape())
                end_points[sc.name + '/fc8'] = net
            return net, end_points


#######################################################################################################################

####################################################################################################################################
'''
def alexnet_v2_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc
'''
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               keep_pro=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2',
               global_pool=False):
    """AlexNet version 2.
      Described in: http://arxiv.org/pdf/1404.5997v2.pdf
      Parameters from:
      github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
      layers-imagenet-1gpu.cfg
      Note: All the fully_connected layers have been transformed to conv2d layers.
            To use in classification mode, resize input to 224x224 or set
            global_pool=True. To use in fully convolutional mode, set
            spatial_squeeze to false.
            The LRN layers have been removed and change the initializers from
            random_normal_initializer to xavier_initializer.
      Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: the number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer are returned instead.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
          layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
          logits. Useful to remove unnecessary dimensions for classification.
        scope: Optional scope for the variables.
        global_pool: Optional boolean flag. If True, the input to the classification
          layer is avgpooled to size 1x1, for any input size. (This is not part
          of the original AlexNet.)
      Returns:
        net: the output of the logits layer (if num_classes is a non-zero integer),
          or the non-dropped-out input to the logits layer (if num_classes is 0
          or None).
        end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            biases_initializer=tf.constant_initializer(0.1),
                            weights_initializer=slim.xavier_initializer(),              ## 原本沒有
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                with slim.arg_scope([slim.max_pool2d], padding='VALID'):
                    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                        outputs_collections=[end_points_collection]):

                        net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
                        net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
                        net = slim.conv2d(net, 192, [5, 5], scope='conv2')
                        net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
                        net = slim.conv2d(net, 384, [3, 3], scope='conv3')
                        net = slim.conv2d(net, 384, [3, 3], scope='conv4')
                        net = slim.conv2d(net, 256, [3, 3], scope='conv5')
                        net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

                        net = slim.conv2d(net, 1024, [5, 5], padding='VALID', scope='fc6')  ### ori 4096
                        net = slim.dropout(net, keep_pro, is_training=is_training,
                                           scope='fc6')
                        net = slim.conv2d(net, 1024, [1, 1], scope='fc7')  ### ori 4096
                        # Convert end_points_collection into a end_point dict.
                        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                        if global_pool:
                            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                            end_points['global_pool'] = net
                        if num_classes:
                            net = slim.dropout(net, keep_pro, is_training=is_training,
                                               scope='dropout7')
                            net = slim.conv2d(net, num_classes, [1, 1],
                                              activation_fn=None,
                                              normalizer_fn=None,
                                              biases_initializer=tf.zeros_initializer,
                                              scope='fc8')
                            print(net.get_shape())
                        if spatial_squeeze:
                            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                            print(net.get_shape())
                        end_points[sc.name + '/fc8'] = net
                    return net, end_points


'''
                        # Use conv2d instead of fully_connected layers.
                        with slim.arg_scope([slim.conv2d],
                                            # weights_initializer=slim.xavier_initializer(),
                                            weights_initializer=trunc_normal(0.005),                    ### ori 0.005
                                            biases_initializer=tf.constant_initializer(0.1)):
                            net = slim.conv2d(net, 2048, [5, 5], padding='VALID', scope='fc6')          ### ori 4096
                            net = slim.dropout(net, keep_pro, is_training=is_training,
                                               scope='fc6')
                            net = slim.conv2d(net, 1024, [1, 1], scope='fc7')                        ### ori 4096
                            # Convert end_points_collection into a end_point dict.
                            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                            if global_pool:
                                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                                end_points['global_pool'] = net
                            if num_classes:
                                net = slim.dropout(net, keep_pro, is_training=is_training,
                                                   scope='dropout7')
                                net = slim.conv2d(net, num_classes, [1, 1],
                                                  activation_fn=None,
                                                  normalizer_fn=None,
                                                  biases_initializer=tf.zeros_initializer,
                                                  scope='fc8')
                                print(net.get_shape())
                            if spatial_squeeze:
                                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                                print(net.get_shape())
                            end_points[sc.name + '/fc8'] = net
                        return net, end_points
                        '''

