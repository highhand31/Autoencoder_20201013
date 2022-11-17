import tensorflow
# from inception_resnet_v1_reduction import inference as inception_resnet_v1_reduction
import numpy as np
from models_VIT import PatchEmbedding,VitLayer,VitLayer_2


#----tensorflow version check
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow as v2
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile
    # import tensorflow_addons as tfa
    # from tensorflow.keras.layers import Activation
    # from tensorflow.keras.utils import get_custom_objects
print("Tensorflow version of {}: {}".format(__file__,tf.__version__))

print_out = True
TCPConnected = False

def say_sth(msg_source, print_out=False):
    if isinstance(msg_source, str):
        msg_source = [msg_source]

    for idx, msg in enumerate(msg_source):
        if print_out:
            print(msg)

#----activations
def tf_mish(inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

#----models
def preprocess(tf_input,preprocess_dict,print_out=False):
    set_dict = {'ct_ratio': 1,'bias': 0.5,'br_ratio': 0}
    msg = 'Pre processed:'

    key_list = list(preprocess_dict.keys())

    for key,value in set_dict.items():
        if key in key_list:
            set_dict[key] = preprocess_dict[key]
        msg += '{}={},'.format(key,set_dict[key])

    # msg = "Pre processed:bias={},br_ratio={},ct_ratio={},rot={}".format(bias, br_ratio, ct_ratio, rot)
    say_sth(msg, print_out=print_out)
    pre_process = (tf_input - set_dict['bias'] * (1 - set_dict['br_ratio'])) * set_dict['ct_ratio']
    pre_process = tf.add(pre_process, set_dict['bias'] * (1 + set_dict['br_ratio']))


    return tf.clip_by_value(pre_process, 0.0, 1.0)

def Conv(input_x,filter,kernel=[3,3],stride=1,activation=tf.nn.relu,padding='same',name=None):
    net = tf.layers.conv2d(
        inputs=input_x,
        filters = filter,
        kernel_size=kernel,
        strides=[stride,stride],
        kernel_regularizer=tf.keras.regularizers.l2(0.08),
        padding=padding,
        activation=activation,
        name=name
    )
    return net

def J_block(input_x,filter,activation=tf.nn.relu):
    net = resnet_block(input_x,filters=filter,activation=activation)
    branch_1 = tf.layers.conv2d(
        inputs=net,
        filters = filter,
        kernel_size=[3,3],
        strides=[2,2],
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        padding="same",
        activation=activation
    )
    branch_2 = tf.layers.conv2d(
        inputs=net,
        filters=filter,
        kernel_size=[5, 5],
        strides=[2, 2],
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        padding="same",
        activation=activation
    )
    net = tf.concat([branch_1,branch_2],axis=-1)
    return net

# def resnet_block(input_x, k_size=3,filters=32,activation=tf.nn.relu):
#     net = tf.layers.conv2d(
#         inputs=input_x,
#         filters = filters,
#         kernel_size=[k_size,k_size],
#         kernel_regularizer=tf.keras.regularizers.l2(0.1),
#         padding="same",
#         activation=activation
#     )
#     net = tf.layers.conv2d(
#         inputs=net,
#         filters=filters,
#         kernel_size=[k_size, k_size],
#         kernel_regularizer=tf.keras.regularizers.l2(0.1),
#         padding="same",
#         activation=activation
#     )
#
#     net_1 = tf.layers.conv2d(
#         inputs=input_x,
#         filters=filters,
#         kernel_size=[k_size, k_size],
#         kernel_regularizer=tf.keras.regularizers.l2(0.1),
#         padding="same",
#         activation=activation
#     )
#
#     add = tf.add(net,net_1)
#
#     add_result = activation(add)
#
#     return add_result

def resnet_block(input_x, k_size=3,filters=32,activation=tf.nn.relu,stride=1,to_cnn_input=True):
    net = Conv(input_x,filters,k_size,activation=tf.nn.relu,padding='same')
    net = Conv(net,filters,k_size,activation=None,padding='same',stride=stride)

    if to_cnn_input:
        net_1 = Conv(input_x,filters,k_size,activation=None,padding='same',stride=stride)
    else:
        net_1 = input_x

    add = v2.add(net, net_1)

    if activation is None:
        return add
    else:
        return activation(add)

def resnet_block_reduction(tf_tensor, k_size=3,filters=32,activation=tf.nn.relu,stride=1,to_cnn_input=True):
    conv1 = Conv(tf_tensor,filters,[k_size,1],activation=None,padding='same')
    conv2 = Conv(tf_tensor,filters,[1,k_size],activation=None,padding='same')
    concat = tf.concat([conv1,conv2],axis=-1)
    net = Conv(concat, filters, [1,1], activation=activation, padding='same')

    conv1 = Conv(net, filters, [k_size, 1], activation=None, padding='same')
    conv2 = Conv(net, filters, [1, k_size], activation=None, padding='same')
    concat = tf.concat([conv1, conv2], axis=-1)
    net = Conv(concat,filters,[1,1],activation=None,padding='same',stride=stride)

    if to_cnn_input:
        conv1 = Conv(tf_tensor, filters, [k_size, 1], activation=None, padding='same')
        conv2 = Conv(tf_tensor, filters, [1, k_size], activation=None, padding='same')
        concat = tf.concat([conv1, conv2], axis=-1)
        net_1 = Conv(concat, filters, [1, 1], activation=None, padding='same', stride=stride)
        #net_1 = Conv(tf_tensor,filters,k_size,activation=None,padding='same',stride=stride)
    else:
        net_1 = tf_tensor

    add = v2.add(net, net_1)

    if activation is None:
        return add
    else:
        return activation(add)

def simple_resnet(tf_input,tf_keep_prob,embed_length):
    net = resnet_block(tf_input,k_size=3,filters=16)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2,2], strides=2)
    print("pool_1 shape:",net.shape)

    net = resnet_block(net, k_size=3, filters=32)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    print("pool_2 shape:", net.shape)

    net = resnet_block(net, k_size=3, filters=48)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    print("pool_3 shape:", net.shape)

    net = resnet_block(net, k_size=3, filters=64)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    print("pool_4 shape:", net.shape)

    #----flatten
    net = tf.layers.flatten(net)
    print("flatten shape:",net.shape)

    #----dropout
    net = tf.nn.dropout(net,keep_prob=tf_keep_prob)

    #----FC
    net = tf.layers.dense(inputs=net,units=embed_length,activation=tf.nn.relu)
    print("FC shape:",net.shape)

    #----output
    # output = tf.layers.dense(inputs=net,units=class_num,activation=None)
    # print("output shape:",output.shape)

    return net

def J_net(input_x,filter_list,tf_keep_prob,embed_length,activation=tf.nn.relu):

    net = J_block(input_x,filter_list[0],activation=activation)
    print("pool1 shape:",net.shape)

    net = J_block(net, filter_list[1], activation=activation)
    print("pool2 shape:", net.shape)

    net = J_block(net, filter_list[2], activation=activation)
    print("pool3 shape:", net.shape)

    net = J_block(net, filter_list[3], activation=activation)
    print("pool4 shape:", net.shape)

    # ----flatten
    net = tf.layers.flatten(net)
    print("flatten shape:", net.shape)

    # ----dropout
    net = tf.nn.dropout(net, keep_prob=tf_keep_prob)

    # ----FC
    net = tf.layers.dense(inputs=net, units=embed_length, activation=activation)
    print("FC shape:", net.shape)

    return net

def J_net_2(input_x,filter_list,tf_keep_prob,embed_length,activation=tf.nn.relu):

    net = resnet_block(input_x,filters=filter_list[0],activation=activation)
    branch_1 = Conv(net,filter_list[0],kernel=[3,3],stride=2,activation=activation)
    branch_2 = Conv(net,filter_list[0],kernel=[5,5],stride=2,activation=activation)
    net = tf.add(branch_1,branch_2)
    print("pool1 shape:",net.shape)

    net = resnet_block(net, filters=filter_list[1], activation=activation)
    branch_1 = Conv(net, filter_list[1], kernel=[3, 3], stride=2, activation=activation)
    branch_2 = Conv(net, filter_list[1], kernel=[5, 5], stride=2, activation=activation)
    net = tf.add(branch_1, branch_2)
    print("pool2 shape:", net.shape)

    net = resnet_block(net, filters=filter_list[2], activation=activation)
    branch_1 = Conv(net, filter_list[2], kernel=[3, 3], stride=2, activation=activation)
    branch_2 = Conv(net, filter_list[2], kernel=[5, 5], stride=2, activation=activation)
    net = tf.add(branch_1, branch_2)
    print("pool3 shape:", net.shape)

    net = resnet_block(net, filters=filter_list[3], activation=activation)
    branch_1 = Conv(net, filter_list[3], kernel=[3, 3], stride=2, activation=activation)
    branch_2 = Conv(net, filter_list[3], kernel=[5, 5], stride=2, activation=activation)
    net = tf.add(branch_1, branch_2)
    print("pool4 shape:", net.shape)

    # ----flatten
    net = tf.layers.flatten(net)
    print("flatten shape:", net.shape)

    # ----dropout
    net = tf.nn.dropout(net, keep_prob=tf_keep_prob)

    # ----FC
    net = tf.layers.dense(inputs=net, units=embed_length, activation=activation)
    print("FC shape:", net.shape)

    return net

def J_net_3(input_x,filter_list,tf_keep_prob,embed_length,activation=tf.nn.relu):

    net = resnet_block(input_x,filters=filter_list[0],activation=activation)
    net = resnet_block(net,filters=filter_list[0],activation=activation)
    branch_1 = Conv(net,filter_list[0],kernel=[3,3],stride=2,activation=activation)
    branch_2 = Conv(net,filter_list[0],kernel=[5,5],stride=2,activation=activation)
    net = tf.concat([branch_1,branch_2],axis=-1)
    print("pool1 shape:",net.shape)

    net = resnet_block(net, filters=filter_list[1], activation=activation)
    net = resnet_block(net, filters=filter_list[1], activation=activation)
    branch_1 = Conv(net, filter_list[1], kernel=[3, 3], stride=2, activation=activation)
    branch_2 = Conv(net, filter_list[1], kernel=[5, 5], stride=2, activation=activation)
    net = tf.concat([branch_1,branch_2],axis=-1)
    print("pool2 shape:", net.shape)

    net = resnet_block(net, filters=filter_list[2], activation=activation)
    net = resnet_block(net, filters=filter_list[2], activation=activation)
    branch_1 = Conv(net, filter_list[2], kernel=[3, 3], stride=2, activation=activation)
    branch_2 = Conv(net, filter_list[2], kernel=[5, 5], stride=2, activation=activation)
    net = tf.concat([branch_1,branch_2],axis=-1)
    print("pool3 shape:", net.shape)

    net = resnet_block(net, filters=filter_list[3], activation=activation)
    net = resnet_block(net, filters=filter_list[3], activation=activation)
    branch_1 = Conv(net, filter_list[3], kernel=[3, 3], stride=2, activation=activation)
    branch_2 = Conv(net, filter_list[3], kernel=[5, 5], stride=2, activation=activation)
    net = tf.concat([branch_1,branch_2],axis=-1)
    print("pool4 shape:", net.shape)

    # ----flatten
    net = tf.layers.flatten(net)
    print("flatten shape:", net.shape)

    # ----dropout
    net = tf.nn.dropout(net, keep_prob=tf_keep_prob)

    # ----FC
    net = tf.layers.dense(inputs=net, units=embed_length, activation=activation)
    print("FC shape:", net.shape)

    return net

def resnet_14(input_x,filter_list,tf_keep_prob,embed_length,activation=tf.nn.relu):

    net = tf.layers.conv2d(
        inputs=input_x,
        filters=filter_list[0],
        kernel_size=[7, 7],
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        padding="same",
        strides=2,
        activation=activation
    )
    #----shape learning
    # net = tf.layers.average_pooling2d(net, pool_size=[2, 2], strides=1, padding='same')
    # net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=1, padding='same')
    # net = tf.layers.average_pooling2d(net, pool_size=[2, 2], strides=2, )

    #----texture learning
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    msg = "pool1 shape = {}".format(net.shape)
    say_sth(msg, print_out=print_out)

    net = resnet_block(net, k_size=3, filters=filter_list[1])
    net = resnet_block(net, k_size=3, filters=filter_list[1])
    # ----shape learning
    # net = tf.layers.average_pooling2d(net, pool_size=[2, 2], strides=1, padding='same')
    # net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=1, padding='same')
    # net = tf.layers.average_pooling2d(net, pool_size=[2, 2], strides=2, )

    # ----texture learning
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    print("pool_2 shape:", net.shape)

    net = resnet_block(net, k_size=3, filters=filter_list[2])
    net = resnet_block(net, k_size=3, filters=filter_list[2])
    # ----shape learning
    # net = tf.layers.average_pooling2d(net, pool_size=[2, 2], strides=1, padding='same')
    # net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=1, padding='same')
    # net = tf.layers.average_pooling2d(net, pool_size=[2, 2], strides=2, )

    # ----texture learning
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    print("pool_3 shape:", net.shape)

    # net = self.resnet_block(net, k_size=3, filters=filter_list[3])
    # net = self.resnet_block(net, k_size=3, filters=filter_list[3])
    # net = tf.layers.average_pooling2d(net, pool_size=[2, 2], strides=1, padding='same')
    # net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=1, padding='same')
    # net = tf.layers.average_pooling2d(net, pool_size=[2, 2], strides=2, )
    # print("pool_4 shape:", net.shape)

    # net = self.resnet_block(net, k_size=3, filters=filter_list[4])
    # net = self.resnet_block(net, k_size=3, filters=filter_list[4])
    # net = tf.layers.average_pooling2d(net, pool_size=[2, 2], strides=1, padding='same')
    # net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=1, padding='same')
    # net = tf.layers.average_pooling2d(net, pool_size=[2, 2], strides=2, )
    # print("pool_5 shape:", net.shape)

    # ----flatten
    net = tf.layers.flatten(net)
    print("flatten shape:", net.shape)

    # ----dropout
    net = tf.nn.dropout(net, keep_prob=tf_keep_prob)

    # ----FC
    net = tf.layers.dense(inputs=net, units=embed_length, activation=tf.nn.relu)
    print("FC shape:", net.shape)

    return net

def resnet_pyramid(tf_input,filter_list,tf_keep_prob,embed_length,activation=tf.nn.relu):


    net_1 = resnet_block(tf_input,k_size=3,filters=filter_list[0],activation=activation)
    net_1 = resnet_block(net_1, k_size=3, filters=filter_list[0], activation=activation)
    net_1 = resnet_block(net_1, k_size=3, filters=filter_list[0], activation=activation)
    net_1 = resnet_block(net_1, k_size=3, filters=filter_list[0], activation=activation)
    net_1 = tf.layers.max_pooling2d(inputs=net_1, pool_size=[2,2], strides=2,padding='same')

    net_2 = resnet_block(tf_input, k_size=3, filters=filter_list[0],activation=activation)
    net_2 = resnet_block(net_2, k_size=3, filters=filter_list[0], activation=activation)
    net_2 = resnet_block(net_2, k_size=3, filters=filter_list[0], activation=activation)
    net_2 = tf.layers.max_pooling2d(inputs=net_2, pool_size=[2, 2], strides=2,padding='same')

    net_3 = resnet_block(tf_input, k_size=3, filters=filter_list[0],activation=activation)
    net_3 = resnet_block(net_3, k_size=3, filters=filter_list[0], activation=activation)
    net_3 = tf.layers.max_pooling2d(inputs=net_3, pool_size=[2, 2], strides=2,padding='same')

    net_4 = resnet_block(tf_input, k_size=3, filters=filter_list[0],activation=activation)
    net_4 = tf.layers.max_pooling2d(inputs=net_4, pool_size=[2, 2], strides=2,padding='same')

    net = tf.concat([net_1,net_2,net_3,net_4],axis=-1)
    print("pool_1 shape:",net.shape)

    net_2 = resnet_block(net, k_size=3, filters=filter_list[1], activation=activation)
    net_2 = resnet_block(net_2, k_size=3, filters=filter_list[1], activation=activation)
    net_2 = resnet_block(net_2, k_size=3, filters=filter_list[1], activation=activation)
    net_2 = tf.layers.max_pooling2d(inputs=net_2, pool_size=[2, 2], strides=2, padding='same')

    net_3 = resnet_block(net, k_size=3, filters=filter_list[1], activation=activation)
    net_3 = resnet_block(net_3, k_size=3, filters=filter_list[1], activation=activation)
    net_3 = tf.layers.max_pooling2d(inputs=net_3, pool_size=[2, 2], strides=2, padding='same')

    net_4 = resnet_block(net, k_size=3, filters=filter_list[1], activation=activation)
    net_4 = tf.layers.max_pooling2d(inputs=net_4, pool_size=[2, 2], strides=2, padding='same')

    net = tf.concat([net_2,net_3,net_4],axis=-1)
    print("pool_2 shape:", net.shape)

    net_3 = resnet_block(net, k_size=3, filters=filter_list[2], activation=activation)
    net_3 = resnet_block(net_3, k_size=3, filters=filter_list[2], activation=activation)
    net_3 = tf.layers.max_pooling2d(inputs=net_3, pool_size=[2, 2], strides=2, padding='same')

    net_4 = resnet_block(net, k_size=3, filters=filter_list[2], activation=activation)
    net_4 = tf.layers.max_pooling2d(inputs=net_4, pool_size=[2, 2], strides=2, padding='same')

    net = tf.concat([net_3, net_4], axis=-1)
    print("pool_3 shape:", net.shape)

    net = resnet_block(net, k_size=3, filters=filter_list[3],activation=activation)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2,padding='same')

    print("pool_4 shape:", net.shape)

    # ----flatten
    net = tf.layers.flatten(net)
    print("flatten shape:", net.shape)

    # ----dropout
    net = tf.nn.dropout(net, keep_prob=tf_keep_prob)

    # ----FC
    net = tf.layers.dense(inputs=net, units=embed_length, activation=tf.nn.relu)
    print("FC shape:", net.shape)

    return net

def resnet_pyramid_2(tf_input,filter_list,tf_keep_prob,embed_length,activation=tf.nn.relu):

        net = tf.layers.conv2d(
            inputs=tf_input,
            filters=filter_list[0],
            kernel_size=[3, 3],
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            padding="same",
            strides=2,
            activation=activation
        )


        # net_1 = self.resnet_block(tf_input,k_size=3,filters=filter_list[0],activation=activation)
        # net_1 = self.resnet_block(net_1, k_size=3, filters=filter_list[0], activation=activation)
        # net_1 = self.resnet_block(net_1, k_size=3, filters=filter_list[0], activation=activation)
        # net_1 = self.resnet_block(net_1, k_size=3, filters=filter_list[0], activation=activation)
        # net_1 = tf.layers.max_pooling2d(inputs=net_1, pool_size=[2,2], strides=2,padding='same')

        net_2 = resnet_block(net, k_size=3, filters=filter_list[0],activation=activation)
        net_2 = resnet_block(net_2, k_size=3, filters=filter_list[0], activation=activation)
        net_2 = resnet_block(net_2, k_size=3, filters=filter_list[0], activation=activation)

        net_3 = resnet_block(net, k_size=3, filters=filter_list[0],activation=activation)
        net_3 = resnet_block(net_3, k_size=3, filters=filter_list[0], activation=activation)

        net_4 = resnet_block(net, k_size=3, filters=filter_list[0],activation=activation)

        net = tf.concat([net_2,net_3,net_4],axis=-1)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        print("pool_1 shape:",net.shape)

        # net_2 = self.resnet_block(net, k_size=3, filters=filter_list[1], activation=activation)
        # net_2 = self.resnet_block(net_2, k_size=3, filters=filter_list[1], activation=activation)
        # net_2 = self.resnet_block(net_2, k_size=3, filters=filter_list[1], activation=activation)
        # net_2 = tf.layers.max_pooling2d(inputs=net_2, pool_size=[2, 2], strides=2, padding='same')

        net_3 = resnet_block(net, k_size=3, filters=filter_list[1], activation=activation)
        net_3 = resnet_block(net_3, k_size=3, filters=filter_list[1], activation=activation)

        net_4 = resnet_block(net, k_size=3, filters=filter_list[1], activation=activation)

        net = tf.concat([net_3,net_4],axis=-1)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        print("pool_2 shape:", net.shape)

        net = resnet_block(net, k_size=3, filters=filter_list[2],activation=activation)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

        print("pool_3 shape:", net.shape)

        # ----flatten
        net = tf.layers.flatten(net)
        print("flatten shape:", net.shape)

        # ----dropout
        net = tf.nn.dropout(net, keep_prob=tf_keep_prob)

        # ----FC
        net = tf.layers.dense(inputs=net, units=embed_length, activation=tf.nn.relu)
        print("FC shape:", net.shape)

        return net

def re_conv4unet(tf_input, kernel_list, filter_list,pool_kernel=2,activation=tf.nn.relu,pool_type=None,rot=False,
            print_out=False):
    #----var
    pool_kernel = [1, pool_kernel, pool_kernel, 1]
    conv_times = len(filter_list)
    msg_list = list()
    conv_list = []
    # ----pool process
    if pool_type is not None:
        if pool_type == 'max':
            tf_pool = tf.nn.max_pool
        elif pool_type == 'ave':
            tf_pool = tf.nn.avg_pool
        elif pool_type == 'cnn':
            pass
        else:
            pool_type = None

    msg = '----Repeat Conv for Unet with pool type {}, kernel {}----'.format(pool_type,pool_kernel[1])
    msg_list.append(msg)
    # say_sth(msg, print_out=print_out)

    #----repeat Conv
    for i in range(conv_times):
        #----normal cnn
        if i == 0:
            net = Conv(tf_input, filter_list[i], kernel=kernel_list[i], activation=activation)
            if rot is True:
                net_1 = rot_cnn(tf_input, filter_list[i], kernel_list[i], activation=activation)
                net = tf.concat([net, net_1], axis=-1)
            net = Conv(net, filter_list[i], kernel=kernel_list[i], activation=None)
            conv_list.append(net)
            net = activation(net)
        else:
            if rot is True:
                net_1 = rot_cnn(net, filter_list[i], kernel_list[i], activation=activation)
            net = Conv(net, filter_list[i], kernel=kernel_list[i], activation=activation)

            if rot is True:
                net = tf.concat([net, net_1], axis=-1)

            net = Conv(net, filter_list[i], kernel=kernel_list[i], activation=None)
            # if i != (conv_times - 1):
            conv_list.append(net)
            net = activation(net)

        #----rot cnn
        # if rot is True:
        #     if i == 0:
        #         net_1 = rot_cnn(tf_input,filter_list[i],kernel_list[i],activation=activation)
        #     else:
        #         net_1 = rot_cnn(net,filter_list[i],kernel_list[i],activation=activation)
        #     net = tf.concat([net,net_1],axis=-1)

        #----pooling
        if pool_type is not None:
            if pool_type == 'cnn':
                net = Conv(net, filter_list[i], kernel=pool_kernel[1:3],stride=2,activation=activation)
            else:
                net = tf_pool(net, ksize=pool_kernel, strides=[1, 2, 2, 1], padding='SAME')


        msg = "encode_{} shape = {}".format(i + 1, net.shape)
        msg_list.append(msg)
        # say_sth(msg, print_out=print_out)

    #----display
    say_sth(msg_list,print_out=print_out)

    return net,conv_list

def re_conv(tf_input, kernel_list, filter_list,pool_kernel=2,activation=tf.nn.relu,pool_type=None,rot=False,
            stride_list=None,print_out=False,to_reduce=False,return_nodes=False):
    #----var
    pool_kernel = [1, pool_kernel, pool_kernel, 1]
    msg_list = list()
    node_list = []
    # ----pool process
    if pool_type is not None:
        if pool_type == 'max':
            tf_pool = tf.nn.max_pool
        elif pool_type == 'ave':
            tf_pool = tf.nn.avg_pool
        elif pool_type == 'cnn':
            pass
        else:
            pool_type = None

    msg = '----Repeat Conv with pool type {}, kernel {}----'.format(pool_type,pool_kernel[1])
    msg_list.append(msg)
    # say_sth(msg, print_out=print_out)
    if isinstance(stride_list, list):
        pass
    else:
        stride_list = [2] * len(filter_list)

    #----repeat Conv
    for i in range(len(filter_list)):
        if pool_type == 'cnn':
            stride = stride_list[i]
        else:
            stride = 1
        #----normal cnn
        if i == 0:
            if to_reduce is True:
                conv1 = Conv(tf_input, filter_list[i], kernel=[kernel_list[i],1],
                             stride=stride, activation=None)
                conv2 = Conv(tf_input, filter_list[i], kernel=[1,kernel_list[i]],
                             stride=stride, activation=None)
                concat = tf.concat([conv1,conv2],axis=-1)
                net = Conv(concat, filter_list[i], kernel=[1,1], activation=activation)
            else:
                net = Conv(tf_input, filter_list[i], kernel=kernel_list[i],
                           stride=stride,activation=None)
                net = activation(net)
                #----append nodes
                if return_nodes:
                    node_list.append(net)


            if rot is True:
                net_1 = rot_cnn(tf_input, filter_list[i], kernel_list[i],
                                stride=stride,activation=activation)
                net = tf.concat([net, net_1], axis=-1)
        else:
            if rot is True:
                net_1 = rot_cnn(net, filter_list[i], kernel_list[i],
                                stride=stride, activation=activation)

            if to_reduce is True:
                conv1 = Conv(net, filter_list[i], kernel=[kernel_list[i], 1],
                             stride=stride, activation=None)
                conv2 = Conv(net, filter_list[i], kernel=[1, kernel_list[i]],
                             stride=stride, activation=None)
                concat = tf.concat([conv1, conv2], axis=-1)
                net = Conv(concat, filter_list[i], kernel=[1, 1], activation=activation)
            else:
                net = Conv(net, filter_list[i], kernel=kernel_list[i],
                           stride=stride, activation=None)
                # ----append nodes
                if return_nodes:
                    node_list.append(net)

                net = activation(net)

            if rot is True:
                net = tf.concat([net, net_1], axis=-1)



        #----pooling
        if pool_type == 'cnn':
            pass
        else:
            net = tf_pool(net, ksize=pool_kernel, strides=[1, stride_list[i], stride_list[i], 1], padding='SAME')



        #----msg
        msg = "encode_{} shape = {}".format(i + 1, net.shape)
        msg_list.append(msg)
        # say_sth(msg, print_out=print_out)

    #----display
    say_sth(msg_list,print_out=print_out)

    if return_nodes:
        return net,node_list
    else:
        return net

def first_layer_process(tf_tensor,data_dict):
    if isinstance(data_dict.get('first_layer'),dict):
        dict_first_layer = data_dict.get('first_layer')
        do_type = dict_first_layer.get('type')
        if do_type == 'resize':
            size = dict_first_layer.get('size')  # (h,w)
            tf_input_layer = v2.image.resize(tf_tensor, size,method='nearest')
        elif do_type == 'CNN_downSampling':
            filter = dict_first_layer.get('filter')
            kernel = dict_first_layer.get('kernel')
            net = v2.keras.layers.Conv2D(filters=filter, kernel_size=kernel, strides=2,padding='same')(
                tf_tensor)
            tf_input_layer = tf.nn.relu(net)
        elif do_type == 'mixed_downSampling':
            filter = dict_first_layer.get('filter')
            kernel = dict_first_layer.get('kernel')
            pool_type_list = dict_first_layer.get('pool_type_list')
            if pool_type_list is None:
                pool_type_list = ['max','ave','cnn']
            net_list = []
            for pool_type in pool_type_list:
                net = None
                if pool_type == 'max':
                    net = v2.keras.layers.MaxPool2D(pool_size=kernel,strides=2,padding='same')(tf_tensor)
                elif pool_type == 'ave':
                    net = v2.keras.layers.AveragePooling2D(pool_size=kernel,strides=2,padding='same')(tf_tensor)
                elif pool_type == 'cnn':
                    net = v2.keras.layers.Conv2D(filters=filter, kernel_size=kernel, strides=2,padding='same')(
                tf_tensor)
                if net is not None:
                    net_list.append(net)
            concat = v2.concat(net_list,axis=-1)
            net = v2.keras.layers.Conv2D(filters=filter, kernel_size=1, strides=1, padding='same')(
                concat)
            tf_input_layer = tf.nn.relu(net)
        elif do_type == 'dilated_downSampling':
            dilated_ratios = dict_first_layer.get('ratio')
            kernel = dict_first_layer.get('kernel')
            filter = dict_first_layer.get('filter')
            if dilated_ratios is None:
                dilated_ratios = [1,2]
            net_list = []
            for ratio in dilated_ratios:
                net = v2.keras.layers.Conv2D(filters=filter, kernel_size=kernel, strides=1,
                                             padding='same',dilation_rate=ratio)(tf_tensor)
                net_list.append(tf.nn.relu(net))
            tf_input_layer = v2.concat(net_list, axis=-1)
        elif do_type == 'patch_embedding':
            patch_size = dict_first_layer.get('patch_size')
            filter = dict_first_layer.get('filter')
            tf_input_layer = v2.keras.layers.Conv2D(filters=filter, kernel_size=patch_size, strides=patch_size)(
                tf_tensor)
        elif do_type == 'resize_patch_embedding':
            patch_size = dict_first_layer.get('patch_size')
            filter = dict_first_layer.get('filter')
            net_patch = v2.keras.layers.Conv2D(filters=filter, kernel_size=patch_size, strides=patch_size)(
                tf_tensor)

            size = dict_first_layer['size']
            net_resize = v2.image.resize(tf_tensor, size)
            tf_input_layer = v2.concat([net_patch, net_resize], axis=-1)
        elif do_type == 'resize_maxpool':
            size = dict_first_layer.get('size')  # (h,w)
            net_resize = v2.image.resize(tf_tensor, size)
            net_maxpool = tf.nn.max_pool(tf_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            tf_input_layer = v2.concat([net_maxpool, net_resize], axis=-1)
        else:
            tf_input_layer = tf_tensor
            do_type = 'do nothing'
    else:
        tf_input_layer = tf_tensor
        do_type = 'do nothing'

    return tf_input_layer,do_type

def cnn_transformer(tf_input, kernel_list, filter_list,stride_list,pool_kernel=2,drop_rate=0.2,activation=tf.nn.relu,
                    pool_type=None,to_Vit=False,print_out=False):

    #----var
    pool_kernel = [1, pool_kernel, pool_kernel, 1]
    msg_list = list()

    #----pool process
    if pool_type is not None:
        if pool_type == 'max':
            tf_pool = tf.nn.max_pool
        elif pool_type == 'ave':
            tf_pool = tf.nn.avg_pool
        elif pool_type == 'cnn':
            pass
        else:
            pool_type = None

    msg = '----Repeat CNN_transformer with pool type {}, kernel {}----'.format(pool_type, pool_kernel[1])
    msg_list.append(msg)

    #----repeat CNN_transformer
    for i in range(len(filter_list)):
        if i == 0:
            data_input = tf_input
        else:
            data_input = net

        net = Conv(data_input, filter_list[i], kernel=kernel_list[i], activation=activation)
        #----down sampling
        if pool_type is not None:
            if pool_type == 'cnn':
                net = Conv(net, filter_list[i], kernel=pool_kernel[1:3], stride=stride_list[i], activation=activation)
            else:
                net = tf_pool(net, ksize=pool_kernel, strides=[1, 2, 2, 1], padding='SAME')
            msg = "CNN encode_{} shape = {}".format(i + 1, net.shape)
            msg_list.append(msg)
        #----transformer
        if to_Vit is True:
            patch_size = [stride_list[i],stride_list[i]]
            net_trans = VitLayer_2(data_input,hidden_size=filter_list[i],patch_size=patch_size,drop_rate=drop_rate)
            msg = "transformer encode_{} shape = {}".format(i + 1, net_trans.shape)
            msg_list.append(msg)

            #----concat
            net = v2.concat([net,net_trans],axis=-1)

    # ----display
    say_sth(msg_list, print_out=print_out)

    return net

def Seg_DifNet(tf_input, tf_input_2, kernel_list, filter_list,pool_kernel_list=2,activation=tf.nn.relu,
                   pool_type_list=None,rot=False,print_out=False,preprocess_dict=None,class_num=3):
    transpose_filter = [1, 1]
    msg_list = list()

    # ----filters vs rot cnn
    if rot is True:
        filter_list = filter_list // 2
        # print("rot is True")

    say_sth('----Seg_DifNet----', print_out=print_out)
    # net = tf.math.abs(tf_input,tf_input_2)#tf.math無法接受有None的4維Tensor
    # net = tf.sqrt(net)#無法進行最佳化

    net = tf.subtract(tf_input,tf_input_2,name='diff')
    net = tf.square(net)
    net = tf.concat([tf_input,tf_input_2,net], axis=-1)
    # net = tf.concat([tf_input,tf_input_2], axis=-1)


    for i,pool_type in enumerate(pool_type_list):
        net = re_conv(net, kernel_list, filter_list, pool_kernel=pool_kernel_list[i], pool_type=pool_type,
                      activation=tf.nn.relu,rot=rot,print_out=print_out)

    for i, filers in enumerate(filter_list[::-1]):
        kernel = kernel_list[::-1][i]
        decode = tf.layers.conv2d_transpose(net, filers, transpose_filter, strides=2, padding='same')

        net = Conv(decode, filers, kernel=kernel, activation=activation)
        if rot is True:
            net_1 = rot_cnn(decode, filers, kernel, activation=activation)
            net = tf.concat([net, net_1], axis=-1)

        msg = "decode_{} shape = {}".format(i + 1, net.shape)
        msg_list.append(msg)

    net = Conv(net, class_num, kernel=kernel_list[0], padding="same", activation=None)  # name='output_AE'
    msg = 'output shape = {}'.format(net.shape)


    say_sth(msg_list, print_out=print_out)

    return net

def Seg_DifNet_V2(tf_input,tf_input_2,encode_dict,decode_dict,class_num=3,print_out=False):
    transpose_filter = [1, 1]
    msg_list = list()
    net_list = list()
    transpose_list = list()

    say_sth('----Seg_DifNet V2----', print_out=print_out)

    net = tf.subtract(tf_input,tf_input_2,name='diff')
    net = tf.square(net)
    net = tf.math.sqrt(net)
    net = tf.concat([tf_input,tf_input_2,net], axis=-1)

    # ----first layer process
    tf_input_layer,do_type = first_layer_process(net,encode_dict)
    msg = "First layer process: {} with shape = {}".format(do_type, tf_input_layer.shape)
    say_sth(msg, print_out=print_out)

    #--------Encode--------
    pool_type_list = encode_dict['pool_type_list']
    filter_list = encode_dict['filter_list']
    kernel_list = encode_dict['kernel_list']
    pool_kernel_list = encode_dict['pool_kernel_list']
    stride_list = encode_dict['stride_list']
    activation = encode_dict.get('activation')
    rot = encode_dict.get('rot')
    if activation is None:
        activation = tf.nn.relu

    # ----filters vs pooling times
    filter_list = np.array(filter_list) // len(pool_type_list)
    if rot:
        filter_list = filter_list // 2

    for i, pool_type in enumerate(pool_type_list):
        net = re_conv(tf_input_layer, kernel_list, filter_list, pool_kernel=pool_kernel_list[i], pool_type=pool_type,
                      stride_list=stride_list, activation=activation, rot=rot,
                      print_out=print_out)
        net_list.append(net)

    #--------Decode--------
    cnn_type = decode_dict['cnn_type']
    filter_list = decode_dict['filter_list']
    kernel_list = decode_dict['kernel_list']
    stride_list = decode_dict['stride_list']
    activation = decode_dict.get('activation')
    if activation is None:
        activation = tf.nn.relu

    # ----filters vs pooling times
    filter_list = np.array(filter_list) // len(pool_type_list)

    for net in net_list:
        for i, filters in enumerate(filter_list):
            if isinstance(stride_list, list):
                stride = stride_list[i]
            else:
                stride = 2
            decode = tf.layers.conv2d_transpose(net, filters, transpose_filter, strides=stride, padding='same')

            if cnn_type == 'resnet':
                net = resnet_block(decode, k_size=kernel_list[i], filters=filters,
                                       activation=activation, stride=1, to_cnn_input=False)
            else:
                net = Conv(decode, filters, kernel=kernel_list[i], activation=activation)
                if rot is True:
                    net_1 = rot_cnn(decode, filters, kernel_list[i], activation=activation)
                    net = tf.concat([net, net_1], axis=-1)

            msg = "decode_{} shape = {}".format(i + 1, net.shape)
            msg_list.append(msg)

        transpose_list.append(net)

    concat = tf.concat(transpose_list, axis=-1)
    msg = "Decode concat shape = {}".format(concat.shape)
    msg_list.append(msg)

    if cnn_type == 'resnet':
        net = resnet_block(concat, k_size=kernel_list[i], filters=filters,
                           activation=activation, to_cnn_input=True)
        net = resnet_block(net, k_size=kernel_list[i], filters=class_num,
                           activation=None, to_cnn_input=True)
    else:
        net = Conv(concat, filters, kernel=kernel_list[i], activation=activation)
        net = Conv(net, class_num, kernel=kernel_list[i], padding="same", activation=None)  # name='output_AE'

    msg = "output shape = {}".format(net.shape)
    msg_list.append(msg)

    say_sth(msg_list, print_out=print_out)

    return net

def Seg_pooling_net_V7(tf_input,tf_input_2,encode_dict,decode_dict,out_channel=3,print_out=False):

    #----var
    net_list = list()
    transpose_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

    # ----dummy output
    pre_embeddings = tf.layers.flatten(tf_input_2)
    embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='dummy_out')

    #----first layer process
    tf_input_layer, do_type = first_layer_process(tf_input, encode_dict)
    msg = "First layer process: {} with shape = {}".format(do_type, tf_input_layer.shape)
    say_sth(msg, print_out=print_out)

    pool_type_list = encode_dict['pool_type_list']
    filter_list = encode_dict['filter_list']
    kernel_list = encode_dict['kernel_list']
    pool_kernel_list = encode_dict['pool_kernel_list']
    stride_list = encode_dict['stride_list']
    activation = encode_dict.get('activation')
    multi_ratio = encode_dict.get('multi_ratio')
    layer_list = encode_dict.get('layer_list')

    if activation is None:
        activation = tf.nn.relu
    if layer_list is None:
        layer_list = [5]

    if multi_ratio is not None:
        filter_list = np.array(filter_list) * multi_ratio
        filter_list = filter_list.astype(np.int16)

    #----filters vs pooling times
    filter_list = np.array(filter_list) // len(pool_type_list)
    for layer in layer_list:
        for i, pool_type in enumerate(pool_type_list):
            net = re_conv(tf_input_layer, kernel_list[:layer], filter_list[:layer], pool_kernel=pool_kernel_list[i],
                          pool_type=pool_type,stride_list=stride_list[:layer],
                          activation=activation, rot=False,
                          return_nodes=False,print_out=print_out)
            net_list.append(net)

    # -----------------------------------------------------------------------
    # --------Decode--------
    # -----------------------------------------------------------------------
    cnn_type = decode_dict['cnn_type']
    filter_list = decode_dict['filter_list']
    kernel_list = decode_dict['kernel_list']
    stride_list = decode_dict['stride_list']
    activation = decode_dict.get('activation')
    multi_ratio = decode_dict.get('multi_ratio')
    if activation is None:
        activation = tf.nn.relu

    if multi_ratio is not None:
        filter_list = np.array(filter_list) * multi_ratio
        filter_list = filter_list.astype(np.int16)

    # ----filters vs pooling times
    filter_list = np.array(filter_list) // len(pool_type_list)
    for idx_layer,layer in enumerate(layer_list):
        msg = "Decode of layer number:{}".format(layer)
        msg_list.append(msg)

        for idx_pool_type,pool_type in enumerate(pool_type_list):
            net = net_list[len(pool_type_list)*idx_layer + idx_pool_type]
            for i in range(layer):
                idx = -layer + i
                filters = filter_list[idx]
                kernel = kernel_list[idx]
                if isinstance(stride_list, list):
                    stride = stride_list[idx]
                else:
                    stride = 2
                decode = tf.layers.conv2d_transpose(net, filters, transpose_filter, strides=stride, padding='same')

                if cnn_type == 'resnet':
                    net = resnet_block(decode, k_size=kernel_list[i], filters=filters,
                                       activation=activation, stride=1, to_cnn_input=False)
                else:
                    net = Conv(decode, filters, kernel=kernel, activation=None)
                    net = activation(net)

                msg = "decode_{} shape = {}".format(i + 1, net.shape)
                msg_list.append(msg)

            transpose_list.append(net)

    concat = tf.concat(transpose_list, axis=-1)
    msg = "concat shape = {}".format(concat.shape)
    msg_list.append(msg)

    if cnn_type == 'resnet':
        # net = resnet_block(concat, k_size=kernel, filters=filters,
        #                    activation=activation,to_cnn_input=True)
        net = resnet_block(concat, k_size=kernel, filters=out_channel,
                           activation=None,to_cnn_input=True)
    else:
        # net = Conv(concat, filters, kernel=kernel_list[i], activation=activation)
        net = Conv(concat, out_channel, kernel=kernel, padding="same", activation=None)  # name='output_AE'

    msg = "output shape = {}".format(net.shape)
    msg_list.append(msg)


    say_sth(msg_list,print_out=print_out)

    return net

def Seg_pooling_net_V8(tf_input,encode_dict,decode_dict,out_channel=3,print_out=False):

    '''
    adopted from Seg_pooling_net_V4
    '''
    #----var
    net_list = list()
    transpose_list = list()
    cnn_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

    #----to grayscal
    batch_image = v2.image.rgb_to_grayscale(tf_input)

    #----first layer process
    tf_input_layer, do_type = first_layer_process(batch_image, encode_dict)
    # tf_input_layer, do_type = first_layer_process(tf.image.rgb_to_grayscale(tf_input), encode_dict)
    msg = "First layer process: {} with shape = {}".format(do_type, tf_input_layer.shape)
    say_sth(msg, print_out=print_out)

    pool_type_list = encode_dict['pool_type_list']
    filter_list = np.array(encode_dict['filter_list'])
    kernel_list = encode_dict['kernel_list']
    pool_kernel_list = encode_dict['pool_kernel_list']
    stride_list = encode_dict['stride_list']
    activation = get_activation(encode_dict.get('activation'))
    multi_ratio = encode_dict.get('multi_ratio')


    if isinstance(multi_ratio,(int,float)):
        filter_list = filter_list * multi_ratio
        filter_list = filter_list.astype(np.int16)

    #----filters vs pooling times
    pool_times = len(pool_type_list)
    if "resize" in pool_type_list:
        pool_times -= 1
    filter_list = filter_list // pool_times

    #----
    layer_num = 0
    for kernel,filters,strides,pool_kernel in zip(kernel_list,filter_list,stride_list,pool_kernel_list):
        if layer_num == 0:
            data_input = tf_input_layer
        else:
            data_input = net

        net = v2.keras.layers.Conv2D(filters,kernel,strides=1,padding='same')(data_input)
        cnn_list.append(net)
        net = activation(net)

        pool_list = []
        for pool_type in pool_type_list:
            if pool_type == 'max':
                net_temp = v2.keras.layers.MaxPool2D(pool_size=pool_kernel,strides=strides,padding='same')(net)
            elif pool_type == 'ave':
                net_temp = v2.keras.layers.AveragePooling2D(pool_size=pool_kernel,strides=strides,padding='same')(net)
            elif pool_type == 'cnn':
                net_temp = v2.keras.layers.Conv2D(filters,kernel,strides=strides,padding='same')(net)
                net_temp = activation(net_temp)
            elif pool_type == "resize":
                h_now = batch_image.shape[1].value
                w_now = batch_image.shape[2].value
                batch_image = v2.image.resize(batch_image, (h_now//2,w_now//2), method='nearest')
                net_temp = batch_image
            else:
                net_temp = None

            if net_temp is not None:
                pool_list.append(net_temp)

        net = v2.concat(pool_list,axis=-1)
        layer_num += 1

        msg_list.append("encode_{} shape: {}".format(layer_num,net.shape))

    # -----------------------------------------------------------------------
    # --------Decode--------
    # -----------------------------------------------------------------------
    cnn_type = decode_dict['cnn_type']
    filter_list = np.array(decode_dict['filter_list'])
    kernel_list = decode_dict['kernel_list']
    stride_list = decode_dict['stride_list']
    activation = get_activation(decode_dict.get('activation'))
    multi_ratio = decode_dict.get('multi_ratio')


    if isinstance(multi_ratio,(int,float)):
        filter_list *= multi_ratio
        filter_list = filter_list.astype(np.int16)

    # ----filters vs pooling times
    filter_list = filter_list // pool_times

    layer_num = 0
    for kernel, filters, strides in zip(kernel_list, filter_list, stride_list):
        # decode = tf.layers.conv2d_transpose(net, filters, transpose_filter, strides=strides, padding='same')
        decode = v2.keras.layers.Conv2DTranspose(filters,1,strides=strides,padding='same')(net)

        if len(cnn_list) == len(kernel_list):
            decode = v2.add(decode,cnn_list[::-1][layer_num])

        if cnn_type == 'resnet':
            net = resnet_block(decode, k_size=kernel, filters=filters,
                               activation=activation, stride=1, to_cnn_input=False)
        else:
            net = v2.keras.layers.Conv2D(filters,kernel,activation=activation,
                                         strides=1,padding='same')(decode)

        layer_num += 1

        msg_list.append("decode_{} shape: {}".format(layer_num, net.shape))

    if cnn_type == 'resnet':
        net = resnet_block(net, k_size=kernel, filters=filters,
                           activation=activation,to_cnn_input=True)
        net = resnet_block(net, k_size=kernel, filters=out_channel,
                           activation=None,to_cnn_input=True)
    else:
        net = Conv(net, filters, kernel=kernel, activation=activation)
        net = Conv(net, out_channel, kernel=kernel, padding="same", activation=None)  # name='output_AE'

    msg = "output shape = {}".format(net.shape)
    msg_list.append(msg)


    say_sth(msg_list,print_out=print_out)

    return net

def Seg_pooling_net_V9(tf_input,tf_input_standard,encode_dict,decode_dict,out_channel=3,to_reduce=False,print_out=False):

    '''
    adopted from Seg_pooling_net_V4
    '''
    #----var
    net_list = list()
    transpose_list = list()
    cnn_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

    #----dummy output
    # pre_embeddings = tf.layers.flatten(tf_input_2)
    # embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='dummy_out')


    #----concat
    input_x = tf.concat([tf_input,tf_input_standard],axis=-1)

    #----first layer process
    tf_input_layer, do_type = first_layer_process(input_x, encode_dict)
    # tf_input_layer, do_type = first_layer_process(tf.image.rgb_to_grayscale(tf_input), encode_dict)
    msg = "First layer process: {} with shape = {}".format(do_type, tf_input_layer.shape)
    say_sth(msg, print_out=print_out)

    pool_type_list = encode_dict['pool_type_list']
    filter_list = encode_dict['filter_list']
    kernel_list = encode_dict['kernel_list']
    pool_kernel_list = encode_dict['pool_kernel_list']
    stride_list = encode_dict['stride_list']
    activation = get_activation(encode_dict.get('activation'))
    multi_ratio = encode_dict.get('multi_ratio')


    if multi_ratio is not None:
        filter_list = np.array(filter_list) * multi_ratio
        filter_list = filter_list.astype(np.int16)

    #----filters vs pooling times
    filter_list = np.array(filter_list) // len(pool_type_list)
    layer_num = 0
    for kernel,filters,strides,pool_kernel in zip(kernel_list,filter_list,stride_list,pool_kernel_list):
        if layer_num == 0:
            data_input = tf_input_layer
        else:
            data_input = net

        net = v2.keras.layers.Conv2D(filters,kernel,strides=1,padding='same')(data_input)
        cnn_list.append(net)
        net = activation(net)

        pool_list = []
        for pool_type in pool_type_list:
            if pool_type == 'max':
                net_temp = v2.keras.layers.MaxPool2D(pool_size=pool_kernel,strides=strides,padding='same')(net)
                pool_list.append(net_temp)
            elif pool_type == 'ave':
                net_temp = v2.keras.layers.AveragePooling2D(pool_size=pool_kernel,strides=strides,padding='same')(net)
                pool_list.append(net_temp)
            elif pool_type == 'cnn':
                net_temp = v2.keras.layers.Conv2D(filters,kernel,strides=strides,padding='same')(net)
                pool_list.append(net_temp)
        net = v2.concat(pool_list,axis=-1)
        layer_num += 1
        msg_list.append("encode_{} shape: {}".format(layer_num+1,net.shape))


    # for i, pool_type in enumerate(pool_type_list):
    #     net = re_conv(tf_input_layer, kernel_list, filter_list, pool_kernel=pool_kernel_list[i], pool_type=pool_type,
    #                   stride_list=stride_list, activation=activation, rot=False, to_reduce=to_reduce,
    #                   print_out=print_out)
    #     net_list.append(net)

    # -----------------------------------------------------------------------
    # --------Decode--------
    # -----------------------------------------------------------------------
    cnn_type = decode_dict['cnn_type']
    filter_list = decode_dict['filter_list']
    kernel_list = decode_dict['kernel_list']
    stride_list = decode_dict['stride_list']
    activation = get_activation(decode_dict.get('activation'))
    multi_ratio = decode_dict.get('multi_ratio')


    if multi_ratio is not None:
        filter_list = np.array(filter_list) * multi_ratio
        filter_list = filter_list.astype(np.int16)

    # ----filters vs pooling times
    filter_list = np.array(filter_list) // len(pool_type_list)

    layer_num = 0
    for kernel, filters, strides in zip(kernel_list, filter_list, stride_list):
        # decode = tf.layers.conv2d_transpose(net, filters, transpose_filter, strides=strides, padding='same')
        decode = v2.keras.layers.Conv2DTranspose(filters,1,strides=strides,padding='same')(net)

        if len(cnn_list) == len(kernel_list):
            decode = v2.add(decode,cnn_list[::-1][layer_num])

        if cnn_type == 'resnet':
            net = resnet_block(decode, k_size=kernel, filters=filters,
                               activation=activation, stride=1, to_cnn_input=False)
        else:
            net = v2.keras.layers.Conv2D(filters,kernel,strides=1,padding='same')(decode)

        layer_num += 1
        msg_list.append("decode_{} shape: {}".format(layer_num + 1, net.shape))

    # for net in net_list:
    #     for i, filters in enumerate(filter_list):
    #         if isinstance(stride_list, list):
    #             stride = stride_list[i]
    #         else:
    #             stride = 2
    #         decode = tf.layers.conv2d_transpose(net, filters, transpose_filter, strides=stride, padding='same')
    #
    #         if cnn_type == 'resnet':
    #             if to_reduce:
    #                 net = resnet_block_reduction(decode, k_size=kernel_list[i], filters=filters,
    #                              activation=activation, stride=1, to_cnn_input=False)
    #             else:
    #                 net = resnet_block(decode,k_size=kernel_list[i],filters=filters,
    #                                    activation=activation,stride=1,to_cnn_input=False)
    #         else:
    #             if to_reduce:
    #                 conv1 = Conv(decode, filters, kernel=[kernel_list[i], 1], activation=None)
    #                 conv2 = Conv(decode, filters, kernel=[1, kernel_list[i]], activation=None)
    #                 concat = tf.concat([conv1, conv2], axis=-1)
    #                 net = Conv(concat, filters, kernel=[1, 1], activation=activation)
    #             else:
    #                 net = Conv(decode, filters, kernel=kernel_list[i], activation=activation)
    #
    #
    #         msg = "decode_{} shape = {}".format(i + 1, net.shape)
    #         msg_list.append(msg)
    #
    #     transpose_list.append(net)
    #
    # concat = tf.concat(transpose_list, axis=-1)
    # msg = "concat shape = {}".format(concat.shape)
    # msg_list.append(msg)

    if cnn_type == 'resnet':
        net = resnet_block(net, k_size=kernel, filters=filters,
                           activation=activation,to_cnn_input=True)
        net = resnet_block(net, k_size=kernel, filters=out_channel,
                           activation=None,to_cnn_input=True)
    else:
        net = Conv(net, filters, kernel=kernel, activation=activation)
        net = Conv(net, out_channel, kernel=kernel, padding="same", activation=None)  # name='output_AE'

    msg = "output shape = {}".format(net.shape)
    msg_list.append(msg)


    say_sth(msg_list,print_out=print_out)

    return net

def Seg_pooling_net_V10(tf_input,tf_input_2,encode_dict,decode_dict,out_channel=3,to_reduce=False,print_out=False):

    '''
    adopted from Seg_pooling_net_V4
    '''
    #----var
    net_list = list()
    transpose_list = list()
    cnn_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

    #----dummy output
    pre_embeddings = tf.layers.flatten(tf_input_2)
    embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='dummy_out')

    input_size = (tf_input.shape[1].value,tf_input.shape[2].value)#(h,w)


    #----first layer process
    tf_input_layer, do_type = first_layer_process(tf_input, encode_dict)
    # tf_input_layer, do_type = first_layer_process(tf.image.rgb_to_grayscale(tf_input), encode_dict)
    msg = "First layer process: {} with shape = {}".format(do_type, tf_input_layer.shape)
    say_sth(msg, print_out=print_out)

    pool_type_list = encode_dict['pool_type_list']
    filter_list = encode_dict['filter_list']
    kernel_list = encode_dict['kernel_list']
    pool_kernel_list = encode_dict['pool_kernel_list']
    stride_list = encode_dict['stride_list']
    activation = get_activation(encode_dict.get('activation'))
    multi_ratio = encode_dict.get('multi_ratio')


    if multi_ratio is not None:
        filter_list = np.array(filter_list) * multi_ratio
        filter_list = filter_list.astype(np.int16)

    #----filters vs pooling times
    filter_list = np.array(filter_list) // len(pool_type_list)
    layer_num = 0
    for kernel,filters,strides,pool_kernel in zip(kernel_list,filter_list,stride_list,pool_kernel_list):
        if layer_num == 0:
            data_input = tf_input_layer
        else:
            data_input = net

        net = v2.keras.layers.Conv2D(filters,kernel,strides=1,padding='same')(data_input)
        cnn_list.append(net)
        net = activation(net)

        pool_list = []
        for pool_type in pool_type_list:
            if pool_type == 'max':
                net_temp = v2.keras.layers.MaxPool2D(pool_size=pool_kernel,strides=strides,padding='same')(net)
                pool_list.append(net_temp)
            elif pool_type == 'ave':
                net_temp = v2.keras.layers.AveragePooling2D(pool_size=pool_kernel,strides=strides,padding='same')(net)
                pool_list.append(net_temp)
            elif pool_type == 'cnn':
                net_temp = v2.keras.layers.Conv2D(filters,kernel,strides=strides,padding='same')(net)
                pool_list.append(net_temp)
        net = v2.concat(pool_list,axis=-1)
        layer_num += 1
        msg_list.append("encode_{} shape: {}".format(layer_num+1,net.shape))

    # -----------------------------------------------------------------------
    # --------Decode--------
    # -----------------------------------------------------------------------
    cnn_type = decode_dict['cnn_type']
    filter_list = decode_dict['filter_list']
    kernel_list = decode_dict['kernel_list']
    stride_list = decode_dict['stride_list']
    activation = get_activation(decode_dict.get('activation'))
    multi_ratio = decode_dict.get('multi_ratio')


    if multi_ratio is not None:
        filter_list = np.array(filter_list) * multi_ratio
        filter_list = filter_list.astype(np.int16)

    # ----filters vs pooling times
    filter_list = np.array(filter_list) // len(pool_type_list)

    #----decode but use resize for the last layer
    layer_num = 0
    for kernel, filters, strides in zip(kernel_list[:-1], filter_list[:-1], stride_list[:-1]):
        # decode = tf.layers.conv2d_transpose(net, filters, transpose_filter, strides=strides, padding='same')
        decode = v2.keras.layers.Conv2DTranspose(filters,1,strides=strides,padding='same')(net)

        if len(cnn_list) == len(kernel_list):
            decode = v2.add(decode,cnn_list[::-1][layer_num])

        if cnn_type == 'resnet':
            net = resnet_block(decode, k_size=kernel, filters=filters,
                               activation=activation, stride=1, to_cnn_input=False)
        else:
            net = v2.keras.layers.Conv2D(filters,kernel,strides=1,padding='same')(decode)

        layer_num += 1
        msg_list.append("decode_{} shape: {}".format(layer_num + 1, net.shape))

    #----resize
    net = v2.image.resize(net, input_size, method='nearest')




    if cnn_type == 'resnet':
        net = resnet_block(net, k_size=kernel, filters=filters,
                           activation=activation,to_cnn_input=True)
        net = resnet_block(net, k_size=kernel, filters=out_channel,
                           activation=None,to_cnn_input=True)
    else:
        net = Conv(net, filters, kernel=kernel, activation=activation)
        net = Conv(net, out_channel, kernel=kernel, padding="same", activation=None)  # name='output_AE'

    msg = "output shape = {}".format(net.shape)
    msg_list.append(msg)


    say_sth(msg_list,print_out=print_out)

    return net

def rot_cnn(net,filters,kernel,stride=1,activation=tf.nn.relu,padding='same',name=None):
    net_rot = v2.image.rot90(net, k=1)
    net_rot = Conv(net_rot, filters, kernel=kernel,
               activation=activation, stride=stride, padding=padding,name=name)
    net_rot = v2.image.rot90(net_rot, k=-1)
    return net_rot

def get_activation(name):
    acti_dict = dict(
        relu=v2.nn.relu,
        gelu=v2.nn.gelu,
        swish=v2.nn.swish,#the best https://arxiv.org/pdf/1710.05941.pdf
        leaky_relu=v2.nn.leaky_relu,
        silu=v2.nn.silu
        )
    if acti_dict.get(name) is None:
        return v2.nn.relu
    else:
        return acti_dict[name]

