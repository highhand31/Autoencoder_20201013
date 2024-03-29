
import numpy as np
# from inception_resnet_v1_reduction import inference as inception_resnet_v1_reduction
# from models_VIT import PatchEmbedding,VitLayer,VitLayer_2


#----tensorflow version check
import tensorflow
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
# print("Tensorflow version of {}: {}".format(__file__,tf.__version__))

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

def resnet_block_2022(input_x,k_size=3,filters=32,activation=tf.nn.relu):
    net_1_add = Conv(input_x, filters, k_size, activation=None, padding='same')
    net_1 = activation(net_1_add)
    net_1 = tf.layers.max_pooling2d(inputs=net_1, pool_size=[3,3], strides=2,padding='same')
    net_1 = Conv(net_1, filters, k_size, activation=None, padding='same')

    net_2_add = Conv(input_x, filters, k_size+2, activation=None, padding='same',stride=1)
    net_2 = activation(net_2_add)
    net_2 = Conv(net_2, filters, k_size, activation=None, padding='same',stride=2)

    net_3_add = Conv(input_x, filters, 1, activation=None, padding='same', stride=1)
    net_3 = activation(net_3_add)
    net_3 = Conv(net_3, filters, k_size, activation=None, padding='same', stride=2)

    net = net_1 + net_2 + net_3
    net_add = net_1_add + net_2_add + net_3_add

    # return net_1_add+net_2_add+net_3_add,activation(net)
    return net_add,activation(net)

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

def Seg_pooling_net_V1(tf_input,tf_input_2,encode_dict,decode_dict,out_channel=3,print_out=False):

    '''
    added on 20221121 from models_Seg.py
    '''
    #----var
    net_list = list()
    transpose_list = list()
    cnn_list = list()
    resize_cnn_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

    # ----dummy output
    pre_embeddings = tf.layers.flatten(tf_input_2)
    embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='dummy_out')

    #----to grayscal
    if encode_dict.get('to_gray'):
        batch_image = v2.image.rgb_to_grayscale(tf_input)
    else:
        batch_image = tf_input

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
    # pool_times = len(pool_type_list)
    # if "resize" in pool_type_list:
    #     pool_times -= 1
    # filter_list = filter_list // pool_times

    #----resnet 2022
    layer_num = 0
    for kernel, filters in zip(kernel_list, filter_list):
        if layer_num == 0:
            data_input = tf_input_layer
        else:
            data_input = net

        net_before_acti,net = resnet_block_2022(data_input,k_size=kernel,filters=filters,
                                               activation=activation)
        # ----resize the original batch images
        # h_now = batch_image.shape[1].value
        # w_now = batch_image.shape[2].value
        # batch_image = v2.image.resize(batch_image, (h_now // 2, w_now // 2), method='nearest')

        #----
        layer_num += 1
        cnn_list.append(net_before_acti)
        msg_list.append("encode_{} shape: {}".format(layer_num, net.shape))

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


    layer_num = 0
    for kernel, filters, stride in zip(kernel_list, filter_list, stride_list):

        net = v2.keras.layers.Conv2DTranspose(filters, kernel, activation=activation,
                                              strides=stride, padding='same')(net)
        net = Conv(net, filters, kernel, activation=None)

        net = tf.add(net, cnn_list[::-1][layer_num])
        net = activation(net)

        layer_num += 1

        msg_list.append("decode_{} shape: {}".format(layer_num, net.shape))

    net = Conv(net, out_channel, kernel=kernel, padding="same", activation=None)  # name='output_AE'

    msg = "output shape = {}".format(net.shape)
    msg_list.append(msg)

    say_sth(msg_list,print_out=print_out)

    return net

def Seg_pooling_net_V4(tf_input,tf_input_2,encode_dict,decode_dict,out_channel=3,to_reduce=False,print_out=False):

    '''
    因為原本AE_pooling_net的re_conv無法加入patch embedding的概念，所以使用該func來加入
    kernel_list, filter_list,pool_kernel_list=2,activation=tf.nn.relu,
                   pool_type_list=None,stride_list=None,print_out=False,patch_size=2
    '''
    #----var
    net_list = list()
    transpose_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

    #----dummy output
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
    if activation is None:
        activation = tf.nn.relu

    if multi_ratio is not None:
        filter_list = np.array(filter_list) * multi_ratio
        filter_list = filter_list.astype(np.int16)

    #----filters vs pooling times
    filter_list = np.array(filter_list) // len(pool_type_list)

    for i, pool_type in enumerate(pool_type_list):
        net = re_conv(tf_input_layer, kernel_list, filter_list, pool_kernel=pool_kernel_list[i], pool_type=pool_type,
                      stride_list=stride_list, activation=activation, rot=False, to_reduce=to_reduce,
                      print_out=print_out)
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

    for net in net_list:
        for i, filters in enumerate(filter_list):
            if isinstance(stride_list, list):
                stride = stride_list[i]
            else:
                stride = 2
            decode = tf.layers.conv2d_transpose(net, filters, transpose_filter, strides=stride, padding='same')

            if cnn_type == 'resnet':
                if to_reduce:
                    net = resnet_block_reduction(decode, k_size=kernel_list[i], filters=filters,
                                 activation=activation, stride=1, to_cnn_input=False)
                else:
                    net = resnet_block(decode,k_size=kernel_list[i],filters=filters,
                                       activation=activation,stride=1,to_cnn_input=False)
            else:
                if to_reduce:
                    conv1 = Conv(decode, filters, kernel=[kernel_list[i], 1], activation=None)
                    conv2 = Conv(decode, filters, kernel=[1, kernel_list[i]], activation=None)
                    concat = tf.concat([conv1, conv2], axis=-1)
                    net = Conv(concat, filters, kernel=[1, 1], activation=activation)
                else:
                    net = Conv(decode, filters, kernel=kernel_list[i], activation=activation)


            msg = "decode_{} shape = {}".format(i + 1, net.shape)
            msg_list.append(msg)

        transpose_list.append(net)

    concat = tf.concat(transpose_list, axis=-1)
    msg = "concat shape = {}".format(concat.shape)
    msg_list.append(msg)

    if cnn_type == 'resnet':
        net = resnet_block(concat, k_size=kernel_list[i], filters=filters,
                           activation=activation,to_cnn_input=True)
        net = resnet_block(net, k_size=kernel_list[i], filters=out_channel,
                           activation=None,to_cnn_input=True)
    else:
        if to_reduce:
            conv1 = Conv(concat, filters, kernel=[kernel_list[i], 1], activation=None)
            conv2 = Conv(concat, filters, kernel=[1, kernel_list[i]], activation=None)
            net = tf.concat([conv1, conv2], axis=-1)
            net = Conv(net, out_channel, kernel=[1, 1], activation=None)
        else:
            net = Conv(concat, filters, kernel=kernel_list[i], activation=activation)
            net = Conv(net, out_channel, kernel=kernel_list[i], padding="same", activation=None)  # name='output_AE'

    msg = "output shape = {}".format(net.shape)
    msg_list.append(msg)


    say_sth(msg_list,print_out=print_out)

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

def Seg_pooling_net_V8(tf_input,tf_input_2,encode_dict,decode_dict,out_channel=3,to_reduce=False,print_out=False):

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

def AE_Seg_net(tf_input, kernel_list, filter_list, pool_kernel_list=2, activation=tf.nn.relu,
                   pool_type_list=None, rot=False, print_out=False, preprocess_dict=None):
    # ----var
    net_list = list()
    transpose_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

    # ----filters vs pooling times
    pool_times = len(pool_type_list)
    filter_list = np.array(filter_list) // pool_times

    # ----filters vs rot cnn
    if rot is True:
        filter_list = filter_list // 2

    # ----preprocess
    if preprocess_dict is not None:
        tf_input = preprocess(tf_input, preprocess_dict, print_out=print_out)

    #----inference
    say_sth('----AE pooling net----',print_out=True)
    for i, pool_type in enumerate(pool_type_list):
        net = re_conv(tf_input, kernel_list, filter_list, pool_kernel=pool_kernel_list[i], pool_type=pool_type,
                      activation=tf.nn.relu, rot=rot, print_out=print_out)
        net_list.append(net)

    net = tf.concat(net_list, axis=-1)

    # net = tf.concat(net_list,axis=-1)

    pre_embeddings = tf.layers.flatten(net, name='pre_embeddings')
    embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='embeddings')
    msg = "embeddings shape:{}".format(embeddings.shape)
    msg_list.append(msg)

    # -----------------------------------------------------------------------
    # --------Decode--------
    # -----------------------------------------------------------------------

    for net in net_list:
        for i, filers in enumerate(filter_list[::-1]):
            kernel = kernel_list[::-1][i]
            decode = tf.layers.conv2d_transpose(net, filers, transpose_filter, strides=2, padding='same')

            net = Conv(decode, filers, kernel=kernel, activation=activation)
            if rot is True:
                net_1 = rot_cnn(decode, filers, kernel, activation=activation)
                net = tf.concat([net, net_1], axis=-1)

            msg = "decode_{} shape = {}".format(i + 1, net.shape)
            msg_list.append(msg)
        transpose_list.append(net)

    concat = tf.concat(transpose_list, axis=-1)
    net = Conv(concat, filers, kernel=kernel, activation=activation)
    if rot is True:
        net_1 = rot_cnn(concat, filers, kernel, activation=activation)
        net = tf.concat([net, net_1], axis=-1)

    # net = Conv(net, 3, kernel=kernel_list[0], padding="same", activation=activation, name='output_AE')
    net = Conv(net, 3, kernel=kernel_list[0], padding="same", activation=None)  # name='output_AE'
    msg = 'output AE shape = {}'.format(net.shape)
    msg_list.append(msg)

    output_AE = tf.identity(net,name='output_AE')

    say_sth('----SEG diff net----', print_out=True)



    say_sth(msg_list, print_out=print_out)

    return net

def AE_pooling_net(tf_input, kernel_list, filter_list,pool_kernel_list=2,activation=tf.nn.relu,
                   pool_type_list=None,stride_list=None,rot=False,print_out=False,preprocess_dict=None):
    #----var
    net_list = list()
    transpose_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

    #----filters vs pooling times
    pool_times = len(pool_type_list)
    filter_list = np.array(filter_list) // pool_times

    #----filters vs rot cnn
    if rot is True:
        filter_list = filter_list // 2

    #----preprocess
    # if preprocess_dict is not None:
    #     tf_input = preprocess(tf_input,preprocess_dict,print_out=print_out)

    # with tf.variable_scope("AE",reuse=tf.AUTO_REUSE):
    for i,pool_type in enumerate(pool_type_list):
        net = re_conv(tf_input, kernel_list, filter_list, pool_kernel=pool_kernel_list[i], pool_type=pool_type,
                      stride_list=stride_list,activation=tf.nn.relu,rot=rot,print_out=print_out)
        net_list.append(net)

    net = tf.concat(net_list,axis=-1)

     # net = tf.concat(net_list,axis=-1)

    pre_embeddings = tf.layers.flatten(net, name='pre_embeddings')
    embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='embeddings')
    msg = "embeddings shape:{}".format(embeddings.shape)
    msg_list.append(msg)

    # -----------------------------------------------------------------------
    # --------Decode--------
    # -----------------------------------------------------------------------

    for net in net_list:
        for i, filers in enumerate(filter_list[::-1]):
            kernel = kernel_list[::-1][i]
            if isinstance(stride_list,list):
                stride = stride_list[::-1][i]
            else:
                stride = 2
            decode = tf.layers.conv2d_transpose(net, filers, transpose_filter, strides=stride, padding='same')

            net = Conv(decode, filers, kernel=kernel, activation=activation)
            if rot is True:
                net_1 = rot_cnn(decode, filers, kernel, activation=activation)
                net = tf.concat([net, net_1], axis=-1)

            msg = "decode_{} shape = {}".format(i + 1, net.shape)
            msg_list.append(msg)
        transpose_list.append(net)

    concat = tf.concat(transpose_list, axis=-1)
    net = Conv(concat, filers, kernel=kernel, activation=activation)
    if rot is True:
        net_1 = rot_cnn(concat, filers, kernel, activation=activation)
        net = tf.concat([net, net_1], axis=-1)

    # net = Conv(net, 3, kernel=kernel_list[0], padding="same", activation=activation, name='output_AE')
    net = Conv(net, 3, kernel=kernel_list[0], padding="same", activation=None)#name='output_AE'

    say_sth(msg_list,print_out=print_out)

    return net

def AE_pooling_net_V3(tf_input, kernel_list, filter_list,pool_kernel_list=2,activation=tf.nn.relu,
                   pool_type_list=None,stride_list=None,print_out=False,patch_size=2):
    '''
    因為原本AE_pooling_net的re_conv無法加入patch embedding的概念，所以使用該func來加入
    '''
    #----var
    net_list = list()
    transpose_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

    #----filters vs pooling times
    pool_times = len(pool_type_list)
    filter_list = np.array(filter_list) // pool_times

    #----filters vs rot cnn
    # if to_Vit is True:
    #     filter_list = filter_list // 2

    #----patch embedding
    if patch_size is not None:
        # tf_input_embed = v2.keras.layers.Conv2D(filters=16,kernel_size=patch_size,strides=patch_size)(tf_input)
        # net_max = tf.nn.max_pool(tf_input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        # net_resize = v2.image.resize(tf_input,[272,416])
        # tf_input_embed = tf.concat([net_max,net_resize],axis=-1)
        #----
        tf_input_embed = v2.image.resize(tf_input,[272,416])

    for i, pool_type in enumerate(pool_type_list):
        net = re_conv(tf_input_embed, kernel_list, filter_list, pool_kernel=pool_kernel_list[i], pool_type=pool_type,
                      stride_list=stride_list, activation=tf.nn.relu, rot=False, print_out=print_out)
        net_list.append(net)

    net = tf.concat(net_list, axis=-1)

    # -----------------------------------------------------------------------
    # --------Decode--------
    # -----------------------------------------------------------------------

    for net in net_list:
        for i, filers in enumerate(filter_list[::-1]):
            kernel = kernel_list[::-1][i]
            if isinstance(stride_list, list):
                stride = stride_list[::-1][i]
            else:
                stride = 2
            decode = tf.layers.conv2d_transpose(net, filers, transpose_filter, strides=stride, padding='same')

            net = Conv(decode, filers, kernel=kernel, activation=activation)


            msg = "decode_{} shape = {}".format(i + 1, net.shape)
            msg_list.append(msg)
        #-----test code
        decode = tf.layers.conv2d_transpose(net, 16, transpose_filter, strides=2, padding='same')
        net = Conv(decode, filers, kernel=kernel, activation=activation)

        msg = "decode_{} shape = {}".format(i + 1, net.shape)
        msg_list.append(msg)

        transpose_list.append(net)

    concat = tf.concat(transpose_list, axis=-1)
    net = Conv(concat, filers, kernel=kernel, activation=activation)


    # net = Conv(net, 3, kernel=kernel_list[0], padding="same", activation=activation, name='output_AE')
    net = Conv(net, 3, kernel=kernel_list[0], padding="same", activation=None)#name='output_AE'

    say_sth(msg_list,print_out=print_out)

    return net

def AE_pooling_net_V4(tf_input,encode_dict,decode_dict,out_channel=3,to_reduce=False,print_out=False):

    '''
    因為原本AE_pooling_net的re_conv無法加入patch embedding的概念，所以使用該func來加入
    kernel_list, filter_list,pool_kernel_list=2,activation=tf.nn.relu,
                   pool_type_list=None,stride_list=None,print_out=False,patch_size=2
    '''
    #----var
    net_list = list()
    transpose_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

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
    if activation is None:
        activation = tf.nn.relu

    if multi_ratio is not None:
        filter_list = np.array(filter_list) * multi_ratio
        filter_list = filter_list.astype(np.int16)

    #----filters vs pooling times
    filter_list = np.array(filter_list) // len(pool_type_list)

    for i, pool_type in enumerate(pool_type_list):
        net = re_conv(tf_input_layer, kernel_list, filter_list, pool_kernel=pool_kernel_list[i], pool_type=pool_type,
                      stride_list=stride_list, activation=activation, rot=False, to_reduce=to_reduce,
                      print_out=print_out)
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

    for net in net_list:
        for i, filters in enumerate(filter_list):
            if isinstance(stride_list, list):
                stride = stride_list[i]
            else:
                stride = 2
            decode = tf.layers.conv2d_transpose(net, filters, transpose_filter, strides=stride, padding='same')

            if cnn_type == 'resnet':
                if to_reduce:
                    net = resnet_block_reduction(decode, k_size=kernel_list[i], filters=filters,
                                 activation=activation, stride=1, to_cnn_input=False)
                else:
                    net = resnet_block(decode,k_size=kernel_list[i],filters=filters,
                                       activation=activation,stride=1,to_cnn_input=False)
            else:
                if to_reduce:
                    conv1 = Conv(decode, filters, kernel=[kernel_list[i], 1], activation=None)
                    conv2 = Conv(decode, filters, kernel=[1, kernel_list[i]], activation=None)
                    concat = tf.concat([conv1, conv2], axis=-1)
                    net = Conv(concat, filters, kernel=[1, 1], activation=activation)
                else:
                    net = Conv(decode, filters, kernel=kernel_list[i], activation=activation)


            msg = "decode_{} shape = {}".format(i + 1, net.shape)
            msg_list.append(msg)

        transpose_list.append(net)

    concat = tf.concat(transpose_list, axis=-1)
    msg = "concat shape = {}".format(concat.shape)
    msg_list.append(msg)

    if cnn_type == 'resnet':
        net = resnet_block(concat, k_size=kernel_list[i], filters=filters,
                           activation=activation,to_cnn_input=True)
        net = resnet_block(net, k_size=kernel_list[i], filters=out_channel,
                           activation=None,to_cnn_input=True)
    else:
        if to_reduce:
            conv1 = Conv(concat, filters, kernel=[kernel_list[i], 1], activation=None)
            conv2 = Conv(concat, filters, kernel=[1, kernel_list[i]], activation=None)
            net = tf.concat([conv1, conv2], axis=-1)
            net = Conv(net, out_channel, kernel=[1, 1], activation=None)
        else:
            net = Conv(concat, filters, kernel=kernel_list[i], activation=activation)
            net = Conv(net, out_channel, kernel=kernel_list[i], padding="same", activation=None)  # name='output_AE'

    msg = "output shape = {}".format(net.shape)
    msg_list.append(msg)


    say_sth(msg_list,print_out=print_out)

    return net

def AE_pooling_net_V5(tf_input,encode_dict,decode_dict,to_reduce=False,print_out=False):

    '''
    因為原本AE_pooling_net的re_conv無法加入patch embedding的概念，所以使用該func來加入
    kernel_list, filter_list,pool_kernel_list=2,activation=tf.nn.relu,
                   pool_type_list=None,stride_list=None,print_out=False,patch_size=2
    '''
    #----var
    net_list = list()
    transpose_list = list()
    transpose_filter = [1, 1]
    msg_list = list()



    #----filters vs rot cnn
    # if to_Vit is True:
    #     filter_list = filter_list // 2

    #----first layer process
    if encode_dict.get('first_layer') is not None:
        dict_first_layer = encode_dict.get('first_layer')
        do_type = dict_first_layer.get('type')
        if do_type == 'resize':
            size = dict_first_layer.get('size')#(h,w)
            tf_input_layer = v2.image.resize(tf_input,size)
        elif do_type == 'CNN_downSampling':
            filter = dict_first_layer.get('filter')
            kernel = dict_first_layer.get('kernel')
            tf_input_layer = v2.keras.layers.Conv2D(filters=filter, kernel_size=kernel, strides=2)(
                tf_input)
        elif do_type == 'patch_embedding':
            patch_size = dict_first_layer.get('patch_size')
            filter = dict_first_layer.get('filter')
            tf_input_layer = v2.keras.layers.Conv2D(filters=filter,kernel_size=patch_size,strides=patch_size)(tf_input)
        elif do_type == 'resize_patch_embedding':
            patch_size = dict_first_layer.get('patch_size')
            filter = dict_first_layer.get('filter')
            net_patch = v2.keras.layers.Conv2D(filters=filter, kernel_size=patch_size, strides=patch_size)(
                tf_input)

            size = dict_first_layer['size']
            net_resize = v2.image.resize(tf_input, size)
            tf_input_layer = v2.concat([net_patch,net_resize],axis=-1)
        elif do_type == 'resize_maxpool':
            size = dict_first_layer.get('size')  # (h,w)
            net_resize = v2.image.resize(tf_input, size)
            net_maxpool = tf.nn.max_pool(tf_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            tf_input_layer = v2.concat([net_maxpool, net_resize], axis=-1)
        else:
            tf_input_layer = tf_input
            do_type = 'do nothing'
    else:
        tf_input_layer = tf_input
        do_type = 'do nothing'

    msg = "first layer:{} with shape = {}".format(do_type,tf_input_layer.shape)
    say_sth(msg, print_out=print_out)

    # pool_type_list = encode_dict['pool_type_list']
    # filter_list = encode_dict['filter_list']
    # kernel_list = encode_dict['kernel_list']
    # pool_kernel_list = encode_dict['pool_kernel_list']
    # stride_list = encode_dict['stride_list']
    # activation = encode_dict.get('activation')
    # if activation is None:
    #     activation = tf.nn.relu
    #
    # #----filters vs pooling times
    # filter_list = np.array(filter_list) // len(pool_type_list)
    #
    # for i, pool_type in enumerate(pool_type_list):
    #
    #     net = re_conv(tf_input_layer, kernel_list, filter_list, pool_kernel=pool_kernel_list[i], pool_type=pool_type,
    #                   stride_list=stride_list, activation=activation, rot=False, to_reduce=to_reduce,
    #                   print_out=print_out)
    #     net_list.append(net)

    # -----------------------------------------------------------------------
    # --------Decode--------
    # -----------------------------------------------------------------------
    cnn_type = decode_dict['cnn_type']
    pool_type_list = encode_dict['pool_type_list']
    filter_list = decode_dict['filter_list']
    kernel_list = decode_dict['kernel_list']
    stride_list = decode_dict['stride_list']
    activation = decode_dict.get('activation')
    if activation is None:
        activation = tf.nn.relu

    # ----filters vs pooling times
    filter_list = np.array(filter_list) // len(pool_type_list)

    for i, pool_type in enumerate(pool_type_list):
        for j in range(len(kernel_list)):
            kernel = kernel_list[j]
            filters = filter_list[j]
            if isinstance(stride_list, list):
                stride = stride_list[j]
            else:
                stride = 2
            if j == 0:
                data_input = tf_input_layer
            else:
                data_input = net
            decode = tf.layers.conv2d_transpose(data_input, filters, transpose_filter, strides=stride, padding='same')
            net = Conv(decode, filters, kernel=kernel, activation=activation)

            msg = "decode_{} shape = {}".format(j + 1, net.shape)
            msg_list.append(msg)

        net_list.append(net)



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

    concat = tf.concat(net_list, axis=-1)
    msg = "concat shape = {}".format(concat.shape)
    msg_list.append(msg)

    if cnn_type == 'resnet':
        net = resnet_block(concat, k_size=kernel_list[i], filters=filters,
                           activation=activation,to_cnn_input=True)
        net = resnet_block(net, k_size=kernel_list[i], filters=3,
                           activation=None,to_cnn_input=True)
    else:
        if to_reduce:
            conv1 = Conv(concat, filters, kernel=[kernel_list[i], 1], activation=None)
            conv2 = Conv(concat, filters, kernel=[1, kernel_list[i]], activation=None)
            net = tf.concat([conv1, conv2], axis=-1)
            net = Conv(net, 3, kernel=[1, 1], activation=None)
        else:
            net = Conv(concat, filters, kernel=kernel_list[i], activation=activation)
            net = Conv(net, 3, kernel=kernel_list[i], padding="same", activation=None)  # name='output_AE'

    msg = "output shape = {}".format(net.shape)
    msg_list.append(msg)


    say_sth(msg_list,print_out=print_out)

    return net

def AE_pooling_net_V6(tf_input,encode_dict,decode_dict,to_reduce=False,print_out=False):

    '''
    將ave pool and max pool concat再進行下一步
    '''
    #----var
    net_list = list()
    transpose_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

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
    if activation is None:
        activation = tf.nn.relu

    #----filters vs pooling times
    filter_list = np.array(filter_list) // len(pool_type_list)

    #----
    net = tf_input
    for i in range(len(kernel_list)):
        net_list = []
        if i == 0:
            data_input = tf_input
        else:
            data_input = concat
        for pool_type in pool_type_list:
            if pool_type == 'cnn':
                stride = stride_list[i]
            else:
                stride = 1

            net = Conv(data_input, filter_list[i], kernel=kernel_list[i],
                       stride=stride, activation=activation)
            #----pooling
            if pool_type == 'max':
                pool_kernel = [1,pool_kernel_list[i], pool_kernel_list[i], 1]
                stride = [1, stride_list[i], stride_list[i], 1]
                net = tf.nn.max_pool(net, ksize=pool_kernel, strides=stride, padding='SAME')
            elif pool_type == 'ave':
                pool_kernel = [1, pool_kernel_list[i], pool_kernel_list[i], 1]
                stride = [1, stride_list[i], stride_list[i], 1]
                net = tf.nn.avg_pool(net, ksize=pool_kernel, strides=stride, padding='SAME')

            net_list.append(net)

        concat = tf.concat(net_list,axis=-1)
        msg = "encode_{} shape = {}".format(i + 1, concat.shape)
        msg_list.append(msg)

    # -----------------------------------------------------------------------
    # --------Decode--------
    # -----------------------------------------------------------------------
    cnn_type = decode_dict['cnn_type']
    pool_type_list = encode_dict['pool_type_list']
    filter_list = decode_dict['filter_list']
    kernel_list = decode_dict['kernel_list']
    stride_list = decode_dict['stride_list']
    activation = decode_dict.get('activation')
    if activation is None:
        activation = tf.nn.relu

    # ----filters vs pooling times
    # filter_list = np.array(filter_list) // len(pool_type_list)
    net = concat
    for j in range(len(kernel_list)):
        kernel = kernel_list[j]
        filters = filter_list[j]
        if isinstance(stride_list, list):
            stride = stride_list[j]
        else:
            stride = 2

        net = tf.layers.conv2d_transpose(net, filters, transpose_filter, strides=stride, padding='same')
        if cnn_type == 'resnet':
            net = resnet_block(net, k_size=kernel, filters=filters,
                               activation=activation, stride=1, to_cnn_input=False)
        else:
            net = Conv(net, filters, kernel=kernel, activation=activation)

        msg = "decode_{} shape = {}".format(j + 1, net.shape)
        msg_list.append(msg)

    if cnn_type == 'resnet':
        net = resnet_block(net, k_size=kernel, filters=filters,
                           activation=activation,to_cnn_input=True)
        net = resnet_block(net, k_size=kernel, filters=3,
                           activation=None,to_cnn_input=True)
    else:
        net = Conv(net, filters, kernel=kernel, activation=activation)
        net = Conv(net, 3, kernel=kernel, padding="same", activation=None)  # name='output_AE'

    msg = "output shape = {}".format(net.shape)
    msg_list.append(msg)

    say_sth(msg_list,print_out=print_out)

    return net

def AE_pooling_net_V7(tf_input,encode_dict,decode_dict,out_channel=3,print_out=False):

    #----var
    net_list = list()
    transpose_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

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

def AE_pooling_net_V8(tf_input,tf_input_standard,encode_dict,decode_dict,out_channel=3,print_out=False):
    #有導入標準照片
    #----var
    net_list = list()
    transpose_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

    #----read a standard image
    # path = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220901_AOI判定OK\OK(多區OK)\train\0_-16_MatchLightSet_NoFailRegion_Ok_1.jpg"
    # path = tf.gfile.FastGFile(path, 'rb').read()
    # img = tf.image.decode_jpeg(path, channels=1)  # 3:color images；1:Gray image
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.image.resize(img, [512, 832])
    # img /= 255.0
    # img = tf.expand_dims(img, axis=0)

    #----concat
    input_x = tf.concat([tf_input,tf_input_standard],axis=-1)

    #----first layer process
    tf_input_layer, do_type = first_layer_process(input_x, encode_dict)
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

def AE_dense_sampling(tf_input,sampling_factor,filters,kernel=5,dilated_ratios=[1,2,5]):
    H = tf_input.shape[1].value
    W = tf_input.shape[2].value
    C = tf_input.shape[3].value

    out_channel = C * sampling_factor**2

    net = v2.reshape(tf_input,[-1,H//sampling_factor,W//sampling_factor,out_channel])
    print("reshape shape:",net.shape)

    for i,ratio in enumerate(dilated_ratios):
        net = v2.keras.layers.Conv2D(filters=filters * out_channel, kernel_size=kernel, strides=1,
                                     padding='same', dilation_rate=ratio)(net)

    net = v2.nn.relu(net)
    print("CNN shape:", net.shape)

    net = v2.reshape(net,[-1,H,W,filters])
    print("reshape shape:", net.shape)

    for i,ratio in enumerate(dilated_ratios):

        net = v2.keras.layers.Conv2D(filters=C, kernel_size=kernel, strides=1,
                                     padding='same', dilation_rate=ratio)(net)
    print("output shape:", net.shape)

    return v2.nn.relu(net)

def AE_Seg_pooling_net(tf_input, kernel_list, filter_list,pool_kernel_list=2,activation=tf.nn.relu,
                   pool_type_list=None,rot=False,print_out=False,preprocess_dict=None,class_num=3):
    #----var
    net_list = list()
    transpose_list = list()
    transpose_filter = [1, 1]
    msg_list = list()

    #----filters vs pooling times
    pool_times = len(pool_type_list)
    filter_list = np.array(filter_list) // pool_times

    #----filters vs rot cnn
    if rot is True:
        filter_list = filter_list // 2

    #----preprocess
    # if preprocess_dict is not None:
    #     tf_input = preprocess(tf_input,preprocess_dict,print_out=print_out)

    for i,pool_type in enumerate(pool_type_list):
        net = re_conv(tf_input, kernel_list, filter_list, pool_kernel=pool_kernel_list[i], pool_type=pool_type,
                      activation=tf.nn.relu,rot=rot,print_out=print_out)
        net_list.append(net)

    net = tf.concat(net_list,axis=-1)

     # net = tf.concat(net_list,axis=-1)

    pre_embeddings = tf.layers.flatten(net, name='pre_embeddings')
    embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='embeddings')
    msg = "embeddings shape:{}".format(embeddings.shape)
    msg_list.append(msg)

    # -----------------------------------------------------------------------
    # --------Decode--------
    # -----------------------------------------------------------------------

    for net in net_list:
        for i, filers in enumerate(filter_list[::-1]):
            kernel = kernel_list[::-1][i]
            decode = tf.layers.conv2d_transpose(net, filers, transpose_filter, strides=2, padding='same')

            net = Conv(decode, filers, kernel=kernel, activation=activation)
            if rot is True:
                net_1 = rot_cnn(decode, filers, kernel, activation=activation)
                net = tf.concat([net, net_1], axis=-1)

            msg = "decode_{} shape = {}".format(i + 1, net.shape)
            msg_list.append(msg)
        transpose_list.append(net)

    concat = tf.concat(transpose_list, axis=-1)
    net = Conv(concat, filers, kernel=kernel, activation=activation)
    if rot is True:
        net_1 = rot_cnn(concat, filers, kernel, activation=activation)
        net = tf.concat([net, net_1], axis=-1)

    # net = Conv(net, 3, kernel=kernel_list[0], padding="same", activation=activation, name='output_AE')
    AE_net = Conv(net, 3, kernel=kernel_list[0], padding="same", activation=None)#name='output_AE'

    diff = tf.subtract(tf_input,AE_net)
    diff = tf.square(diff)
    Seg_net = tf.concat([AE_net, tf_input,diff], axis=-1)
    Seg_net = Conv(Seg_net, class_num, kernel=kernel_list[0], padding="same", activation=None)#name='output_SEG'

    say_sth(msg_list,print_out=print_out)

    return AE_net,Seg_net

def AE_transpose_4layer(tf_input, kernel_list, filter_list,pool_kernel=2,activation=tf.nn.relu,pool_type=None):
        #----var
        pool_kernel = [1,pool_kernel,pool_kernel,1]
        transpose_filter = [1, 1]

        #----pool process
        if pool_type is not None:
            if pool_type == 'max':
                tf_pool = tf.nn.max_pool
            elif pool_type == 'ave':
                tf_pool = tf.nn.avg_pool
            else:
                pool_type = None

        msg = '----AE_transpose_4layer_struct_2----'
        say_sth(msg, print_out=print_out)

        net = Conv(tf_input, filter_list[0],kernel=kernel_list[0],activation=activation)
        if pool_type is not None:
            net = tf_pool(net, ksize=pool_kernel, strides=[1, 2, 2, 1], padding='SAME')
            #----add features
            # net_1 = Conv(tf_input, filter_list[0], kernel=kernel_list[0], stride=2, activation=activation)
            net_1 = tf.nn.avg_pool(tf_input, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')
            net = tf.concat([net,net_1],axis=-1)

        msg = "encode_1 shape = {}".format(net.shape)
        say_sth(msg, print_out=print_out)
        # -----------------------------------------------------------------------
        for i in range(1, len(filter_list), 1):
            net = Conv(net, filter_list[i], kernel=kernel_list[i], activation=activation)
            if pool_type is not None:
                net_1 = tf_pool(net, ksize=pool_kernel, strides=[1, 2, 2, 1], padding='SAME')
                # ----add features
                # net_2 = Conv(net, filter_list[i], kernel=kernel_list[i], stride=2, activation=activation)
                net_2 = tf.nn.avg_pool(net, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')
                net = tf.concat([net_1, net_2], axis=-1)

            msg = "encode_{} shape = {}".format(i + 1, net.shape)
            say_sth(msg, print_out=print_out)

        pre_embeddings = tf.layers.flatten(net,name='pre_embeddings')
        embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='embeddings')
        print("embeddings shape:", embeddings.shape)

        # -----------------------------------------------------------------------
        # --------Decode--------
        # -----------------------------------------------------------------------

        for i, filers in enumerate(filter_list[::-1]):
            kernel = kernel_list[::-1][i]
            net = tf.layers.conv2d_transpose(net, filers, transpose_filter, strides=2, padding='same')
            net = Conv(net, filers, kernel=kernel, activation=activation)
            msg = "decode_{} shape = {}".format(i + 1, net.shape)
            say_sth(msg, print_out=print_out)

        net = Conv(net, 3, kernel=kernel_list[0], padding="same", activation=activation, name='output_AE')

        # net = tf.layers.conv2d_transpose(net, filter_list[3], transpose_filter, strides=2, padding='same')
        # #net = tf.concat([net, U_4_point], axis=3)
        # net = Conv(net, filter_list[3],kernel=kernel_list[3],activation=activation)
        # msg = "decode_1 shape = {}".format(net.shape)
        # say_sth(msg, print_out=print_out)
        # # -----------------------------------------------------------------------
        # # data= 8 x 8 x 64
        # net = tf.layers.conv2d_transpose(net, filter_list[2], transpose_filter, strides=2, padding='same')
        # # net = tf.concat([net, U_3_point], axis=3)
        # net = Conv(net, filter_list[2],kernel=kernel_list[2],activation=activation)
        # msg = "decode_2 shape = {}".format(net.shape)
        # say_sth(msg, print_out=print_out)
        # # -----------------------------------------------------------------------
        #
        # net = tf.layers.conv2d_transpose(net, filter_list[1], transpose_filter, strides=2, padding='same')
        # # net = tf.concat([net, U_2_point], axis=3)
        # net = Conv(net, filter_list[1],kernel=kernel_list[1],activation=activation)
        # msg = "decode_3 shape = {}".format(net.shape)
        # say_sth(msg, print_out=print_out)
        # # -----------------------------------------------------------------------
        # # data= 32 x 32 x 64
        #
        # net = tf.layers.conv2d_transpose(net, filter_list[0], transpose_filter, strides=2, padding='same')
        # # net = tf.concat([net, U_1_point], axis=3)
        # net = Conv(net, filter_list[0],kernel=kernel_list[0],activation=activation)
        # msg = "decode_2 shape = {}".format(net.shape)
        # say_sth(msg, print_out=print_out)
        #
        # net = tf.layers.conv2d(
        #     inputs=net,
        #     filters=3,
        #     kernel_size=kernel_list[0],
        #     # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
        #     kernel_regularizer=tf.keras.regularizers.l2(0.08),
        #     padding="same",
        #     activation=activation,
        #     name='output_AE')
        # msg = "output shape = {}".format(net.shape)
        # say_sth(msg, print_out=print_out)
        # -----------------------------------------------------------------------
        # data= 64 x 64 x 3

        return ne

def AE_transpose_4layer_2(tf_input, kernel_list, filter_list, maxpool_kernel=2, activation=tf.nn.relu):
    # ----var
    maxpool_kernel = [1, maxpool_kernel, maxpool_kernel, 1]
    transpose_filter = [1, 1]

    msg = '----AE_transpose_4layer_struct_2----'
    say_sth(msg, print_out=print_out)

    net = Conv(tf_input, filter_list[0], kernel=kernel_list[0], activation=activation)
    U_1_point = net
    net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')

    msg = "encode_1 shape = {}".format(net.shape)
    say_sth(msg, print_out=print_out)
    # -----------------------------------------------------------------------
    net = Conv(net, filter_list[1], kernel=kernel_list[1], activation=activation)
    U_2_point = net
    net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
    msg = "encode_2 shape = {}".format(net.shape)
    say_sth(msg, print_out=print_out)
    # -----------------------------------------------------------------------

    net = Conv(net, filter_list[2], kernel=kernel_list[2], activation=activation)
    U_3_point = net
    net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')

    msg = "encode_3 shape = {}".format(net.shape)
    say_sth(msg, print_out=print_out)
    # -----------------------------------------------------------------------
    net = Conv(net, filter_list[3], kernel=kernel_list[3], activation=activation)
    U_4_point = net
    net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')

    msg = "encode_4 shape = {}".format(net.shape)
    say_sth(msg, print_out=print_out)

    net = Conv(net, filter_list[4], kernel=kernel_list[4], activation=activation)
    # shape = [-1]
    # shape.extend(net.shape[1:])

    flatten = tf.layers.flatten(net)
    #length = net.shape[-1]

    # ----dropout
    #net = tf.nn.dropout(net, keep_prob=0.8)

    # ----FC
    # net = tf.layers.dense(inputs=net, units=128, activation=activation)
    # print("FC shape:", net.shape)

    embeddings = tf.nn.l2_normalize(flatten, 1, 1e-10, name='embeddings')
    print("embeddings shape:", embeddings.shape)

    # net = tf.layers.dense(inputs=net, units=length, activation=activation)
    # print("FC shape:", net.shape)
    #
    # net = tf.reshape(net, shape)
    # print("reshape shape:", net.shape)

    # net = tf.layers.dense(inputs=prelogits, units=units, activation=None)
    # print("net shape:",net.shape)
    # net = tf.reshape(net,shape)
    # -----------------------------------------------------------------------
    # --------Decode--------
    # -----------------------------------------------------------------------

    # data= 4 x 4 x 64
    net = tf.layers.conv2d_transpose(net, filter_list[3], transpose_filter, strides=2, padding='same')
    # net = tf.concat([net, U_4_point], axis=3)
    net = Conv(net, filter_list[3], kernel=kernel_list[3], activation=activation)
    msg = "decode_1 shape = {}".format(net.shape)
    say_sth(msg, print_out=print_out)
    # -----------------------------------------------------------------------
    # data= 8 x 8 x 64
    net = tf.layers.conv2d_transpose(net, filter_list[2], transpose_filter, strides=2, padding='same')
    # net = tf.concat([net, U_3_point], axis=3)
    net = Conv(net, filter_list[2], kernel=kernel_list[2], activation=activation)
    msg = "decode_2 shape = {}".format(net.shape)
    say_sth(msg, print_out=print_out)
    # -----------------------------------------------------------------------

    net = tf.layers.conv2d_transpose(net, filter_list[1], transpose_filter, strides=2, padding='same')
    # net = tf.concat([net, U_2_point], axis=3)
    net = Conv(net, filter_list[1], kernel=kernel_list[1], activation=activation)
    msg = "decode_3 shape = {}".format(net.shape)
    say_sth(msg, print_out=print_out)
    # -----------------------------------------------------------------------
    # data= 32 x 32 x 64

    net = tf.layers.conv2d_transpose(net, filter_list[0], transpose_filter, strides=2, padding='same')
    # net = tf.concat([net, U_1_point], axis=3)
    net = Conv(net, filter_list[0], kernel=kernel_list[0], activation=activation)
    msg = "decode_2 shape = {}".format(net.shape)
    say_sth(msg, print_out=print_out)

    net = tf.layers.conv2d(
        inputs=net,
        filters=3,
        kernel_size=kernel_list[0],
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
        padding="same",
        activation=activation,
        name='output_AE')
    msg = "output shape = {}".format(net.shape)
    say_sth(msg, print_out=print_out)
    # -----------------------------------------------------------------------
    # data= 64 x 64 x 3
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

