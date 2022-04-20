import cv2,sys,shutil,os,json,time,math
import numpy as np
import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow as v2
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile
print(tf.__version__)
sys.path.append(r'G:\我的雲端硬碟\Python\Code\Pycharm\utility')
from Utility import tools,file_transfer,file_decode_v2
from models_AE import AE_transpose_4layer,tf_mish,AE_JNet,AE_Resnet_Rot,AE_pooling_net,AE_Unet,AE_VIT,AE_pooling_net_V2


print_out = True
SockConnected = False
img_format = {"jpg", 'png', 'bmp', 'JPG','tif','TIF'}

#----functions
def say_sth(msg_source, print_out=False,header=None,delay=0.005):
    end = '\n'
    if isinstance(msg_source,str):
        msg_source = [msg_source]

    for idx,msg in enumerate(msg_source):
        if print_out:
            print(msg)
        # if SockConnected:
        #     if isinstance(header,list):
        #         h = TCPIP.protol_dict.get(header[idx])
        #     elif isinstance(header,str):
        #         h = TCPIP.protol_dict.get(header)
        #     else:
        #         h = '$M00'
        #     if h is None:
        #         h = '$M00'
        #     #print("header:",header)
        #     Sock.send(h + msg + end)
        #     time.sleep(delay)

def GPU_setting(GPU_ratio):
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    if GPU_ratio is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

    return config

def weights_check(sess,saver,save_dir,encript_flag=True,encode_num=87,encode_header=[24, 97, 28, 98]):
    #----var
    error_msg = '模型參數初始化失敗'
    status = True

    files = [file.path for file in os.scandir(save_dir) if file.name.split(".")[-1] == 'meta']
    if len(files) == 0:
        # model_vars = [v for v in tf.trainable_variables()]
        # lookahead = BaseLookAhead(model_vars, k=5, alpha=0.5)
        # self.train_op += lookahead.get_ops()
        # print("no previous model param can be used!")
        msg = "沒有之前的權重檔，重新訓練!"
        say_sth(msg, print_out=print_out)
        try:
            sess.run(tf.global_variables_initializer())
        except:
            say_sth(error_msg,print_out=print_out)
            status = False
    else:
        # ----get file created time and use the largest one
        ctime_list = list()
        for file in files:
            ctime_list.append(os.path.getctime(file))
        arg_max = int(np.argmax(ctime_list))

        # ----get the relative data file
        data_file = files[arg_max].rstrip('.meta')
        data_file += '.data-00000-of-00001'
        if not os.path.exists(data_file):
            msg = "Warning:之前的權重資料檔不存在:{}，進行重新訓練!".format(data_file)
            say_sth(msg, print_out=print_out)
            try:
                sess.run(tf.global_variables_initializer())
            except:
                say_sth(error_msg,print_out=print_out)
                status = False
        else:
            # ----file decode
            if encript_flag is True:
                # data_file = [file.path for file in os.scandir(self.save_dir) if
                #              file.name.split(".")[-1] == 'data-00000-of-00001']
                # print("data_file = ", data_file)
                file_decode_v2(data_file, random_num_range=encode_num, header=encode_header)
                time.sleep(0.1)
                file_decode_v2(files[arg_max], random_num_range=encode_num, header=encode_header)
                time.sleep(0.1)
            check_name = files[arg_max].split("\\")[-1].split(".")[0]
            model_path = os.path.join(save_dir, check_name)
            # print("model path:", model_path)
            try:
                saver.restore(sess, model_path)
                msg = "使用之前的權重檔:{}".format(model_path)
                say_sth(msg, print_out=print_out)
                # ----file encode
                if encript_flag is True:
                    file_transfer(data_file, random_num_range=encode_num, header=encode_header)
                    file_transfer(files[arg_max], random_num_range=encode_num, header=encode_header)
            except:
                msg = "恢復模型時產生錯誤"
                say_sth(msg, print_out=print_out)
                status = False
                try:
                    sess.run(tf.global_variables_initializer())
                    status = True
                except:
                    say_sth(error_msg)
                    status = False

    return status

def save_pb_file(sess,pb_save_list,pb_save_path,encode=False,random_num_range=87,header=[24,97,28,98]):
    graph = tf.get_default_graph().as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph, pb_save_list)
    with gfile.GFile(pb_save_path, 'wb')as f:
        f.write(output_graph_def.SerializeToString())

    # time.sleep(0.5)
    if encode is True:
        file_transfer(pb_save_path,random_num_range=random_num_range,header=header)

    msg = "儲存權重檔至 {}".format(pb_save_path)
    say_sth(msg,print_out=print_out)

def encode_CKPT(model_save_path,encode_num=87,encode_header=[24, 97, 28, 98]):
    file_transfer(model_save_path + '.meta', random_num_range=encode_num, header=encode_header)
    data_file = [file.path for file in os.scandir(os.path.dirname(model_save_path)) if
                 file.name.split(".")[-1] == 'data-00000-of-00001']
    file_transfer(data_file[0], random_num_range=encode_num, header=encode_header)

#----main class
class AE():
    def __init__(self,para_dict):

        #----config check(if exe)
        status, para_dict = self.config_check(para_dict)

        if status is True:
            #----var parsing
            train_img_dir = para_dict['train_img_dir']
            test_img_dir = para_dict.get('test_img_dir')
            recon_img_dir = para_dict.get('recon_img_dir')
            special_img_dir = para_dict.get('special_img_dir')
            show_data_qty = para_dict.get('show_data_qty')
            print_out = para_dict.get('print_out')

            #----local var
            recon_flag = False
            msg_list = list()

            #----train image paths
            tl = tools()
            self.train_paths,self.train_path_qty = tl.get_paths(train_img_dir)
            if self.train_path_qty == 0:
                say_sth("Error:訓練資料集沒有圖片",print_out=print_out)
                status = False
            else:
                msg = "訓練集圖片數量:{}".format(self.train_path_qty)
                msg_list.append(msg)

            #----test image paths
            if status is True:
                if test_img_dir is None:
                    msg = "沒有輸入驗證集路徑"
                else:
                    self.test_paths,self.test_path_qty = tl.get_paths(test_img_dir)
                    msg = "驗證集圖片數量:{}".format(self.test_path_qty)
                msg_list.append(msg)
                # say_sth(msg,print_out=print_out)


                if special_img_dir is None:
                    msg = "沒有輸入加強學習集路徑"
                else:
                    self.sp_paths,self.sp_path_qty = tl.get_paths(special_img_dir)
                    msg = "加強學習圖片數量:{}".format(self.sp_path_qty)
                msg_list.append(msg)
                # say_sth(msg, print_out=print_out)

                if recon_img_dir is None:
                    msg = "沒有輸入重建圖集路徑"
                else:
                    self.recon_paths,self.recon_path_qty = tl.get_paths(recon_img_dir)
                    if self.recon_path_qty > 0:
                        recon_flag = True
                    msg = "重建圖片數量:{}".format(self.recon_path_qty)

                msg_list.append(msg)



                #----display data info
                if show_data_qty is True:
                    for msg in msg_list:
                        say_sth(msg, print_out=print_out)


            # if recon_img_dir is not None:
            #     recon_paths,recon_path_qty = tl.get_paths(recon_img_dir)
            #     self.recon_img_dir = recon_img_dir
            #
            #     if len(recon_paths) > 0:
            #         self.recon_paths = recon_paths
            #         recon_flag = True
            #     else:
            #         print("recon img dir:{} has no files".format(recon_img_dir))

            #----log update
            content = dict()
            content = self.log_update(content, para_dict)

            #----local var to global
            self.train_img_dir = train_img_dir
            self.test_img_dir = test_img_dir
            self.special_img_dir = special_img_dir
            self.recon_img_dir = recon_img_dir
            self.content = content
            self.recon_flag = recon_flag
        self.status = status

    def model_init(self,para_dict):
        #----var parsing
        model_shape = para_dict['model_shape']  # [N,H,W,C]
        infer_method = para_dict['infer_method']
        activation = para_dict['activation']
        pool_kernel = para_dict['pool_kernel']
        kernel_list = para_dict['kernel_list']
        filter_list = para_dict['filter_list']
        conv_time = para_dict['conv_time']
        pool_type = para_dict.get('pool_type')
        loss_method = para_dict['loss_method']
        loss_method_2 = para_dict.get('loss_method_2')
        opti_method = para_dict['opti_method']
        lr = para_dict['learning_rate']
        save_dir = para_dict['save_dir']
        save_pb_name = para_dict['save_pb_name']
        embed_length = para_dict['embed_length']
        encript_flag = para_dict['encript_flag']
        scaler = para_dict.get('scaler')
        preprocess_dict = para_dict.get('preprocess_dict')
        rot = para_dict.get('rot')
        process_dict = para_dict['process_dict']
        print_out = para_dict.get('print_out')
        add_name_tail = para_dict.get('add_name_tail')
        stride_list = para_dict.get('stride_list')
        to_Vit = para_dict.get('to_Vit')
        dtype = para_dict.get('dtype')

        #----var
        #rot = False
        # bias = 0.5
        # br_ratio = 0
        # ct_ratio = 1
        pb_extension = 'pb'
        log_extension = 'json'
        pb_save_list = []

        #----var process
        if encript_flag is True:
            pb_extension = 'nst'
            log_extension = 'nst'
        if add_name_tail is None:
            add_name_tail = True
        if dtype is None:
            dtype = 'float32'

        #----random patch
        rdm_patch = False
        if process_dict.get('rdm_patch') is True:
            rdm_patch = True

        #----filer scaling process
        if scaler is not None:
            filter_list = (np.array(filter_list) / scaler).astype(np.uint16)

        # ----tf_placeholder declaration
        tf_input = tf.placeholder(dtype, shape=model_shape, name='input')
        tf_keep_prob = tf.placeholder(dtype=dtype, name="keep_prob")

        if rdm_patch is True:
            self.tf_input_ori = tf.placeholder(dtype, shape=model_shape, name='input_ori')

        #tf_label_batch = tf.placeholder(dtype=tf.int32, shape=[None], name="label_batch")
        #tf_phase_train = tf.placeholder(tf.bool, name="phase_train")

        # ----activation selection
        if activation == 'relu':
            acti_func = tf.nn.relu
        elif activation == 'mish':
            acti_func = tf_mish
        else:
            acti_func = tf.nn.relu

        avepool_out = self.__avepool(tf_input, k_size=5, strides=1)

        #----pre process
        #====pre-process dict process
        # if preprocess_dict is None:
        #     preprocess_dict = {'rot': rot, 'bias': bias, 'br_ratio': br_ratio, 'ct_ratio': ct_ratio}
        # else:
        #     if preprocess_dict.get('rot') is None:
        #         preprocess_dict['rot'] = rot
        #     if preprocess_dict.get('bias') is None:
        #         preprocess_dict['bias'] = bias
        #     if preprocess_dict.get('ct_ratio') is None:
        #         preprocess_dict['ct_ratio'] = ct_ratio
        #     if preprocess_dict.get('br_ratio') is None:
        #         preprocess_dict['br_ratio'] = br_ratio
        #
        # msg = "Pre processed:bias={},br_ratio={},ct_ratio={},rot={}".format(bias, br_ratio, ct_ratio, rot)
        # say_sth(msg,print_out=print_out)
        # pre_process = (tf_input - bias * (1 - br_ratio)) * ct_ratio
        # pre_process = tf.add(pre_process, bias * (1 + br_ratio))
        # tf_input_2 = tf.clip_by_value(pre_process, 0.0, 1.0, name='preprocess')

        #----inference selection
        if infer_method == "AE_transpose_4layer":
            recon = AE_transpose_4layer(tf_input, kernel_list, filter_list,activation=acti_func,
                                              pool_kernel=pool_kernel,pool_type=pool_type)
            recon = tf.identity(recon, name='output_AE')
            #(tf_input, kernel_list, filter_list,pool_kernel=2,activation=tf.nn.relu,pool_type=None)
            # recon = AE_refinement(temp,96)
        elif infer_method == "AE_VIT":
            recon = AE_VIT(tf_input, kernel_list, filter_list,activation=acti_func,
                                              pool_kernel_list=pool_kernel,pool_type_list=pool_type,
                                   rot=rot,print_out=print_out,preprocess_dict=preprocess_dict)
            recon = tf.identity(recon, name='output_AE')
        elif infer_method == "AE_pooling_net":
            recon = AE_pooling_net(tf_input, kernel_list, filter_list,activation=acti_func,
                                              pool_kernel_list=pool_kernel,pool_type_list=pool_type,
                                   stride_list=stride_list,rot=rot,print_out=print_out)
            recon = tf.identity(recon,name='output_AE')
        elif infer_method == "AE_pooling_net_V2":#add the transformer
            recon = AE_pooling_net_V2(tf_input, kernel_list, filter_list,stride_list,activation=acti_func,
                                              pool_kernel_list=pool_kernel,pool_type_list=pool_type,
                                   to_Vit=to_Vit,print_out=print_out)
            recon = tf.identity(recon,name='output_AE')
        elif infer_method == "AE_Unet":
            recon = AE_Unet(tf_input, kernel_list, filter_list,activation=acti_func,
                                              pool_kernel_list=pool_kernel,pool_type_list=pool_type,
                                   rot=rot,print_out=print_out,preprocess_dict=preprocess_dict)
            recon = tf.identity(recon,name='output_AE')
        elif infer_method == "AE_JNet":
            recon = AE_JNet(tf_input, kernel_list, filter_list,activation=acti_func,
                                            rot=rot,pool_kernel=pool_kernel,pool_type=pool_type)
        # elif infer_method == "test":
        #     recon = self.__AE_transpose_4layer_test(tf_input, kernel_list, filter_list,
        #                                        conv_time=conv_time,maxpool_kernel=maxpool_kernel)
        # elif infer_method == 'inception_resnet_v1_reduction':
        #     recon = AE_incep_resnet_v1(tf_input=tf_input,tf_keep_prob=tf_keep_prob,embed_length=embed_length,
        #                                scaler=scaler,kernel_list=kernel_list,filter_list=filter_list,
        #                                activation=acti_func,)
        elif infer_method == "Resnet_Rot":
            filter_list = [12, 16, 24, 36, 48, 196]
            recon = AE_Resnet_Rot(tf_input,filter_list,tf_keep_prob,embed_length,activation=acti_func,
                                  print_out=True,rot=True)

        #----loss method selection
        if loss_method == 'mse':
            loss_AE = tf.reduce_mean(tf.pow(recon - tf_input,2), name="loss_AE")
        elif loss_method =='ssim':
            # self.loss_AE = tf.reduce_mean(tf.image.ssim_multiscale(tf.image.rgb_to_grayscale(self.tf_input),tf.image.rgb_to_grayscale(self.recon),2),name='loss')

            if rdm_patch is True:
                loss_AE = tf.reduce_mean(tf.image.ssim(self.tf_input_ori, recon, 2.0), name='loss_AE')
            else:
                loss_AE = tf.reduce_mean(tf.image.ssim(tf_input, recon, 2.0), name='loss_AE')
        elif loss_method == "huber":
            loss_AE = tf.reduce_sum(tf.losses.huber_loss(tf_input, recon, delta=1.35),name='loss_AE')
        elif loss_method == 'ssim+mse':
            loss_1 = tf.reduce_mean(tf.pow(recon - tf_input, 2))
            loss_2 = tf.reduce_mean(tf.image.ssim(tf_input, recon, 2.0))
            loss_AE = tf.subtract(loss_2,loss_1,name='loss_AE')
        elif loss_method == 'cross_entropy':
            loss_AE = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.layers.flatten(tf_input),
                                                           logits=tf.layers.flatten(recon)),name="loss_AE")
        elif loss_method == 'kl_d':
            epsilon = 1e-8
            # generation loss(cross entropy)
            loss_AE = tf.reduce_mean(
                tf_input * tf.subtract(tf.log(epsilon + tf_input), tf.log(epsilon + recon)) , name='loss_AE')

        #----Second loss method selection
        # if loss_method_2 is not None:
        #     if loss_method_2 == 'mse':
        #         loss_AE_2 = tf.reduce_mean(tf.pow(recon_1 - tf_input,2), name="loss_AE_2")
        #     elif loss_method_2 == 'ssim':
        #         if rdm_patch is True:
        #             loss_AE_2 = tf.reduce_mean(tf.image.ssim(self.tf_input_ori, recon_1, 2.0), name='loss_AE_2')
        #         else:
        #             loss_AE_2 = tf.reduce_mean(tf.image.ssim(tf_input, recon_1, 2.0), name='loss_AE_2')

        #----optimizer selection
        if opti_method == "adam":
            if loss_method.find('ssim') >= 0:
                opt_AE = tf.train.AdamOptimizer(learning_rate=lr).minimize(-loss_AE)
            else:
                opt_AE = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_AE)

            #----for second loss
            # if loss_method_2 is not None:
            #     if loss_method_2.find('ssim') >= 0:
            #         opt_AE_2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(-loss_AE_2)
            #     else:
            #         opt_AE_2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_AE_2)

        # ----create the dir to save model weights(CKPT, PB)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.recon_flag is True:
            new_recon_dir = "img_recon-" + os.path.basename(save_dir)
            #----if recon_img_dir is list format
            if isinstance(self.recon_img_dir,list):
                self.recon_img_dir = self.recon_img_dir[0]
            self.new_recon_dir = os.path.join(self.recon_img_dir, new_recon_dir)
            if not os.path.exists(self.new_recon_dir):
                os.makedirs(self.new_recon_dir)

        out_dir_prefix = os.path.join(save_dir, "model")
        saver = tf.train.Saver(max_to_keep=2)

        # ----PB file save filename
        if add_name_tail is True:
            xtime = time.localtime()
            name_tailer = ''
            for i in range(6):
                string = str(xtime[i])
                if len(string) == 1:
                    string = '0' + string
                name_tailer += string
            pb_save_path = "{}_{}.{}".format(save_pb_name, name_tailer, pb_extension)
        else:
            pb_save_path = "{}.{}".format(save_pb_name, pb_extension)
        pb_save_path = os.path.join(save_dir, pb_save_path)

        # ----appoint PB node names
        pb_save_list.extend(['output_AE', "loss_AE"])

        # ----create the log(JSON)
        count = 0
        for i in range(1000):
            log_path = "{}_{}.{}".format('train_result', count, log_extension)
            log_path = os.path.join(save_dir, log_path)
            if not os.path.exists(log_path):
                break
            count += 1
        self.content = self.log_update(self.content, para_dict)

        # ----local var to global
        self.model_shape = model_shape
        self.tf_input = tf_input
        self.tf_keep_prob = tf_keep_prob
        self.avepool_out = avepool_out
        self.recon = recon
        self.loss_AE = loss_AE
        self.opt_AE = opt_AE
        self.out_dir_prefix = out_dir_prefix
        self.saver = saver
        self.save_dir = save_dir
        self.pb_save_path = pb_save_path
        self.pb_save_list = pb_save_list
        self.pb_extension = pb_extension
        self.log_path = log_path
        self.loss_method = loss_method
        self.loss_method_2 = loss_method_2
        self.dtype = dtype
        # if loss_method_2 is not None:
        #     self.loss_AE_2 = loss_AE_2
        #     self.opt_AE_2 = opt_AE_2

    def train(self,para_dict):
        # ----var parsing
        epochs = para_dict['epochs']
        GPU_ratio = para_dict.get('GPU_ratio')
        aug_times = para_dict.get('aug_times')
        batch_size = para_dict['batch_size']
        ratio = para_dict.get('ratio')
        process_dict = para_dict.get('process_dict')
        eval_epochs = para_dict.get('eval_epochs')
        setting_dict = para_dict.get('setting_dict')
        encode_header = para_dict.get('encode_header')
        encode_num = para_dict.get('encode_num')
        encript_flag = para_dict.get('encript_flag')
        save_period = para_dict.get('save_period')
        target_dict = para_dict.get('target')
        print_out = para_dict.get('print_out')
        to_read_manual_cmd = para_dict.get('to_read_manual_cmd')

        #----special_img_dir
        if self.special_img_dir is not None:
            special_img_ratio = para_dict.get('special_img_ratio')
            if special_img_ratio is None:
                special_img_ratio = 0.04
            elif special_img_ratio > 1.0:
                special_img_ratio = 1.0
            elif special_img_ratio < 0:
                special_img_ratio = 0.01

        if encode_header is None:
            encode_header = [24,97,28,98]
        if encode_num is None:
            encode_num = 87
        if save_period is None:
            save_period = 1

        # ----local var
        train_loss_list = list()
        train_acc_list = list()
        test_loss_list = list()
        test_acc_list = list()
        epoch_time_list = list()
        img_quantity = 0
        aug_enable = False
        break_flag = False
        error_dict = {'GPU_resource_error': False}
        train_result_dict = {'loss': 0,"loss_method":self.loss_method}
        test_result_dict = {'loss': 0,"loss_method":self.loss_method}
        tl = tools()
        record_type = 'loss'
        pb_save_path_old = ''
        qty_sp = 0

        #----set target
        tl.set_target(target_dict)

        # ratio = 1.0
        batch_size_test = batch_size

        # ----check if the augmentation(image processing) is enabled
        if isinstance(process_dict, dict):
            if True in process_dict.values():
                aug_enable = True
                if aug_times is None:
                    aug_times = 2
                batch_size = batch_size // aug_times  # the batch size must be integer!!

        if aug_enable is True:
            tl.set_process(process_dict,setting_dict)

        #----update content
        self.content = self.log_update(self.content, para_dict)

        #----read the manual cmd
        if to_read_manual_cmd is True:
            j_path = os.path.join(self.save_dir, 'manual_cmd.json')


        # ----calculate iterations of one epoch
        # train_ites = math.ceil(img_quantity / batch_size)
        # test_ites = math.ceil(len(self.test_paths) / batch_size)

        t_train_start = time.time()
        #----GPU setting
        config = GPU_setting(GPU_ratio)
        with tf.Session(config=config) as sess:
            status = weights_check(sess, self.saver, self.save_dir, encript_flag=encript_flag,
                                   encode_num=encode_num, encode_header=encode_header)
            if status is False:
                error_dict['GPU_resource_error'] = True
            elif status is True:


            # ----info display

                #----train set quantity
                qty_train = self.train_path_qty
                if ratio is not None:
                    if ratio <= 1.0 and ratio > 0:
                        qty_train = int(self.train_path_qty * ratio)
                        qty_train = np.maximum(1, qty_train)

                msg = "訓練集資料總共取數量{}".format(qty_train)
                say_sth(msg, print_out=print_out)

                #----special set quantity
                if self.special_img_dir is not None:
                    qty_sp = int(qty_train * special_img_ratio)
                    msg = "加強學習資料總共取數量{}".format(qty_sp)
                    say_sth(msg, print_out=print_out)

                # ----calculate iterations of one epoch
                img_quantity = qty_train + qty_sp
                train_ites = math.ceil(img_quantity / batch_size)
                if self.test_img_dir is not None:
                    test_ites = math.ceil(self.test_path_qty / batch_size_test)

                # ----epoch training
                for epoch in range(epochs):
                    # ----read manual cmd
                    if to_read_manual_cmd is True:
                        if os.path.exists(j_path):
                            with open(j_path,'r') as f:
                                cmd_dict = json.load(f)
                            if cmd_dict.get('to_stop_training') is True:
                                break_flag = True
                                msg = "接收到manual cmd: stop the training!"
                                say_sth(msg, print_out=print_out)

                    # ----break the training
                    if break_flag:
                        break_flag = False
                        break

                    # ----error check
                    if True in list(error_dict.values()):
                        break
                    # ----record the start time
                    d_t = time.time()

                    train_loss = 0
                    train_loss_2 = 0
                    train_acc = 0
                    test_loss = 0
                    test_loss_2 = 0
                    test_acc = 0

                    #----train img dir


                    #----shuffle
                    indice = np.random.permutation(self.train_path_qty)
                    self.train_paths = self.train_paths[indice]
                    train_paths_ori = self.train_paths[:qty_train]

                    #----special img dir
                    if self.special_img_dir is not None:
                        #----shuffle for special set
                        indice = np.random.permutation(self.sp_path_qty)
                        self.sp_paths = self.sp_paths[indice]

                        if self.sp_path_qty < qty_sp:
                            multi_ratio = math.ceil(qty_sp / self.sp_path_qty)
                            sp_paths = np.array(list(self.sp_paths) * multi_ratio)
                        else:
                            sp_paths = self.sp_paths

                        train_paths_ori = np.concatenate([train_paths_ori, sp_paths[:qty_sp]], axis=-1)

                        #-----shuffle for (train set + special set)
                        indice = np.random.permutation(img_quantity)
                        train_paths_ori = train_paths_ori[indice]


                    if aug_enable is True:
                        train_paths_aug = train_paths_ori[::-1]
                        #train_labels_aug = train_labels_ori[::-1]

                    # ----optimizations(train set)
                    for index in range(train_ites):
                        # ----command process
                        if SockConnected:
                            # print(Sock.Message)
                            if len(Sock.Message):
                                if Sock.Message[-1][:4] == "$S00":
                                    # Sock.send("Protocol:Ack\n")
                                    model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)
                                    # ----encode ckpt file
                                    if encript_flag is True:
                                        file = model_save_path + '.meta'
                                        if os.path.exists(file):
                                            file_transfer(file, random_num_range=encode_num, header=encode_header)
                                        else:
                                            msg = "Warning:找不到權重檔:{}進行處理".format(file)
                                            say_sth(msg, print_out=print_out)
                                        # data_file = [file.path for file in os.scandir(self.save_dir) if
                                        #              file.name.split(".")[-1] == 'data-00000-of-00001']
                                        data_file = model_save_path + '.data-00000-of-00001'
                                        if os.path.exists(data_file):
                                            file_transfer(data_file, random_num_range=encode_num, header=encode_header)
                                        else:
                                            msg = "Warning:找不到權重檔:{}進行處理".format(data_file)
                                            say_sth(msg, print_out=print_out)

                                    # msg = "儲存訓練權重檔至{}".format(model_save_path)
                                    # say_sth(msg, print_out=print_out)
                                    save_pb_file(sess, self.pb_save_list, self.pb_save_path, encode=encript_flag,
                                                      random_num_range=encode_num, header=encode_header)
                                    break_flag = True
                                    break



                        # ----get image start and end numbers
                        ori_paths = tl.get_ite_data(train_paths_ori,index,batch_size=batch_size)
                        aug_paths = tl.get_ite_data(train_paths_aug,index,batch_size=batch_size)

                        # ----get 4-D data
                        if aug_enable is True:
                            # ----ori data
                            # ori_data = get_4D_data(ori_paths, self.model_shape[1:],process_dict=None)
                            ori_data = tl.get_4D_data(ori_paths, self.model_shape[1:], to_norm=True, to_rgb=True,
                                                        to_process=False,dtype=self.dtype)

                            #ori_labels = train_labels_ori[num_start:num_end]
                            # ----aug data
                            if process_dict.get('rdm_patch'):
                                # aug_data_no_patch,aug_data = get_4D_data(aug_paths, self.model_shape[1:],
                                #                         process_dict=process_dict,setting_dict=setting_dict)
                                aug_data_no_patch, aug_data = tl.get_4D_data(aug_paths, self.model_shape[1:],
                                                                             to_norm=True,
                                                                             to_rgb=True,
                                                                             to_process=True,
                                                                             dtype=self.dtype)
                                batch_data_no_patch = np.concatenate([ori_data, aug_data_no_patch], axis=0)
                            else:
                                # aug_data = get_4D_data(aug_paths, self.model_shape[1:],
                                #                         process_dict=process_dict,setting_dict=setting_dict)
                                aug_data = tl.get_4D_data(aug_paths, self.model_shape[1:],
                                                          to_norm=True,
                                                          to_rgb=True,
                                                          to_process=True,
                                                          dtype=self.dtype)
                            #aug_labels = train_labels_aug[num_start:num_end]
                            # ----data concat
                            batch_data = np.concatenate([ori_data, aug_data], axis=0)
                            # if process_dict.get('rdm_patch'):
                            #     batch_data_ori = np.concatenate([ori_data, ori_data], axis=0)
                            #batch_labels = np.concatenate([ori_labels, aug_labels], axis=0)
                        else:
                            # batch_data = get_4D_data(ori_paths, self.model_shape[1:])
                            batch_data = tl.get_4D_data(ori_paths, self.model_shape[1:],dtype=self.dtype)
                            #batch_labels = train_labels_ori[num_start:num_end]

                        #----put all data to tf placeholders
                        if process_dict.get('rdm_patch') is True:
                            feed_dict = {self.tf_input: batch_data, self.tf_input_ori: batch_data_no_patch,
                                         self.tf_keep_prob: 0.5}
                        else:
                            feed_dict = {self.tf_input: batch_data, self.tf_keep_prob: 0.5}


                        # ----session run
                        try:
                            sess.run(self.opt_AE,feed_dict=feed_dict)
                        except:
                            error_dict['GPU_resource_error'] = True
                            msg = "Error:權重最佳化時產生錯誤"
                            say_sth(msg, print_out=print_out)
                            break
                        # if self.loss_method_2 is not None:
                        #     sess.run(self.opt_AE_2, feed_dict=feed_dict)

                        # ----evaluation(training set)
                        feed_dict[self.tf_keep_prob] = 1.0
                        # feed_dict[self.tf_phase_train] = False
                        loss_temp = sess.run(self.loss_AE, feed_dict=feed_dict)
                        # if self.loss_method_2 is not None:
                        #     loss_temp_2 = sess.run(self.loss_AE_2, feed_dict=feed_dict)

                        # ----calculate the loss and accuracy
                        train_loss += loss_temp
                        # if self.loss_method_2 is not None:
                        #     train_loss_2 += loss_temp_2

                    train_loss /= train_ites
                    train_result_dict['loss'] = train_loss
                    # if self.loss_method_2 is not None:
                    #     train_loss_2 /= train_ites

                    # ----break the training
                    if break_flag:
                        break_flag = False
                        break
                    if True in list(error_dict.values()):
                        break


                    #----evaluation(test set)
                    if self.test_img_dir is not None:
                        for index in range(test_ites):
                            # ----get image start and end numbers
                            ite_paths = tl.get_ite_data(self.test_paths, index, batch_size=batch_size_test)

                            # batch_data = get_4D_data(ite_paths, self.model_shape[1:])
                            batch_data = tl.get_4D_data(ite_paths, self.model_shape[1:],dtype=self.dtype)

                            # ----put all data to tf placeholders
                            if process_dict.get('rdm_patch'):
                                feed_dict = {self.tf_input: batch_data, self.tf_input_ori: batch_data,
                                             self.tf_keep_prob: 1.0}
                            else:
                                feed_dict = {self.tf_input: batch_data, self.tf_keep_prob: 1.0}

                            # ----session run
                            #sess.run(self.opt_AE, feed_dict=feed_dict)

                            # ----evaluation(training set)
                            # feed_dict[self.tf_phase_train] = False
                            try:
                                loss_temp = sess.run(self.loss_AE, feed_dict=feed_dict)
                            except:
                                error_dict['GPU_resource_error'] = True
                                msg = "Error:推論驗證集時產生錯誤"
                                say_sth(msg, print_out=print_out)
                                break
                            # if self.loss_method_2 is not None:
                            #     loss_temp_2 = sess.run(self.loss_AE_2, feed_dict=feed_dict)

                            # ----calculate the loss and accuracy
                            test_loss += loss_temp
                            # if self.loss_method_2 is not None:
                            #     test_loss_2 += loss_temp_2

                        test_loss /= test_ites
                        test_result_dict['loss'] = test_loss
                        # if self.loss_method_2 is not None:
                        #     test_loss_2 /= test_ites

                    #----save ckpt, pb files
                    if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
                        model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)

                        #----encode ckpt file
                        if encript_flag is True:
                            encode_CKPT(model_save_path, encode_num=encode_num, encode_header=encode_header)

                        #----save pb(normal)
                        save_pb_file(sess, self.pb_save_list, self.pb_save_path, encode=encript_flag,
                                     random_num_range=encode_num, header=encode_header)

                    # ----record the end time
                    d_t = time.time() - d_t

                    #----save results in the log file
                    train_loss_list.append(float(train_loss))
                    self.content["train_loss_list"] = train_loss_list
                    #train_acc_list.append(float(train_acc))
                    #self.content["train_acc_list"] = train_acc_list

                    if self.test_img_dir is not None:
                        test_loss_list.append(float(test_loss))
                        #test_acc_list.append(float(test_acc))
                        self.content["test_loss_list"] = test_loss_list

                    # if self.test_img_dir is not None:
                    #     # self.content["test_loss_list"] = test_loss_list
                    #     self.content["test_acc_list"] = test_acc_list

                    epoch_time_list.append(d_t)
                    total_train_time = time.time() - t_train_start
                    self.content['total_train_time'] = float(total_train_time)
                    self.content['ave_epoch_time'] = float(np.average(epoch_time_list))

                    with open(self.log_path, 'w') as f:
                        json.dump(self.content, f)

                    if encript_flag is True:
                        if os.path.exists(self.log_path):
                            file_transfer(self.log_path, cut_num_range=30, random_num_range=10)
                    msg = "儲存訓練結果數據至{}".format(self.log_path)
                    say_sth(msg, print_out=print_out)

                    #----display training results
                    msg_list = list()
                    msg_list.append("\n----訓練週期 {} 與相關結果如下----".format(epoch))
                    msg_list.append("訓練集loss:{}".format(np.round(train_loss, 4)))
                    if self.test_img_dir is not None:
                        msg_list.append("驗證集loss:{}".format(np.round(test_loss, 4)))

                    msg_list.append("此次訓練所花時間: {}秒".format(np.round(d_t, 2)))
                    msg_list.append("累積訓練時間: {}秒".format(np.round(total_train_time, 2)))
                    say_sth(msg_list, print_out=print_out, header='msg')

                    #----send protocol data(for C++ to draw)
                    msg_list, header_list = self.collect_data2ui(epoch, train_result_dict,
                                                                 test_result_dict)
                    say_sth(msg_list, print_out=print_out, header=header_list)

                    #----find the best performance
                    new_value = self.__get_result_value(record_type, train_result_dict, test_result_dict)
                    if epoch == 0:
                        record_value = new_value
                    else:

                        go_flag = False
                        if self.loss_method == 'ssim':
                            if new_value > record_value:
                                go_flag = True
                        else:
                            if new_value < record_value:
                                go_flag = True

                        if go_flag is True:
                            # ----delete the previous pb
                            if os.path.exists(pb_save_path_old):
                                os.remove(pb_save_path_old)

                            #----save the better one
                            pb_save_path = "infer_{}.{}".format(new_value, self.pb_extension)
                            pb_save_path = os.path.join(self.save_dir, pb_save_path)

                            save_pb_file(sess, self.pb_save_list, pb_save_path, encode=encript_flag,
                                              random_num_range=encode_num, header=encode_header)
                            #----update record value
                            record_value = new_value
                            pb_save_path_old = pb_save_path

                    #----Check if stops the training
                    if target_dict['type'] == 'loss':
                        if self.test_img_dir is not None:
                            re = tl.target_compare(test_result_dict)
                            name = "驗證集"
                        else:
                            re = tl.target_compare(train_result_dict)
                            name = "訓練集"

                        if re is True:
                            msg = '模型訓練結束:\n{}{}已經達到設定目標:{}累積達{}次'.format(
                                name, target_dict['type'],target_dict['value'], target_dict['hit_target_times'])
                            say_sth(msg, print_out=print_out)
                            break

                    # ----test image reconstruction
                    if self.recon_flag is True:
                        # if (epoch + 1) % eval_epochs == 0 and train_loss > 0.80:
                        if (epoch + 1) % eval_epochs == 0:
                            for filename in self.recon_paths:
                                test_img = self.__img_read(filename, self.model_shape[1:],dtype=self.dtype)
                                # ----session run
                                img_sess_out = sess.run(self.recon, feed_dict={self.tf_input: test_img,
                                                                               self.tf_keep_prob:1.0})
                                # ----process of sess-out
                                img_sess_out = img_sess_out[0] * 255
                                img_sess_out = cv2.convertScaleAbs(img_sess_out)
                                if self.model_shape[3] == 1:
                                    img_sess_out = np.reshape(img_sess_out, (self.model_shape[1], self.model_shape[2]))
                                else:
                                    img_sess_out = cv2.cvtColor(img_sess_out, cv2.COLOR_RGB2BGR)

                                # if loss_method != 'ssim':
                                #     img_sess_out = cv2.cvtColor(img_sess_out, cv2.COLOR_RGB2BGR)

                                # ----save recon image
                                splits = filename.split("\\")[-1]
                                new_filename = splits.split('.')[0] + '_sess-out.' + splits.split('.')[-1]

                                new_filename = os.path.join(self.new_recon_dir, new_filename)
                                # cv2.imwrite(new_filename, img_sess_out)
                                cv2.imencode('.'+splits.split('.')[-1], img_sess_out)[1].tofile(new_filename)
                                # ----img diff method
                                img_diff = self.__img_diff_method(filename, img_sess_out, diff_th=15, cc_th=15)
                                img_diff = cv2.convertScaleAbs(img_diff)
                                new_filename = filename.split("\\")[-1]
                                new_filename = new_filename.split(".")[0] + '_diff.' + new_filename.split(".")[-1]
                                new_filename = os.path.join(self.new_recon_dir, new_filename)
                                # cv2.imwrite(new_filename, img_diff)
                                cv2.imencode('.' + splits.split('.')[-1], img_diff)[1].tofile(new_filename)

                                # ----img avepool diff method
                                img_diff = self.__img_patch_diff_method(filename, img_sess_out, sess, diff_th=30, cc_th=15)
                                img_diff = cv2.convertScaleAbs(img_diff)
                                new_filename = filename.split("\\")[-1]
                                new_filename = new_filename.split(".")[0] + '_avepool_diff.' + new_filename.split(".")[-1]
                                new_filename = os.path.join(self.new_recon_dir, new_filename)
                                # cv2.imwrite(new_filename, img_diff)
                                cv2.imencode('.' + splits.split('.')[-1], img_diff)[1].tofile(new_filename)

                                # ----SSIM method
                                # img_ssim = self.__ssim_method(filename, img_sess_out)
                                # img_ssim = cv2.convertScaleAbs(img_ssim)
                                # new_filename = filename.split("\\")[-1]
                                # new_filename = new_filename.split(".")[0] + '_ssim.' + new_filename.split(".")[-1]
                                # new_filename = os.path.join(self.new_recon_dir, new_filename)
                                # cv2.imwrite(new_filename, img_ssim)

            #----error messages
            if True in list(error_dict.values()):
                for key, value in error_dict.items():
                    if value is True:
                        say_sth('', print_out=print_out, header=key)
            else:
                say_sth('AI Engine結束!!期待下次再相見', print_out=print_out, header='AIE_end')

    #----functions
    def config_check(self, config_dict):
        # ----var
        must_list = ['train_img_dir', 'model_name', 'save_dir', 'epochs']
        # must_list = ['train_img_dir', 'test_img_dir', 'save_dir', 'epochs']
        must_flag = True
        default_dict = {"model_shape": [None, 192, 192, 3],
                        'model_name': "type_1_0",
                        'loss_method': 'ssim',
                        'activation': 'relu',
                        'save_pb_name': 'inference',
                        'opti_method': 'adam',
                        'pool_type': ['max', 'ave'],
                        'pool_kernel': [7, 2],
                        'embed_length': 144,
                        'learning_rate': 1e-4,
                        'batch_size': 8,
                        'ratio': 1.0,
                        'aug_times': 2,
                        'hit_target_times': 2,
                        'eval_epochs': 2,
                        'save_period': 2,
                        'kernel_list': [7, 5, 3, 3, 3],
                        'filter_list': [32, 64, 96, 128, 256],
                        'conv_time': 1,
                        'rot': False,
                        'scaler': 1,
                        'dtype': 'float32',
                        'process_dict': {"rdm_flip": True, 'rdm_br': True, 'rdm_blur': True,
                                         'rdm_angle': True,
                                         'rdm_noise': False,
                                         'rdm_shift': True,
                                         'rdm_patch': True,
                                         },
                        'setting_dict': {'rdm_shift': 0.1, 'rdm_angle': 10, 'rdm_patch': [0.25, 0.3, 10]},
                        # rdm_patch:[margin_ratio,patch_ratio,size_min]
                        'show_data_qty': True
                        }

        # ----get the must list
        if config_dict.get('must_list') is not None:
            must_list = config_dict.get('must_list')
        # ----collect all keys of config_dict
        config_key_list = list(config_dict.keys())

        # ----check the must list
        if config_dict.get("J_mode") is not True:

            # ----check of must items
            for item in must_list:
                if not item in config_key_list:
                    msg = "Error: could you plz give me parameters -> {}".format(item)
                    say_sth(msg, print_out=print_out)
                    if must_flag is True:
                        must_flag = False

        # ----parameters parsing
        if must_flag is True:
            # ----model name
            if config_dict.get("J_mode") is not True:
                infer_num = config_dict['model_name'].split("_")[-1]
                if infer_num == '0':  #
                    config_dict['infer_method'] = "AE_pooling_net"
                elif infer_num == '1':  #
                    config_dict['infer_method'] = "AE_Unet"
                elif infer_num == "2":
                    config_dict['infer_method'] = "AE_VIT"
                elif infer_num == "3":
                    config_dict['infer_method'] = "AE_pooling_net_V2"
                else:
                    config_dict['infer_method'] = "AE_transpose_4layer"

            # ----optional parameters
            for key, value in default_dict.items():
                if not key in config_key_list:
                    config_dict[key] = value

        return must_flag, config_dict

    def __img_patch_diff_method(self,img_source_1, img_source_2, sess,diff_th=30, cc_th=30):

        temp = np.array([1., 2., 3.])
        re = None
        # ----read img source 1
        if isinstance(temp, type(img_source_1)):
            img_1 = img_source_1
        elif os.path.isfile(img_source_1):
            # img_1 = cv2.imread(img_source_1)
            img_1 = cv2.imdecode(np.fromfile(img_source_1, dtype=np.uint8), 1)
            img_1 = cv2.resize(img_1,(self.model_shape[2],self.model_shape[1]))
            # img_1 = img_1.astype('float32')

        else:
            print("The type of img_source_1 is not supported")

        # ----read img source 2
        if isinstance(temp, type(img_source_2)):
            img_2 = img_source_2
        elif os.path.isfile(img_source_2):
            # img_2 = cv2.imread(img_source_2)
            img_2 = cv2.imdecode(np.fromfile(img_source_2, dtype=np.uint8), 1)
            # img_2 = img_2.astype('float32')
        else:
            print("The type of img_source_2 is not supported")

        # ----subtraction
        if img_1 is not None and img_2 is not None:
            img_1_ave_pool = sess.run(self.avepool_out,feed_dict={self.tf_input:np.expand_dims(img_1,axis=0)})
            img_2_ave_pool = sess.run(self.avepool_out,feed_dict={self.tf_input:np.expand_dims(img_2,axis=0)})
            # print("img_1 shape = {}, dtype = {}".format(img_1.shape, img_1.dtype))
            # print("img_2 shape = {}, dtype = {}".format(img_2.shape, img_2.dtype))
            img_diff = cv2.absdiff(img_1_ave_pool[0], img_2_ave_pool[0])  # img_diff = np.abs(img - img_sess)#不能使用，因為相減若<0，會取補數
            if img_1.shape[-1] == 3:
                img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)  # 利用轉成灰階，減少RGB的相加計算量

            # 連通
            img_compare = cv2.compare(img_diff, diff_th, cv2.CMP_GT)
            retval, labels = cv2.connectedComponents(img_compare)
            max_label_num = np.max(labels) + 1

            img_1_copy = img_1.copy()
            for i in range(0, max_label_num):  # label = 0是背景，所以從1開始
                y, x = np.where(labels == i)  # y,x代表類別=i的像素座標值
                if (y.shape[0] > cc_th) and (img_compare[y[0], x[0]] == 255):
                    for j in range(y.shape[0]):
                        img_1_copy.itemset((y[j], x[j], 0), 0)
                        img_1_copy.itemset((y[j], x[j], 1), 0)
                        img_1_copy.itemset((y[j], x[j], 2), 255)

            re = img_1_copy
            return re

    def __img_diff_method(self,img_source_1, img_source_2, diff_th=30, cc_th=30):

        temp = np.array([1., 2., 3.])
        re = None
        # ----read img source 1
        if isinstance(temp, type(img_source_1)):
            img_1 = img_source_1
        elif os.path.isfile(img_source_1):
            # img_1 = cv2.imread(img_source_1)
            img_1 = cv2.imdecode(np.fromfile(img_source_1, dtype=np.uint8), 1)
            img_1 = cv2.resize(img_1,(self.model_shape[2],self.model_shape[1]))
            # img_1 = img_1.astype('float32')

        else:
            print("The type of img_source_1 is not supported")

        # ----read img source 2
        if isinstance(temp, type(img_source_2)):
            img_2 = img_source_2
        elif os.path.isfile(img_source_2):
            # img_2 = cv2.imread(img_source_2)
            img_2 = cv2.imdecode(np.fromfile(img_source_2, dtype=np.uint8), 1)
            # img_2 = img_2.astype('float32')
        else:
            print("The type of img_source_2 is not supported")

        # ----substraction
        if img_1 is not None and img_2 is not None:
            # print("img_1 shape = {}, dtype = {}".format(img_1.shape, img_1.dtype))
            # print("img_2 shape = {}, dtype = {}".format(img_2.shape, img_2.dtype))
            img_diff = cv2.absdiff(img_1, img_2)  # img_diff = np.abs(img - img_sess)#不能使用，因為相減若<0，會取補數
            if img_1.shape[-1] == 3:
                img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)  # 利用轉成灰階，減少RGB的相加計算量

            # 連通
            img_compare = cv2.compare(img_diff, diff_th, cv2.CMP_GT)
            retval, labels = cv2.connectedComponents(img_compare)
            max_label_num = np.max(labels) + 1

            img_1_copy = img_1.copy()
            for i in range(0, max_label_num):  # label = 0是背景，所以從1開始
                y, x = np.where(labels == i)  # y,x代表類別=i的像素座標值
                if (y.shape[0] > cc_th) and (img_compare[y[0], x[0]] == 255):
                    for j in range(y.shape[0]):
                        img_1_copy.itemset((y[j], x[j], 0), 0)
                        img_1_copy.itemset((y[j], x[j], 1), 0)
                        img_1_copy.itemset((y[j], x[j], 2), 255)

            re = img_1_copy
            return re

    def __avepool(self,input_x,k_size=3,strides=1):
        kernel = [1,k_size,k_size,1]
        stride_kernel = [1,strides,strides,1]
        return tf.nn.avg_pool(input_x, ksize=kernel, strides=stride_kernel, padding='SAME')

    def __Conv(self,input_x,kernel=[3,3],filter=32,conv_times=2,stride=1):
        net = None
        for i in range(conv_times):
            if i == 0:
                net = tf.layers.conv2d(
                    inputs=input_x,
                    filters=filter,
                    kernel_size=kernel,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                    strides=stride,
                    padding="same",
                    activation=tf.nn.relu)
            else:
                net = tf.layers.conv2d(
                    inputs=net,
                    filters=filter,
                    kernel_size=kernel,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                    strides=stride,
                    padding="same",
                    activation=tf.nn.relu)
        return net

    def say_sth(self,msg, print_out=False):
        if print_out:
            print(msg)

    def log_update(self,content,para_dict):
        for key, value in para_dict.items():
            content[key] = value

        return content

    def __img_read(self, img_path, shape,dtype='float32'):

        # img = cv2.imread(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        if img is None:
            print("Read failed:",img_path)
            return None
        else:
            img = cv2.resize(img,(shape[1],shape[0]))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img.astype(dtype)
            img /= 255

            return np.expand_dims(img,axis=0)

    def __get_result_value(self,record_type,train_dict,test_dict):
        value = None
        if record_type == 'loss':
            if self.test_img_dir is not None:
                value = test_dict['loss']
            else:
                value = train_dict['loss']

            if self.loss_method == 'ssim':
                value = np.round(value * 100, 2)
            else:
                value = np.round(value, 2)

        return value

    def collect_data2ui(self,epoch,train_dict,test_dict):
        #----var
        msg_list = list()
        header_list = list()
        #----process
        msg_list.append('{},{}'.format(epoch, train_dict['loss']))
        header_list.append('train_loss')
        # msg_list.append('{},{}'.format(epoch, train_dict['accuracy']))
        # header_list.append('train_acc')
        if self.test_img_dir is not None:
            msg_list.append('{},{}'.format(epoch, test_dict['loss']))
            header_list.append('val_loss')
            # msg_list.append('{},{}'.format(epoch, test_dict['accuracy']))
            # header_list.append('val_acc')


        return msg_list,header_list
    #----models

    def __AE_transpose_4layer_test(self, input_x, kernel_list, filter_list,conv_time=1,maxpool_kernel=2):
        #----var
        maxpool_kernel = [1,maxpool_kernel,maxpool_kernel,1]
        transpose_filter = [1, 1]

        msg = '----AE_transpose_4layer_struct_2----'
        self.say_sth(msg, print_out=self.print_out)

        net = self.__Conv(input_x, kernel=kernel_list[0], filter=filter_list[0], conv_times=conv_time)
        U_1_point = net
        #net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
        net = tf.layers.max_pooling2d(net,pool_size=[2,2],strides=2,padding='SAME')

        msg = "encode_1 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        net = self.__Conv(net, kernel=kernel_list[1], filter=filter_list[1], conv_times=conv_time)
        U_2_point = net
        # net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
        msg = "encode_2 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------

        net = self.__Conv(net, kernel=kernel_list[2], filter=filter_list[2], conv_times=conv_time)
        U_3_point = net
        # net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')

        msg = "encode_3 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        net = self.__Conv(net, kernel=kernel_list[3], filter=filter_list[3], conv_times=conv_time)
        U_4_point = net
        # net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')

        msg = "encode_4 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)

        net = self.__Conv(net, kernel=kernel_list[4], filter=filter_list[4], conv_times=conv_time)


        flatten = tf.layers.flatten(net)

        embeddings = tf.nn.l2_normalize(flatten, 1, 1e-10, name='embeddings')
        print("embeddings shape:",embeddings.shape)
        # net = tf.layers.dense(inputs=prelogits, units=units, activation=None)
        # print("net shape:",net.shape)
        # net = tf.reshape(net,shape)
        # -----------------------------------------------------------------------
        # --------Decode--------
        # -----------------------------------------------------------------------

        # data= 4 x 4 x 64

        net = tf.layers.conv2d_transpose(net, filter_list[3], transpose_filter, strides=2, padding='same')
        #net = tf.concat([net, U_4_point], axis=3)
        net = self.__Conv(net, kernel=kernel_list[3], filter=filter_list[3], conv_times=conv_time)
        msg = "decode_1 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        # data= 8 x 8 x 64
        net = tf.layers.conv2d_transpose(net, filter_list[2], transpose_filter, strides=2, padding='same')
        # net = tf.concat([net, U_3_point], axis=3)
        net = self.__Conv(net, kernel=kernel_list[2], filter=filter_list[2], conv_times=conv_time)
        msg = "decode_2 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------

        net = tf.layers.conv2d_transpose(net, filter_list[1], transpose_filter, strides=2, padding='same')
        # net = tf.concat([net, U_2_point], axis=3)
        net = self.__Conv(net, kernel=kernel_list[1], filter=filter_list[1], conv_times=conv_time)
        msg = "decode_3 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        # data= 32 x 32 x 64

        net = tf.layers.conv2d_transpose(net, filter_list[0], transpose_filter, strides=2, padding='same')
        # net = tf.concat([net, U_1_point], axis=3)
        net = self.__Conv(net, kernel=kernel_list[0], filter=filter_list[0], conv_times=conv_time)
        msg = "decode_2 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)

        net = tf.layers.conv2d(
            inputs=net,
            filters=3,
            kernel_size=kernel_list[0],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu,
            name='output_AE')
        msg = "output shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        # data= 64 x 64 x 3
        return net

if __name__ == "__main__":
    para_dict = dict()
    dirs = list()


    #----class init
    para_dict['train_img_source'] = [
        r"D:\dataset\optotech\CMIT_009IRC\only_OK\train\L1_OK",
        r"D:\dataset\optotech\CMIT_009IRC\only_OK\train\L2_OK",
        r"D:\dataset\optotech\CMIT_009IRC\only_OK\train\L4_OK",
                ]
    para_dict['vali_img_source'] = [
        r'D:\dataset\optotech\CMIT_009IRC\only_OK\test\L1_OK',
        r'D:\dataset\optotech\CMIT_009IRC\only_OK\test\L2_OK',
        r'D:\dataset\optotech\CMIT_009IRC\only_OK\test\L4_OK',
        ]
    para_dict['recon_img_dir'] = r"D:\dataset\optotech\CMIT_009IRC\only_OK\recon"



    #----model init
    para_dict['model_shape'] = [None, 192, 192, 3]
    para_dict['preprocess_dict'] = {"rot": False, 'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}
    para_dict['infer_method'] = "AE_pooling_net"#"AE_JNet"#"AE_transpose_4layer"
    para_dict['kernel_list'] = [3,3,3,3,3]
    # para_dict['filter_list'] = [64,96,144,192,256]
    para_dict['filter_list'] = [32,64,96,128,256]
    para_dict['conv_time'] = 1
    para_dict['embed_length'] = 144
    para_dict['scaler'] = 1
    para_dict['pool_type'] = ['max','ave']
    para_dict['pool_kernel'] = [5,2]
    para_dict['activation'] = 'relu'
    para_dict['loss_method'] = "ssim"
    para_dict['loss_method_2'] = None
    para_dict['opti_method'] = "adam"
    para_dict['learning_rate'] = 1e-4
    para_dict['save_dir'] = r"D:\code\model_saver\AE_Rot_28"



    para_dict['epochs'] = 150
    #----train
    para_dict['eval_epochs'] = 2
    para_dict['GPU_ratio'] = None
    para_dict['batch_size'] = 16
    para_dict['ratio'] = 1.0
    para_dict['setting_dict'] = {'rdm_shift': 0.1, 'rdm_angle': 5,'rdm_patch':[0.25,0.3]}#rdm_patch:[margin_ratio,patch_ratio]

    process_dict = {"rdm_flip": True, 'rdm_br': True, 'rdm_crop': False, 'rdm_blur': True,
                    'rdm_angle': True,
                    'rdm_noise': False,
                    'rdm_shift': True,
                    'rdm_patch': True,
                    }

    if True in process_dict.values():
        pass
    else:
        process_dict = None
    para_dict['process_dict'] = process_dict

    AE_train = AE(para_dict)
    AE_train.model_init(para_dict)
    AE_train.train(para_dict)




