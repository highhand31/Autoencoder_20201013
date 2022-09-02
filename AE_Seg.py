import cv2,sys,shutil,os,json,time,math,imgviz
import numpy as np
import matplotlib.pyplot as plt
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
# sys.path.append(r'G:\我的雲端硬碟\Python\Code\Pycharm\utility')
from Utility import tools,file_transfer,file_decode_v2,Seg_performance,get_classname_id_color,\
    get_latest_json_content,dict_transform


from models_AE import AE_transpose_4layer,tf_mish,AE_JNet,AE_Resnet_Rot,AE_pooling_net,\
    AE_Unet,Seg_DifNet,AE_Seg_pooling_net,preprocess,AE_pooling_net_V3,AE_pooling_net_V4,Seg_DifNet_V2,\
    AE_pooling_net_V5,AE_dense_sampling,AE_pooling_net_V6,AE_pooling_net_V7,Seg_pooling_net_V4,Seg_pooling_net_V7,\
    Seg_pooling_net_V8

import config_mit
from models_MiT import MixVisionTransformer,MiTDecoder


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

def create_pb_filename(save_pb_name,extension,save_dir,add_name_tail=False):
    if save_pb_name is None:
        save_pb_name = 'pb_model'
    if add_name_tail is True:
        xtime = time.localtime()
        name_tailer = ''
        for i in range(6):
            string = str(xtime[i])
            if len(string) == 1:
                string = '0' + string
            name_tailer += string
        pb_save_path = "{}_{}.{}".format(save_pb_name, name_tailer, extension)
    else:
        pb_save_path = "{}.{}".format(save_pb_name, extension)
    return os.path.join(save_dir, pb_save_path)

#----main class
class AE_Seg():
    def __init__(self,para_dict):
        #----use previous settings
        if para_dict.get('use_previous_settings'):
            dict_t = get_latest_json_content(para_dict['save_dir'])
            self.para_dict = None
            if isinstance(dict_t,dict):
                msg_list = list()
                msg_list.append("Use previous settings")
                #----exclude dict process
                update_dict = para_dict.get('update_dict')
                if isinstance(update_dict,dict):
                    for key, value in update_dict.items():
                        if value is None:
                            dict_t[key] = para_dict[key]
                            msg = '{} update to {}'.format(key,para_dict[key])
                            msg_list.append(msg)
                        else:
                            dict_t[key][value] = para_dict[key][value]
                            msg = '{}_{} update to {}'.format(key,value,para_dict[key][value])
                            msg_list.append(msg)

                self.para_dict = dict_t

                for msg in msg_list:
                    say_sth(msg, print_out=True)

        # ----common var
        show_data_qty = para_dict.get('show_data_qty')
        print_out = para_dict.get('print_out')

        # ----local var
        recon_flag = False
        msg_list = list()
        to_train_ae = False
        to_train_seg = False

        # ----tools class init
        tl = tools(print_out=print_out)

        #----AE process
        ae_var = para_dict.get('ae_var')
        if isinstance(ae_var,dict) is True:
            #----config check(if exe)
            status = True
            # status, para_dict = self.config_check(para_dict)
            if status:
                train_img_dir = ae_var.get('train_img_dir')
                test_img_dir = ae_var.get('test_img_dir')
                recon_img_dir = ae_var.get('recon_img_dir')
                special_img_dir = ae_var.get('special_img_dir')

                # ----AE image path process
                if train_img_dir is None:
                    msg_list.append('沒有輸入AE訓練集')
                else:
                    self.train_paths, self.train_path_qty = tl.get_paths(train_img_dir)

                    if self.train_path_qty == 0:
                        say_sth("Error:AE訓練資料集沒有圖片", print_out=print_out)
                    else:
                        to_train_ae = True
                        msg = "AE訓練集圖片數量:{}".format(self.train_path_qty)
                        msg_list.append(msg)

                        # ----test image paths
                        if test_img_dir is None:
                            msg = "沒有輸入AE驗證集路徑"
                        else:
                            self.test_paths, self.test_path_qty = tl.get_paths(test_img_dir)
                            msg = "AE驗證集圖片數量:{}".format(self.test_path_qty)
                        msg_list.append(msg)
                        # say_sth(msg,print_out=print_out)

                        # ----special image paths
                        if special_img_dir is None:
                            msg = "沒有輸入加強AE學習集路徑"
                        else:
                            self.sp_paths, self.sp_path_qty = tl.get_paths(special_img_dir)
                            msg = "AE加強學習圖片數量:{}".format(self.sp_path_qty)
                        msg_list.append(msg)
                        # say_sth(msg, print_out=print_out)

                        # ----recon image paths
                        if recon_img_dir is None:
                            msg = "沒有輸入AE重建圖集路徑"
                        else:
                            self.recon_paths, self.recon_path_qty = tl.get_paths(recon_img_dir)
                            if self.recon_path_qty > 0:
                                recon_flag = True
                            msg = "AE重建圖片數量:{}".format(self.recon_path_qty)
                        msg_list.append(msg)

        #----SEG process
        seg_var = para_dict.get('seg_var')
        if isinstance(seg_var,dict) is True:
            #----SEG config check(if exe)
            status = True
            # status, para_dict = self.config_check(para_dict)
            if status:
                # ----SEG
                seg_var = para_dict.get('seg_var')
                train_img_seg_dir = seg_var.get('train_img_seg_dir')
                test_img_seg_dir = seg_var.get('test_img_seg_dir')
                predict_img_dir = seg_var.get('predict_img_dir')
                json_check = seg_var.get('json_check')
                to_train_w_AE_paths = seg_var.get('to_train_w_AE_paths')
                id2class_name = seg_var.get('id2class_name')
                select_OK_ratio = 0.2


                # ----SEG train image path process
                if train_img_seg_dir is None:
                    msg_list.append('沒有輸入SEG訓練集')
                else:
                    if json_check:
                        self.seg_train_paths, self.seg_train_json_paths, self.seg_train_qty = tl.get_subdir_paths_withJsonCheck(
                            train_img_seg_dir)
                    else:
                        self.seg_train_paths, self.seg_train_qty = tl.get_paths(train_img_seg_dir)

                    msg = "SEG訓練集圖片數量:{}".format(self.seg_train_qty)
                    msg_list.append(msg)

                    # ----Seg img path qty check
                    if self.seg_train_qty > 0:
                        to_train_seg = True

                        #----read class names
                        if isinstance(id2class_name,dict):
                            id2class_name = dict_transform(id2class_name,set_key=True)
                            source = id2class_name
                        elif os.path.isfile(id2class_name):
                            source = id2class_name
                        else:
                            source = os.path.dirname(train_img_seg_dir[0])

                        class_names, class_name2id, id2class_name, id2color,_ = get_classname_id_color(source, print_out=print_out)

                        class_num = len(class_names)
                        if class_num == 0:
                            say_sth("Error:沒有取到SEG類別數目等資料，無法進行SEG訓練", print_out=True)
                            to_train_seg = False
                            to_train_ae = False
                        else:
                            #----train with AE ok images
                            if to_train_w_AE_paths:
                                if to_train_ae:
                                    if json_check is not True:
                                        select_num = np.minimum(self.train_path_qty,int(self.seg_train_qty * select_OK_ratio))
                                        temp_paths = np.random.choice(self.train_paths,size=select_num,replace=False)
                                        self.seg_train_paths = list(self.seg_train_paths)
                                        self.seg_train_paths.extend(temp_paths)
                                        self.seg_train_paths = np.array(self.seg_train_paths)
                                        self.seg_train_qty += select_num
                                        msg = "to_train_w_AE_paths，SEG訓練集圖片數量:{}，實際為{}".\
                                            format(self.seg_train_qty,self.seg_train_paths.shape)
                                        msg_list.append(msg)

                            #----check test images if test seg qty > 0
                            if test_img_seg_dir is None:
                                self.seg_test_qty = 0
                                msg_list.append('沒有輸入SEG驗證集')
                            else:
                                if json_check:
                                    self.seg_test_paths, self.seg_test_json_paths, self.seg_test_qty = tl.get_subdir_paths_withJsonCheck(
                                        test_img_seg_dir)
                                else:
                                    self.seg_test_paths, self.seg_test_qty = tl.get_paths(test_img_seg_dir)
                                msg = "SEG驗證集圖片數量:{}".format(self.seg_test_qty)
                                msg_list.append(msg)

                            #----check predict images if predict qty > 0
                            if predict_img_dir is None:
                                self.seg_predict_qty = 0
                                msg_list.append('沒有輸入SEG預測集')
                            else:
                                self.seg_predict_paths, self.seg_predict_qty = tl.get_paths(predict_img_dir)
                                msg = "SEG預測集圖片數量:{}".format(self.seg_predict_qty)
                                msg_list.append(msg)

        #----status dicision
        if to_train_seg or to_train_ae:
            status = True
        else:
            status = False

        #----display data info
        if show_data_qty is True:
            for msg in msg_list:
                say_sth(msg, print_out=print_out)

        #----log update
        content = dict()
        content = self.log_update(content, para_dict)
        #====record id, classname, and color
        if to_train_seg:
            content['class_names'] = class_names
            content['class_name2id'] = class_name2id
            content['id2class_name'] = id2class_name
            content['id2color'] = id2color

        #----local var to global
        self.to_train_ae = to_train_ae
        self.to_train_seg = to_train_seg
        self.status = status
        self.content = content
        if to_train_ae:
            self.train_img_dir = train_img_dir
            self.test_img_dir = test_img_dir
            self.special_img_dir = special_img_dir
            self.recon_img_dir = recon_img_dir
            self.recon_flag = recon_flag
        if to_train_seg:
            self.train_img_seg_dir = train_img_seg_dir
            self.test_img_seg_dir = test_img_seg_dir
            self.predict_img_dir = predict_img_dir
            self.class_num = class_num
            self.class_names = class_names
            self.class_name2id = class_name2id
            self.id2class_name = id2class_name
            self.id2color = id2color

    def model_init(self,para_dict):
        #----var parsing

        # ----use previous settings
        if para_dict.get('use_previous_settings'):
            if isinstance(self.para_dict,dict):
                para_dict = self.para_dict
        #----AE
        if self.to_train_ae:
            ae_var = para_dict['ae_var']
            model_shape = ae_var.get('model_shape')  # [N,H,W,C]
            infer_method = ae_var['infer_method']
            acti = ae_var['activation']
            pool_kernel = ae_var['pool_kernel']
            kernel_list = ae_var['kernel_list']
            filter_list = ae_var['filter_list']
            conv_time = ae_var['conv_time']
            pool_type = ae_var.get('pool_type')
            loss_method = ae_var['loss_method']
            opti_method = ae_var['opti_method']
            embed_length = ae_var['embed_length']
            stride_list = ae_var.get('stride_list')
            scaler = ae_var.get('scaler')
            process_dict = ae_var['process_dict']

            rot = ae_var.get('rot')

        #----SEG
        if self.to_train_seg:
            seg_var = para_dict['seg_var']
            infer_method4Seg = seg_var.get('infer_method')
            pool_kernel4Seg = seg_var['pool_kernel']
            pool_type4Seg = seg_var.get('pool_type')
            kernel_list4Seg = seg_var['kernel_list']
            filter_list4Seg = seg_var['filter_list']
            loss_method4Seg = seg_var.get('loss_method')
            opti_method4Seg = seg_var.get('opti_method')
            preprocess_dict4Seg = seg_var.get('preprocess_dict')
            rot4Seg = seg_var.get('rot')
            acti_seg = seg_var.get('activation')
            if model_shape is None:
                model_shape = seg_var['model_shape']

        #----common var
        preprocess_dict = para_dict.get('preprocess_dict')
        lr = para_dict['learning_rate']
        save_dir = para_dict['save_dir']
        save_pb_name = para_dict.get('save_pb_name')
        encript_flag = para_dict.get('encript_flag')
        print_out = para_dict.get('print_out')
        add_name_tail = para_dict.get('add_name_tail')
        dtype = para_dict.get('dtype')


        #----var
        #rot = False
        # bias = 0.5
        # br_ratio = 0
        # ct_ratio = 1
        pb_extension = 'pb'
        log_extension = 'json'
        acti_dict = {'relu': tf.nn.relu, 'mish': tf_mish, None: tf.nn.relu}
        pb_save_list = list()

        #----var process
        if encript_flag is True:
            pb_extension = 'nst'
            log_extension = 'nst'
        else:
            pb_extension = 'pb'
            log_extension = 'json'

        if add_name_tail is None:
            add_name_tail = True
        if dtype is None:
            dtype = 'float32'


        # ----tf_placeholder declaration
        tf_input = tf.placeholder(dtype, shape=model_shape, name='input')
        tf_keep_prob = tf.placeholder(dtype=dtype, name="keep_prob")


        if self.to_train_ae:
            #----random patch
            rdm_patch = False
            if process_dict.get('rdm_patch') is True:
                rdm_patch = True

            #----filer scaling process
            if scaler is not None:
                filter_list = (np.array(filter_list) / scaler).astype(np.uint16)

            if rdm_patch is True:
                self.tf_input_ori = tf.placeholder(dtype, shape=model_shape, name='input_ori')

            #tf_label_batch = tf.placeholder(dtype=tf.int32, shape=[None], name="label_batch")
            #tf_phase_train = tf.placeholder(tf.bool, name="phase_train")

            # ----activation selection
            acti_func = acti_dict[acti]


            avepool_out = self.__avepool(tf_input, k_size=5, strides=1)
            #----preprocess
            if preprocess_dict is None:
                tf_input_process = tf.identity(tf_input,name='preprocess')
                if rdm_patch is True:
                    tf_input_ori_no_patch = tf.identity(self.tf_input_ori, name='tf_input_ori_no_patch')

            else:
                tf_temp = preprocess(tf_input, preprocess_dict, print_out=print_out)
                tf_input_process = tf.identity(tf_temp,name='preprocess')

                if rdm_patch is True:
                    tf_temp_2 = preprocess(self.tf_input_ori, preprocess_dict, print_out=print_out)
                    tf_input_ori_no_patch = tf.identity(tf_temp_2, name='tf_input_ori_no_patch')


            #----AE inference selection
            if infer_method == "AE_transpose_4layer":
                recon = AE_transpose_4layer(tf_input, kernel_list, filter_list,activation=acti_func,
                                                  pool_kernel=pool_kernel,pool_type=pool_type)
                recon = tf.identity(recon, name='output_AE')
                #(tf_input, kernel_list, filter_list,pool_kernel=2,activation=tf.nn.relu,pool_type=None)
                # recon = AE_refinement(temp,96)
            elif infer_method.find('mit') >= 0:
                cfg = config_mit.config[infer_method.split("_")[-1]]
                model = MixVisionTransformer(
                    embed_dims=cfg['embed_dims'],
                    num_stages=cfg['num_stages'],
                    num_layers=cfg['num_layers'],
                    num_heads=cfg['num_heads'],
                    patch_sizes=cfg['patch_sizes'],
                    strides=cfg['strides'],
                    sr_ratios=cfg['sr_ratios'],
                    mlp_ratio=cfg['mlp_ratio'],
                    ffn_dropout_keep_ratio=1.0,
                    dropout_keep_rate=1.0)
                # model.init_weights()
                outs = model(tf_input_process)
                # print("outs shape:",outs.shape)
                mitDec = MiTDecoder(channels=cfg['num_heads'][-1] * cfg['embed_dims'],
                                    dropout_ratio=0, num_classes=3)
                logits = mitDec(outs)
                recon = tf.identity(logits, name='output_AE')

            elif infer_method == "AE_pooling_net_V3":
                recon = AE_pooling_net_V3(tf_input_process, kernel_list, filter_list, activation=acti_func,
                                          pool_kernel_list=pool_kernel, pool_type_list=pool_type,
                                          stride_list=stride_list, print_out=print_out)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_pooling_net_V4":
                recon = AE_pooling_net_V4(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
                                          to_reduce=ae_var.get('to_reduce'),print_out=print_out)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_pooling_net_V5":
                recon = AE_pooling_net_V5(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
                                          to_reduce=ae_var.get('to_reduce'),print_out=print_out)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_pooling_net_V6":
                recon = AE_pooling_net_V6(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
                                          to_reduce=ae_var.get('to_reduce'),print_out=print_out)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_pooling_net_V7":

                recon = AE_pooling_net_V7(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
                                         print_out=print_out)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_dense_sampling":
                sampling_factor = 16
                filters = 2
                recon = AE_dense_sampling(tf_input_process,sampling_factor,filters)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_pooling_net":

                recon = AE_pooling_net(tf_input_process, kernel_list, filter_list,activation=acti_func,
                                      pool_kernel_list=pool_kernel,pool_type_list=pool_type,
                                      stride_list=stride_list,rot=rot,print_out=print_out)
                recon = tf.identity(recon,name='output_AE')
            elif infer_method == "AE_Seg_pooling_net":
                AE_out,Seg_out = AE_Seg_pooling_net(tf_input, kernel_list, filter_list,activation=acti_func,
                                                  pool_kernel_list=pool_kernel,pool_type_list=pool_type,
                                       rot=rot,print_out=print_out,preprocess_dict=preprocess_dict,
                                           class_num=self.class_num)
                recon = tf.identity(AE_out,name='output_AE')
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

            # ----AE loss method selection
            if loss_method == 'mse':
                loss_AE = tf.reduce_mean(tf.pow(recon - tf_input, 2), name="loss_AE")
            elif loss_method == 'ssim':
                # self.loss_AE = tf.reduce_mean(tf.image.ssim_multiscale(tf.image.rgb_to_grayscale(self.tf_input),tf.image.rgb_to_grayscale(self.recon),2),name='loss')

                if rdm_patch is True:
                    loss_AE = tf.reduce_mean(tf.image.ssim(tf_input_ori_no_patch, recon, 2.0), name='loss_AE')
                else:
                    loss_AE = tf.reduce_mean(tf.image.ssim(tf_input_process, recon, 2.0), name='loss_AE')

            elif loss_method == "huber":
                loss_AE = tf.reduce_sum(tf.losses.huber_loss(tf_input, recon, delta=1.35), name='loss_AE')
            elif loss_method == 'ssim+mse':
                loss_1 = tf.reduce_mean(tf.pow(recon - tf_input, 2))
                loss_2 = tf.reduce_mean(tf.image.ssim(tf_input, recon, 2.0))
                loss_AE = tf.subtract(loss_2, loss_1, name='loss_AE')
            elif loss_method == 'cross_entropy':
                loss_AE = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.layers.flatten(tf_input),
                                                               logits=tf.layers.flatten(recon)), name="loss_AE")
            elif loss_method == 'kl_d':
                epsilon = 1e-8
                # generation loss(cross entropy)
                loss_AE = tf.reduce_mean(
                    tf_input * tf.subtract(tf.log(epsilon + tf_input), tf.log(epsilon + recon)), name='loss_AE')

            # ----AE optimizer selection
            if opti_method == "adam":
                if loss_method.find('ssim') >= 0:
                    opt_AE = tf.train.AdamOptimizer(learning_rate=lr).minimize(-loss_AE)
                else:
                    opt_AE = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_AE)

            if self.recon_flag is True:
                new_dir_name = "img_recon-" + os.path.basename(save_dir)
                # ----if recon_img_dir is list format
                if isinstance(self.recon_img_dir, list):
                    dir_path = self.recon_img_dir[0]
                else:
                    dir_path = self.recon_img_dir
                self.new_recon_dir = os.path.join(dir_path, new_dir_name)
                if not os.path.exists(self.new_recon_dir):
                    os.makedirs(self.new_recon_dir)

            # ----appoint PB node names
            pb_save_list.extend(['output_AE', "loss_AE"])

            #----pb filename for AE model
            if isinstance(save_pb_name,str):
                filename = save_pb_name + "_ae"
            else:
                filename = 'pb_model_ae'
            pb4ae_save_path = create_pb_filename(filename, pb_extension, save_dir, add_name_tail=add_name_tail)

        #----Seg inference selection
        if self.to_train_seg:
            tf_input_recon = tf.placeholder(dtype, shape=model_shape, name='input_recon')
            tf_label_batch = tf.placeholder(tf.int32, shape=model_shape[:-1], name='label_batch')
            tf_dropout = tf.placeholder(dtype=tf.float32, name="dropout")

            # ----activation selection
            acti_func = acti_dict[acti_seg]

            #----filer scaling process
            filter_list4Seg = np.array(filter_list4Seg)
            if seg_var.get('scaler') is not None:
                filter_list4Seg /= seg_var.get('scaler')
                filter_list4Seg = filter_list4Seg.astype(np.uint16)

            if infer_method4Seg == "Seg_DifNet":
                logits_Seg = Seg_DifNet(tf_input_process,tf_input_recon, kernel_list4Seg, filter_list4Seg,activation=acti_func,
                                   pool_kernel_list=pool_kernel4Seg,pool_type_list=pool_type4Seg,
                                   rot=rot4Seg,print_out=print_out,preprocess_dict=preprocess_dict4Seg,class_num=self.class_num)
                softmax_Seg = tf.nn.softmax(logits_Seg,name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'Seg_DifNet_V2':
                logits_Seg = Seg_DifNet_V2(tf_input_process,tf_input_recon,seg_var['encode_dict'],seg_var['decode_dict'],
                              class_num=self.class_num,print_out=print_out)
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg.find('mit') >= 0:
                cfg = config_mit.config[infer_method4Seg.split("_")[-1]]
                model = MixVisionTransformer(
                    embed_dims=cfg['embed_dims'],
                    num_stages=cfg['num_stages'],
                    num_layers=cfg['num_layers'],
                    num_heads=cfg['num_heads'],
                    patch_sizes=cfg['patch_sizes'],
                    strides=cfg['strides'],
                    sr_ratios=cfg['sr_ratios'],
                    mlp_ratio=cfg['mlp_ratio'],
                    drop_rate=0,
                    attn_drop_rate=0)
                # model.init_weights()
                outs = model(tf_input_process)
                # print("outs shape:",outs.shape)
                mitDec = MiTDecoder(channels=cfg['num_heads'][-1] * cfg['embed_dims'],
                                    dropout_ratio=0, num_classes=self.class_num)
                logits_Seg = mitDec(outs)
                logits_Seg = tf.image.resize(logits_Seg,model_shape[1:-1])
                print("logits_Seg shape:", logits_Seg.shape)
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'Seg_pooling_net_V4':
                logits_Seg = Seg_pooling_net_V4(tf_input_process,tf_input_recon, seg_var['encode_dict'], seg_var['decode_dict'],
                                          to_reduce=seg_var.get('to_reduce'),out_channel=self.class_num,
                                          print_out=print_out)
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'Seg_pooling_net_V7':
                logits_Seg = Seg_pooling_net_V7(tf_input_process,tf_input_recon, seg_var['encode_dict'], seg_var['decode_dict'],
                                          out_channel=self.class_num,
                                          print_out=print_out)
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'Seg_pooling_net_V8':
                logits_Seg = Seg_pooling_net_V8(tf_input_process, tf_input_recon, seg_var['encode_dict'],
                                                seg_var['decode_dict'],
                                                to_reduce=seg_var.get('to_reduce'), out_channel=self.class_num,
                                                print_out=print_out)
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'AE_Seg_pooling_net':
                logits_Seg = Seg_out
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg,tf.uint8, name='predict_Seg')

            #----Seg loss method selection
            if loss_method4Seg == "cross_entropy":
                loss_Seg = tf.reduce_mean(v2.nn.sparse_softmax_cross_entropy_with_logits(tf_label_batch,logits_Seg),name='loss_Seg')

            #----Seg optimizer selection
            if opti_method4Seg == "adam":
                opt_Seg = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_Seg)
                # var_list = []
                # for var in tf.global_variables():
                #     if var.trainable:
                #         var_list.append(var)
                # opt_Seg = v2.optimizers.Adam(learning_rate=lr).minimize(loss_Seg,var_list)
            # elif opti_method4Seg == 'sgd':


            # ----appoint PB node names
            pb_save_list.extend(['predict_Seg'])
            pb_save_list.extend(['dummy_out'])

            #----pb filename for SEG model
            if isinstance(save_pb_name,str):
                filename = save_pb_name + "_seg"
            else:
                filename = "pb_model_seg"
            pb4seg_save_path = create_pb_filename(filename, pb_extension, save_dir, add_name_tail=add_name_tail)

        # ----pb filename(common)
        if isinstance(save_pb_name, str):
            filename = save_pb_name
        else:
            filename = "pb_model"
        pb_save_path = create_pb_filename(filename, pb_extension, save_dir, add_name_tail=add_name_tail)

        # ----create the dir to save model weights(CKPT, PB)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.to_train_seg is True and self.seg_predict_qty > 0:
            new_dir_name = "img_seg-" + os.path.basename(save_dir)
            #----if recon_img_dir is list format
            if isinstance(self.predict_img_dir,list):
                dir_path = self.predict_img_dir[0]
            else:
                dir_path = self.predict_img_dir
            self.new_predict_dir = os.path.join(dir_path, new_dir_name)
            if not os.path.exists(self.new_predict_dir):
                os.makedirs(self.new_predict_dir)

        out_dir_prefix = os.path.join(save_dir, "model")
        saver = tf.train.Saver(max_to_keep=2)

        # ----PB file save filename
        # if save_pb_name is None:
        #     save_pb_name = 'pb_model'
        # if add_name_tail is True:
        #     xtime = time.localtime()
        #     name_tailer = ''
        #     for i in range(6):
        #         string = str(xtime[i])
        #         if len(string) == 1:
        #             string = '0' + string
        #         name_tailer += string
        #     pb_save_path = "{}_{}.{}".format(save_pb_name, name_tailer, pb_extension)
        # else:
        #     pb_save_path = "{}.{}".format(save_pb_name, pb_extension)
        # pb_save_path = os.path.join(save_dir, pb_save_path)

        # ----create the log(JSON)
        count = 0
        for i in range(1000):
            log_path = "{}_{}.{}".format('train_result', count, log_extension)
            log_path = os.path.join(save_dir, log_path)
            if not os.path.exists(log_path):
                break
            count += 1
        self.content = self.log_update(self.content, para_dict)
        self.content['pb_save_list'] = pb_save_list

        # ----local var to global
        self.model_shape = model_shape
        self.tf_input = tf_input
        self.tf_keep_prob = tf_keep_prob
        self.saver = saver
        self.save_dir = save_dir
        self.pb_save_path = pb_save_path


        self.pb_save_list = pb_save_list
        self.pb_extension = pb_extension
        self.log_path = log_path
        self.dtype = dtype
        if self.to_train_ae:
            self.avepool_out = avepool_out
            self.recon = recon
            self.loss_AE = loss_AE
            self.opt_AE = opt_AE
            self.out_dir_prefix = out_dir_prefix
            self.loss_method = loss_method
            self.pb4ae_save_path = pb4ae_save_path

        if self.to_train_seg is True:
            self.tf_label_batch = tf_label_batch
            self.tf_input_recon = tf_input_recon
            self.tf_prediction_Seg = prediction_Seg
            self.infer_method4Seg = infer_method4Seg
            self.logits_Seg = logits_Seg
            self.loss_Seg = loss_Seg
            self.opt_Seg = opt_Seg
            self.prediction_Seg = prediction_Seg
            self.loss_method4Seg = loss_method4Seg
            self.pb4seg_save_path = pb4seg_save_path
            self.tf_dropout = tf_dropout

    def train(self,para_dict):
        # ----use previous settings
        if para_dict.get('use_previous_settings'):
            if isinstance(self.para_dict, dict):
                para_dict = self.para_dict
        # ----var parsing
        epochs = para_dict['epochs']
        GPU_ratio = para_dict.get('GPU_ratio')
        print_out = para_dict.get('print_out')
        encode_header = para_dict.get('encode_header')
        encode_num = para_dict.get('encode_num')
        encript_flag = para_dict.get('encript_flag')
        eval_epochs = para_dict.get('eval_epochs')
        to_fix_ae = para_dict.get('to_fix_ae')
        to_fix_seg = para_dict.get('to_fix_seg')

        #----AE
        if self.to_train_ae:
            ae_var = para_dict['ae_var']
            aug_times = ae_var.get('aug_times')
            batch_size = ae_var['batch_size']
            ratio = ae_var.get('ratio')
            process_dict = ae_var.get('process_dict')
            setting_dict = ae_var.get('setting_dict')
            save_period = ae_var.get('save_period')
            target_dict = ae_var.get('target')
            pause_opt_ae = ae_var.get('pause_opt_ae')

        #----SEG
        if self.to_train_seg:
            seg_var = para_dict['seg_var']
            ratio_seg = seg_var.get('ratio_seg')
            process_seg_dict = seg_var.get('process_dict')
            setting_seg_dict = seg_var.get('setting_dict')
            aug_seg_times = seg_var.get('aug_times')
            json_check = seg_var.get('json_check')
            batch_size_seg = seg_var['batch_size']
            pause_opt_seg = seg_var.get('pause_opt_seg')
            self_create_label = seg_var.get('self_create_label')

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
        LV = dict()
        train_loss_list = list()
        seg_train_loss_list = list()
        train_acc_list = list()
        test_loss_list = list()
        seg_test_loss_list = list()
        test_acc_list = list()
        epoch_time_list = list()
        img_quantity = 0
        aug_enable = False
        break_flag = False
        error_dict = {'GPU_resource_error': False}
        train_result_dict = {'loss': 0,"loss_method":self.loss_method}
        test_result_dict = {'loss': 0, "loss_method": self.loss_method}
        if self.to_train_seg:
            seg_train_result_dict = {'loss': 0,"loss_method":self.loss_method4Seg}
            seg_test_result_dict = {'loss': 0,"loss_method":self.loss_method4Seg}

        record_type = 'loss'
        qty_sp = 0
        LV['pb_save_path_old'] = ''
        LV['record_value'] = 0
        LV['pb_seg_save_path_old'] = ''
        LV['record_value_seg'] = 0
        keep_prob = 0.7

        #----AE hyper-parameters
        if self.to_train_ae:
            tl = tools()
            # ----set target
            tl.set_target(target_dict)
            batch_size_test = batch_size

            #----check if the augmentation(image processing) is enabled
            if isinstance(process_dict, dict):
                if True in process_dict.values():
                    aug_enable = True
                    tl.set_process(process_dict, setting_dict, print_out=print_out)

                    if aug_times is None:
                        aug_times = 2
                    batch_size = batch_size // aug_times  # the batch size must be integer!!

        #----SEG hyper-parameters
        if self.to_train_seg:
            tl_seg = tools()
            tl_seg.class_name2id = self.class_name2id
            tl_seg.id2color = self.id2color
            titles = ['prediction', 'answer']
            batch_size_seg_test = batch_size_seg
            seg_p = Seg_performance(len(self.class_name2id),print_out=print_out)

            # ----check if the augmentation(image processing) is enabled
            if isinstance(process_seg_dict, dict):
                if True in process_seg_dict.values():
                    aug_seg_enable = True
                    tl_seg.set_process(process_seg_dict, setting_seg_dict)
                    if aug_seg_times is None:
                        aug_seg_times = 2
                    batch_size_seg = batch_size_seg // aug_seg_times  # the batch size must be integer!!
                    if batch_size_seg < 1:
                        batch_size_seg = 1

        #----update content
        self.content = self.log_update(self.content, para_dict)

        # ----read the manual cmd
        if para_dict.get('to_read_manual_cmd') is True:
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
                if self.to_train_ae is True:
                    #----AE train set quantity
                    qty_train = self.get_train_qty(self.train_path_qty,ratio,print_out=print_out,name='AE')
                    # qty_train = self.train_path_qty
                    # if ratio is not None:
                    #     if ratio <= 1.0 and ratio > 0:
                    #         qty_train = int(self.train_path_qty * ratio)
                    #         qty_train = np.maximum(1, qty_train)
                    #
                    # msg = "AE訓練集資料總共取數量{}".format(qty_train)
                    # say_sth(msg, print_out=print_out)

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

                #----SEG
                if self.to_train_seg is True:
                    # ----SEG train set quantity
                    qty_train_seg = self.get_train_qty(self.seg_train_qty, ratio_seg, print_out=print_out,
                                                       name='SEG')

                    # ----calculate iterations of one epoch
                    train_ites_seg = math.ceil(self.seg_train_qty / batch_size_seg)
                    if self.seg_test_qty > 0:
                        test_ites_seg = math.ceil(self.seg_test_qty / batch_size_seg_test)
                    if self.seg_predict_qty > 0:
                        predict_ites_seg = math.ceil(self.seg_predict_qty / batch_size_seg_test)

                # ----epoch training
                for epoch in range(epochs):
                    # ----read manual cmd
                    if para_dict.get('to_read_manual_cmd') is True:
                        if os.path.exists(j_path):
                            with open(j_path, 'r') as f:
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
                    train_loss_seg = 0
                    test_loss_seg = 0
                    train_acc = 0
                    test_loss = 0
                    test_loss_2 = 0
                    test_acc = 0

                    #----AE part
                    if self.to_train_ae is True:
                        if to_fix_ae is True:
                            pass
                        else:
                            #tf_var_AE = tf.trainable_variables(scope='AE')
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

                            # ----optimizations(AE train set)
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
                                                 self.tf_keep_prob: keep_prob}
                                else:
                                    feed_dict = {self.tf_input: batch_data, self.tf_keep_prob: keep_prob}

                                # ----optimization
                                try:
                                    if pause_opt_ae is not True:
                                        sess.run(self.opt_AE, feed_dict=feed_dict)
                                except:
                                    error_dict['GPU_resource_error'] = True
                                    msg = "Error:權重最佳化時產生錯誤，可能GPU資源不夠導致"
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
                            # if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
                            #     model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)
                            #
                            #     #----encode ckpt file
                            #     if encript_flag is True:
                            #         encode_CKPT(model_save_path, encode_num=encode_num, encode_header=encode_header)
                            #
                            #     #----save pb(normal)
                            #     save_pb_file(sess, self.pb_save_list, self.pb_save_path, encode=encript_flag,
                            #                  random_num_range=encode_num, header=encode_header)


                            #----save results in the log file
                            train_loss_list.append(float(train_loss))
                            self.content["train_loss_list"] = train_loss_list
                            #train_acc_list.append(float(train_acc))
                            #self.content["train_acc_list"] = train_acc_list

                            if self.test_img_dir is not None:
                                test_loss_list.append(float(test_loss))
                                #test_acc_list.append(float(test_acc))
                                self.content["test_loss_list"] = test_loss_list

                            #----display training results
                            msg_list = list()
                            msg_list.append("\n----訓練週期 {} 與相關結果如下----".format(epoch))
                            msg_list.append("AE訓練集loss:{}".format(np.round(train_loss, 4)))
                            if self.test_img_dir is not None:
                                msg_list.append("AE驗證集loss:{}".format(np.round(test_loss, 4)))

                            # msg_list.append("此次訓練所花時間: {}秒".format(np.round(d_t, 2)))
                            # msg_list.append("累積訓練時間: {}秒".format(np.round(total_train_time, 2)))
                            say_sth(msg_list, print_out=print_out, header='msg')

                            #----send protocol data(for C++ to draw)
                            msg_list, header_list = self.collect_data2ui(epoch, train_result_dict,
                                                                         test_result_dict)
                            say_sth(msg_list, print_out=print_out, header=header_list)

                            #----find the best performance
                            new_value = self.__get_result_value(record_type, train_result_dict, test_result_dict)
                            if epoch == 0:
                                LV['record_value'] = new_value
                            else:

                                go_flag = False
                                if self.loss_method == 'ssim':
                                    if new_value > LV['record_value']:
                                        go_flag = True
                                else:
                                    if new_value < LV['record_value']:
                                        go_flag = True

                                if go_flag is True:
                                    # ----delete the previous pb
                                    if os.path.exists(LV['pb_save_path_old']):
                                        os.remove(LV['pb_save_path_old'])

                                    #----save the better one
                                    pb_save_path = "infer_{}.{}".format(new_value, self.pb_extension)
                                    pb_save_path = os.path.join(self.save_dir, pb_save_path)

                                    save_pb_file(sess, self.pb_save_list, pb_save_path, encode=encript_flag,
                                                      random_num_range=encode_num, header=encode_header)
                                    #----update record value
                                    LV['record_value'] = new_value
                                    LV['pb_save_path_old'] = pb_save_path

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

                            # ----save ckpt
                            # if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
                            #     save_pb_file(sess, self.pb_save_list, self.pb4ae_save_path,
                            #                  encode=encript_flag,
                            #                  random_num_range=encode_num, header=encode_header)

                    #----SEG part
                    if self.to_train_seg is True:
                        if to_fix_seg is True:
                            pass
                        else:
                            #----
                            seg_p.reset_arg()
                            seg_p.reset_defect_stat()
                            #----
                            indice = np.random.permutation(self.seg_train_qty)
                            self.seg_train_paths = self.seg_train_paths[indice]
                            if json_check:
                                self.seg_train_json_paths = self.seg_train_json_paths[indice]

                            seg_train_paths_ori = self.seg_train_paths[:qty_train_seg]
                            if json_check:
                                seg_train_json_paths_ori = self.seg_train_json_paths[:qty_train_seg]

                            if aug_enable is True:
                                seg_train_paths_aug = seg_train_paths_ori[::-1]
                                if json_check:
                                    seg_train_json_paths_aug = seg_train_json_paths_ori[::-1]

                            # ----optimizations(SEG train set)
                            for idx_seg in range(train_ites_seg):
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
                                                    file_transfer(data_file, random_num_range=encode_num,
                                                                  header=encode_header)
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
                                ori_seg_paths = tl_seg.get_ite_data(seg_train_paths_ori, idx_seg, batch_size=batch_size_seg)
                                aug_seg_paths = tl_seg.get_ite_data(seg_train_paths_aug, idx_seg, batch_size=batch_size_seg)
                                if json_check:
                                    ori_seg_json_paths = tl_seg.get_ite_data(seg_train_json_paths_ori, idx_seg,
                                                                             batch_size=batch_size_seg)
                                    aug_seg_json_paths = tl_seg.get_ite_data(seg_train_json_paths_aug, idx_seg,
                                                                             batch_size=batch_size_seg)
                                else:
                                    ori_seg_json_paths = None
                                    aug_seg_json_paths = None
                                    # print("ori_seg_json_paths:",ori_seg_json_paths)
                                    # print("aug_seg_json_paths:",aug_seg_json_paths)

                                #----get 4-D data
                                if aug_enable is True:
                                    if self_create_label:
                                        # ----ori data
                                        #(self,paths, output_shape,to_norm=True,to_rgb=True,to_process=False,dtype='float32')
                                        ori_data, ori_label = tl_seg.get_4D_data_create_mask(ori_seg_paths,
                                                                                           self.model_shape[1:],
                                                                                           to_norm=True, to_rgb=True,
                                                                                           to_process=False,
                                                                                           dtype=self.dtype)

                                        # ----aug data
                                        aug_data, aug_label = tl_seg.get_4D_data_create_mask(aug_seg_paths,
                                                                                           self.model_shape[1:],
                                                                                           to_norm=True,
                                                                                           to_rgb=True,
                                                                                           to_process=True,
                                                                                           dtype=self.dtype)
                                    else:
                                        # ----ori data
                                        ori_data,ori_label = tl_seg.get_4D_img_label_data(ori_seg_paths,self.model_shape[1:],
                                                                                          json_paths=ori_seg_json_paths,
                                                                            to_norm=True, to_rgb=True,
                                                                            to_process=False, dtype=self.dtype)

                                        # ----aug data
                                        aug_data,aug_label = tl_seg.get_4D_img_label_data(aug_seg_paths,self.model_shape[1:],
                                                                                          json_paths=aug_seg_json_paths,
                                                                  to_norm=True,
                                                                  to_rgb=True,
                                                                  to_process=True,
                                                                  dtype=self.dtype)

                                    #----data concat
                                    batch_data = np.concatenate([ori_data, aug_data], axis=0)
                                    batch_label = np.concatenate([ori_label, aug_label], axis=0)
                                else:
                                    if self_create_label:
                                        batch_data, batch_label = tl_seg.get_4D_data_create_mask(ori_seg_paths,
                                                                                               self.model_shape[1:],
                                                                                               dtype=self.dtype)
                                    else:
                                        batch_data,batch_label = tl_seg.get_4D_img_label_data(ori_seg_paths,self.model_shape[1:],
                                                                                    json_paths=ori_seg_json_paths,
                                                                                   dtype=self.dtype)

                                #----put all data to tf placeholders
                                # recon = sess.run(self.recon,feed_dict={self.tf_input:batch_data,self.tf_keep_prob: 1.0})
                                # feed_dict = {self.tf_input: batch_data, self.tf_input_recon: recon,
                                #              self.tf_label_batch: batch_label,
                                #              self.tf_keep_prob: 0.5}


                                # if self.infer_method4Seg.find('mit') >= 0:
                                #     batch_label = tl_seg.batch_resize_label(batch_label,
                                #                                             (self.model_shape[2]//4,self.model_shape[1]//4))



                                recon = sess.run(self.recon, feed_dict={self.tf_input: batch_data})
                                feed_dict = {self.tf_input: batch_data,self.tf_input_recon:recon,
                                             self.tf_label_batch: batch_label,
                                             self.tf_keep_prob: keep_prob,
                                             self.tf_dropout: 0.3}

                                #----session run
                                # print("idx_seg:",idx_seg)
                                try:
                                    if pause_opt_seg is not True:
                                        sess.run(self.opt_Seg, feed_dict=feed_dict)
                                except:
                                    error_dict['GPU_resource_error'] = True
                                    msg = "Error:SEG權重最佳化時產生錯誤，可能GPU資源不夠導致"
                                    say_sth(msg, print_out=print_out)
                                    break

                                #----evaluation(training set)
                                feed_dict[self.tf_keep_prob] = 1.0
                                feed_dict[self.tf_dropout] = 0.0
                                # feed_dict[self.tf_phase_train] = False

                                loss_temp = sess.run(self.loss_Seg, feed_dict=feed_dict)
                                predict_label = sess.run(self.prediction_Seg, feed_dict=feed_dict)
                                # predict_label = np.argmax(predict_label,axis=-1).astype(np.uint8)

                                #----calculate the loss and accuracy
                                train_loss_seg += loss_temp
                                seg_p.cal_intersection_union(predict_label,batch_label)
                                _ = seg_p.cal_label_defect_by_acc_v2(predict_label, batch_label)
                                _ = seg_p.cal_predict_defect_by_acc_v2(predict_label, batch_label)

                            train_loss_seg /= train_ites_seg
                            seg_train_result_dict['loss'] = train_loss_seg
                            #----save results in the log file
                            seg_train_loss_list.append(float(train_loss_seg))
                            self.content["seg_train_loss_list"] = seg_train_loss_list
                            train_iou_seg, train_acc_seg, train_all_acc_seg = seg_p.cal_iou_acc(save_dict=self.content,name='train')
                            train_defect_recall = seg_p.cal_defect_recall(save_dict=self.content,name='train')
                            train_defect_sensitivity = seg_p.cal_defect_sensitivity(save_dict=self.content,name='train')
                            #print("iou:{}, acc:{}, all_acc:{}".format(iou, acc, all_acc))
                            #print("train_loss_seg:",train_loss_seg)

                            #----evaluation(test set)
                            if self.seg_test_qty > 0:
                                seg_p.reset_arg()
                                seg_p.reset_defect_stat()
                                for idx_seg in range(test_ites_seg):
                                    #----get batch paths
                                    seg_paths = tl_seg.get_ite_data(self.seg_test_paths, idx_seg,
                                                                        batch_size=batch_size_seg_test)
                                    if json_check:
                                        seg_json_paths = tl_seg.get_ite_data(self.seg_test_json_paths, idx_seg,
                                                                                 batch_size=batch_size_seg_test)
                                    else:
                                        seg_json_paths = None
                                    #----get batch data
                                    if self_create_label:
                                        batch_data, batch_label = tl_seg.get_4D_data_create_mask(seg_paths,
                                                                                                 self.model_shape[1:],
                                                                                                 to_process=True,
                                                                                                 dtype=self.dtype)
                                    else:
                                        batch_data, batch_label = tl_seg.get_4D_img_label_data(seg_paths,
                                                                                               self.model_shape[1:],
                                                                                               json_paths=seg_json_paths,
                                                                                               dtype=self.dtype)
                                    #----put all data to tf placeholders
                                    recon = sess.run(self.recon,feed_dict={self.tf_input: batch_data})

                                    feed_dict = {self.tf_input: batch_data, self.tf_input_recon: recon,
                                                 self.tf_label_batch: batch_label,self.tf_keep_prob: 0}

                                    loss_temp = sess.run(self.loss_Seg, feed_dict=feed_dict)
                                    predict_label = sess.run(self.prediction_Seg, feed_dict=feed_dict)
                                    # predict_label = np.argmax(predict_label, axis=-1).astype(np.uint8)

                                    # ----calculate the loss and accuracy
                                    test_loss_seg += loss_temp

                                    seg_p.cal_intersection_union(predict_label, batch_label)
                                    _ = seg_p.cal_label_defect_by_acc_v2(predict_label,batch_label)
                                    _ = seg_p.cal_predict_defect_by_acc_v2(predict_label,batch_label)

                                test_loss_seg /= test_ites_seg
                                seg_test_result_dict['loss'] = test_loss_seg
                                # ----save results in the log file
                                seg_test_loss_list.append(float(test_loss_seg))
                                self.content["seg_test_loss_list"] = seg_test_loss_list
                                test_iou_seg, test_acc_seg, test_all_acc_seg = seg_p.cal_iou_acc(save_dict=self.content,
                                                                                                    name='test')
                                test_defect_recall = seg_p.cal_defect_recall(save_dict=self.content, name='test')
                                test_defect_sensitivity = seg_p.cal_defect_sensitivity(save_dict=self.content, name='test')

                                #----find the best performance(SEG)
                                target_of_best = seg_var.get('target_of_best')
                                print("target_of_best:",target_of_best)
                                if target_of_best == 'defect_recall':
                                    new_value = seg_p.sum_defect_recall()
                                elif target_of_best == 'defect_sensitivity':
                                    new_value = seg_p.sum_defect_sensitivity()
                                elif target_of_best == 'recall+sensitivity':
                                    new_value = seg_p.sum_defect_recall() + seg_p.sum_defect_sensitivity()
                                else:
                                    new_value = seg_p.sum_iou_acc()

                                if epoch == 0:
                                    LV['record_value_seg'] = new_value
                                else:
                                    if new_value > LV['record_value_seg']:
                                        # ----delete the previous pb
                                        if os.path.exists(LV['pb_seg_save_path_old']):
                                            os.remove(LV['pb_seg_save_path_old'])

                                        # ----save the better one
                                        pb_save_path = "infer_best_epoch{}.{}".format(epoch, self.pb_extension)
                                        pb_save_path = os.path.join(self.save_dir, pb_save_path)

                                        save_pb_file(sess, self.pb_save_list, pb_save_path, encode=encript_flag,
                                                     random_num_range=encode_num, header=encode_header)
                                        # ----update record value
                                        LV['record_value_seg'] = new_value
                                        LV['pb_seg_save_path_old'] = pb_save_path

                            #----display training results
                            msg_list = list()
                            msg_list.append("\n----訓練週期 {} 與相關結果如下----".format(epoch))
                            msg_list.append("Seg訓練集loss:{}".format(np.round(train_loss_seg, 4)))
                            if self.test_img_dir is not None:
                                msg_list.append("Seg驗證集loss:{}".format(np.round(test_loss_seg, 4)))

                            # msg_list.append("此次訓練所花時間: {}秒".format(np.round(d_t, 2)))
                            # msg_list.append("累積訓練時間: {}秒".format(np.round(total_train_time, 2)))
                            say_sth(msg_list, print_out=print_out, header='msg')

                            self.display_results(self.id2class_name,print_out,'訓練集',
                                                 iou=train_iou_seg,acc=train_acc_seg,
                                                 defect_recall=train_defect_recall,all_acc=train_all_acc_seg,
                                                 defect_sensitivity=train_defect_sensitivity
                                                 )
                            self.display_results(self.id2class_name, print_out, '驗證集',
                                                 iou=test_iou_seg, acc=test_acc_seg,
                                                 defect_recall=test_defect_recall, all_acc=test_all_acc_seg,
                                                 defect_sensitivity=test_defect_sensitivity
                                                 )
                            # self.display_iou_acc(train_iou_seg, train_acc_seg,train_defect_recall, train_all_acc_seg,
                            #                      self.id2class_name,name='訓練集',print_out=print_out)
                            # self.display_iou_acc(test_iou_seg, test_acc_seg,test_defect_recall, test_all_acc_seg,
                            #                      self.id2class_name, name='驗證集', print_out=print_out)

                            #----send protocol data(for C++ to draw)
                            msg_list, header_list = self.collect_data2ui(epoch, seg_train_result_dict,
                                                                         seg_test_result_dict)
                            say_sth(msg_list, print_out=print_out, header=header_list)

                            #----prediction for selected images
                            if self.seg_predict_qty > 0:
                                if (epoch + 1 ) % eval_epochs == 0:
                                    for idx_seg in range(predict_ites_seg):
                                        # ----get batch paths
                                        seg_paths = tl_seg.get_ite_data(self.seg_predict_paths, idx_seg,
                                                                        batch_size=batch_size_seg_test)

                                        #----get batch data
                                        batch_data, batch_label = tl_seg.get_4D_img_label_data(seg_paths,
                                                                                               self.model_shape[1:],
                                                                                               json_paths=None,
                                                                                               dtype=self.dtype)
                                        recon = sess.run(self.recon,feed_dict={self.tf_input:batch_data})
                                        predict_label = sess.run(self.tf_prediction_Seg,
                                                                 feed_dict={self.tf_input: batch_data,
                                                                            self.tf_input_recon:recon})
                                        # predict_label = np.argmax(predict_label, axis=-1).astype(np.uint8)

                                        batch_data *= 255
                                        batch_data = batch_data.astype(np.uint8)

                                        for i in range(len(predict_label)):
                                            img = batch_data[i]
                                            # label = batch_label[i]
                                            # predict_label = predict_label[i]

                                            #----label to color
                                            zeros = np.zeros_like(batch_data[i])
                                            for label_num in np.unique(predict_label[i]):
                                                if label_num != 0:
                                                    # print(label_num)
                                                    coors = np.where(predict_label[i] == label_num)
                                                    try:
                                                        zeros[coors] = self.id2color[label_num]
                                                    except:
                                                        print("error")

                                            predict_png = cv2.addWeighted(img, 0.5, zeros, 0.5, 0)
                                            #----create answer png
                                            path = self.seg_predict_paths[batch_size_seg_test * idx_seg + i]
                                            ext = path.split(".")[-1]
                                            json_path = path.strip(ext) + 'json'
                                            show_imgs = [predict_png]
                                            if os.path.exists(json_path):
                                                answer_png = tl_seg.get_single_label_png(path, json_path)
                                                show_imgs.append(answer_png)
                                            qty_show = len(show_imgs)
                                            plt.figure(num=1,figsize=(5*qty_show, 5*qty_show), clear=True)

                                            for i, show_img in enumerate(show_imgs):
                                                plt.subplot(1, qty_show, i + 1)
                                                plt.imshow(show_img)
                                                plt.axis('off')
                                                plt.title(titles[i])


                                            save_path = os.path.join(self.new_predict_dir, path.split("\\")[-1])
                                            plt.savefig(save_path)

                            #----save ckpt
                            # if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
                            #     save_pb_file(sess, self.pb_save_list, self.pb4seg_save_path,
                            #                  encode=encript_flag,
                            #                  random_num_range=encode_num, header=encode_header)

                    #----save ckpt, pb files
                    if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
                            model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)

                            # ----encode ckpt file
                            if encript_flag is True:
                                encode_CKPT(model_save_path, encode_num=encode_num, encode_header=encode_header)

                            #----save pb(normal)
                            save_pb_file(sess, self.pb_save_list, self.pb_save_path, encode=encript_flag,
                                         random_num_range=encode_num, header=encode_header)

                    #----record the end time
                    d_t = time.time() - d_t

                    epoch_time_list.append(d_t)
                    total_train_time = time.time() - t_train_start
                    self.content['total_train_time'] = float(total_train_time)
                    self.content['ave_epoch_time'] = float(np.average(epoch_time_list))

                    msg_list = []
                    msg_list.append("此次訓練所花時間: {}秒".format(np.round(d_t, 2)))
                    msg_list.append("累積訓練時間: {}秒".format(np.round(total_train_time, 2)))
                    say_sth(msg_list, print_out=print_out, header='msg')

                    with open(self.log_path, 'w') as f:
                        json.dump(self.content, f)

                    if encript_flag is True:
                        if os.path.exists(self.log_path):
                            file_transfer(self.log_path, cut_num_range=30, random_num_range=10)
                    msg = "儲存訓練結果數據至{}".format(self.log_path)
                    say_sth(msg, print_out=print_out)

            #----error messages
            if True in list(error_dict.values()):
                for key, value in error_dict.items():
                    if value is True:
                        say_sth('', print_out=print_out, header=key)
            else:
                say_sth('AI Engine結束!!期待下次再相見', print_out=print_out, header='AIE_end')

    #----functions
    def get_train_qty(self,ori_qty,ratio,print_out=False,name=''):
        if ratio is None:
            qty_train = ori_qty
        else:
            if ratio <= 1.0 and ratio > 0:
                qty_train = int(ori_qty * ratio)
                qty_train = np.maximum(1, qty_train)

        msg = "{}訓練集資料總共取數量{}".format(name,qty_train)
        say_sth(msg, print_out=print_out)

        return qty_train

    def config_check(self,config_dict):
        #----var
        must_list = ['train_img_dir', 'model_name', 'save_dir', 'epochs']
        # must_list = ['train_img_dir', 'test_img_dir', 'save_dir', 'epochs']
        must_flag = True
        default_dict = {"model_shape":[None,192,192,3],
                        'model_name':"type_1_0",
                        'loss_method':'ssim',
                        'activation':'relu',
                        'save_pb_name':'inference',
                        'opti_method':'adam',
                        'pool_type':['max', 'ave'],
                        'pool_kernel':[7, 2],
                        'embed_length':144,
                        'learning_rate':1e-4,
                        'batch_size':8,
                        'ratio':1.0,
                        'aug_times':2,
                        'hit_target_times':2,
                        'eval_epochs':2,
                        'save_period':2,
                        'kernel_list':[7,5,3,3,3],
                        'filter_list':[32,64,96,128,256],
                        'conv_time':1,
                        'rot':False,
                        'scaler':1,
                        #'preprocess_dict':{'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}
                        }


        #----get the must list
        if config_dict.get('must_list') is not None:
            must_list = config_dict.get('must_list')
        #----collect all keys of config_dict
        config_key_list = list(config_dict.keys())

        #----check the must list
        if config_dict.get("J_mode") is not True:

            #----check of must items
            for item in must_list:
                if not item in config_key_list:
                    msg = "Error: could you plz give me parameters -> {}".format(item)
                    say_sth(msg,print_out=print_out)
                    if must_flag is True:
                        must_flag = False

        #----parameters parsing
        if must_flag is True:
            #----model name
            if config_dict.get("J_mode") is not True:
                infer_num = config_dict['model_name'].split("_")[-1]
                if infer_num == '0':#
                    config_dict['infer_method'] = "AE_pooling_net"
                elif infer_num == '1':#
                    config_dict['infer_method'] = "AE_Unet"
                else:
                    config_dict['infer_method'] = "AE_transpose_4layer"

            #----optional parameters
            for key,value in default_dict.items():
                if not key in config_key_list:
                    config_dict[key] = value

        return must_flag,config_dict

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

    def dict_update(self,main_content,add_content):
        for key, value in add_content.items():
            main_content[key] = value

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

    # def display_iou_acc(self,iou,acc,defect_recall,all_acc,id2name,name='',print_out=False):
    def display_results(self,id2name,print_out,dataset_name,**kwargs):
        msg_list = []
        class_names = list(id2name.values())
        #a_dict = {'iou':iou, 'acc':acc,'defect_recall':defect_recall}
        for key,value_list in kwargs.items():
            if key == 'all_acc':
                msg_list.append("Seg{}_all_acc: {}".format(dataset_name,value_list))
            else:
                msg_list.append("Seg{}_{}:".format(dataset_name,key))
                msg_list.append("{}:".format(class_names))
                msg_list.append("{}:".format(value_list))

            # for i,value in enumerate(value_list):
            #     msg_list.append(" {}: {}".format(id2name[i],value))



        for msg in msg_list:
            say_sth(msg,print_out=print_out)

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

    AE_train = AE_Seg(para_dict)
    AE_train.model_init(para_dict)
    AE_train.train(para_dict)




