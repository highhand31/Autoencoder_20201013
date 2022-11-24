import cv2,sys,shutil,os,json,time,math,imgviz,copy
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
import AE_Seg_Util
# from AE_Seg_Util import Seg_performance,get_classname_id_color,\
#     get_latest_json_content,dict_transform,DataLoader4Seg

import models_AE_Seg
# from models_AE_Seg import AE_transpose_4layer,tf_mish,AE_JNet,AE_Resnet_Rot,AE_pooling_net,\
#     AE_Unet,Seg_DifNet,AE_Seg_pooling_net,preprocess,AE_pooling_net_V3,AE_pooling_net_V4,Seg_DifNet_V2,\
#     AE_pooling_net_V5,AE_dense_sampling,AE_pooling_net_V6,AE_pooling_net_V7,Seg_pooling_net_V4,Seg_pooling_net_V7,\
#     Seg_pooling_net_V8,Seg_pooling_net_V9,AE_pooling_net_V8,Seg_pooling_net_V10

import config_mit
from models_MiT import MixVisionTransformer,MiTDecoder


print_out = True
SockConnected = False
img_format = {'png','PNG','jpg','JPG','JPEG','jpeg','bmp','BMP','webp','tiff','TIFF'}

#----functions
def dtype_transform(**kwargs):
    re = dict()
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            v = v.astype(float)
            v = v.tolist()
            re[k] = v
        elif isinstance(v, list):
            new_v = list()
            for digit in v:
                new_v.append(float(digit))
            re[k] = new_v
        # elif isinstance(v,(int,str)):
        #     re[k] = v
        else:
            re[k] = v

    return re

def file_transfer(file_path, cut_num_range=100, random_num_range=87,header=[24, 97, 28, 98],save_name=None):
    print_out = False
    with open(file_path, 'rb') as f:
        content = f.read()
    # data reverse
    content = content[::-1]

    leng = len(content)
    msg = "file length = ".format(leng)
    #self.say_sth(msg,print_out=print_out)

    cut_num = np.random.randint(10, cut_num_range)
    msg = "cut_num = ".format(cut_num)
    #self.say_sth(msg, print_out=print_out)

    slice_num = math.ceil(leng / cut_num)
    msg = "slice_num = ".format(slice_num)
    #self.say_sth(msg, print_out=print_out)

    compliment_num = slice_num * cut_num - leng
    msg = "compliment_num = ".format(compliment_num)
    #self.say_sth(msg, print_out=print_out)
    compliment_data = np.random.randint(0, 128, compliment_num, dtype=np.uint8)

    seq = [i for i in range(cut_num)]
    np.random.shuffle(seq)
    #print(seq)

    header = np.array(header, dtype=np.uint8)
    random_num = np.random.randint(0, 128, random_num_range, dtype=np.uint8)  # 根據數量製造假資料
    msg = "bytes of header len = ".format(len(bytes(header)))
    #self.say_sth(msg, print_out=print_out)
    msg = "bytes of random_num len = ".format(len(bytes(random_num)))
    #self.say_sth(msg, print_out=print_out)

    # new_filename = os.path.join(os.path.dirname(file_path), 'encode_data.dat')
    if save_name is None:
        new_filename = file_path
    else:
        new_filename = save_name

    with open(new_filename, 'wb') as f:
        f.write(bytes(header))
        f.write(bytes(random_num))
        f.write(cut_num.to_bytes(4, byteorder='little', signed=False))
        f.write(compliment_num.to_bytes(4, byteorder='little', signed=False))
        f.write(bytes(seq))
        add_complement = False
        for i in seq:
            #print("i = ", i)
            num_start = slice_num * i
            num_end = num_start + slice_num
            if num_end > leng:
                num_end = leng
                add_complement = True
                msg = "num_end over leng"
                #self.say_sth(msg, print_out=print_out)
            msg = "num_start = ".format(num_start)
            # self.say_sth(msg, print_out=print_out)
            msg = "num_end = ".format(num_end)
            # self.say_sth(msg, print_out=print_out)

            f.write(bytes(content[num_start:num_end]))
            if add_complement is True:
                f.write(bytes(compliment_data))
                add_complement = False
                msg = "add complement ok"
                #self.say_sth(msg, print_out=print_out)
    msg = "encode data is completed in {}".format(new_filename)
    # self.say_sth(msg, print_out=print_out)

def file_decode_v2(file_path, random_num_range=87,header = [24, 97, 28, 98],
                   save_dir=None,return_value=False,to_save=True,print_f=None):
    #----var
    print_out = True
    header_len = 4
    decode_flag = True


    if print_f is None:
        print_f = print

    #----read the file
    with open(file_path, 'rb') as f:
        content = f.read()

    #----check headers
    for i in range(4):
        try:
            # print(int(content[i]))
            if int(content[i]) != header[i]:
                decode_flag = False
        except:
            decode_flag = False
    # msg = "decode_flag:{}".format(decode_flag)
    # print_f(msg)

    #----decode process
    if decode_flag is False:
        if return_value is True:
            return None
    else:
        # print("execute decode")
        leng = len(content)
        #msg = "file length = {}".format(leng)
        # say_sth(msg, print_out=print_out)
        #print_f(msg)

        cut_num_start = random_num_range + header_len
        cut_num = int.from_bytes(content[cut_num_start:cut_num_start + 4], byteorder="little", signed=False)
        #msg = "cut_num = {}".format(cut_num)
        # say_sth(msg, print_out=print_out)
        #print_f(msg)

        compliment_num_start = cut_num_start + 4
        compliment_num = int.from_bytes(content[compliment_num_start:compliment_num_start + 4], byteorder="little",
                                        signed=False)
        # msg = "compliment_num = {}".format(compliment_num)
        # # say_sth(msg, print_out=print_out)
        # print_f(msg)

        seq_num_start = compliment_num_start + 4
        seq = content[seq_num_start:seq_num_start + cut_num]

        pb_fragment = content[seq_num_start + cut_num:]
        leng = len(pb_fragment)
        # msg = "pb_fragment size = {}".format(pb_fragment)
        # say_sth(msg, print_out=print_out)

        slice_num = math.ceil(leng / cut_num)
        # msg = "slice_num = {}".format(slice_num)
        # # say_sth(msg, print_out=print_out)
        # print_f(msg)

        seq2num = list()
        for i in seq:
            seq2num.append(int(i))
        sort_seq = np.argsort(seq2num)

        for num, i in enumerate(sort_seq):
            num_start = slice_num * i
            num_end = num_start + slice_num
            if num == cut_num - 1:
                num_end -= compliment_num
                # print("subtract compliment")
            # print("num_start = {}, num_end = {}".format(num_start, num_end))
            if num == 0:
                temp = pb_fragment[num_start:num_end]
            else:
                temp += pb_fragment[num_start:num_end]
        temp = temp[::-1]  # data reverse

        if to_save is True:
            #----output filename
            if save_dir is None:
                new_filename = file_path
            else:
                new_filename = os.path.join(save_dir,file_path.split("\\")[-1])

            #----save the file
            with open(new_filename, 'wb') as f:
                f.write(temp)
            # msg = "decode data is completed in {}".format(new_filename)
            # # say_sth(msg, print_out=print_out)
            # print_f(msg)

        if return_value is True:
            return temp
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

def transmit_data2ui(epoch,print_out,**kwargs):
    #----var
    msg_list = list()
    header_list = list()

    for header,values in kwargs.items():
        msg = ""
        if isinstance(values,(list,np.ndarray)):
            leng = len(values)
            for i,value in enumerate(values):
                str_value = str(value)
                splits = str_value.split(".")

                if len(splits[-1]) > 8:
                    msg += "{}.{}".format(splits[0],splits[-1][:8])
                else:
                    msg += str_value

                if (i+1) < leng:
                    msg += ','
        elif isinstance(values,float):
            msg = "{:.8f}".format(values)
        else:
            msg = str(values)

        msg_list.append('{},{}'.format(epoch, msg))
        header_list.append(header)

    say_sth(msg_list, print_out=print_out, header=header_list)

def GPU_setting(GPU_ratio):
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    if GPU_ratio is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

    return config

def weights_check(sess,saver,save_dir,encript_flag=True,encode_num=87,encode_header=[24, 97, 28, 98],
                  print_out=True):
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
        except Exception as err:
            say_sth(f"Error:{err}", print_out=True)
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
            except Exception as err:
                say_sth(f"Error:{err}", print_out=True)
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
            except Exception as err:
                say_sth(f"Error:{err}", print_out=True)
                # msg = "恢復模型時產生錯誤"
                # say_sth(msg, print_out=print_out)
                status = False
                try:
                    sess.run(tf.global_variables_initializer())
                    status = True
                except:
                    say_sth(error_msg)
                    status = False

    return status

def save_pb_file(sess,pb_save_list,pb_save_path,encode=False,random_num_range=87,header=[24,97,28,98],
                 print_out=True):
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

def create_pb_path(save_pb_name, save_dir, to_encode=False,add_name_tail=False):
    ext = 'pb'
    if not isinstance(save_pb_name, str):
        save_pb_name = "pb_model"

    if to_encode:
        ext = 'nst'

    if add_name_tail is True:
        name_tailer = get_time4tailer()
        pb_save_path = "{}_{}.{}".format(save_pb_name, name_tailer, ext)
    else:
        pb_save_path = "{}.{}".format(save_pb_name, ext)

    return os.path.join(save_dir, pb_save_path)

def create_save_dir(save_dir):
    if isinstance(save_dir,str):
        pass
    else:
        msg = "輸入的權重資料夾路徑有誤，系統自動建立資料夾!"
        say_sth(msg,print_out=True)

        #----create a default save dir
        save_dir = os.path.join(os.getcwd(),"model_saver")

    if os.path.exists(save_dir):
        msg = "儲存權重資料夾已存在:{}".format(save_dir)
    else:
        os.makedirs(save_dir)
        msg = "儲存權重資料夾已建立:{}".format(save_dir)
    say_sth(msg,print_out=True)

    return save_dir

def get_time4tailer():
    xtime = time.localtime()
    name_tailer = ''
    for i in range(6):
        string = str(xtime[i])
        if len(string) == 1:
            string = '0' + string
        name_tailer += string

    return name_tailer

def update_dict(new_dict,old_dict):
    re_dict = copy.deepcopy(old_dict)
    for key, value in new_dict.items():
        re_dict[key] = value
    return re_dict

def get_paths(img_source,ext=None):
    # ----var
    paths = list()
    try:
        if img_source is None:
            say_sth("Warning: 沒有輸入資料夾路徑",print_out=True)
        else:
            if not isinstance(img_source, list):
                img_source = [img_source]

            if isinstance(ext,str):
                ext_list = [ext]
            else:
                ext_list = img_format

            for img_dir in img_source:
                temp = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] in ext_list]
                if len(temp) > 0:
                    paths.extend(temp)

            #----path qty check
            if len(paths) == 0:
                say_sth("Warning:沒有找到支援的圖片檔案:{}".format(img_dir),print_out=True)
    except Exception as err:
        paths = list()
        print(f"Warning:{err}")

    return np.array(paths),len(paths)

def break_signal_check():
    break_flag = False
    if SockConnected:
        if len(Sock.Message):
            if Sock.Message[-1][:4] == "$S00":
                break_flag = True

    return break_flag

def display_results(class_names,print_out,**kwargs):
    msg_list = []
    if class_names is not None:
        msg_list.append("{}:".format(class_names))
    for k, v in kwargs.items():
        if isinstance(v, list):
            msg_list.append(f"{k}:")
            msg_list.append(f"{v}")
        else:
            msg_list.append(f"{k}: {v}")

    for msg in msg_list:
        say_sth(msg, print_out=print_out)

class TrainDataRecord():
    def __init__(self,**kwargs):
        self.content = kwargs

    def update(self,**kwargs):
        key_list = list(self.content.keys())
        for k,v in kwargs.items():
            if k in key_list:
                if isinstance(self.content[k],list):
                    self.content[k].append(v)
                else:
                    self.content[k] = v
            else:
                self.content[k] = v

class TrainLog():
    def __init__(self,save_dir,to_encode=False,filename='train_result',
                 cut_num_range=30,random_num_range=10):
        self.save_dir = save_dir
        self.to_encode = to_encode
        self.filename = filename
        self.content = dict()
        self.cut_num_range = cut_num_range
        self.random_num_range = random_num_range
        self.create_log_path()

    def create_log_path(self,):
        count = 0
        ext = "json"

        if self.to_encode:
            ext = "nst"

        while(True):
            log_path = "{}_{}.{}".format(self.filename, count, ext)
            log_path = os.path.join(self.save_dir, log_path)
            if not os.path.exists(log_path):
                say_sth("本次建立的log路徑:{}".format(log_path),print_out=True)
                break
            count += 1

        self.log_path = log_path

    def update(self,**kwargs):
        for k,v in kwargs.items():
            self.content[k] = v

    def save(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.content, f)

        if self.to_encode:
            if os.path.exists(self.log_path):
                file_transfer(self.log_path, cut_num_range=self.cut_num_range,
                              random_num_range=self.random_num_range)

        msg = "儲存訓練結果數據至{}".format(self.log_path)
        say_sth(msg, print_out=True)

class AE_Seg():
    def __init__(self,para_dict,user_dict=None):

        #----config process
        if isinstance(user_dict,dict):
            para_dict = self.update_config_dict(user_dict,para_dict)

        #----common var
        self.print_out = para_dict.get('print_out')
        self.encript_flag = para_dict.get('encript_flag')

        #----AE process
        to_train_ae = False
        recon_flag = False
        ae_var = para_dict.get('ae_var')
        if isinstance(ae_var,dict):
            train_img_dir = ae_var.get('train_img_dir')
            test_img_dir = ae_var.get('test_img_dir')
            recon_img_dir = ae_var.get('recon_img_dir')
            # special_img_dir = ae_var.get('special_img_dir')

            # ----AE image path process
            self.train_paths, self.train_path_qty = get_paths(train_img_dir)
            self.test_paths, self.test_path_qty = get_paths(test_img_dir)
            # self.sp_paths, self.sp_path_qty = get_paths(special_img_dir)
            self.recon_paths, self.recon_path_qty = get_paths(recon_img_dir)
            qty_status_list = self.path_qty_process(
                dict(AE訓練集=self.train_path_qty,
                     AE驗證集=self.test_path_qty,
                     # AE加強學習集=self.sp_path_qty,
                     AE重建圖集=self.recon_path_qty
                     ))
            recon_flag = qty_status_list[-1]
            to_train_ae = bool(np.sum(qty_status_list[:-1]))

        #----SEG process
        to_train_seg = False
        seg_var = para_dict.get('seg_var')
        if isinstance(seg_var,dict) is True:
            train_img_seg_dir = seg_var.get('train_img_seg_dir')
            test_img_seg_dir = seg_var.get('test_img_seg_dir')
            # predict_img_dir = seg_var.get('predict_img_dir')
            to_train_w_AE_paths = seg_var.get('to_train_w_AE_paths')
            id2class_name = seg_var.get('id2class_name')
            select_OK_ratio = 0.2


            #----SEG train image path process
            self.seg_train_paths, self.seg_train_qty = get_paths(train_img_seg_dir)
            self.seg_test_paths, self.seg_test_qty = get_paths(test_img_seg_dir)
            # self.seg_predict_paths, self.seg_predict_qty = get_paths(predict_img_dir)

            qty_status_list = self.path_qty_process(
                dict(SEG訓練集=self.seg_train_qty,
                     SEG驗證集=self.seg_test_qty,
                     # SEG預測集=self.seg_predict_qty
                     ))
            to_train_seg = bool(np.sum(qty_status_list))

            #----read class names
            classname_id_color_dict = AE_Seg_Util.get_classname_id_color_v2(id2class_name,print_out=self.print_out)

            # ----train with AE ok images
            self.seg_path_change_process(to_train_w_AE_paths, to_train_ae, select_OK_ratio=select_OK_ratio)

        # #----log update
        # log = classname_id_color_dict.copy()

        #----local var to global
        self.para_dict = para_dict
        self.to_train_ae = to_train_ae
        self.to_train_seg = to_train_seg
        self.status = bool(to_train_seg+to_train_ae)
        # self.log = log
        if to_train_ae:
            self.train_img_dir = train_img_dir
            self.test_img_dir = test_img_dir
            # self.special_img_dir = special_img_dir
            self.recon_img_dir = recon_img_dir
            self.recon_flag = recon_flag
        if to_train_seg:
            self.train_img_seg_dir = train_img_seg_dir
            self.test_img_seg_dir = test_img_seg_dir
            # self.predict_img_dir = predict_img_dir
            self.classname_id_color_dict = classname_id_color_dict
            self.class_num = len(classname_id_color_dict['class_names'])

    def model_init(self):
        #----var parsing
        para_dict = self.para_dict

        #----common var
        model_shape = para_dict.get('model_shape')
        preprocess_dict = para_dict.get('preprocess_dict')
        lr = para_dict['learning_rate']
        dtype = para_dict.get('dtype')


        #----var
        acti_dict = {'relu': tf.nn.relu, 'mish': models_AE_Seg.tf_mish, None: tf.nn.relu}
        pb_save_list = list()

        #----var process
        if dtype is None:
            dtype = 'float32'

        # ----create the dir to save model weights(CKPT, PB)
        save_dir = create_save_dir(para_dict.get('save_dir'))

        # ----tf_placeholder declaration
        tf_input = tf.placeholder(dtype, shape=model_shape, name='input')
        tf_keep_prob = tf.placeholder(dtype=dtype, name="keep_prob")


        if self.to_train_ae:
            ae_var = para_dict['ae_var']
            infer_method = ae_var.get('infer_method')
            model_name = ae_var.get('model_name')
            acti = ae_var['activation']
            pool_kernel = ae_var.get('pool_kernel')
            kernel_list = ae_var.get('kernel_list')
            filter_list = ae_var.get('filter_list')
            pool_type = ae_var.get('pool_type')
            loss_method = ae_var['loss_method']
            opti_method = ae_var['opti_method']
            embed_length = ae_var.get('embed_length')
            stride_list = ae_var.get('stride_list')
            scaler = ae_var.get('scaler')
            # process_dict = ae_var['process_dict']

            rot = ae_var.get('rot')
            special_process_list = ['rdm_patch', 'rdm_perlin', 'rdm_light_defect']
            return_ori_data = False
            #----if return ori data or not

            # for name in special_process_list:
            #     if process_dict.get(name) is True:
            #         return_ori_data = True
            #         break
            #----random patch
            # rdm_patch = False
            # if process_dict.get('rdm_patch') is True:
            #     rdm_patch = True

            #----filer scaling process
            if scaler is not None:
                filter_list = (np.array(filter_list) / scaler).astype(np.uint16)

            if return_ori_data is True:
                self.tf_input_ori = tf.placeholder(dtype, shape=model_shape, name='input_ori')

            #tf_label_batch = tf.placeholder(dtype=tf.int32, shape=[None], name="label_batch")
            #tf_phase_train = tf.placeholder(tf.bool, name="phase_train")

            # ----activation selection
            acti_func = acti_dict[acti]


            avepool_out = self.__avepool(tf_input, k_size=5, strides=1)
            #----preprocess
            if preprocess_dict is None:
                tf_input_process = tf.identity(tf_input,name='preprocess')
                if return_ori_data is True:
                    tf_input_ori_no_patch = tf.identity(self.tf_input_ori, name='tf_input_ori_no_patch')
            else:
                tf_temp = models_AE_Seg.preprocess(tf_input, preprocess_dict, print_out=self.print_out)
                tf_input_process = tf.identity(tf_temp,name='preprocess')

                if return_ori_data is True:
                    tf_temp_2 = models_AE_Seg.preprocess(self.tf_input_ori, preprocess_dict, print_out=self.print_out)
                    tf_input_ori_no_patch = tf.identity(tf_temp_2, name='tf_input_ori_no_patch')

            #----AIE model mapping
            if model_name is not None:
                if model_name.find("type_5") >= 0:
                    infer_method = "AE_pooling_net_V" + model_name.split("_")[-1]

            #----AE inference selection
            if infer_method == "AE_transpose_4layer":
                recon = models_AE_Seg.AE_transpose_4layer(tf_input, kernel_list, filter_list,activation=acti_func,
                                                  pool_kernel=pool_kernel,pool_type=pool_type)
                recon = tf.identity(recon, name='output_AE')
                #(tf_input, kernel_list, filter_list,pool_kernel=2,activation=tf.nn.relu,pool_type=None)
                # recon = AE_refinement(temp,96)
            elif infer_method == "AE_pooling_net_V3":
                recon = models_AE_Seg.AE_pooling_net_V3(tf_input_process, kernel_list, filter_list, activation=acti_func,
                                          pool_kernel_list=pool_kernel, pool_type_list=pool_type,
                                          stride_list=stride_list, print_out=self.print_out)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_pooling_net_V4":
                recon = models_AE_Seg.AE_pooling_net_V4(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
                                          to_reduce=ae_var.get('to_reduce'),print_out=self.print_out)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_pooling_net_V5":
                recon = models_AE_Seg.AE_pooling_net_V5(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
                                          to_reduce=ae_var.get('to_reduce'),print_out=self.print_out)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_pooling_net_V6":
                recon = models_AE_Seg.AE_pooling_net_V6(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
                                          to_reduce=ae_var.get('to_reduce'),print_out=self.print_out)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_pooling_net_V7":

                recon = models_AE_Seg.AE_pooling_net_V7(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
                                         print_out=self.print_out)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_pooling_net_V8":

                self.tf_input_standard = tf.placeholder(dtype, shape=model_shape, name='input_standard')
                recon = models_AE_Seg.AE_pooling_net_V8(tf_input_process,self.tf_input_standard,ae_var['encode_dict'],ae_var['decode_dict'],
                                         print_out=self.print_out)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_dense_sampling":
                sampling_factor = 16
                filters = 2
                recon = models_AE_Seg.AE_dense_sampling(tf_input_process,sampling_factor,filters)
                recon = tf.identity(recon, name='output_AE')
            elif infer_method == "AE_pooling_net":

                recon = models_AE_Seg.AE_pooling_net(tf_input_process, kernel_list, filter_list,activation=acti_func,
                                      pool_kernel_list=pool_kernel,pool_type_list=pool_type,
                                      stride_list=stride_list,rot=rot,print_out=self.print_out)
                recon = tf.identity(recon,name='output_AE')
            elif infer_method == "AE_Seg_pooling_net":
                AE_out,Seg_out = models_AE_Seg.AE_Seg_pooling_net(tf_input, kernel_list, filter_list,activation=acti_func,
                                                  pool_kernel_list=pool_kernel,pool_type_list=pool_type,
                                       rot=rot,print_out=self.print_out,preprocess_dict=preprocess_dict,
                                           class_num=self.class_num)
                recon = tf.identity(AE_out,name='output_AE')
            else:
                if model_name is not None:
                    display_name = model_name
                else:
                    display_name = infer_method
                say_sth(f"Error:AE model doesn't exist-->{display_name}", print_out=True)

            # ----AE loss method selection
            if loss_method == 'mse':
                loss_AE = tf.reduce_mean(tf.pow(recon - tf_input, 2), name="loss_AE")
            elif loss_method == 'ssim':
                # self.loss_AE = tf.reduce_mean(tf.image.ssim_multiscale(tf.image.rgb_to_grayscale(self.tf_input),tf.image.rgb_to_grayscale(self.recon),2),name='loss')

                if return_ori_data is True:
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

        #----Seg inference selection
        if self.to_train_seg:
            seg_var = para_dict['seg_var']
            infer_method4Seg = seg_var.get('infer_method')
            model_name4seg = seg_var.get('model_name')
            pool_kernel4Seg = seg_var['pool_kernel']
            pool_type4Seg = seg_var.get('pool_type')
            kernel_list4Seg = seg_var['kernel_list']
            filter_list4Seg = seg_var['filter_list']
            loss_method4Seg = seg_var.get('loss_method')
            opti_method4Seg = seg_var.get('opti_method')
            preprocess_dict4Seg = seg_var.get('preprocess_dict')
            rot4Seg = seg_var.get('rot')
            acti_seg = seg_var.get('activation')


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

            #----AIE model mapping
            if model_name4seg is not None:
                if model_name4seg.find("type_5") >= 0:
                    infer_method4Seg = "Seg_pooling_net_V" + model_name4seg.split("_")[-1]

            #----Seg model selection
            if infer_method4Seg == "Seg_DifNet":
                logits_Seg = models_AE_Seg.Seg_DifNet(tf_input_process,tf_input_recon, kernel_list4Seg, filter_list4Seg,activation=acti_func,
                                   pool_kernel_list=pool_kernel4Seg,pool_type_list=pool_type4Seg,
                                   rot=rot4Seg,print_out=self.print_out,preprocess_dict=preprocess_dict4Seg,class_num=self.class_num)
                softmax_Seg = tf.nn.softmax(logits_Seg,name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'Seg_DifNet_V2':
                logits_Seg = models_AE_Seg.Seg_DifNet_V2(tf_input_process,tf_input_recon,seg_var['encode_dict'],seg_var['decode_dict'],
                              class_num=self.class_num,print_out=self.print_out)
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'Seg_pooling_net_V1':
                logits_Seg = models_AE_Seg.Seg_pooling_net_V1(tf_input_process, tf_input_recon, seg_var['encode_dict'],
                                                seg_var['decode_dict'],
                                                out_channel=self.class_num,
                                                print_out=self.print_out)
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'Seg_pooling_net_V4':
                logits_Seg = models_AE_Seg.Seg_pooling_net_V4(tf_input_process,tf_input_recon, seg_var['encode_dict'], seg_var['decode_dict'],
                                          to_reduce=seg_var.get('to_reduce'),out_channel=self.class_num,
                                          print_out=self.print_out)
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'Seg_pooling_net_V7':
                logits_Seg = models_AE_Seg.Seg_pooling_net_V7(tf_input_process,tf_input_recon, seg_var['encode_dict'], seg_var['decode_dict'],
                                          out_channel=self.class_num,
                                          print_out=self.print_out)
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'Seg_pooling_net_V8':
                logits_Seg = models_AE_Seg.Seg_pooling_net_V8(tf_input_process, tf_input_recon, seg_var['encode_dict'],
                                                seg_var['decode_dict'],
                                                to_reduce=seg_var.get('to_reduce'), out_channel=self.class_num,
                                                print_out=self.print_out)
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'Seg_pooling_net_V9':
                logits_Seg = models_AE_Seg.Seg_pooling_net_V9(tf_input_process, tf_input_recon, seg_var['encode_dict'],
                                                seg_var['decode_dict'],
                                                to_reduce=seg_var.get('to_reduce'), out_channel=self.class_num,
                                                print_out=self.print_out)
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'Seg_pooling_net_V10':
                logits_Seg = models_AE_Seg.Seg_pooling_net_V10(tf_input_process, tf_input_recon, seg_var['encode_dict'],
                                                seg_var['decode_dict'],
                                                to_reduce=seg_var.get('to_reduce'), out_channel=self.class_num,
                                                print_out=self.print_out)
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
            elif infer_method4Seg == 'AE_Seg_pooling_net':
                logits_Seg = Seg_out
                softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
                prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
                prediction_Seg = tf.cast(prediction_Seg,tf.uint8, name='predict_Seg')
            else:
                if model_name4seg is not None:
                    display_name = model_name4seg
                else:
                    display_name = infer_method4Seg
                say_sth(f"Error:Seg model doesn't exist-->{display_name}", print_out=True)

            #----Seg loss method selection
            if loss_method4Seg == "cross_entropy":
                loss_Seg = tf.reduce_mean(v2.nn.sparse_softmax_cross_entropy_with_logits(tf_label_batch,logits_Seg),name='loss_Seg')

            #----Seg optimizer selection
            if opti_method4Seg == "adam":
                opt_Seg = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_Seg)


            # ----appoint PB node names
            pb_save_list.extend(['predict_Seg','dummy_out'])

        # ----pb filename(common)
        pb_save_path = create_pb_path(para_dict.get('save_pb_name'), save_dir,
                                      to_encode=self.encript_flag,
                                      add_name_tail=para_dict.get('add_name_tail'))


        #----save SEG coloer index image
        _ = AE_Seg_Util.draw_color_index(self.classname_id_color_dict['class_names'], save_dir=save_dir)

        # if self.to_train_seg is True and self.seg_predict_qty > 0:
        #     new_dir_name = "img_seg-" + os.path.basename(save_dir)
        #     #----if recon_img_dir is list format
        #     if isinstance(self.predict_img_dir,list):
        #         dir_path = self.predict_img_dir[0]
        #     else:
        #         dir_path = self.predict_img_dir
        #     self.new_predict_dir = os.path.join(dir_path, new_dir_name)
        #     if not os.path.exists(self.new_predict_dir):
        #         os.makedirs(self.new_predict_dir)

        out_dir_prefix = os.path.join(save_dir, "model")
        self.saver = tf.train.Saver(max_to_keep=2)

        #----create train log module
        self.log = TrainLog(save_dir, to_encode=self.encript_flag, filename="train_result")
        self.log.update(**self.para_dict)
        self.log.update(**self.classname_id_color_dict)
        self.log.update(pb_save_list=pb_save_list)


        # ----local var to global
        self.model_shape = model_shape
        self.tf_input = tf_input
        self.tf_keep_prob = tf_keep_prob
        self.save_dir = save_dir
        self.pb_save_path = pb_save_path
        self.pb_save_list = pb_save_list
        # self.pb_extension = pb_extension
        self.dtype = dtype
        if self.to_train_ae:
            self.avepool_out = avepool_out
            self.recon = recon
            self.loss_AE = loss_AE
            self.opt_AE = opt_AE
            self.out_dir_prefix = out_dir_prefix
            self.loss_method = loss_method
            self.return_ori_data = return_ori_data
            self.infer_method = infer_method

        if self.to_train_seg:
            self.tf_label_batch = tf_label_batch
            self.tf_input_recon = tf_input_recon
            self.tf_prediction_Seg = prediction_Seg
            self.infer_method4Seg = infer_method4Seg
            self.logits_Seg = logits_Seg
            self.loss_Seg = loss_Seg
            self.opt_Seg = opt_Seg
            self.prediction_Seg = prediction_Seg
            self.loss_method4Seg = loss_method4Seg
            # self.pb4seg_save_path = pb4seg_save_path
            self.tf_dropout = tf_dropout
            self.infer_method4Seg = infer_method4Seg

    def train(self):
        para_dict = self.para_dict

        # ----var parsing
        epochs = para_dict['epochs']
        eval_epochs = para_dict.get('eval_epochs')
        to_fix_ae = para_dict.get('to_fix_ae')
        to_fix_seg = para_dict.get('to_fix_seg')
        save_period = para_dict.get('save_period')


        self.encode_header_process(para_dict.get('encode_header'), para_dict.get('encode_num'))

        if save_period is None:
            save_period = 1

        # ----local var
        LV = dict(
            pb_save_path_old="",
            pb_seg_save_path_old="",
            record_value=0,
            record_value_seg=0
            )
        self.break_flag = False
        self.error_dict = {'GPU_resource_error': False}


        #----AE hyper-parameters
        if self.to_train_ae:
            ae_var = para_dict['ae_var']
            self.set_target_AE(**ae_var.get('target'))
            dataloader_AE_train = AE_Seg_Util.DataLoader4Seg(self.train_paths,
                                                       only_img=True,
                                                       batch_size=ae_var['batch_size'],
                                                       pipelines=ae_var.get("train_pipelines"),
                                                       to_shuffle=True,
                                                       print_out=self.print_out)
            dataloader_AE_val = AE_Seg_Util.DataLoader4Seg(self.test_paths,
                                                             only_img=True,
                                                             batch_size=ae_var['batch_size'],
                                                             pipelines=ae_var.get("val_pipelines"),
                                                             to_shuffle=False,
                                                             print_out=self.print_out)
            train_AE_Data = TrainDataRecord(
                train_AE_loss=list(),
                val_AE_loss=list(),
            )

        #----SEG hyper-parameters
        if self.to_train_seg:
            seg_var = para_dict['seg_var']
            class_names = self.classname_id_color_dict['class_names']
            dataloader_SEG_train = AE_Seg_Util.DataLoader4Seg(self.seg_train_paths,
                                                             only_img=False,
                                                             batch_size=seg_var['batch_size'],
                                                             pipelines=seg_var.get("train_pipelines"),
                                                             to_shuffle=True,
                                                             print_out=self.print_out)

            dataloader_SEG_val = AE_Seg_Util.DataLoader4Seg(self.seg_test_paths,
                                                              only_img=False,
                                                              batch_size=seg_var['batch_size'],
                                                              pipelines=seg_var.get("val_pipelines"),
                                                              to_shuffle=False,
                                                              print_out=self.print_out)
            # if self.seg_predict_qty > 0:
            #     dataloader_SEG_predict = AE_Seg_Util.DataLoader4Seg(self.seg_predict_paths,
            #                                                     only_img=False,
            #                                                     batch_size=seg_var['batch_size'],
            #                                                     pipelines=seg_var.get("val_pipelines"),
            #                                                     to_shuffle=False,
            #                                                     print_out=self.print_out)
            #     dataloader_SEG_predict.set_classname_id_color(**self.classname_id_color_dict)

            dataloader_SEG_train.set_classname_id_color(**self.classname_id_color_dict)
            dataloader_SEG_val.set_classname_id_color(**self.classname_id_color_dict)

            #titles = ['prediction', 'answer']
            # batch_size_seg_test = batch_size_seg
            self.seg_p = AE_Seg_Util.Seg_performance(print_out=self.print_out)
            self.seg_p.set_classname_id_color(**self.classname_id_color_dict)

            train_SEG_Data = TrainDataRecord(
                train_SEG_loss=list(),
                train_SEG_acc=list(),
                train_SEG_iou=list(),
                train_SEG_defect_recall=list(),
                train_SEG_defect_sensitivity=list(),

                val_SEG_loss=list(),
                val_SEG_acc=list(),
                val_SEG_iou=list(),
                val_SEG_defect_recall=list(),
                val_SEG_defect_sensitivity=list(),
            )

        self.set_time_start()

        #----GPU setting
        config = GPU_setting(para_dict.get('GPU_ratio'))
        self.sess = tf.Session(config=config)
        # with tf.Session(config=config) as sess:
        status = weights_check(self.sess, self.saver, self.save_dir, encript_flag=self.encript_flag,
                               encode_num=self.encode_num, encode_header=self.encode_header,
                               )
        if status is False:
            self.error_dict['GPU_resource_error'] = True
        elif status is True:

            # ----epoch training
            for epoch in range(epochs):
                #----read manual cmd
                # break_flag = self.read_manual_cmd(para_dict.get('to_read_manual_cmd'))

                #----record the epoch start time
                self.set_epoch_start_time()

                #----AE part
                if self.to_train_ae is True:
                    if to_fix_ae is False:
                        #----optimizations(AE train set)
                        self.ae_opti_by_dataloader(dataloader_AE_train)
                        if self.break_flag:
                            break

                        #----evaluation(AE train set)
                        train_eval_dict = self.ae_eval_by_dataloader(dataloader_AE_train,name='train')

                        #----data update
                        train_AE_Data.update(**train_eval_dict)
                        self.log.update(**train_AE_Data.content)

                        #----evaluation(validation set)
                        val_eval_dict = self.ae_eval_by_dataloader(dataloader_AE_val,name='val')

                        #----data update
                        train_AE_Data.update(**val_eval_dict)
                        self.log.update(**train_AE_Data.content)

                        #----display training results
                        msg = "\n----訓練週期 {} 與相關結果如下----".format(epoch)
                        say_sth(msg, print_out=self.print_out)
                        display_results(None, self.print_out,**train_eval_dict)
                        display_results(None, self.print_out,**val_eval_dict)

                        #----send protocol data(for UI to draw)
                        transmit_data2ui(epoch, self.print_out,**train_eval_dict)
                        transmit_data2ui(epoch, self.print_out,**val_eval_dict)

                        #----find the best performance
                        self.performance_process_AE(epoch,val_eval_dict["val_AE_loss"], LV)

                        #----Check if stops the training
                        goal = self.target_comparison_AE(val_eval_dict["val_AE_loss"],loss_method=self.loss_method)
                        if goal is True:
                            break

                        # ----test image reconstruction
                        if self.recon_flag is True:
                            # if (epoch + 1) % eval_epochs == 0 and train_loss > 0.80:
                            if (epoch + 1) % eval_epochs == 0:
                                for filename in self.recon_paths:
                                    test_img = self.__img_read(filename, self.model_shape[1:],dtype=self.dtype)
                                    # if self.infer_method == 'AE_pooling_net_V8':
                                    #     img_standard = tl.get_4D_data([img_standard_path] * len(test_img),
                                    #                                   self.model_shape[1:], dtype=self.dtype)
                                    #     feed_dict[self.tf_input_standard] = img_standard
                                    # ----session run
                                    feed_dict = {self.tf_input: test_img, self.tf_keep_prob: 1.0}
                                    #feed_dict[self.tf_input] = test_img
                                    img_sess_out = self.sess.run(self.recon, feed_dict=feed_dict)
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
                                    img_diff = self.__img_patch_diff_method(filename, img_sess_out, self.sess, diff_th=30, cc_th=15)
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

                if self.break_flag:
                    self.save_ckpt_and_pb(epoch)
                    break

                #----SEG part
                if self.to_train_seg is True:
                    if to_fix_seg is False:
                        # say_sth("SEG opti ing", print_out=True)
                        # ----optimizations(SEG train set)
                        self.seg_opti_by_dataloader(dataloader_SEG_train)
                        if self.break_flag:
                            break

                        #----evaluation(SEG train set)
                        train_eval_dict = self.seg_eval_by_dataloader(dataloader_SEG_train, name='train')
                        train_eval_dict = dtype_transform(**train_eval_dict)

                        # ----data update
                        train_SEG_Data.update(**train_eval_dict)
                        self.log.update(**train_SEG_Data.content)

                        val_eval_dict = self.seg_eval_by_dataloader(dataloader_SEG_val, name='val')
                        val_eval_dict = dtype_transform(**val_eval_dict)

                        # ----data update
                        train_SEG_Data.update(**val_eval_dict)
                        self.log.update(**train_SEG_Data.content)

                        if self.break_flag is False:
                            #----find the best performance(SEG)
                            new_value = self.get_value_from_performance(seg_var.get('target_of_best'))
                            self.performance_process_SEG(epoch, new_value, LV)

                            #----display training results

                            msg = "\n----訓練週期 {} 與相關結果如下----".format(epoch)
                            say_sth(msg, print_out=self.print_out)
                            display_results(class_names, self.print_out, **train_eval_dict)
                            display_results(class_names, self.print_out, **val_eval_dict)
                            # display_results(None,self.print_out,
                            #                      train_SEG_loss=train_loss_seg,
                            #                      val_SEG_loss=test_loss_seg,
                            #                      )
                            #
                            # display_results(class_names, self.print_out,
                            #                      train_SEG_iou=train_iou_seg,
                            #                      val_SEG_iou=test_iou_seg,
                            #                      train_SEG_accuracy=train_acc_seg,
                            #                      val_SEG_accuracy=test_acc_seg,
                            #                      train_SEG_defect_recall=train_defect_recall,
                            #                      val_SEG_defect_recall=test_defect_recall,
                            #                      train_SEG_defect_sensitivity=train_defect_sensitivity,
                            #                      val_SEG_defect_sensitivity=test_defect_sensitivity
                            #                      )

                            # ----send protocol data(for UI to draw)
                            transmit_data2ui(epoch, self.print_out, **train_eval_dict)
                            transmit_data2ui(epoch, self.print_out, **val_eval_dict)
                            # transmit_data2ui(epoch, self.print_out,
                            #                       train_loss=train_loss_seg,
                            #                       train_SEG_iou=train_iou_seg,
                            #                       train_SEG_accuracy=train_acc_seg,
                            #                       train_SEG_defect_recall=train_defect_recall,
                            #                       train_SEG_defect_sensitivity=train_defect_sensitivity,
                            #                       val_loss=test_loss_seg,
                            #                       val_SEG_iou=test_iou_seg,
                            #                       val_SEG_accuracy=test_acc_seg,
                            #                       val_SEG_defect_recall=test_defect_recall,
                            #                       val_SEG_defect_sensitivity=test_defect_sensitivity
                            #                       )

                            # ----prediction for selected images
                            # if self.seg_predict_qty > 0:
                            #     if (epoch + 1) % eval_epochs == 0:
                            #         for batch_paths, batch_data, batch_label in dataloader_SEG_predict:
                            #             predict_labels = self.sess.run(self.tf_prediction_Seg,
                            #                                       feed_dict={self.tf_input: batch_data,
                            #                                                  self.tf_input_recon: batch_data})
                            #             batch_data *= 255
                            #             for i in range(len(predict_labels)):
                            #                 img_ori = cv2.convertScaleAbs(batch_data[i])
                            #
                            #                 # ----image with prediction
                            #                 img_predict = dataloader_SEG_predict.combine_img_label(
                            #                     img_ori,
                            #                     predict_labels[i],
                            #                 )
                            #
                            #                 show_imgs = [img_predict]
                            #
                            #                 # ----image with label
                            #                 img_answer = dataloader_SEG_predict.combine_img_label(
                            #                     img_ori,
                            #                     batch_label[i],
                            #                 )
                            #
                            #                 show_imgs.append(img_answer)
                            #
                            #                 qty_show = len(show_imgs)
                            #                 plt.figure(num=1, figsize=(5 * qty_show, 5 * qty_show), clear=True)
                            #
                            #                 for k, show_img in enumerate(show_imgs):
                            #                     plt.subplot(1, qty_show, k + 1)
                            #                     plt.imshow(show_img)
                            #                     plt.axis('off')
                            #                     plt.title(titles[i])
                            #                 save_path = os.path.join(self.new_predict_dir, batch_paths[i].split("\\")[-1])
                            #                 plt.savefig(save_path)

                if self.break_flag:
                    self.save_ckpt_and_pb(epoch)
                    break

                #----save ckpt, pb files
                if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
                    self.save_ckpt_and_pb(epoch)

                #----record the end time
                time_dict = self.record_time()
                self.log.update(**time_dict)

                #----save the log
                self.log.save()


        #----error messages
        if True in list(self.error_dict.values()):
            for key, value in self.error_dict.items():
                if value is True:
                    say_sth('', print_out=self.print_out, header=key)

        #----close the session
        self.sess.close()

    #----functions
    def encode_header_process(self,encode_header,encode_num):
        if isinstance(encode_header,list):
            self.encode_header = encode_header
        else:
            self.encode_header = [24,97,28,98]

        if isinstance(encode_num,int):
            self.encode_num = encode_num
        else:
            self.encode_num = 87

    def update_config_dict(self,new_dict, ori_dict):

        name_list = ['ae_var', 'seg_var']
        for name in name_list:
            new_dict[name] = update_dict(new_dict[name], ori_dict[name])
        combine_dict = update_dict(new_dict, ori_dict)
        # ----pipeline modification
        p = AE_Seg_pipeline_modification()
        p(config_dict=combine_dict)

        return combine_dict

    def save_ckpt_and_pb(self,epoch):
        model_save_path = self.saver.save(self.sess, self.out_dir_prefix, global_step=epoch)

        # ----encode ckpt file
        if self.encript_flag is True:
            encode_CKPT(model_save_path, encode_num=self.encode_num, encode_header=self.encode_header)

        # ----save pb(normal)
        save_pb_file(self.sess, self.pb_save_list, self.pb_save_path, encode=self.encript_flag,
                     random_num_range=self.encode_num, header=self.encode_header)

    def seg_opti_by_dataloader(self,dataloader,keep_prob=0.5,dropout=0.3):
        if self.break_flag is False:
            dataloader.reset()
            for batch_paths, batch_data, batch_label in dataloader:

                feed_dict = {self.tf_input: batch_data, self.tf_input_recon: batch_data,
                             self.tf_label_batch: batch_label,
                             self.tf_keep_prob: keep_prob,
                             self.tf_dropout: dropout}
                try:
                    self.sess.run(self.opt_Seg, feed_dict=feed_dict)

                    #----check break signal from TCPIP
                    self.break_flag = break_signal_check()
                    if self.break_flag:
                        # self.save_ckpt_and_pb(epoch)
                        break
                except Exception as err:
                    self.error_dict['GPU_resource_error'] = True
                    # msg = "Error:SEG權重最佳化時產生錯誤"
                    say_sth(f"Error:{err}", print_out=True)
                    self.break_flag = True
                    break

    def seg_eval_by_dataloader(self, dataloader, name='train'):
        loss = None
        iou = None
        acc = None
        defect_recall = None
        defect_sensitivity = None
        if self.break_flag is False:
            self.seg_p.reset_arg()
            self.seg_p.reset_defect_stat()
            loss = 0
            dataloader.reset()
            for batch_paths, batch_data, batch_label in dataloader:

                feed_dict = {self.tf_input: batch_data,
                             self.tf_label_batch: batch_label,
                             self.tf_keep_prob: 1.0,
                             self.tf_dropout: 0.0}
                loss_temp = self.sess.run(self.loss_Seg, feed_dict=feed_dict)
                predict_label = self.sess.run(self.prediction_Seg, feed_dict=feed_dict)

                # ----calculate the loss and accuracy
                loss += loss_temp
                self.seg_p.cal_intersection_union(predict_label, batch_label)
                _ = self.seg_p.cal_label_defect_by_acc_v2(predict_label, batch_label)
                _ = self.seg_p.cal_predict_defect_by_acc_v2(predict_label, batch_label)

                # ----check the break signal
                self.break_flag = break_signal_check()
                if self.break_flag:
                    break

            iou, acc, all_acc_seg = self.seg_p.cal_iou_acc()
            defect_recall = self.seg_p.cal_defect_recall()
            defect_sensitivity = self.seg_p.cal_defect_sensitivity()

            if self.break_flag:
                loss /= (dataloader.ite_num + 1)
            else:
                loss /= dataloader.iterations

        return {
            f"{name}_SEG_loss": loss,
            f"{name}_SEG_iou": iou,
            f"{name}_SEG_acc": acc,
            f"{name}_SEG_defect_recall": defect_recall,
            f"{name}_SEG_defect_sensitivity": defect_sensitivity,
        }

    def ae_opti_by_dataloader(self, dataloader,keep_prob=0.5):
        dataloader.reset()
        if self.break_flag is False:
            for batch_paths, batch_data in dataloader:
                feed_dict = {self.tf_input: batch_data, self.tf_keep_prob: keep_prob}
                # ----optimization
                try:
                    self.sess.run(self.opt_AE, feed_dict=feed_dict)

                    #----check break signal from TCPIP
                    self.break_flag = break_signal_check()
                    if self.break_flag:
                        # self.save_ckpt_and_pb(epoch)
                        break
                except Exception as err:
                    self.error_dict['GPU_resource_error'] = True
                    # msg = "Error:權重最佳化時產生錯誤"
                    say_sth(f"Error:{err}", print_out=True)
                    self.break_flag = True
                    break

    def ae_eval_by_dataloader(self, dataloader,name='train'):
        loss = 0

        if self.break_flag is False:
            dataloader.reset()
            for batch_paths, batch_data in dataloader:
                feed_dict = {self.tf_input: batch_data, self.tf_keep_prob: 1.0}
                loss_temp = self.sess.run(self.loss_AE, feed_dict=feed_dict)
                loss += loss_temp

                #----check the break signal
                self.break_flag = break_signal_check()
                if self.break_flag:
                    break

            if self.break_flag:
                loss /= (dataloader.ite_num + 1)
            else:
                loss /= dataloader.iterations



        return {f"{name}_AE_loss":loss}

    def record_time(self,):
        now_time = time.time()
        d_t = now_time - self.epoch_start_time
        total_train_time = now_time - self.t_train_start

        self.epoch_time_list.append(d_t)

        msg_list = []
        msg_list.append("此次訓練所花時間: {}秒".format(np.round(d_t, 2)))
        msg_list.append("累積訓練時間: {}秒".format(np.round(total_train_time, 2)))
        say_sth(msg_list, print_out=self.print_out, header='msg')

        return dict(total_train_time=float(total_train_time),
                    ave_epoch_time=float(np.average(self.epoch_time_list)))

    def set_time_start(self):
        self.epoch_time_list = list()
        self.t_train_start = time.time()

    def set_epoch_start_time(self):
        self.epoch_start_time = time.time()

    def set_target_AE(self,**kwargs):
        methods = {'value':float,"hit_target_times":int,'hit_times':int}
        name_list = list(methods.keys())
        AE_target_dict = dict(
            type='loss',
            value=99.7,
            hit_target_times=2,
            hit_times=0
        )

        for key,value in kwargs.items():
            if key in name_list:
                value = methods[key](value)

            AE_target_dict[key] = value

        self.AE_target_dict = AE_target_dict

    def target_comparison_AE(self,loss_value,loss_method='ssim'):
        goal = False
        if self.break_flag is False:
            target = self.AE_target_dict['value']
            set_t = self.AE_target_dict['hit_target_times']
            t = self.AE_target_dict['hit_times']

            if loss_method == 'ssim':
                if loss_value * 100 >= target:
                    t += 1
            else:
                if loss_value <= target:
                    t += 1

            if t >= set_t:
                goal = True
            else:
                self.AE_target_dict['hit_times'] = t

            if goal:
                msg = '模型訓練結束:\n{}已經達到設定目標:{}累積達{}次'.format(
                     self.AE_target_dict['type'],target, set_t)
                say_sth(msg, print_out=self.print_out)

        return goal

    def seg_path_change_process(self,to_train_w_AE_paths,to_train_ae,select_OK_ratio=0.2):
        if to_train_w_AE_paths:
            if to_train_ae:
                select_num = np.minimum(self.train_path_qty, int(self.seg_train_qty * select_OK_ratio))
                temp_paths = np.random.choice(self.train_paths, size=select_num, replace=False)
                self.seg_train_paths = list(self.seg_train_paths)
                self.seg_train_paths.extend(temp_paths)
                self.seg_train_paths = np.array(self.seg_train_paths)
                self.seg_train_qty += select_num
                msg = f"SEG額外加入AE的OK圖片進行訓練，SEG訓練集圖片集數量增加後為{self.seg_train_qty}"
                say_sth(msg,print_out=self.print_out)

    def path_qty_process(self,imgDirPathDict):
        msg_list = []
        status_list = []
        for img_dir_name,qty in imgDirPathDict.items():
            if qty == 0:
                msg = f"{img_dir_name}沒有圖片"
                stat = False
            else:
                msg = "{}圖片數量:{}".format(img_dir_name,qty)
                stat = True
            msg_list.append(msg)
            status_list.append(stat)
        say_sth(msg_list,print_out=self.print_out)

        return status_list

    def read_manual_cmd(self,to_read_manual_cmd):
        break_flag = False
        if to_read_manual_cmd:
            j_path = os.path.join(self.save_dir, 'manual_cmd.json')
            if os.path.exists(j_path):
                with open(j_path, 'r') as f:
                    cmd_dict = json.load(f)
                if cmd_dict.get('to_stop_training') is True:
                    break_flag = True
                    msg = "接收到manual cmd: stop the training!"
                    say_sth(msg, print_out=True)

        return break_flag

    def performance_process_AE(self,epoch,loss_value,LV):
        if self.break_flag is False:
            if self.loss_method == 'ssim':
                new_value = np.round(loss_value * 100, 2)
            else:
                new_value = np.round(loss_value, 2)

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

                    # ----save the better one
                    pb_save_path = create_pb_path("infer_{}".format(new_value), self.save_dir,
                                                  to_encode=self.encript_flag)
                    save_pb_file(self.sess, self.pb_save_list, pb_save_path, encode=self.encript_flag,
                                 random_num_range=self.encode_num, header=self.encode_header)
                    # ----update record value
                    LV['record_value'] = new_value
                    LV['pb_save_path_old'] = pb_save_path

    def performance_process_SEG(self, epoch, new_value, LV):
        if epoch == 0:
            LV['record_value_seg'] = new_value
        else:
            if new_value > LV['record_value_seg']:
                # ----delete the previous pb
                if os.path.exists(LV['pb_seg_save_path_old']):
                    os.remove(LV['pb_seg_save_path_old'])

                # ----save the better one
                pb_save_path = create_pb_path(f"infer_best_epoch{epoch}", self.save_dir,
                                              to_encode=self.encript_flag)
                save_pb_file(self.sess, self.pb_save_list, pb_save_path, encode=self.encript_flag,
                             random_num_range=self.encode_num, header=self.encode_header)
                # ----update record value
                LV['record_value_seg'] = new_value
                LV['pb_seg_save_path_old'] = pb_save_path

    def get_value_from_performance(self,target_of_best="defect_recall"):
        if target_of_best == 'defect_recall':
            new_value = self.seg_p.sum_defect_recall()
        elif target_of_best == 'defect_sensitivity':
            new_value = self.seg_p.sum_defect_sensitivity()
        elif target_of_best == 'recall+sensitivity':
            new_value = self.seg_p.sum_defect_recall() + self.seg_p.sum_defect_sensitivity()
        else:
            new_value = self.seg_p.sum_iou_acc()

        return new_value

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

class AE_Seg_pipeline_modification():
    # def __init__(self):
    #     print("Pipeline_modification init")

    def __call__(self, *args, **kwargs):
        # model_type = kwargs.get('model_type')
        config_dict = kwargs.get('config_dict')

        #if model_type == "5":
        self.__modify_type_5__(config_dict)
        say_sth("execute type_5 Pipeline_modification",print_out=True)

    def __modify_type_5__(self,config_dict):
        dict_list = ['ae_var','seg_var']
        key_list = ["train_pipelines","val_pipelines"]
        model_shape = config_dict.get('model_shape')

        h,w = model_shape[1:3]

        for dict_name in dict_list:
            var_dict = config_dict.get(dict_name)
            if isinstance(var_dict,dict):
                for key in key_list:
                    if var_dict.get(key) is not None:
                        self.height_width_check(var_dict[key],h,w)

    def height_width_check(self,pipelines,height,width):
        '''
        dict(type='Resize', height=int(height*1.05), width=int(width*1.05)),
        dict(type='RandomCrop', height=height, width=width),
        '''
        #----get all types
        type_list = []
        for p_dict in pipelines:
            if p_dict.get('type') is not None:
                type_list.append(p_dict['type'])

        #----modify height and width
        for p_dict in pipelines:
            if p_dict.get('type') == 'Resize':
                if "RandomCrop" in type_list:
                    p_dict['height'] = int(height * 1.05)
                    p_dict['width'] = int(width * 1.05)
                else:
                    p_dict['height'] = height
                    p_dict['width'] = width
            elif p_dict.get('type') == 'RandomCrop':
                p_dict['height'] = height
                p_dict['width'] = width

# class AE_Seg():
#     def __init__(self,para_dict):
#
#         #----use previous settings
#         if para_dict.get('use_previous_settings'):
#             dict_t = AE_Seg_Util.get_latest_json_content(para_dict['save_dir'])
#             self.para_dict = None
#             if isinstance(dict_t,dict):
#                 msg_list = list()
#                 msg_list.append("Use previous settings")
#                 #----exclude dict process
#                 update_dict = para_dict.get('update_dict')
#                 if isinstance(update_dict,dict):
#                     for key, value in update_dict.items():
#                         if value is None:
#                             dict_t[key] = para_dict[key]
#                             msg = '{} update to {}'.format(key,para_dict[key])
#                             msg_list.append(msg)
#                         else:
#                             dict_t[key][value] = para_dict[key][value]
#                             msg = '{}_{} update to {}'.format(key,value,para_dict[key][value])
#                             msg_list.append(msg)
#
#                 self.para_dict = dict_t
#
#                 for msg in msg_list:
#                     say_sth(msg, print_out=True)
#
#         #----common var
#         print_out = para_dict.get('print_out')
#         self.print_out = print_out
#
#         # ----local var
#
#         # ----tools class init
#         tl = AE_Seg_Util.tools(print_out=print_out)
#
#         #----AE process
#         to_train_ae = False
#         recon_flag = False
#         ae_var = para_dict.get('ae_var')
#         if isinstance(ae_var,dict):
#             train_img_dir = ae_var.get('train_img_dir')
#             test_img_dir = ae_var.get('test_img_dir')
#             recon_img_dir = ae_var.get('recon_img_dir')
#             special_img_dir = ae_var.get('special_img_dir')
#
#             # ----AE image path process
#             self.train_paths, self.train_path_qty = get_paths(train_img_dir)
#             self.test_paths, self.test_path_qty = get_paths(test_img_dir)
#             self.sp_paths, self.sp_path_qty = get_paths(special_img_dir)
#             self.recon_paths, self.recon_path_qty = get_paths(recon_img_dir)
#             qty_status_list = self.path_qty_process(
#                 dict(AE訓練集=self.train_path_qty,
#                      AE驗證集=self.test_path_qty,
#                      AE加強學習集=self.sp_path_qty,
#                      AE重建圖集=self.recon_path_qty
#                      ))
#             to_train_ae, _, _, recon_flag = qty_status_list
#
#         #----SEG process
#         to_train_seg = False
#         seg_var = para_dict.get('seg_var')
#         if isinstance(seg_var,dict) is True:
#             train_img_seg_dir = seg_var.get('train_img_seg_dir')
#             test_img_seg_dir = seg_var.get('test_img_seg_dir')
#             predict_img_dir = seg_var.get('predict_img_dir')
#             to_train_w_AE_paths = seg_var.get('to_train_w_AE_paths')
#             id2class_name = seg_var.get('id2class_name')
#             select_OK_ratio = 0.2
#
#
#             #----SEG train image path process
#             self.seg_train_paths, self.seg_train_qty = get_paths(train_img_seg_dir)
#             self.seg_test_paths, self.seg_test_qty = get_paths(test_img_seg_dir)
#             self.seg_predict_paths, self.seg_predict_qty = get_paths(predict_img_dir)
#
#             qty_status_list = self.path_qty_process(
#                 dict(SEG訓練集=self.seg_train_qty,
#                      SEG驗證集=self.seg_test_qty,
#                      SEG預測集=self.seg_predict_qty
#                      ))
#             to_train_seg = qty_status_list[0]
#
#             #----read class names
#             classname_id_color_dict = AE_Seg_Util.get_classname_id_color_v2(id2class_name,print_out=print_out)
#
#             # ----train with AE ok images
#             self.seg_path_change_process(to_train_w_AE_paths, to_train_ae, select_OK_ratio=select_OK_ratio)
#
#         #----log update
#         log = classname_id_color_dict.copy()
#
#         #----local var to global
#         self.to_train_ae = to_train_ae
#         self.to_train_seg = to_train_seg
#         self.status = bool(to_train_seg+to_train_ae)
#         self.log = log
#         if to_train_ae:
#             self.train_img_dir = train_img_dir
#             self.test_img_dir = test_img_dir
#             self.special_img_dir = special_img_dir
#             self.recon_img_dir = recon_img_dir
#             self.recon_flag = recon_flag
#         if to_train_seg:
#             self.train_img_seg_dir = train_img_seg_dir
#             self.test_img_seg_dir = test_img_seg_dir
#             self.predict_img_dir = predict_img_dir
#             self.classname_id_color_dict = classname_id_color_dict
#
#
#     def model_init(self,para_dict):
#         #----var parsing
#
#         # ----use previous settings
#         if para_dict.get('use_previous_settings'):
#             if isinstance(self.para_dict,dict):
#                 para_dict = self.para_dict
#
#         #----common var
#         model_shape = para_dict.get('model_shape')
#         preprocess_dict = para_dict.get('preprocess_dict')
#         lr = para_dict['learning_rate']
#         save_dir = para_dict['save_dir']
#         save_pb_name = para_dict.get('save_pb_name')
#         encript_flag = para_dict.get('encript_flag')
#         print_out = para_dict.get('print_out')
#         add_name_tail = para_dict.get('add_name_tail')
#         dtype = para_dict.get('dtype')
#
#
#         #----var
#         acti_dict = {'relu': tf.nn.relu, 'mish': models_AE_Seg.tf_mish, None: tf.nn.relu}
#         pb_save_list = list()
#
#         #----var process
#
#         if add_name_tail is None:
#             add_name_tail = True
#         if dtype is None:
#             dtype = 'float32'
#
#
#         # ----tf_placeholder declaration
#         tf_input = tf.placeholder(dtype, shape=model_shape, name='input')
#         tf_keep_prob = tf.placeholder(dtype=dtype, name="keep_prob")
#
#
#         if self.to_train_ae:
#             ae_var = para_dict['ae_var']
#             infer_method = ae_var.get('infer_method')
#             model_name = ae_var.get('model_name')
#             acti = ae_var['activation']
#             pool_kernel = ae_var['pool_kernel']
#             kernel_list = ae_var['kernel_list']
#             filter_list = ae_var['filter_list']
#             conv_time = ae_var['conv_time']
#             pool_type = ae_var.get('pool_type')
#             loss_method = ae_var['loss_method']
#             opti_method = ae_var['opti_method']
#             embed_length = ae_var['embed_length']
#             stride_list = ae_var.get('stride_list')
#             scaler = ae_var.get('scaler')
#             # process_dict = ae_var['process_dict']
#
#             rot = ae_var.get('rot')
#             special_process_list = ['rdm_patch', 'rdm_perlin', 'rdm_light_defect']
#             return_ori_data = False
#             #----if return ori data or not
#
#             # for name in special_process_list:
#             #     if process_dict.get(name) is True:
#             #         return_ori_data = True
#             #         break
#             #----random patch
#             # rdm_patch = False
#             # if process_dict.get('rdm_patch') is True:
#             #     rdm_patch = True
#
#             #----filer scaling process
#             if scaler is not None:
#                 filter_list = (np.array(filter_list) / scaler).astype(np.uint16)
#
#             if return_ori_data is True:
#                 self.tf_input_ori = tf.placeholder(dtype, shape=model_shape, name='input_ori')
#
#             #tf_label_batch = tf.placeholder(dtype=tf.int32, shape=[None], name="label_batch")
#             #tf_phase_train = tf.placeholder(tf.bool, name="phase_train")
#
#             # ----activation selection
#             acti_func = acti_dict[acti]
#
#
#             avepool_out = self.__avepool(tf_input, k_size=5, strides=1)
#             #----preprocess
#             if preprocess_dict is None:
#                 tf_input_process = tf.identity(tf_input,name='preprocess')
#                 if return_ori_data is True:
#                     tf_input_ori_no_patch = tf.identity(self.tf_input_ori, name='tf_input_ori_no_patch')
#             else:
#                 tf_temp = models_AE_Seg.preprocess(tf_input, preprocess_dict, print_out=print_out)
#                 tf_input_process = tf.identity(tf_temp,name='preprocess')
#
#                 if return_ori_data is True:
#                     tf_temp_2 = models_AE_Seg.preprocess(self.tf_input_ori, preprocess_dict, print_out=print_out)
#                     tf_input_ori_no_patch = tf.identity(tf_temp_2, name='tf_input_ori_no_patch')
#
#             #----AIE model mapping
#             if model_name is not None:
#                 if model_name.find("type_5") >= 0:
#                     infer_method = "AE_pooling_net_V" + model_name.split("_")[-1]
#
#             #----AE inference selection
#             if infer_method == "AE_transpose_4layer":
#                 recon = models_AE_Seg.AE_transpose_4layer(tf_input, kernel_list, filter_list,activation=acti_func,
#                                                   pool_kernel=pool_kernel,pool_type=pool_type)
#                 recon = tf.identity(recon, name='output_AE')
#                 #(tf_input, kernel_list, filter_list,pool_kernel=2,activation=tf.nn.relu,pool_type=None)
#                 # recon = AE_refinement(temp,96)
#             elif infer_method.find('mit') >= 0:
#                 cfg = config_mit.config[infer_method.split("_")[-1]]
#                 model = MixVisionTransformer(
#                     embed_dims=cfg['embed_dims'],
#                     num_stages=cfg['num_stages'],
#                     num_layers=cfg['num_layers'],
#                     num_heads=cfg['num_heads'],
#                     patch_sizes=cfg['patch_sizes'],
#                     strides=cfg['strides'],
#                     sr_ratios=cfg['sr_ratios'],
#                     mlp_ratio=cfg['mlp_ratio'],
#                     ffn_dropout_keep_ratio=1.0,
#                     dropout_keep_rate=1.0)
#                 # model.init_weights()
#                 outs = model(tf_input_process)
#                 # print("outs shape:",outs.shape)
#                 mitDec = MiTDecoder(channels=cfg['num_heads'][-1] * cfg['embed_dims'],
#                                     dropout_ratio=0, num_classes=3)
#                 logits = mitDec(outs)
#                 recon = tf.identity(logits, name='output_AE')
#             elif infer_method == "AE_pooling_net_V3":
#                 recon = models_AE_Seg.AE_pooling_net_V3(tf_input_process, kernel_list, filter_list, activation=acti_func,
#                                           pool_kernel_list=pool_kernel, pool_type_list=pool_type,
#                                           stride_list=stride_list, print_out=print_out)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_pooling_net_V4":
#                 recon = models_AE_Seg.AE_pooling_net_V4(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
#                                           to_reduce=ae_var.get('to_reduce'),print_out=print_out)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_pooling_net_V5":
#                 recon = models_AE_Seg.AE_pooling_net_V5(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
#                                           to_reduce=ae_var.get('to_reduce'),print_out=print_out)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_pooling_net_V6":
#                 recon = models_AE_Seg.AE_pooling_net_V6(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
#                                           to_reduce=ae_var.get('to_reduce'),print_out=print_out)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_pooling_net_V7":
#
#                 recon = models_AE_Seg.AE_pooling_net_V7(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
#                                          print_out=print_out)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_pooling_net_V8":
#
#                 self.tf_input_standard = tf.placeholder(dtype, shape=model_shape, name='input_standard')
#                 recon = models_AE_Seg.AE_pooling_net_V8(tf_input_process,self.tf_input_standard,ae_var['encode_dict'],ae_var['decode_dict'],
#                                          print_out=print_out)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_dense_sampling":
#                 sampling_factor = 16
#                 filters = 2
#                 recon = models_AE_Seg.AE_dense_sampling(tf_input_process,sampling_factor,filters)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_pooling_net":
#
#                 recon = models_AE_Seg.AE_pooling_net(tf_input_process, kernel_list, filter_list,activation=acti_func,
#                                       pool_kernel_list=pool_kernel,pool_type_list=pool_type,
#                                       stride_list=stride_list,rot=rot,print_out=print_out)
#                 recon = tf.identity(recon,name='output_AE')
#             elif infer_method == "AE_Seg_pooling_net":
#                 AE_out,Seg_out = models_AE_Seg.AE_Seg_pooling_net(tf_input, kernel_list, filter_list,activation=acti_func,
#                                                   pool_kernel_list=pool_kernel,pool_type_list=pool_type,
#                                        rot=rot,print_out=print_out,preprocess_dict=preprocess_dict,
#                                            class_num=self.class_num)
#                 recon = tf.identity(AE_out,name='output_AE')
#             elif infer_method == "AE_Unet":
#                 recon = models_AE_Seg.AE_Unet(tf_input, kernel_list, filter_list,activation=acti_func,
#                                                   pool_kernel_list=pool_kernel,pool_type_list=pool_type,
#                                        rot=rot,print_out=print_out,preprocess_dict=preprocess_dict)
#                 recon = tf.identity(recon,name='output_AE')
#             elif infer_method == "AE_JNet":
#                 recon = models_AE_Seg.AE_JNet(tf_input, kernel_list, filter_list,activation=acti_func,
#                                                 rot=rot,pool_kernel=pool_kernel,pool_type=pool_type)
#             # elif infer_method == "test":
#             #     recon = self.__AE_transpose_4layer_test(tf_input, kernel_list, filter_list,
#             #                                        conv_time=conv_time,maxpool_kernel=maxpool_kernel)
#             # elif infer_method == 'inception_resnet_v1_reduction':
#             #     recon = AE_incep_resnet_v1(tf_input=tf_input,tf_keep_prob=tf_keep_prob,embed_length=embed_length,
#             #                                scaler=scaler,kernel_list=kernel_list,filter_list=filter_list,
#             #                                activation=acti_func,)
#             elif infer_method == "Resnet_Rot":
#                 filter_list = [12, 16, 24, 36, 48, 196]
#                 recon = models_AE_Seg.AE_Resnet_Rot(tf_input,filter_list,tf_keep_prob,embed_length,activation=acti_func,
#                                       print_out=True,rot=True)
#             else:
#                 if model_name is not None:
#                     display_name = model_name
#                 else:
#                     display_name = infer_method
#                 say_sth(f"Error:AE model doesn't exist-->{display_name}", print_out=True)
#
#             # ----AE loss method selection
#             if loss_method == 'mse':
#                 loss_AE = tf.reduce_mean(tf.pow(recon - tf_input, 2), name="loss_AE")
#             elif loss_method == 'ssim':
#                 # self.loss_AE = tf.reduce_mean(tf.image.ssim_multiscale(tf.image.rgb_to_grayscale(self.tf_input),tf.image.rgb_to_grayscale(self.recon),2),name='loss')
#
#                 if return_ori_data is True:
#                     loss_AE = tf.reduce_mean(tf.image.ssim(tf_input_ori_no_patch, recon, 2.0), name='loss_AE')
#                 else:
#                     loss_AE = tf.reduce_mean(tf.image.ssim(tf_input_process, recon, 2.0), name='loss_AE')
#             elif loss_method == "huber":
#                 loss_AE = tf.reduce_sum(tf.losses.huber_loss(tf_input, recon, delta=1.35), name='loss_AE')
#             elif loss_method == 'ssim+mse':
#                 loss_1 = tf.reduce_mean(tf.pow(recon - tf_input, 2))
#                 loss_2 = tf.reduce_mean(tf.image.ssim(tf_input, recon, 2.0))
#                 loss_AE = tf.subtract(loss_2, loss_1, name='loss_AE')
#             elif loss_method == 'cross_entropy':
#                 loss_AE = tf.reduce_mean(
#                     tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.layers.flatten(tf_input),
#                                                                logits=tf.layers.flatten(recon)), name="loss_AE")
#             elif loss_method == 'kl_d':
#                 epsilon = 1e-8
#                 # generation loss(cross entropy)
#                 loss_AE = tf.reduce_mean(
#                     tf_input * tf.subtract(tf.log(epsilon + tf_input), tf.log(epsilon + recon)), name='loss_AE')
#
#             # ----AE optimizer selection
#             if opti_method == "adam":
#                 if loss_method.find('ssim') >= 0:
#                     opt_AE = tf.train.AdamOptimizer(learning_rate=lr).minimize(-loss_AE)
#                 else:
#                     opt_AE = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_AE)
#
#             if self.recon_flag is True:
#                 new_dir_name = "img_recon-" + os.path.basename(save_dir)
#                 # ----if recon_img_dir is list format
#                 if isinstance(self.recon_img_dir, list):
#                     dir_path = self.recon_img_dir[0]
#                 else:
#                     dir_path = self.recon_img_dir
#                 self.new_recon_dir = os.path.join(dir_path, new_dir_name)
#                 if not os.path.exists(self.new_recon_dir):
#                     os.makedirs(self.new_recon_dir)
#
#             # ----appoint PB node names
#             pb_save_list.extend(['output_AE', "loss_AE"])
#
#         #----Seg inference selection
#         if self.to_train_seg:
#             seg_var = para_dict['seg_var']
#             infer_method4Seg = seg_var.get('infer_method')
#             model_name4seg = seg_var.get('model_name')
#             pool_kernel4Seg = seg_var['pool_kernel']
#             pool_type4Seg = seg_var.get('pool_type')
#             kernel_list4Seg = seg_var['kernel_list']
#             filter_list4Seg = seg_var['filter_list']
#             loss_method4Seg = seg_var.get('loss_method')
#             opti_method4Seg = seg_var.get('opti_method')
#             preprocess_dict4Seg = seg_var.get('preprocess_dict')
#             rot4Seg = seg_var.get('rot')
#             acti_seg = seg_var.get('activation')
#             class_num = len(self.classname_id_color_dict['class_names'])
#
#             tf_input_recon = tf.placeholder(dtype, shape=model_shape, name='input_recon')
#             tf_label_batch = tf.placeholder(tf.int32, shape=model_shape[:-1], name='label_batch')
#             tf_dropout = tf.placeholder(dtype=tf.float32, name="dropout")
#
#             # ----activation selection
#             acti_func = acti_dict[acti_seg]
#
#             #----filer scaling process
#             filter_list4Seg = np.array(filter_list4Seg)
#             if seg_var.get('scaler') is not None:
#                 filter_list4Seg /= seg_var.get('scaler')
#                 filter_list4Seg = filter_list4Seg.astype(np.uint16)
#
#             #----AIE model mapping
#             if model_name4seg is not None:
#                 if model_name4seg.find("type_5") >= 0:
#                     infer_method4Seg = "Seg_pooling_net_V" + model_name4seg.split("_")[-1]
#
#             #----Seg model selection
#             if infer_method4Seg == "Seg_DifNet":
#                 logits_Seg = models_AE_Seg.Seg_DifNet(tf_input_process,tf_input_recon, kernel_list4Seg, filter_list4Seg,activation=acti_func,
#                                    pool_kernel_list=pool_kernel4Seg,pool_type_list=pool_type4Seg,
#                                    rot=rot4Seg,print_out=print_out,preprocess_dict=preprocess_dict4Seg,class_num=self.class_num)
#                 softmax_Seg = tf.nn.softmax(logits_Seg,name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'Seg_DifNet_V2':
#                 logits_Seg = models_AE_Seg.Seg_DifNet_V2(tf_input_process,tf_input_recon,seg_var['encode_dict'],seg_var['decode_dict'],
#                               class_num=self.class_num,print_out=print_out)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg.find('mit') >= 0:
#                 cfg = config_mit.config[infer_method4Seg.split("_")[-1]]
#                 model = MixVisionTransformer(
#                     embed_dims=cfg['embed_dims'],
#                     num_stages=cfg['num_stages'],
#                     num_layers=cfg['num_layers'],
#                     num_heads=cfg['num_heads'],
#                     patch_sizes=cfg['patch_sizes'],
#                     strides=cfg['strides'],
#                     sr_ratios=cfg['sr_ratios'],
#                     mlp_ratio=cfg['mlp_ratio'],
#                     drop_rate=0,
#                     attn_drop_rate=0)
#                 # model.init_weights()
#                 outs = model(tf_input_process)
#                 # print("outs shape:",outs.shape)
#                 mitDec = MiTDecoder(channels=cfg['num_heads'][-1] * cfg['embed_dims'],
#                                     dropout_ratio=0, num_classes=self.class_num)
#                 logits_Seg = mitDec(outs)
#                 logits_Seg = tf.image.resize(logits_Seg,model_shape[1:-1])
#                 print("logits_Seg shape:", logits_Seg.shape)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'Seg_pooling_net_V4':
#                 logits_Seg = models_AE_Seg.Seg_pooling_net_V4(tf_input_process,tf_input_recon, seg_var['encode_dict'], seg_var['decode_dict'],
#                                           to_reduce=seg_var.get('to_reduce'),out_channel=self.class_num,
#                                           print_out=print_out)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'Seg_pooling_net_V7':
#                 logits_Seg = models_AE_Seg.Seg_pooling_net_V7(tf_input_process,tf_input_recon, seg_var['encode_dict'], seg_var['decode_dict'],
#                                           out_channel=self.class_num,
#                                           print_out=print_out)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'Seg_pooling_net_V8':
#                 logits_Seg = models_AE_Seg.Seg_pooling_net_V8(tf_input_process, tf_input_recon, seg_var['encode_dict'],
#                                                 seg_var['decode_dict'],
#                                                 to_reduce=seg_var.get('to_reduce'), out_channel=self.class_num,
#                                                 print_out=print_out)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'Seg_pooling_net_V9':
#                 logits_Seg = models_AE_Seg.Seg_pooling_net_V9(tf_input_process, tf_input_recon, seg_var['encode_dict'],
#                                                 seg_var['decode_dict'],
#                                                 to_reduce=seg_var.get('to_reduce'), out_channel=self.class_num,
#                                                 print_out=print_out)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'Seg_pooling_net_V10':
#                 logits_Seg = models_AE_Seg.Seg_pooling_net_V10(tf_input_process, tf_input_recon, seg_var['encode_dict'],
#                                                 seg_var['decode_dict'],
#                                                 to_reduce=seg_var.get('to_reduce'), out_channel=class_num,
#                                                 print_out=print_out)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'AE_Seg_pooling_net':
#                 logits_Seg = Seg_out
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg,tf.uint8, name='predict_Seg')
#             else:
#                 if model_name4seg is not None:
#                     display_name = model_name4seg
#                 else:
#                     display_name = infer_method4Seg
#                 say_sth(f"Error:Seg model doesn't exist-->{display_name}", print_out=True)
#
#             #----Seg loss method selection
#             if loss_method4Seg == "cross_entropy":
#                 loss_Seg = tf.reduce_mean(v2.nn.sparse_softmax_cross_entropy_with_logits(tf_label_batch,logits_Seg),name='loss_Seg')
#
#             #----Seg optimizer selection
#             if opti_method4Seg == "adam":
#                 opt_Seg = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_Seg)
#
#
#             # ----appoint PB node names
#             pb_save_list.extend(['predict_Seg','dummy_out'])
#
#         # ----pb filename(common)
#         pb_save_path = create_pb_path(save_pb_name, save_dir,
#                                       to_encode=encript_flag,add_name_tail=add_name_tail)
#
#         # ----create the dir to save model weights(CKPT, PB)
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#
#         #----save SEG coloer index image
#         _ = AE_Seg_Util.draw_color_index(self.classname_id_color_dict['class_names'], save_dir=save_dir)
#
#         if self.to_train_seg is True and self.seg_predict_qty > 0:
#             new_dir_name = "img_seg-" + os.path.basename(save_dir)
#             #----if recon_img_dir is list format
#             if isinstance(self.predict_img_dir,list):
#                 dir_path = self.predict_img_dir[0]
#             else:
#                 dir_path = self.predict_img_dir
#             self.new_predict_dir = os.path.join(dir_path, new_dir_name)
#             if not os.path.exists(self.new_predict_dir):
#                 os.makedirs(self.new_predict_dir)
#
#         out_dir_prefix = os.path.join(save_dir, "model")
#         saver = tf.train.Saver(max_to_keep=2)
#
#         # ----create the log(JSON)
#         log_path = create_log_path(save_dir,to_encode=encript_flag,filename="train_result")
#         self.log['pb_save_list'] = pb_save_list
#         self.log = update_dict(self.log,para_dict)
#
#         # ----local var to global
#         self.model_shape = model_shape
#         self.tf_input = tf_input
#         self.tf_keep_prob = tf_keep_prob
#         self.saver = saver
#         self.save_dir = save_dir
#         self.pb_save_path = pb_save_path
#
#
#
#         self.pb_save_list = pb_save_list
#         # self.pb_extension = pb_extension
#         self.log_path = log_path
#         self.dtype = dtype
#         if self.to_train_ae:
#             self.avepool_out = avepool_out
#             self.recon = recon
#             self.loss_AE = loss_AE
#             self.opt_AE = opt_AE
#             self.out_dir_prefix = out_dir_prefix
#             self.loss_method = loss_method
#             self.return_ori_data = return_ori_data
#             self.infer_method = infer_method
#
#         if self.to_train_seg is True:
#             self.tf_label_batch = tf_label_batch
#             self.tf_input_recon = tf_input_recon
#             self.tf_prediction_Seg = prediction_Seg
#             self.infer_method4Seg = infer_method4Seg
#             self.logits_Seg = logits_Seg
#             self.loss_Seg = loss_Seg
#             self.opt_Seg = opt_Seg
#             self.prediction_Seg = prediction_Seg
#             self.loss_method4Seg = loss_method4Seg
#             # self.pb4seg_save_path = pb4seg_save_path
#             self.tf_dropout = tf_dropout
#             self.infer_method4Seg = infer_method4Seg
#
#     def train(self,para_dict):
#         # ----use previous settings
#         if para_dict.get('use_previous_settings'):
#             if isinstance(self.para_dict, dict):
#                 para_dict = self.para_dict
#         # ----var parsing
#         epochs = para_dict['epochs']
#         GPU_ratio = para_dict.get('GPU_ratio')
#         print_out = para_dict.get('print_out')
#         encode_header = para_dict.get('encode_header')
#         encode_num = para_dict.get('encode_num')
#         encript_flag = para_dict.get('encript_flag')
#         eval_epochs = para_dict.get('eval_epochs')
#         to_fix_ae = para_dict.get('to_fix_ae')
#         to_fix_seg = para_dict.get('to_fix_seg')
#
#         #----AE
#         if self.to_train_ae:
#             ae_var = para_dict['ae_var']
#             aug_times = ae_var.get('aug_times')
#             batch_size = ae_var['batch_size']
#             ratio = ae_var.get('ratio')
#             process_dict = ae_var.get('process_dict')
#             setting_dict = ae_var.get('setting_dict')
#             save_period = ae_var.get('save_period')
#             target_dict = ae_var.get('target')
#             pause_opt_ae = ae_var.get('pause_opt_ae')
#             img_standard_path = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220901_AOI判定OK\OK(多區OK)\train\0_-16_MatchLightSet_NoFailRegion_Ok_1.jpg"
#
#
#         #----SEG
#         if self.to_train_seg:
#             seg_var = para_dict['seg_var']
#             ratio_seg = seg_var.get('ratio_seg')
#             process_seg_dict = seg_var.get('process_dict')
#             setting_seg_dict = seg_var.get('setting_dict')
#             aug_seg_times = seg_var.get('aug_times')
#             json_check = seg_var.get('json_check')
#             batch_size_seg = seg_var['batch_size']
#             self_create_label = seg_var.get('self_create_label')
#             train_with_aug_v2 = seg_var.get('train_with_aug_v2')
#
#
#         #----special_img_dir
#         if self.special_img_dir is not None:
#             special_img_ratio = para_dict.get('special_img_ratio')
#             if special_img_ratio is None:
#                 special_img_ratio = 0.04
#             elif special_img_ratio > 1.0:
#                 special_img_ratio = 1.0
#             elif special_img_ratio < 0:
#                 special_img_ratio = 0.01
#
#         if encode_header is None:
#             encode_header = [24,97,28,98]
#         if encode_num is None:
#             encode_num = 87
#         self.encode_header = encode_header
#         self.encode_num = encode_num
#
#         if save_period is None:
#             save_period = 1
#
#
#         # ----local var
#         LV = dict()
#         train_loss_list = list()
#         seg_train_loss_list = list()
#         train_acc_list = list()
#         test_loss_list = list()
#         seg_test_loss_list = list()
#         test_acc_list = list()
#         epoch_time_list = list()
#         img_quantity = 0
#         aug_enable = False
#         break_flag = False
#         error_dict = {'GPU_resource_error': False}
#         train_result_dict = {'loss': 0,"loss_method":self.loss_method}
#         test_result_dict = {'loss': 0, "loss_method": self.loss_method}
#
#         record_type = 'loss'
#         qty_sp = 0
#         LV['pb_save_path_old'] = ''
#         LV['record_value'] = 0
#         LV['pb_seg_save_path_old'] = ''
#         LV['record_value_seg'] = 0
#         keep_prob = 0.7
#
#         #----AE hyper-parameters
#         if self.to_train_ae:
#             dataloader_AE_train = AE_Seg_Util.DataLoader4Seg(self.train_paths,
#                                                        only_img=True,
#                                                        batch_size=batch_size,
#                                                        pipelines=ae_var.get("train_pipelines"),
#                                                        to_shuffle=True,
#                                                        print_out=self.print_out)
#             dataloader_AE_val = AE_Seg_Util.DataLoader4Seg(self.test_paths,
#                                                              only_img=True,
#                                                              batch_size=batch_size,
#                                                              pipelines=ae_var.get("val_pipelines"),
#                                                              to_shuffle=False,
#                                                              print_out=self.print_out)
#             tl = AE_Seg_Util.tools()
#             # ----set target
#             tl.set_target(target_dict)
#             batch_size_test = batch_size
#
#             #----check if the augmentation(image processing) is enabled
#             if isinstance(process_dict, dict):
#                 if True in process_dict.values():
#                     aug_enable = True
#                     tl.set_process(process_dict, setting_dict, print_out=print_out)
#
#                     if aug_times is None:
#                         aug_times = 2
#                     batch_size = batch_size // aug_times  # the batch size must be integer!!
#
#         #----SEG hyper-parameters
#         if self.to_train_seg:
#             tl_seg = AE_Seg_Util.tools()
#             tl_seg.set_classname_id_color(**self.classname_id_color_dict)
#
#             #get seg ok paths for Aug vivid defect
#             if train_with_aug_v2:
#                 ok_img_seg_dir = seg_var.get('ok_img_seg_dir')
#                 if ok_img_seg_dir is None:
#                     train_with_aug_v2 = False
#                 else:
#                     ok_img_seg_paths,ok_img_seg_qty = tl_seg.get_paths(ok_img_seg_dir)
#                     if ok_img_seg_qty == 0:
#                         train_with_aug_v2 = False
#
#
#             if self_create_label or train_with_aug_v2:
#                 tl_seg_v2 = AE_Seg_Util.tools_v2(pipelines=seg_var['train_pipelines'],print_out=print_out)
#                 tl_seg_v2.set_classname_id_color(**self.classname_id_color_dict)
#
#
#             titles = ['prediction', 'answer']
#             batch_size_seg_test = batch_size_seg
#             seg_p = AE_Seg_Util.Seg_performance(print_out=print_out)
#             seg_p.set_classname_id_color(**self.classname_id_color_dict)
#
#             # ----check if the augmentation(image processing) is enabled
#             if isinstance(process_seg_dict, dict):
#                 if True in process_seg_dict.values():
#                     aug_seg_enable = True
#                     tl_seg.set_process(process_seg_dict, setting_seg_dict)
#                     if aug_seg_times is None:
#                         aug_seg_times = 2
#                     batch_size_seg = batch_size_seg // aug_seg_times  # the batch size must be integer!!
#                     if batch_size_seg < 1:
#                         batch_size_seg = 1
#
#         #----update content
#         # self.content = self.log_update(self.content, para_dict)
#
#
#
#         # ----calculate iterations of one epoch
#         # train_ites = math.ceil(img_quantity / batch_size)
#         # test_ites = math.ceil(len(self.test_paths) / batch_size)
#
#         t_train_start = time.time()
#         #----GPU setting
#         config = GPU_setting(GPU_ratio)
#         with tf.Session(config=config) as sess:
#             status = weights_check(sess, self.saver, self.save_dir, encript_flag=encript_flag,
#                                    encode_num=encode_num, encode_header=encode_header)
#             if status is False:
#                 error_dict['GPU_resource_error'] = True
#             elif status is True:
#                 if self.to_train_ae is True:
#                     #----AE train set quantity
#                     qty_train = self.get_train_qty(self.train_path_qty,ratio,print_out=print_out,name='AE')
#
#
#                     # qty_train = self.train_path_qty
#                     # if ratio is not None:
#                     #     if ratio <= 1.0 and ratio > 0:
#                     #         qty_train = int(self.train_path_qty * ratio)
#                     #         qty_train = np.maximum(1, qty_train)
#                     #
#                     # msg = "AE訓練集資料總共取數量{}".format(qty_train)
#                     # say_sth(msg, print_out=print_out)
#
#                     #----special set quantity
#                     if self.special_img_dir is not None:
#                         qty_sp = int(qty_train * special_img_ratio)
#                         msg = "加強學習資料總共取數量{}".format(qty_sp)
#                         say_sth(msg, print_out=print_out)
#
#                     # ----calculate iterations of one epoch
#                     img_quantity = qty_train + qty_sp
#                     train_ites = math.ceil(img_quantity / batch_size)
#                     if self.test_img_dir is not None:
#                         test_ites = math.ceil(self.test_path_qty / batch_size_test)
#
#                 #----SEG
#                 if self.to_train_seg is True:
#                     # ----SEG train set quantity
#                     qty_train_seg = self.get_train_qty(self.seg_train_qty, ratio_seg, print_out=print_out,
#                                                        name='SEG')
#
#                     # ----calculate iterations of one epoch
#                     train_ites_seg = math.ceil(self.seg_train_qty / batch_size_seg)
#                     if self.seg_test_qty > 0:
#                         test_ites_seg = math.ceil(self.seg_test_qty / batch_size_seg_test)
#                     if self.seg_predict_qty > 0:
#                         predict_ites_seg = math.ceil(self.seg_predict_qty / batch_size_seg_test)
#
#                 # ----epoch training
#                 for epoch in range(epochs):
#                     # ----read manual cmd
#                     break_flag = self.read_manual_cmd(para_dict.get('to_read_manual_cmd'))
#                     # ----break the training
#                     if break_flag:
#                         break_flag = False
#                         break
#
#                     # ----error check
#                     if True in list(error_dict.values()):
#                         break
#                     # ----record the start time
#                     d_t = time.time()
#
#
#                     train_loss_seg = 0
#                     test_loss_seg = 0
#                     train_acc = 0
#
#                     test_loss_2 = 0
#                     test_acc = 0
#
#                     #----AE part
#                     if self.to_train_ae is True:
#                         if to_fix_ae is True:
#                             pass
#                         else:
#                             #tf_var_AE = tf.trainable_variables(scope='AE')
#                             #----shuffle
#                             indice = np.random.permutation(self.train_path_qty)
#                             self.train_paths = self.train_paths[indice]
#                             train_paths_ori = self.train_paths[:qty_train]
#
#                             #----special img dir
#                             if self.special_img_dir is not None:
#                                 #----shuffle for special set
#                                 indice = np.random.permutation(self.sp_path_qty)
#                                 self.sp_paths = self.sp_paths[indice]
#
#                                 if self.sp_path_qty < qty_sp:
#                                     multi_ratio = math.ceil(qty_sp / self.sp_path_qty)
#                                     sp_paths = np.array(list(self.sp_paths) * multi_ratio)
#                                 else:
#                                     sp_paths = self.sp_paths
#
#                                 train_paths_ori = np.concatenate([train_paths_ori, sp_paths[:qty_sp]], axis=-1)
#
#                                 #-----shuffle for (train set + special set)
#                                 indice = np.random.permutation(img_quantity)
#                                 train_paths_ori = train_paths_ori[indice]
#
#                             if aug_enable is True:
#                                 train_paths_aug = train_paths_ori[::-1]
#                                 #train_labels_aug = train_labels_ori[::-1]
#
#                             #----optimizations(AE train set)
#                             for batch_paths, batch_data in dataloader_AE_train:
#                                 feed_dict = {self.tf_input: batch_data, self.tf_keep_prob: keep_prob}
#                                 # ----optimization
#                                 try:
#                                     sess.run(self.opt_AE, feed_dict=feed_dict)
#                                 except:
#                                     error_dict['GPU_resource_error'] = True
#                                     msg = "Error:權重最佳化時產生錯誤，可能GPU資源不夠導致"
#                                     say_sth(msg, print_out=print_out)
#                                     break
#
#                             #----evaluation(training set)
#                             train_loss = 0
#                             for batch_paths, batch_data in dataloader_AE_train:
#                                 feed_dict = {self.tf_input: batch_data, self.tf_keep_prob: 1.0}
#                                 loss_temp = sess.run(self.loss_AE, feed_dict=feed_dict)
#                                 train_loss += loss_temp
#
#                             train_loss /= dataloader_AE_train.iterations
#                             train_loss_list.append(float(train_loss))
#                             self.log["train_loss_list"] = train_loss_list
#                             #----evaluation(validation set)
#                             test_loss = 0
#                             for batch_paths, batch_data in dataloader_AE_val:
#                                 feed_dict = {self.tf_input: batch_data, self.tf_keep_prob: 1.0}
#                                 loss_temp = sess.run(self.loss_AE, feed_dict=feed_dict)
#                                 test_loss += loss_temp
#
#                             test_loss /= dataloader_AE_val.iterations
#                             test_loss_list.append(float(test_loss))
#                             self.log["test_loss_list"] = test_loss_list
#
#                             # for index in range(train_ites):
#                             #     # ----command process
#                             #     break_flag = self.receive_msg_process(sess,epoch,encript_flag=encript_flag)
#                             #     if break_flag:
#                             #         break
#                             #
#                             #     # ----get image start and end numbers
#                             #     ori_paths = tl.get_ite_data(train_paths_ori,index,batch_size=batch_size)
#                             #     aug_paths = tl.get_ite_data(train_paths_aug,index,batch_size=batch_size)
#                             #
#                             #     # ----get 4-D data
#                             #     if aug_enable is True:
#                             #         # ----ori data
#                             #         # ori_data = get_4D_data(ori_paths, self.model_shape[1:],process_dict=None)
#                             #         ori_data = tl.get_4D_data(ori_paths, self.model_shape[1:], to_norm=True, to_rgb=True,
#                             #                                     to_process=False,dtype=self.dtype)
#                             #
#                             #         #ori_labels = train_labels_ori[num_start:num_end]
#                             #         # ----aug data
#                             #
#                             #         if self.return_ori_data:
#                             #             # aug_data_no_patch,aug_data = get_4D_data(aug_paths, self.model_shape[1:],
#                             #             #                         process_dict=process_dict,setting_dict=setting_dict)
#                             #             aug_data_no_patch, aug_data = tl.get_4D_data(aug_paths, self.model_shape[1:],
#                             #                                                          to_norm=True,
#                             #                                                          to_rgb=True,
#                             #                                                          to_process=True,
#                             #                                                          dtype=self.dtype)
#                             #             batch_data_no_patch = np.concatenate([ori_data, aug_data_no_patch], axis=0)
#                             #         else:
#                             #             # aug_data = get_4D_data(aug_paths, self.model_shape[1:],
#                             #             #                         process_dict=process_dict,setting_dict=setting_dict)
#                             #             aug_data = tl.get_4D_data(aug_paths, self.model_shape[1:],
#                             #                                       to_norm=True,
#                             #                                       to_rgb=True,
#                             #                                       to_process=True,
#                             #                                       dtype=self.dtype)
#                             #
#                             #         #aug_labels = train_labels_aug[num_start:num_end]
#                             #         # ----data concat
#                             #         batch_data = np.concatenate([ori_data, aug_data], axis=0)
#                             #         # if process_dict.get('rdm_patch'):
#                             #         #     batch_data_ori = np.concatenate([ori_data, ori_data], axis=0)
#                             #         #batch_labels = np.concatenate([ori_labels, aug_labels], axis=0)
#                             #     else:
#                             #         # batch_data = get_4D_data(ori_paths, self.model_shape[1:])
#                             #         batch_data = tl.get_4D_data(ori_paths, self.model_shape[1:],dtype=self.dtype)
#                             #         #batch_labels = train_labels_ori[num_start:num_end]
#                             #
#                             #     #----put all data to tf placeholders
#                             #     if self.return_ori_data:
#                             #         feed_dict = {self.tf_input: batch_data, self.tf_input_ori: batch_data_no_patch,
#                             #                      self.tf_keep_prob: keep_prob}
#                             #     else:
#                             #         feed_dict = {self.tf_input: batch_data, self.tf_keep_prob: keep_prob}
#                             #
#                             #     if self.infer_method == 'AE_pooling_net_V8':
#                             #         img_standard = tl.get_4D_data([img_standard_path]*len(batch_data), self.model_shape[1:], dtype=self.dtype)
#                             #         feed_dict[self.tf_input_standard] = img_standard
#                             #
#                             #     # ----optimization
#                             #     try:
#                             #         if pause_opt_ae is not True:
#                             #             sess.run(self.opt_AE, feed_dict=feed_dict)
#                             #     except:
#                             #         error_dict['GPU_resource_error'] = True
#                             #         msg = "Error:權重最佳化時產生錯誤，可能GPU資源不夠導致"
#                             #         say_sth(msg, print_out=print_out)
#                             #         break
#                             #     # if self.loss_method_2 is not None:
#                             #     #     sess.run(self.opt_AE_2, feed_dict=feed_dict)
#                             #
#                             #     # ----evaluation(training set)
#                             #     feed_dict[self.tf_keep_prob] = 1.0
#                             #     # feed_dict[self.tf_phase_train] = False
#                             #     loss_temp = sess.run(self.loss_AE, feed_dict=feed_dict)
#                             #     # if self.loss_method_2 is not None:
#                             #     #     loss_temp_2 = sess.run(self.loss_AE_2, feed_dict=feed_dict)
#                             #
#                             #     # ----calculate the loss and accuracy
#                             #     train_loss += loss_temp
#                             #     # if self.loss_method_2 is not None:
#                             #     #     train_loss_2 += loss_temp_2
#
#                             # train_loss /= train_ites
#                             # train_result_dict['loss'] = train_loss
#                             # # if self.loss_method_2 is not None:
#                             # #     train_loss_2 /= train_ites
#                             #
#                             # # ----break the training
#                             # if break_flag:
#                             #     break_flag = False
#                             #     break
#                             # if True in list(error_dict.values()):
#                             #     break
#                             #
#                             # #----evaluation(test set)
#                             # if self.test_img_dir is not None:
#                             #     for index in range(test_ites):
#                             #         # ----get image start and end numbers
#                             #         ite_paths = tl.get_ite_data(self.test_paths, index, batch_size=batch_size_test)
#                             #
#                             #         # batch_data = get_4D_data(ite_paths, self.model_shape[1:])
#                             #         batch_data = tl.get_4D_data(ite_paths, self.model_shape[1:],dtype=self.dtype)
#                             #
#                             #
#                             #         # ----put all data to tf placeholders
#                             #         if self.return_ori_data:
#                             #             feed_dict[self.tf_input] = batch_data
#                             #             feed_dict[self.tf_input_ori] = batch_data
#                             #             # feed_dict = {self.tf_input: batch_data, self.tf_input_ori: batch_data,
#                             #             #              self.tf_keep_prob: 1.0}
#                             #         else:
#                             #             feed_dict[self.tf_input] = batch_data
#                             #             # feed_dict = {self.tf_input: batch_data, self.tf_keep_prob: 1.0}
#                             #
#                             #         if self.infer_method == 'AE_pooling_net_V8':
#                             #             img_standard = tl.get_4D_data([img_standard_path] * len(batch_data),
#                             #                                           self.model_shape[1:], dtype=self.dtype)
#                             #             feed_dict[self.tf_input_standard] = img_standard
#                             #
#                             #         # ----session run
#                             #         #sess.run(self.opt_AE, feed_dict=feed_dict)
#                             #
#                             #         # ----evaluation(training set)
#                             #         # feed_dict[self.tf_phase_train] = False
#                             #         try:
#                             #             loss_temp = sess.run(self.loss_AE, feed_dict=feed_dict)
#                             #         except:
#                             #             error_dict['GPU_resource_error'] = True
#                             #             msg = "Error:推論驗證集時產生錯誤"
#                             #             say_sth(msg, print_out=print_out)
#                             #             break
#                             #         # if self.loss_method_2 is not None:
#                             #         #     loss_temp_2 = sess.run(self.loss_AE_2, feed_dict=feed_dict)
#                             #
#                             #         # ----calculate the loss and accuracy
#                             #         test_loss += loss_temp
#                             #         # if self.loss_method_2 is not None:
#                             #         #     test_loss_2 += loss_temp_2
#                             #
#                             #     test_loss /= test_ites
#                             #     test_result_dict['loss'] = test_loss
#
#                             #----save results in the log file
#                             # train_loss_list.append(float(train_loss))
#                             # self.log["train_loss_list"] = train_loss_list
#                             #
#                             # if self.test_img_dir is not None:
#                             #     test_loss_list.append(float(test_loss))
#                             #     #test_acc_list.append(float(test_acc))
#                             #     self.log["test_loss_list"] = test_loss_list
#
#                             #----display training results
#                             msg = "\n----訓練週期 {} 與相關結果如下----".format(epoch)
#                             say_sth(msg, print_out=print_out)
#                             self.display_results(None, print_out,
#                                                  train_loss=train_loss,
#                                                  val_loss=test_loss,
#                                                  )
#
#                             #----send protocol data(for UI to draw)
#                             self.transmit_data2ui(epoch, print_out,
#                                                   train_AE_loss=train_loss,
#                                                   val_AE_loss=test_loss,
#                                                   )
#
#                             #----find the best performance
#                             new_value = self.__get_result_value(record_type, train_result_dict, test_result_dict)
#                             if epoch == 0:
#                                 LV['record_value'] = new_value
#                             else:
#
#                                 go_flag = False
#                                 if self.loss_method == 'ssim':
#                                     if new_value > LV['record_value']:
#                                         go_flag = True
#                                 else:
#                                     if new_value < LV['record_value']:
#                                         go_flag = True
#
#                                 if go_flag is True:
#                                     # ----delete the previous pb
#                                     if os.path.exists(LV['pb_save_path_old']):
#                                         os.remove(LV['pb_save_path_old'])
#
#                                     #----save the better one
#                                     pb_save_path = create_pb_path("infer_{}".format(new_value),self.save_dir,
#                                                    to_encode=encript_flag)
#                                     save_pb_file(sess, self.pb_save_list, pb_save_path, encode=encript_flag,
#                                                       random_num_range=encode_num, header=encode_header)
#                                     #----update record value
#                                     LV['record_value'] = new_value
#                                     LV['pb_save_path_old'] = pb_save_path
#
#                             #----Check if stops the training
#                             if target_dict['type'] == 'loss':
#                                 if self.test_img_dir is not None:
#                                     re = tl.target_compare(test_result_dict)
#                                     name = "驗證集"
#                                 else:
#                                     re = tl.target_compare(train_result_dict)
#                                     name = "訓練集"
#
#                                 if re is True:
#                                     msg = '模型訓練結束:\n{}{}已經達到設定目標:{}累積達{}次'.format(
#                                         name, target_dict['type'],target_dict['value'], target_dict['hit_target_times'])
#                                     say_sth(msg, print_out=print_out)
#                                     break
#
#                             # ----test image reconstruction
#                             if self.recon_flag is True:
#                                 # if (epoch + 1) % eval_epochs == 0 and train_loss > 0.80:
#                                 if (epoch + 1) % eval_epochs == 0:
#                                     for filename in self.recon_paths:
#                                         test_img = self.__img_read(filename, self.model_shape[1:],dtype=self.dtype)
#                                         if self.infer_method == 'AE_pooling_net_V8':
#                                             img_standard = tl.get_4D_data([img_standard_path] * len(test_img),
#                                                                           self.model_shape[1:], dtype=self.dtype)
#                                             feed_dict[self.tf_input_standard] = img_standard
#                                         # ----session run
#                                         feed_dict[self.tf_input] = test_img
#                                         img_sess_out = sess.run(self.recon, feed_dict=feed_dict)
#                                         # ----process of sess-out
#                                         img_sess_out = img_sess_out[0] * 255
#                                         img_sess_out = cv2.convertScaleAbs(img_sess_out)
#                                         if self.model_shape[3] == 1:
#                                             img_sess_out = np.reshape(img_sess_out, (self.model_shape[1], self.model_shape[2]))
#                                         else:
#                                             img_sess_out = cv2.cvtColor(img_sess_out, cv2.COLOR_RGB2BGR)
#
#                                         # if loss_method != 'ssim':
#                                         #     img_sess_out = cv2.cvtColor(img_sess_out, cv2.COLOR_RGB2BGR)
#
#                                         # ----save recon image
#                                         splits = filename.split("\\")[-1]
#                                         new_filename = splits.split('.')[0] + '_sess-out.' + splits.split('.')[-1]
#
#                                         new_filename = os.path.join(self.new_recon_dir, new_filename)
#                                         # cv2.imwrite(new_filename, img_sess_out)
#                                         cv2.imencode('.'+splits.split('.')[-1], img_sess_out)[1].tofile(new_filename)
#                                         # ----img diff method
#                                         img_diff = self.__img_diff_method(filename, img_sess_out, diff_th=15, cc_th=15)
#                                         img_diff = cv2.convertScaleAbs(img_diff)
#                                         new_filename = filename.split("\\")[-1]
#                                         new_filename = new_filename.split(".")[0] + '_diff.' + new_filename.split(".")[-1]
#                                         new_filename = os.path.join(self.new_recon_dir, new_filename)
#                                         # cv2.imwrite(new_filename, img_diff)
#                                         cv2.imencode('.' + splits.split('.')[-1], img_diff)[1].tofile(new_filename)
#
#                                         # ----img avepool diff method
#                                         img_diff = self.__img_patch_diff_method(filename, img_sess_out, sess, diff_th=30, cc_th=15)
#                                         img_diff = cv2.convertScaleAbs(img_diff)
#                                         new_filename = filename.split("\\")[-1]
#                                         new_filename = new_filename.split(".")[0] + '_avepool_diff.' + new_filename.split(".")[-1]
#                                         new_filename = os.path.join(self.new_recon_dir, new_filename)
#                                         # cv2.imwrite(new_filename, img_diff)
#                                         cv2.imencode('.' + splits.split('.')[-1], img_diff)[1].tofile(new_filename)
#
#                                         # ----SSIM method
#                                         # img_ssim = self.__ssim_method(filename, img_sess_out)
#                                         # img_ssim = cv2.convertScaleAbs(img_ssim)
#                                         # new_filename = filename.split("\\")[-1]
#                                         # new_filename = new_filename.split(".")[0] + '_ssim.' + new_filename.split(".")[-1]
#                                         # new_filename = os.path.join(self.new_recon_dir, new_filename)
#                                         # cv2.imwrite(new_filename, img_ssim)
#
#                             # ----save ckpt
#                             # if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
#                             #     save_pb_file(sess, self.pb_save_list, self.pb4ae_save_path,
#                             #                  encode=encript_flag,
#                             #                  random_num_range=encode_num, header=encode_header)
#
#                     #----SEG part
#                     if self.to_train_seg is True:
#                         if to_fix_seg is True:
#                             pass
#                         else:
#                             #----
#                             seg_p.reset_arg()
#                             seg_p.reset_defect_stat()
#                             #----
#                             indice = np.random.permutation(self.seg_train_qty)
#                             self.seg_train_paths = self.seg_train_paths[indice]
#                             if json_check:
#                                 self.seg_train_json_paths = self.seg_train_json_paths[indice]
#
#                             seg_train_paths_ori = self.seg_train_paths[:qty_train_seg]
#                             if json_check:
#                                 seg_train_json_paths_ori = self.seg_train_json_paths[:qty_train_seg]
#
#                             if aug_enable is True:
#                                 seg_train_paths_aug = seg_train_paths_ori[::-1]
#                                 if json_check:
#                                     seg_train_json_paths_aug = seg_train_json_paths_ori[::-1]
#
#                             # ----optimizations(SEG train set)
#                             for idx_seg in range(train_ites_seg):
#                                 # ----command process
#                                 break_flag = self.receive_msg_process(sess, epoch, encript_flag=encript_flag)
#                                 if break_flag:
#                                     break
#
#                                 # ----get image start and end numbers
#                                 ori_seg_paths = tl_seg.get_ite_data(seg_train_paths_ori, idx_seg, batch_size=batch_size_seg)
#                                 aug_seg_paths = tl_seg.get_ite_data(seg_train_paths_aug, idx_seg, batch_size=batch_size_seg)
#                                 if json_check:
#                                     ori_seg_json_paths = tl_seg.get_ite_data(seg_train_json_paths_ori, idx_seg,
#                                                                              batch_size=batch_size_seg)
#                                     aug_seg_json_paths = tl_seg.get_ite_data(seg_train_json_paths_aug, idx_seg,
#                                                                              batch_size=batch_size_seg)
#                                 else:
#                                     ori_seg_json_paths = None
#                                     aug_seg_json_paths = None
#                                     # print("ori_seg_json_paths:",ori_seg_json_paths)
#                                     # print("aug_seg_json_paths:",aug_seg_json_paths)
#
#                                 #----get 4-D data
#                                 if aug_enable is True:
#                                     if self_create_label:
#                                         # ----ori data
#                                         #(self,paths, output_shape,to_norm=True,to_rgb=True,to_process=False,dtype='float32')
#                                         ori_data, ori_label = tl_seg.get_4D_data_create_mask(ori_seg_paths,
#                                                                                            self.model_shape[1:],
#                                                                                            to_norm=True, to_rgb=True,
#                                                                                            to_process=False,
#                                                                                            dtype=self.dtype)
#
#                                         # ----aug data
#                                         # aug_data, aug_label = tl_seg.get_4D_data_create_mask(aug_seg_paths,
#                                         #                                                    self.model_shape[1:],
#                                         #                                                    to_norm=True,
#                                         #                                                    to_rgb=True,
#                                         #                                                    to_process=True,
#                                         #                                                    dtype=self.dtype)
#                                         aug_data, aug_label = tl_seg_v2.get_seg_batch_data(aug_seg_paths)
#                                     else:
#                                         # ----ori data
#                                         ori_data,ori_label = tl_seg.get_4D_img_label_data(ori_seg_paths,self.model_shape[1:],
#                                                                                           json_paths=ori_seg_json_paths,
#                                                                             to_norm=True, to_rgb=True,
#                                                                             to_process=False, dtype=self.dtype)
#
#                                         # ----aug data
#                                         aug_data,aug_label = tl_seg.get_4D_img_label_data(aug_seg_paths,self.model_shape[1:],
#                                                                                           json_paths=aug_seg_json_paths,
#                                                                   to_norm=True,
#                                                                   to_rgb=True,
#                                                                   to_process=True,
#                                                                   dtype=self.dtype)
#                                         if train_with_aug_v2:
#                                             ok_batch_paths = np.random.choice(ok_img_seg_paths,batch_size_seg,replace=False)
#                                             aug_data_2, aug_label_2 = tl_seg_v2.get_seg_batch_data(ok_batch_paths)
#                                             # a_dir = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\temp"
#                                             # temp_img = aug_data_2[0] * 255
#                                             # temp_img = cv2.convertScaleAbs(temp_img)
#                                             # save_path = ok_batch_paths[0].split("\\")[-1].split(".")[0]
#                                             # save_path += '.bmp'
#                                             # save_path = os.path.join(a_dir, save_path)
#                                             # cv2.imencode('.bmp', temp_img)[1].tofile(save_path)
#
#                                     #----data concat
#                                     batch_data = np.concatenate([ori_data, aug_data], axis=0)
#                                     batch_label = np.concatenate([ori_label, aug_label], axis=0)
#                                     if train_with_aug_v2:
#                                         batch_data = np.concatenate([batch_data, aug_data_2], axis=0)
#                                         batch_label = np.concatenate([batch_label, aug_label_2], axis=0)
#                                 else:
#                                     if self_create_label:
#                                         batch_data, batch_label = tl_seg.get_4D_data_create_mask(ori_seg_paths,
#                                                                                                self.model_shape[1:],
#                                                                                                dtype=self.dtype)
#                                     else:
#                                         batch_data,batch_label = tl_seg.get_4D_img_label_data(ori_seg_paths,self.model_shape[1:],
#                                                                                     json_paths=ori_seg_json_paths,
#                                                                                    dtype=self.dtype)
#
#
#                                 feed_dict = {self.tf_input: batch_data,self.tf_input_recon:batch_data,
#                                              self.tf_label_batch: batch_label,
#                                              self.tf_keep_prob: keep_prob,
#                                              self.tf_dropout: 0.3}
#
#                                 #----session run
#                                 try:
#                                     sess.run(self.opt_Seg, feed_dict=feed_dict)
#                                 except:
#                                     error_dict['GPU_resource_error'] = True
#                                     msg = "Error:SEG權重最佳化時產生錯誤，可能GPU資源不夠導致"
#                                     say_sth(msg, print_out=print_out)
#                                     break
#
#                                 #----evaluation(training set)
#                                 feed_dict[self.tf_keep_prob] = 1.0
#                                 feed_dict[self.tf_dropout] = 0.0
#                                 # feed_dict[self.tf_phase_train] = False
#
#                                 loss_temp = sess.run(self.loss_Seg, feed_dict=feed_dict)
#                                 predict_label = sess.run(self.prediction_Seg, feed_dict=feed_dict)
#                                 # predict_label = np.argmax(predict_label,axis=-1).astype(np.uint8)
#
#                                 #----calculate the loss and accuracy
#                                 train_loss_seg += loss_temp
#                                 seg_p.cal_intersection_union(predict_label,batch_label)
#                                 _ = seg_p.cal_label_defect_by_acc_v2(predict_label, batch_label)
#                                 _ = seg_p.cal_predict_defect_by_acc_v2(predict_label, batch_label)
#
#                             train_loss_seg /= train_ites_seg
#                             # seg_train_result_dict['loss'] = train_loss_seg
#                             #----save results in the log file
#                             seg_train_loss_list.append(float(train_loss_seg))
#                             self.log["seg_train_loss_list"] = seg_train_loss_list
#                             train_iou_seg, train_acc_seg, train_all_acc_seg = seg_p.cal_iou_acc(save_dict=self.log,name='train')
#                             train_defect_recall = seg_p.cal_defect_recall(save_dict=self.log,name='train')
#                             train_defect_sensitivity = seg_p.cal_defect_sensitivity(save_dict=self.log,name='train')
#                             #print("iou:{}, acc:{}, all_acc:{}".format(iou, acc, all_acc))
#                             #print("train_loss_seg:",train_loss_seg)
#
#                             #----evaluation(test set)
#                             if self.seg_test_qty > 0:
#                                 seg_p.reset_arg()
#                                 seg_p.reset_defect_stat()
#                                 for idx_seg in range(test_ites_seg):
#                                     #----get batch paths
#                                     seg_paths = tl_seg.get_ite_data(self.seg_test_paths, idx_seg,
#                                                                         batch_size=batch_size_seg_test)
#                                     if json_check:
#                                         seg_json_paths = tl_seg.get_ite_data(self.seg_test_json_paths, idx_seg,
#                                                                                  batch_size=batch_size_seg_test)
#                                     else:
#                                         seg_json_paths = None
#                                     #----get batch data
#                                     # if self_create_label:
#                                     #     batch_data, batch_label = tl_seg.get_4D_data_create_mask(seg_paths,
#                                     #                                                              self.model_shape[1:],
#                                     #                                                              to_process=True,
#                                     #                                                              dtype=self.dtype)
#                                     # else:
#                                     batch_data, batch_label = tl_seg.get_4D_img_label_data(seg_paths,
#                                                                                            self.model_shape[1:],
#                                                                                            json_paths=seg_json_paths,
#                                                                                            dtype=self.dtype)
#                                     #----put all data to tf placeholders
#                                     # recon = sess.run(self.recon,feed_dict={self.tf_input: batch_data})
#
#                                     feed_dict = {self.tf_input: batch_data, self.tf_input_recon: batch_data,
#                                                  self.tf_label_batch: batch_label,self.tf_keep_prob: 0}
#
#                                     loss_temp = sess.run(self.loss_Seg, feed_dict=feed_dict)
#                                     predict_label = sess.run(self.prediction_Seg, feed_dict=feed_dict)
#                                     # predict_label = np.argmax(predict_label, axis=-1).astype(np.uint8)
#
#                                     # ----calculate the loss and accuracy
#                                     test_loss_seg += loss_temp
#
#                                     seg_p.cal_intersection_union(predict_label, batch_label)
#                                     _ = seg_p.cal_label_defect_by_acc_v2(predict_label,batch_label)
#                                     _ = seg_p.cal_predict_defect_by_acc_v2(predict_label,batch_label)
#
#                                 test_loss_seg /= test_ites_seg
#                                 # seg_test_result_dict['loss'] = test_loss_seg
#
#                                 # ----save results in the log file
#                                 seg_test_loss_list.append(float(test_loss_seg))
#                                 self.log["seg_test_loss_list"] = seg_test_loss_list
#                                 test_iou_seg, test_acc_seg, test_all_acc_seg = seg_p.cal_iou_acc(save_dict=self.log,
#                                                                                                     name='test')
#                                 test_defect_recall = seg_p.cal_defect_recall(save_dict=self.log, name='test')
#                                 test_defect_sensitivity = seg_p.cal_defect_sensitivity(save_dict=self.log, name='test')
#
#                                 #----find the best performance(SEG)
#                                 new_value = self.get_value_from_performance(seg_p,seg_var.get('target_of_best'))
#                                 self.performance_process(epoch, new_value, LV, sess, encript_flag)
#
#
#                             #----display training results
#                             msg = "\n----訓練週期 {} 與相關結果如下----".format(epoch)
#                             say_sth(msg, print_out=print_out)
#
#                             self.display_results(self.class_names,print_out,
#                                                  train_SEG_loss=train_loss_seg,
#                                                  train_SEG_iou=train_iou_seg,
#                                                  train_SEG_accuracy=train_acc_seg,
#                                                  train_SEG_defect_recall=train_defect_recall,
#                                                  train_SEG_defect_sensitivity=train_defect_sensitivity
#                                                  )
#                             self.display_results(self.class_names,print_out,
#                                                  val_SEG_loss=test_loss_seg,
#                                                  val_SEG_iou=test_iou_seg,
#                                                  val_SEG_accuracy=test_acc_seg,
#                                                  val_SEG_defect_recall=test_defect_recall,
#                                                  val_SEG_defect_sensitivity=test_defect_sensitivity
#                                                  )
#
#                             #----send protocol data(for UI to draw)
#                             self.transmit_data2ui(epoch,print_out,
#                                                   train_loss=train_loss_seg,
#                                                   train_SEG_iou=train_iou_seg,
#                                                   train_SEG_accuracy=train_acc_seg,
#                                                   train_SEG_defect_recall=train_defect_recall,
#                                                   train_SEG_defect_sensitivity=train_defect_sensitivity,
#                                                   val_loss=test_loss_seg,
#                                                   val_SEG_iou=test_iou_seg,
#                                                   val_SEG_accuracy=test_acc_seg,
#                                                   val_SEG_defect_recall=test_defect_recall,
#                                                   val_SEG_defect_sensitivity=test_defect_sensitivity
#                                                   )
#
#                             #----prediction for selected images
#                             if self.seg_predict_qty > 0:
#                                 if (epoch + 1 ) % eval_epochs == 0:
#                                     for idx_seg in range(predict_ites_seg):
#                                         # ----get batch paths
#                                         seg_paths = tl_seg.get_ite_data(self.seg_predict_paths, idx_seg,
#                                                                         batch_size=batch_size_seg_test)
#
#                                         #----get batch data
#                                         batch_data, batch_label = tl_seg.get_4D_img_label_data(seg_paths,
#                                                                                                self.model_shape[1:],
#                                                                                                json_paths=None,
#                                                                                                dtype=self.dtype)
#                                         predict_labels = sess.run(self.tf_prediction_Seg,
#                                                                  feed_dict={self.tf_input: batch_data,
#                                                                             self.tf_input_recon:batch_data})
#                                         batch_data *= 255
#                                         for i in range(len(predict_labels)):
#                                             img_ori = cv2.convertScaleAbs(batch_data[i])
#
#                                             #----image with prediction
#                                             img_predict = tl_seg.combine_img_label(img_ori,
#                                                                                    predict_labels[i],
#                                                                                    self.id2color)
#                                             show_imgs = [img_predict]
#
#                                             #----image with label
#                                             json_path = tl_seg.get_json_path_from_img_path(seg_paths[i])
#                                             seg_label = tl_seg.read_seg_label(json_path)
#                                             if seg_label is not None:
#                                                 img_answer = tl_seg.combine_img_label(img_ori,
#                                                                                        seg_label,
#                                                                                        self.id2color)
#                                                 show_imgs.append(img_answer)
#
#                                             qty_show = len(show_imgs)
#                                             plt.figure(num=1,figsize=(5*qty_show, 5*qty_show), clear=True)
#
#                                             for k, show_img in enumerate(show_imgs):
#                                                 plt.subplot(1, qty_show, k + 1)
#                                                 plt.imshow(show_img)
#                                                 plt.axis('off')
#                                                 plt.title(titles[i])
#                                             save_path = os.path.join(self.new_predict_dir, seg_paths[i].split("\\")[-1])
#                                             plt.savefig(save_path)
#
#
#                     #----save ckpt, pb files
#                     if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
#                             model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)
#
#                             # ----encode ckpt file
#                             if encript_flag is True:
#                                 encode_CKPT(model_save_path, encode_num=encode_num, encode_header=encode_header)
#
#                             #----save pb(normal)
#                             save_pb_file(sess, self.pb_save_list, self.pb_save_path, encode=encript_flag,
#                                          random_num_range=encode_num, header=encode_header)
#
#                     #----record the end time
#                     d_t = time.time() - d_t
#
#                     epoch_time_list.append(d_t)
#                     total_train_time = time.time() - t_train_start
#                     self.log['total_train_time'] = float(total_train_time)
#                     self.log['ave_epoch_time'] = float(np.average(epoch_time_list))
#
#                     msg_list = []
#                     msg_list.append("此次訓練所花時間: {}秒".format(np.round(d_t, 2)))
#                     msg_list.append("累積訓練時間: {}秒".format(np.round(total_train_time, 2)))
#                     say_sth(msg_list, print_out=print_out, header='msg')
#
#                     with open(self.log_path, 'w') as f:
#                         json.dump(self.log, f)
#
#                     if encript_flag is True:
#                         if os.path.exists(self.log_path):
#                             file_transfer(self.log_path, cut_num_range=30, random_num_range=10)
#                     msg = "儲存訓練結果數據至{}".format(self.log_path)
#                     say_sth(msg, print_out=print_out)
#
#             #----error messages
#             if True in list(error_dict.values()):
#                 for key, value in error_dict.items():
#                     if value is True:
#                         say_sth('', print_out=print_out, header=key)
#             else:
#                 say_sth('AI Engine結束!!期待下次再相見', print_out=print_out, header='AIE_end')
#
#     #----functions
#     def seg_path_change_process(self,to_train_w_AE_paths,to_train_ae,select_OK_ratio=0.2):
#         if to_train_w_AE_paths:
#             if to_train_ae:
#                 select_num = np.minimum(self.train_path_qty, int(self.seg_train_qty * select_OK_ratio))
#                 temp_paths = np.random.choice(self.train_paths, size=select_num, replace=False)
#                 self.seg_train_paths = list(self.seg_train_paths)
#                 self.seg_train_paths.extend(temp_paths)
#                 self.seg_train_paths = np.array(self.seg_train_paths)
#                 self.seg_train_qty += select_num
#                 msg = f"SEG額外加入AE的OK圖片進行訓練，SEG訓練集圖片集數量增加後為{self.seg_train_qty}"
#                 say_sth(msg,print_out=self.print_out)
#
#     def path_qty_process(self,imgDirPathDict):
#         msg_list = []
#         status_list = []
#         for img_dir_name,qty in imgDirPathDict.items():
#             if qty == 0:
#                 msg = f"{img_dir_name}沒有圖片"
#                 stat = False
#             else:
#                 msg = "{}圖片數量:{}".format(img_dir_name,qty)
#                 stat = True
#             msg_list.append(msg)
#             status_list.append(stat)
#         say_sth(msg_list,print_out=self.print_out)
#
#         return status_list
#
#     def receive_msg_process(self,sess,epoch,encript_flag=True):
#         break_flag = False
#         if SockConnected:
#             # print(Sock.Message)
#             if len(Sock.Message):
#                 if Sock.Message[-1][:4] == "$S00":
#                     # Sock.send("Protocol:Ack\n")
#                     model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)
#                     # ----encode ckpt file
#                     if encript_flag:
#                         file = model_save_path + '.meta'
#                         if os.path.exists(file):
#                             file_transfer(file, random_num_range=self.encode_num, header=self.encode_header)
#                         else:
#                             msg = "Warning:找不到權重檔:{}進行處理".format(file)
#                             say_sth(msg, print_out=print_out)
#                         # data_file = [file.path for file in os.scandir(self.save_dir) if
#                         #              file.name.split(".")[-1] == 'data-00000-of-00001']
#                         data_file = model_save_path + '.data-00000-of-00001'
#                         if os.path.exists(data_file):
#                             file_transfer(data_file, random_num_range=self.encode_num, header=self.encode_header)
#                         else:
#                             msg = "Warning:找不到權重檔:{}進行處理".format(data_file)
#                             say_sth(msg, print_out=print_out)
#
#                     # msg = "儲存訓練權重檔至{}".format(model_save_path)
#                     # say_sth(msg, print_out=print_out)
#                     save_pb_file(sess, self.pb_save_list, self.pb_save_path, encode=encript_flag,
#                                  random_num_range=self.encode_num, header=self.encode_header)
#                     break_flag = True
#
#         return break_flag
#
#     def read_manual_cmd(self,to_read_manual_cmd,print_out=False):
#         break_flag = False
#         if to_read_manual_cmd:
#             j_path = os.path.join(self.save_dir, 'manual_cmd.json')
#             if os.path.exists(j_path):
#                 with open(j_path, 'r') as f:
#                     cmd_dict = json.load(f)
#                 if cmd_dict.get('to_stop_training') is True:
#                     break_flag = True
#                     msg = "接收到manual cmd: stop the training!"
#                     say_sth(msg, print_out=print_out)
#
#         return break_flag
#
#     def performance_process(self,epoch,new_value,LV,sess,encript_flag):
#         if epoch == 0:
#             LV['record_value_seg'] = new_value
#         else:
#             if new_value > LV['record_value_seg']:
#                 # ----delete the previous pb
#                 if os.path.exists(LV['pb_seg_save_path_old']):
#                     os.remove(LV['pb_seg_save_path_old'])
#
#                 # ----save the better one
#                 pb_save_path = create_pb_path(f"infer_best_epoch{epoch}", self.save_dir,
#                                               to_encode=encript_flag)
#                 save_pb_file(sess, self.pb_save_list, pb_save_path, encode=encript_flag,
#                              random_num_range=self.encode_num, header=self.encode_header)
#                 # ----update record value
#                 LV['record_value_seg'] = new_value
#                 LV['pb_seg_save_path_old'] = pb_save_path
#
#     def get_value_from_performance(self,seg_p,target_of_best="defect_recall"):
#         if target_of_best == 'defect_recall':
#             new_value = seg_p.sum_defect_recall()
#         elif target_of_best == 'defect_sensitivity':
#             new_value = seg_p.sum_defect_sensitivity()
#         elif target_of_best == 'recall+sensitivity':
#             new_value = seg_p.sum_defect_recall() + seg_p.sum_defect_sensitivity()
#         else:
#             new_value = seg_p.sum_iou_acc()
#
#         return new_value
#
#     def config_check(self,config_dict):
#         #----var
#         must_list = ['train_img_dir', 'model_name', 'save_dir', 'epochs']
#         # must_list = ['train_img_dir', 'test_img_dir', 'save_dir', 'epochs']
#         must_flag = True
#         default_dict = {"model_shape":[None,192,192,3],
#                         'model_name':"type_1_0",
#                         'loss_method':'ssim',
#                         'activation':'relu',
#                         'save_pb_name':'inference',
#                         'opti_method':'adam',
#                         'pool_type':['max', 'ave'],
#                         'pool_kernel':[7, 2],
#                         'embed_length':144,
#                         'learning_rate':1e-4,
#                         'batch_size':8,
#                         'ratio':1.0,
#                         'aug_times':2,
#                         'hit_target_times':2,
#                         'eval_epochs':2,
#                         'save_period':2,
#                         'kernel_list':[7,5,3,3,3],
#                         'filter_list':[32,64,96,128,256],
#                         'conv_time':1,
#                         'rot':False,
#                         'scaler':1,
#                         #'preprocess_dict':{'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}
#                         }
#
#
#         #----get the must list
#         if config_dict.get('must_list') is not None:
#             must_list = config_dict.get('must_list')
#         #----collect all keys of config_dict
#         config_key_list = list(config_dict.keys())
#
#         #----check the must list
#         if config_dict.get("J_mode") is not True:
#
#             #----check of must items
#             for item in must_list:
#                 if not item in config_key_list:
#                     msg = "Error: could you plz give me parameters -> {}".format(item)
#                     say_sth(msg,print_out=print_out)
#                     if must_flag is True:
#                         must_flag = False
#
#         #----parameters parsing
#         if must_flag is True:
#             #----model name
#             if config_dict.get("J_mode") is not True:
#                 infer_num = config_dict['model_name'].split("_")[-1]
#                 if infer_num == '0':#
#                     config_dict['infer_method'] = "AE_pooling_net"
#                 else:
#                     config_dict['infer_method'] = "AE_transpose_4layer"
#
#             #----optional parameters
#             for key,value in default_dict.items():
#                 if not key in config_key_list:
#                     config_dict[key] = value
#
#         return must_flag,config_dict
#
#     def get_train_qty(self,ori_qty,ratio,print_out=False,name=''):
#         if ratio is None:
#             qty_train = ori_qty
#         else:
#             if ratio <= 1.0 and ratio > 0:
#                 qty_train = int(ori_qty * ratio)
#                 qty_train = np.maximum(1, qty_train)
#
#         msg = "{}訓練集資料總共取數量{}".format(name,qty_train)
#         say_sth(msg, print_out=print_out)
#
#         return qty_train
#
#     def __img_patch_diff_method(self,img_source_1, img_source_2, sess,diff_th=30, cc_th=30):
#
#         temp = np.array([1., 2., 3.])
#         re = None
#         # ----read img source 1
#         if isinstance(temp, type(img_source_1)):
#             img_1 = img_source_1
#         elif os.path.isfile(img_source_1):
#             # img_1 = cv2.imread(img_source_1)
#             img_1 = cv2.imdecode(np.fromfile(img_source_1, dtype=np.uint8), 1)
#             img_1 = cv2.resize(img_1,(self.model_shape[2],self.model_shape[1]))
#             # img_1 = img_1.astype('float32')
#
#         else:
#             print("The type of img_source_1 is not supported")
#
#         # ----read img source 2
#         if isinstance(temp, type(img_source_2)):
#             img_2 = img_source_2
#         elif os.path.isfile(img_source_2):
#             # img_2 = cv2.imread(img_source_2)
#             img_2 = cv2.imdecode(np.fromfile(img_source_2, dtype=np.uint8), 1)
#             # img_2 = img_2.astype('float32')
#         else:
#             print("The type of img_source_2 is not supported")
#
#         # ----subtraction
#         if img_1 is not None and img_2 is not None:
#             img_1_ave_pool = sess.run(self.avepool_out,feed_dict={self.tf_input:np.expand_dims(img_1,axis=0)})
#             img_2_ave_pool = sess.run(self.avepool_out,feed_dict={self.tf_input:np.expand_dims(img_2,axis=0)})
#             # print("img_1 shape = {}, dtype = {}".format(img_1.shape, img_1.dtype))
#             # print("img_2 shape = {}, dtype = {}".format(img_2.shape, img_2.dtype))
#             img_diff = cv2.absdiff(img_1_ave_pool[0], img_2_ave_pool[0])  # img_diff = np.abs(img - img_sess)#不能使用，因為相減若<0，會取補數
#             if img_1.shape[-1] == 3:
#                 img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)  # 利用轉成灰階，減少RGB的相加計算量
#
#             # 連通
#             img_compare = cv2.compare(img_diff, diff_th, cv2.CMP_GT)
#             retval, labels = cv2.connectedComponents(img_compare)
#             max_label_num = np.max(labels) + 1
#
#             img_1_copy = img_1.copy()
#             for i in range(0, max_label_num):  # label = 0是背景，所以從1開始
#                 y, x = np.where(labels == i)  # y,x代表類別=i的像素座標值
#                 if (y.shape[0] > cc_th) and (img_compare[y[0], x[0]] == 255):
#                     for j in range(y.shape[0]):
#                         img_1_copy.itemset((y[j], x[j], 0), 0)
#                         img_1_copy.itemset((y[j], x[j], 1), 0)
#                         img_1_copy.itemset((y[j], x[j], 2), 255)
#
#             re = img_1_copy
#             return re
#
#     def __img_diff_method(self,img_source_1, img_source_2, diff_th=30, cc_th=30):
#
#         temp = np.array([1., 2., 3.])
#         re = None
#         # ----read img source 1
#         if isinstance(temp, type(img_source_1)):
#             img_1 = img_source_1
#         elif os.path.isfile(img_source_1):
#             # img_1 = cv2.imread(img_source_1)
#             img_1 = cv2.imdecode(np.fromfile(img_source_1, dtype=np.uint8), 1)
#             img_1 = cv2.resize(img_1,(self.model_shape[2],self.model_shape[1]))
#             # img_1 = img_1.astype('float32')
#
#         else:
#             print("The type of img_source_1 is not supported")
#
#         # ----read img source 2
#         if isinstance(temp, type(img_source_2)):
#             img_2 = img_source_2
#         elif os.path.isfile(img_source_2):
#             # img_2 = cv2.imread(img_source_2)
#             img_2 = cv2.imdecode(np.fromfile(img_source_2, dtype=np.uint8), 1)
#             # img_2 = img_2.astype('float32')
#         else:
#             print("The type of img_source_2 is not supported")
#
#         # ----substraction
#         if img_1 is not None and img_2 is not None:
#             # print("img_1 shape = {}, dtype = {}".format(img_1.shape, img_1.dtype))
#             # print("img_2 shape = {}, dtype = {}".format(img_2.shape, img_2.dtype))
#             img_diff = cv2.absdiff(img_1, img_2)  # img_diff = np.abs(img - img_sess)#不能使用，因為相減若<0，會取補數
#             if img_1.shape[-1] == 3:
#                 img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)  # 利用轉成灰階，減少RGB的相加計算量
#
#             # 連通
#             img_compare = cv2.compare(img_diff, diff_th, cv2.CMP_GT)
#             retval, labels = cv2.connectedComponents(img_compare)
#             max_label_num = np.max(labels) + 1
#
#             img_1_copy = img_1.copy()
#             for i in range(0, max_label_num):  # label = 0是背景，所以從1開始
#                 y, x = np.where(labels == i)  # y,x代表類別=i的像素座標值
#                 if (y.shape[0] > cc_th) and (img_compare[y[0], x[0]] == 255):
#                     for j in range(y.shape[0]):
#                         img_1_copy.itemset((y[j], x[j], 0), 0)
#                         img_1_copy.itemset((y[j], x[j], 1), 0)
#                         img_1_copy.itemset((y[j], x[j], 2), 255)
#
#             re = img_1_copy
#             return re
#
#     def __avepool(self,input_x,k_size=3,strides=1):
#         kernel = [1,k_size,k_size,1]
#         stride_kernel = [1,strides,strides,1]
#         return tf.nn.avg_pool(input_x, ksize=kernel, strides=stride_kernel, padding='SAME')
#
#     def __Conv(self,input_x,kernel=[3,3],filter=32,conv_times=2,stride=1):
#         net = None
#         for i in range(conv_times):
#             if i == 0:
#                 net = tf.layers.conv2d(
#                     inputs=input_x,
#                     filters=filter,
#                     kernel_size=kernel,
#                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#                     strides=stride,
#                     padding="same",
#                     activation=tf.nn.relu)
#             else:
#                 net = tf.layers.conv2d(
#                     inputs=net,
#                     filters=filter,
#                     kernel_size=kernel,
#                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#                     strides=stride,
#                     padding="same",
#                     activation=tf.nn.relu)
#         return net
#
#     def say_sth(self,msg, print_out=False):
#         if print_out:
#             print(msg)
#
#     def log_update(self,content,para_dict):
#         for key, value in para_dict.items():
#             content[key] = value
#
#         return content
#
#     def dict_update(self,main_content,add_content):
#         for key, value in add_content.items():
#             main_content[key] = value
#
#     def __img_read(self, img_path, shape,dtype='float32'):
#
#         # img = cv2.imread(img_path)
#         img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
#         if img is None:
#             print("Read failed:",img_path)
#             return None
#         else:
#             img = cv2.resize(img,(shape[1],shape[0]))
#             img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#             img = img.astype(dtype)
#             img /= 255
#
#             return np.expand_dims(img,axis=0)
#
#     def __get_result_value(self,record_type,train_dict,test_dict):
#         value = None
#         if record_type == 'loss':
#             if self.test_img_dir is not None:
#                 value = test_dict['loss']
#             else:
#                 value = train_dict['loss']
#
#             if self.loss_method == 'ssim':
#                 value = np.round(value * 100, 2)
#             else:
#                 value = np.round(value, 2)
#
#         return value
#
#     def collect_data2ui(self,epoch,train_dict,test_dict):
#         #----var
#         msg_list = list()
#         header_list = list()
#         #----process
#         msg_list.append('{},{}'.format(epoch, train_dict['loss']))
#         header_list.append('train_loss')
#         # msg_list.append('{},{}'.format(epoch, train_dict['accuracy']))
#         # header_list.append('train_acc')
#         if self.test_img_dir is not None:
#             msg_list.append('{},{}'.format(epoch, test_dict['loss']))
#             header_list.append('val_loss')
#             # msg_list.append('{},{}'.format(epoch, test_dict['accuracy']))
#             # header_list.append('val_acc')
#
#
#         return msg_list,header_list
#
#     def transmit_data2ui(self,epoch,print_out,**kwargs):
#         #----var
#         msg_list = list()
#         header_list = list()
#
#         for header,values in kwargs.items():
#             msg = ""
#             if isinstance(values,list) or isinstance(values,np.ndarray):
#                 leng = len(values)
#                 for i,value in enumerate(values):
#                     str_value = str(value)
#                     splits = str_value.split(".")
#
#                     if len(splits[-1]) > 8:
#                         msg += "{}.{}".format(splits[0],splits[-1][:8])
#                     else:
#                         msg += str_value
#
#
#
#                     # if value % 1.0 == 0:
#                     #     msg += str(value)
#                     # else:
#                     #     msg += "{:.8f}".format(value)
#
#                     if (i+1) < leng:
#                         msg += ','
#             elif isinstance(values,float):
#                 msg = "{:.8f}".format(values)
#             else:
#                 msg = str(values)
#
#             msg_list.append('{},{}'.format(epoch, msg))
#             header_list.append(header)
#
#         say_sth(msg_list, print_out=print_out, header=header_list)
#
#
#     # def display_iou_acc(self,iou,acc,defect_recall,all_acc,id2name,name='',print_out=False):
#     # def display_results(self,id2name,print_out,dataset_name,**kwargs):
#     #     msg_list = []
#     #     class_names = list(id2name.values())
#     #     #a_dict = {'iou':iou, 'acc':acc,'defect_recall':defect_recall}
#     #     for key,value_list in kwargs.items():
#     #         if key == 'all_acc':
#     #             msg_list.append("Seg{}_all_acc: {}".format(dataset_name,value_list))
#     #         else:
#     #             msg_list.append("Seg{}_{}:".format(dataset_name,key))
#     #             msg_list.append("{}:".format(class_names))
#     #             msg_list.append("{}:".format(value_list))
#     #
#     #         # for i,value in enumerate(value_list):
#     #         #     msg_list.append(" {}: {}".format(id2name[i],value))
#     #
#     #
#     #
#     #     for msg in msg_list:
#     #         say_sth(msg,print_out=print_out)
#     def display_results(self,class_names,print_out,**kwargs):
#         msg_list = []
#         for key,value_list in kwargs.items():
#                 msg_list.append(f"{key}:")
#                 if class_names is not None:
#                     msg_list.append("{}:".format(class_names))
#                 msg_list.append("{}:".format(value_list))
#         for msg in msg_list:
#             say_sth(msg,print_out=print_out)
#
#     #----models
#
#     def __AE_transpose_4layer_test(self, input_x, kernel_list, filter_list,conv_time=1,maxpool_kernel=2):
#         #----var
#         maxpool_kernel = [1,maxpool_kernel,maxpool_kernel,1]
#         transpose_filter = [1, 1]
#
#         msg = '----AE_transpose_4layer_struct_2----'
#         self.say_sth(msg, print_out=self.print_out)
#
#         net = self.__Conv(input_x, kernel=kernel_list[0], filter=filter_list[0], conv_times=conv_time)
#         U_1_point = net
#         #net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
#         net = tf.layers.max_pooling2d(net,pool_size=[2,2],strides=2,padding='SAME')
#
#         msg = "encode_1 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#         net = self.__Conv(net, kernel=kernel_list[1], filter=filter_list[1], conv_times=conv_time)
#         U_2_point = net
#         # net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
#         net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
#         msg = "encode_2 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#
#         net = self.__Conv(net, kernel=kernel_list[2], filter=filter_list[2], conv_times=conv_time)
#         U_3_point = net
#         # net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
#         net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
#
#         msg = "encode_3 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#         net = self.__Conv(net, kernel=kernel_list[3], filter=filter_list[3], conv_times=conv_time)
#         U_4_point = net
#         # net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
#         net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
#
#         msg = "encode_4 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#
#         net = self.__Conv(net, kernel=kernel_list[4], filter=filter_list[4], conv_times=conv_time)
#
#
#         flatten = tf.layers.flatten(net)
#
#         embeddings = tf.nn.l2_normalize(flatten, 1, 1e-10, name='embeddings')
#         print("embeddings shape:",embeddings.shape)
#         # net = tf.layers.dense(inputs=prelogits, units=units, activation=None)
#         # print("net shape:",net.shape)
#         # net = tf.reshape(net,shape)
#         # -----------------------------------------------------------------------
#         # --------Decode--------
#         # -----------------------------------------------------------------------
#
#         # data= 4 x 4 x 64
#
#         net = tf.layers.conv2d_transpose(net, filter_list[3], transpose_filter, strides=2, padding='same')
#         #net = tf.concat([net, U_4_point], axis=3)
#         net = self.__Conv(net, kernel=kernel_list[3], filter=filter_list[3], conv_times=conv_time)
#         msg = "decode_1 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#         # data= 8 x 8 x 64
#         net = tf.layers.conv2d_transpose(net, filter_list[2], transpose_filter, strides=2, padding='same')
#         # net = tf.concat([net, U_3_point], axis=3)
#         net = self.__Conv(net, kernel=kernel_list[2], filter=filter_list[2], conv_times=conv_time)
#         msg = "decode_2 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#
#         net = tf.layers.conv2d_transpose(net, filter_list[1], transpose_filter, strides=2, padding='same')
#         # net = tf.concat([net, U_2_point], axis=3)
#         net = self.__Conv(net, kernel=kernel_list[1], filter=filter_list[1], conv_times=conv_time)
#         msg = "decode_3 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#         # data= 32 x 32 x 64
#
#         net = tf.layers.conv2d_transpose(net, filter_list[0], transpose_filter, strides=2, padding='same')
#         # net = tf.concat([net, U_1_point], axis=3)
#         net = self.__Conv(net, kernel=kernel_list[0], filter=filter_list[0], conv_times=conv_time)
#         msg = "decode_2 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#
#         net = tf.layers.conv2d(
#             inputs=net,
#             filters=3,
#             kernel_size=kernel_list[0],
#             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#             padding="same",
#             activation=tf.nn.relu,
#             name='output_AE')
#         msg = "output shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#         # data= 64 x 64 x 3
#         return net

# class AE_Seg_v2():
#     def __init__(self,para_dict):
#         #----use previous settings
#         if para_dict.get('use_previous_settings'):
#             dict_t = AE_Seg_Util.get_latest_json_content(para_dict['save_dir'])
#             self.para_dict = None
#             if isinstance(dict_t,dict):
#                 msg_list = list()
#                 msg_list.append("Use previous settings")
#                 #----exclude dict process
#                 update_dict = para_dict.get('update_dict')
#                 if isinstance(update_dict,dict):
#                     for key, value in update_dict.items():
#                         if value is None:
#                             dict_t[key] = para_dict[key]
#                             msg = '{} update to {}'.format(key,para_dict[key])
#                             msg_list.append(msg)
#                         else:
#                             dict_t[key][value] = para_dict[key][value]
#                             msg = '{}_{} update to {}'.format(key,value,para_dict[key][value])
#                             msg_list.append(msg)
#
#                 self.para_dict = dict_t
#
#                 for msg in msg_list:
#                     say_sth(msg, print_out=True)
#
#         # ----common var
#         show_data_qty = para_dict.get('show_data_qty')
#         print_out = para_dict.get('print_out')
#
#         # ----local var
#         recon_flag = False
#         msg_list = list()
#         to_train_ae = False
#         to_train_seg = False
#
#         # ----tools class init
#         tl = AE_Seg_Util.tools(print_out=print_out)
#
#         #----AE process
#         ae_var = para_dict.get('ae_var')
#         if isinstance(ae_var,dict) is True:
#             #----config check(if exe)
#             status = True
#             # status, para_dict = self.config_check(para_dict)
#             if status:
#                 train_img_dir = ae_var.get('train_img_dir')
#                 test_img_dir = ae_var.get('test_img_dir')
#                 recon_img_dir = ae_var.get('recon_img_dir')
#                 special_img_dir = ae_var.get('special_img_dir')
#
#                 # ----AE image path process
#                 if train_img_dir is None:
#                     msg_list.append('沒有輸入AE訓練集')
#                 else:
#                     self.train_paths, self.train_path_qty = tl.get_paths(train_img_dir)
#
#                     if self.train_path_qty == 0:
#                         say_sth("Error:AE訓練資料集沒有圖片", print_out=print_out)
#                     else:
#                         to_train_ae = True
#                         msg = "AE訓練集圖片數量:{}".format(self.train_path_qty)
#                         msg_list.append(msg)
#
#                         # ----test image paths
#                         if test_img_dir is None:
#                             msg = "沒有輸入AE驗證集路徑"
#                         else:
#                             self.test_paths, self.test_path_qty = tl.get_paths(test_img_dir)
#                             msg = "AE驗證集圖片數量:{}".format(self.test_path_qty)
#                         msg_list.append(msg)
#                         # say_sth(msg,print_out=print_out)
#
#                         # ----special image paths
#                         if special_img_dir is None:
#                             msg = "沒有輸入加強AE學習集路徑"
#                         else:
#                             self.sp_paths, self.sp_path_qty = tl.get_paths(special_img_dir)
#                             msg = "AE加強學習圖片數量:{}".format(self.sp_path_qty)
#                         msg_list.append(msg)
#                         # say_sth(msg, print_out=print_out)
#
#                         # ----recon image paths
#                         if recon_img_dir is None:
#                             msg = "沒有輸入AE重建圖集路徑"
#                         else:
#                             self.recon_paths, self.recon_path_qty = tl.get_paths(recon_img_dir)
#                             if self.recon_path_qty > 0:
#                                 recon_flag = True
#                             msg = "AE重建圖片數量:{}".format(self.recon_path_qty)
#                         msg_list.append(msg)
#
#         #----SEG process
#         seg_var = para_dict.get('seg_var')
#         if isinstance(seg_var,dict) is True:
#             #----SEG config check(if exe)
#             status = True
#             # status, para_dict = self.config_check(para_dict)
#             if status:
#                 # ----SEG
#                 seg_var = para_dict.get('seg_var')
#                 train_img_seg_dir = seg_var.get('train_img_seg_dir')
#                 test_img_seg_dir = seg_var.get('test_img_seg_dir')
#                 predict_img_dir = seg_var.get('predict_img_dir')
#                 json_check = seg_var.get('json_check')
#                 to_train_w_AE_paths = seg_var.get('to_train_w_AE_paths')
#                 id2class_name = seg_var.get('id2class_name')
#                 select_OK_ratio = 0.2
#
#
#                 # ----SEG train image path process
#                 if train_img_seg_dir is None:
#                     msg_list.append('沒有輸入SEG訓練集')
#                 else:
#                     if json_check:
#                         self.seg_train_paths, self.seg_train_json_paths, self.seg_train_qty = tl.get_subdir_paths_withJsonCheck(
#                             train_img_seg_dir)
#                     else:
#                         self.seg_train_paths, self.seg_train_qty = tl.get_paths(train_img_seg_dir)
#
#                     msg = "SEG訓練集圖片數量:{}".format(self.seg_train_qty)
#                     msg_list.append(msg)
#
#                     # ----Seg img path qty check
#                     if self.seg_train_qty > 0:
#                         to_train_seg = True
#
#                         #----read class names
#                         if isinstance(id2class_name,dict):
#                             id2class_name = AE_Seg_Util.dict_transform(id2class_name,set_key=True)
#                             source = id2class_name
#                         elif os.path.isfile(id2class_name):
#                             source = id2class_name
#                         else:
#                             source = os.path.dirname(train_img_seg_dir[0])
#
#                         class_names, class_name2id, id2class_name, id2color,_ = AE_Seg_Util.get_classname_id_color(source, print_out=print_out)
#
#                         class_num = len(class_names)
#                         if class_num == 0:
#                             say_sth("Error:沒有取到SEG類別數目等資料，無法進行SEG訓練", print_out=True)
#                             to_train_seg = False
#                             to_train_ae = False
#                         else:
#                             #----train with AE ok images
#                             if to_train_w_AE_paths:
#                                 if to_train_ae:
#                                     if json_check is not True:
#                                         select_num = np.minimum(self.train_path_qty,int(self.seg_train_qty * select_OK_ratio))
#                                         temp_paths = np.random.choice(self.train_paths,size=select_num,replace=False)
#                                         self.seg_train_paths = list(self.seg_train_paths)
#                                         self.seg_train_paths.extend(temp_paths)
#                                         self.seg_train_paths = np.array(self.seg_train_paths)
#                                         self.seg_train_qty += select_num
#                                         msg = "to_train_w_AE_paths，SEG訓練集圖片數量:{}，實際為{}".\
#                                             format(self.seg_train_qty,self.seg_train_paths.shape)
#                                         msg_list.append(msg)
#
#                             #----check test images if test seg qty > 0
#                             if test_img_seg_dir is None:
#                                 self.seg_test_qty = 0
#                                 msg_list.append('沒有輸入SEG驗證集')
#                             else:
#                                 if json_check:
#                                     self.seg_test_paths, self.seg_test_json_paths, self.seg_test_qty = tl.get_subdir_paths_withJsonCheck(
#                                         test_img_seg_dir)
#                                 else:
#                                     self.seg_test_paths, self.seg_test_qty = tl.get_paths(test_img_seg_dir)
#                                 msg = "SEG驗證集圖片數量:{}".format(self.seg_test_qty)
#                                 msg_list.append(msg)
#
#                             #----check predict images if predict qty > 0
#                             if predict_img_dir is None:
#                                 self.seg_predict_qty = 0
#                                 msg_list.append('沒有輸入SEG預測集')
#                             else:
#                                 self.seg_predict_paths, self.seg_predict_qty = tl.get_paths(predict_img_dir)
#                                 msg = "SEG預測集圖片數量:{}".format(self.seg_predict_qty)
#                                 msg_list.append(msg)
#
#         #----status dicision
#         if to_train_seg or to_train_ae:
#             status = True
#         else:
#             status = False
#
#         #----display data info
#         if show_data_qty is True:
#             for msg in msg_list:
#                 say_sth(msg, print_out=print_out)
#
#         #----log update
#         content = dict()
#         content = self.log_update(content, para_dict)
#         #====record id, classname, and color
#         if to_train_seg:
#             content['class_names'] = class_names
#             content['class_name2id'] = class_name2id
#             content['id2class_name'] = id2class_name
#             content['id2color'] = id2color
#
#         #----local var to global
#         self.to_train_ae = to_train_ae
#         self.to_train_seg = to_train_seg
#         self.status = status
#         self.content = content
#         if to_train_ae:
#             self.train_img_dir = train_img_dir
#             self.test_img_dir = test_img_dir
#             self.special_img_dir = special_img_dir
#             self.recon_img_dir = recon_img_dir
#             self.recon_flag = recon_flag
#         if to_train_seg:
#             self.train_img_seg_dir = train_img_seg_dir
#             self.test_img_seg_dir = test_img_seg_dir
#             self.predict_img_dir = predict_img_dir
#             self.class_num = class_num
#             self.class_names = class_names
#             self.class_name2id = class_name2id
#             self.id2class_name = id2class_name
#             self.id2color = id2color
#
#     def model_init(self,para_dict):
#         #----var parsing
#
#         # ----use previous settings
#         if para_dict.get('use_previous_settings'):
#             if isinstance(self.para_dict,dict):
#                 para_dict = self.para_dict
#         #----AE
#         if self.to_train_ae:
#             ae_var = para_dict['ae_var']
#             model_shape = ae_var.get('model_shape')  # [N,H,W,C]
#             infer_method = ae_var['infer_method']
#             acti = ae_var['activation']
#             pool_kernel = ae_var['pool_kernel']
#             kernel_list = ae_var['kernel_list']
#             filter_list = ae_var['filter_list']
#             conv_time = ae_var['conv_time']
#             pool_type = ae_var.get('pool_type')
#             loss_method = ae_var['loss_method']
#             opti_method = ae_var['opti_method']
#             embed_length = ae_var['embed_length']
#             stride_list = ae_var.get('stride_list')
#             scaler = ae_var.get('scaler')
#             process_dict = ae_var['process_dict']
#
#             rot = ae_var.get('rot')
#
#         #----SEG
#         if self.to_train_seg:
#             seg_var = para_dict['seg_var']
#             infer_method4Seg = seg_var.get('infer_method')
#             pool_kernel4Seg = seg_var['pool_kernel']
#             pool_type4Seg = seg_var.get('pool_type')
#             kernel_list4Seg = seg_var['kernel_list']
#             filter_list4Seg = seg_var['filter_list']
#             loss_method4Seg = seg_var.get('loss_method')
#             opti_method4Seg = seg_var.get('opti_method')
#             preprocess_dict4Seg = seg_var.get('preprocess_dict')
#             rot4Seg = seg_var.get('rot')
#             acti_seg = seg_var.get('activation')
#             if model_shape is None:
#                 model_shape = seg_var['model_shape']
#
#         #----common var
#         preprocess_dict = para_dict.get('preprocess_dict')
#         lr = para_dict['learning_rate']
#         save_dir = para_dict['save_dir']
#         save_pb_name = para_dict.get('save_pb_name')
#         encript_flag = para_dict.get('encript_flag')
#         print_out = para_dict.get('print_out')
#         add_name_tail = para_dict.get('add_name_tail')
#         dtype = para_dict.get('dtype')
#
#
#         #----var
#         #rot = False
#         # bias = 0.5
#         # br_ratio = 0
#         # ct_ratio = 1
#         pb_extension = 'pb'
#         log_extension = 'json'
#         acti_dict = {'relu': tf.nn.relu, 'mish': models_AE_Seg.tf_mish, None: tf.nn.relu}
#         pb_save_list = list()
#
#         #----var process
#         if encript_flag is True:
#             pb_extension = 'nst'
#             log_extension = 'nst'
#         else:
#             pb_extension = 'pb'
#             log_extension = 'json'
#
#         if add_name_tail is None:
#             add_name_tail = True
#         if dtype is None:
#             dtype = 'float32'
#
#
#         # ----tf_placeholder declaration
#         tf_input = tf.placeholder(dtype, shape=model_shape, name='input')
#         tf_keep_prob = tf.placeholder(dtype=dtype, name="keep_prob")
#
#
#         if self.to_train_ae:
#             special_process_list = ['rdm_patch', 'rdm_perlin', 'rdm_light_defect']
#             return_ori_data = False
#             #----if return ori data or not
#
#             for name in special_process_list:
#                 if process_dict.get(name) is True:
#                     return_ori_data = True
#                     break
#             #----random patch
#             # rdm_patch = False
#             # if process_dict.get('rdm_patch') is True:
#             #     rdm_patch = True
#
#             #----filer scaling process
#             if scaler is not None:
#                 filter_list = (np.array(filter_list) / scaler).astype(np.uint16)
#
#             if return_ori_data is True:
#                 self.tf_input_ori = tf.placeholder(dtype, shape=model_shape, name='input_ori')
#
#             #tf_label_batch = tf.placeholder(dtype=tf.int32, shape=[None], name="label_batch")
#             #tf_phase_train = tf.placeholder(tf.bool, name="phase_train")
#
#             # ----activation selection
#             acti_func = acti_dict[acti]
#
#
#             avepool_out = self.__avepool(tf_input, k_size=5, strides=1)
#             #----preprocess
#             if preprocess_dict is None:
#                 tf_input_process = tf.identity(tf_input,name='preprocess')
#                 if return_ori_data is True:
#                     tf_input_ori_no_patch = tf.identity(self.tf_input_ori, name='tf_input_ori_no_patch')
#
#             else:
#                 tf_temp = models_AE_Seg.preprocess(tf_input, preprocess_dict, print_out=print_out)
#                 tf_input_process = tf.identity(tf_temp,name='preprocess')
#
#                 if return_ori_data is True:
#                     tf_temp_2 = models_AE_Seg.preprocess(self.tf_input_ori, preprocess_dict, print_out=print_out)
#                     tf_input_ori_no_patch = tf.identity(tf_temp_2, name='tf_input_ori_no_patch')
#
#
#             #----AE inference selection
#             if infer_method == "AE_transpose_4layer":
#                 recon = models_AE_Seg.AE_transpose_4layer(tf_input, kernel_list, filter_list,activation=acti_func,
#                                                   pool_kernel=pool_kernel,pool_type=pool_type)
#                 recon = tf.identity(recon, name='output_AE')
#                 #(tf_input, kernel_list, filter_list,pool_kernel=2,activation=tf.nn.relu,pool_type=None)
#                 # recon = AE_refinement(temp,96)
#             elif infer_method.find('mit') >= 0:
#                 cfg = config_mit.config[infer_method.split("_")[-1]]
#                 model = MixVisionTransformer(
#                     embed_dims=cfg['embed_dims'],
#                     num_stages=cfg['num_stages'],
#                     num_layers=cfg['num_layers'],
#                     num_heads=cfg['num_heads'],
#                     patch_sizes=cfg['patch_sizes'],
#                     strides=cfg['strides'],
#                     sr_ratios=cfg['sr_ratios'],
#                     mlp_ratio=cfg['mlp_ratio'],
#                     ffn_dropout_keep_ratio=1.0,
#                     dropout_keep_rate=1.0)
#                 # model.init_weights()
#                 outs = model(tf_input_process)
#                 # print("outs shape:",outs.shape)
#                 mitDec = MiTDecoder(channels=cfg['num_heads'][-1] * cfg['embed_dims'],
#                                     dropout_ratio=0, num_classes=3)
#                 logits = mitDec(outs)
#                 recon = tf.identity(logits, name='output_AE')
#             elif infer_method == "AE_pooling_net_V3":
#                 recon = models_AE_Seg.AE_pooling_net_V3(tf_input_process, kernel_list, filter_list, activation=acti_func,
#                                           pool_kernel_list=pool_kernel, pool_type_list=pool_type,
#                                           stride_list=stride_list, print_out=print_out)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_pooling_net_V4":
#                 recon = models_AE_Seg.AE_pooling_net_V4(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
#                                           to_reduce=ae_var.get('to_reduce'),print_out=print_out)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_pooling_net_V5":
#                 recon = models_AE_Seg.AE_pooling_net_V5(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
#                                           to_reduce=ae_var.get('to_reduce'),print_out=print_out)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_pooling_net_V6":
#                 recon = models_AE_Seg.AE_pooling_net_V6(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
#                                           to_reduce=ae_var.get('to_reduce'),print_out=print_out)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_pooling_net_V7":
#
#                 recon = models_AE_Seg.AE_pooling_net_V7(tf_input_process,ae_var['encode_dict'],ae_var['decode_dict'],
#                                          print_out=print_out)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_pooling_net_V8":
#
#                 self.tf_input_standard = tf.placeholder(dtype, shape=model_shape, name='input_standard')
#                 recon = models_AE_Seg.AE_pooling_net_V8(tf_input_process,self.tf_input_standard,ae_var['encode_dict'],ae_var['decode_dict'],
#                                          print_out=print_out)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_dense_sampling":
#                 sampling_factor = 16
#                 filters = 2
#                 recon = models_AE_Seg.AE_dense_sampling(tf_input_process,sampling_factor,filters)
#                 recon = tf.identity(recon, name='output_AE')
#             elif infer_method == "AE_pooling_net":
#
#                 recon = models_AE_Seg.AE_pooling_net(tf_input_process, kernel_list, filter_list,activation=acti_func,
#                                       pool_kernel_list=pool_kernel,pool_type_list=pool_type,
#                                       stride_list=stride_list,rot=rot,print_out=print_out)
#                 recon = tf.identity(recon,name='output_AE')
#             elif infer_method == "AE_Seg_pooling_net":
#                 AE_out,Seg_out = models_AE_Seg.AE_Seg_pooling_net(tf_input, kernel_list, filter_list,activation=acti_func,
#                                                   pool_kernel_list=pool_kernel,pool_type_list=pool_type,
#                                        rot=rot,print_out=print_out,preprocess_dict=preprocess_dict,
#                                            class_num=self.class_num)
#                 recon = tf.identity(AE_out,name='output_AE')
#             elif infer_method == "AE_Unet":
#                 recon = models_AE_Seg.AE_Unet(tf_input, kernel_list, filter_list,activation=acti_func,
#                                                   pool_kernel_list=pool_kernel,pool_type_list=pool_type,
#                                        rot=rot,print_out=print_out,preprocess_dict=preprocess_dict)
#                 recon = tf.identity(recon,name='output_AE')
#             elif infer_method == "AE_JNet":
#                 recon = models_AE_Seg.AE_JNet(tf_input, kernel_list, filter_list,activation=acti_func,
#                                                 rot=rot,pool_kernel=pool_kernel,pool_type=pool_type)
#             # elif infer_method == "test":
#             #     recon = self.__AE_transpose_4layer_test(tf_input, kernel_list, filter_list,
#             #                                        conv_time=conv_time,maxpool_kernel=maxpool_kernel)
#             # elif infer_method == 'inception_resnet_v1_reduction':
#             #     recon = AE_incep_resnet_v1(tf_input=tf_input,tf_keep_prob=tf_keep_prob,embed_length=embed_length,
#             #                                scaler=scaler,kernel_list=kernel_list,filter_list=filter_list,
#             #                                activation=acti_func,)
#             elif infer_method == "Resnet_Rot":
#                 filter_list = [12, 16, 24, 36, 48, 196]
#                 recon = models_AE_Seg.AE_Resnet_Rot(tf_input,filter_list,tf_keep_prob,embed_length,activation=acti_func,
#                                       print_out=True,rot=True)
#
#             # ----AE loss method selection
#             if loss_method == 'mse':
#                 loss_AE = tf.reduce_mean(tf.pow(recon - tf_input, 2), name="loss_AE")
#             elif loss_method == 'ssim':
#                 # self.loss_AE = tf.reduce_mean(tf.image.ssim_multiscale(tf.image.rgb_to_grayscale(self.tf_input),tf.image.rgb_to_grayscale(self.recon),2),name='loss')
#
#                 if return_ori_data is True:
#                     loss_AE = tf.reduce_mean(tf.image.ssim(tf_input_ori_no_patch, recon, 2.0), name='loss_AE')
#                 else:
#                     loss_AE = tf.reduce_mean(tf.image.ssim(tf_input_process, recon, 2.0), name='loss_AE')
#
#             elif loss_method == "huber":
#                 loss_AE = tf.reduce_sum(tf.losses.huber_loss(tf_input, recon, delta=1.35), name='loss_AE')
#             elif loss_method == 'ssim+mse':
#                 loss_1 = tf.reduce_mean(tf.pow(recon - tf_input, 2))
#                 loss_2 = tf.reduce_mean(tf.image.ssim(tf_input, recon, 2.0))
#                 loss_AE = tf.subtract(loss_2, loss_1, name='loss_AE')
#             elif loss_method == 'cross_entropy':
#                 loss_AE = tf.reduce_mean(
#                     tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.layers.flatten(tf_input),
#                                                                logits=tf.layers.flatten(recon)), name="loss_AE")
#             elif loss_method == 'kl_d':
#                 epsilon = 1e-8
#                 # generation loss(cross entropy)
#                 loss_AE = tf.reduce_mean(
#                     tf_input * tf.subtract(tf.log(epsilon + tf_input), tf.log(epsilon + recon)), name='loss_AE')
#
#             # ----AE optimizer selection
#             if opti_method == "adam":
#                 if loss_method.find('ssim') >= 0:
#                     opt_AE = tf.train.AdamOptimizer(learning_rate=lr).minimize(-loss_AE)
#                 else:
#                     opt_AE = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_AE)
#
#             if self.recon_flag is True:
#                 new_dir_name = "img_recon-" + os.path.basename(save_dir)
#                 # ----if recon_img_dir is list format
#                 if isinstance(self.recon_img_dir, list):
#                     dir_path = self.recon_img_dir[0]
#                 else:
#                     dir_path = self.recon_img_dir
#                 self.new_recon_dir = os.path.join(dir_path, new_dir_name)
#                 if not os.path.exists(self.new_recon_dir):
#                     os.makedirs(self.new_recon_dir)
#
#             # ----appoint PB node names
#             pb_save_list.extend(['output_AE', "loss_AE"])
#
#             #----pb filename for AE model
#             if isinstance(save_pb_name,str):
#                 filename = save_pb_name + "_ae"
#             else:
#                 filename = 'pb_model_ae'
#             pb4ae_save_path = create_pb_filename(filename, pb_extension, save_dir, add_name_tail=add_name_tail)
#
#         #----Seg inference selection
#         if self.to_train_seg:
#             tf_input_recon = tf.placeholder(dtype, shape=model_shape, name='input_recon')
#             tf_label_batch = tf.placeholder(tf.int32, shape=model_shape[:-1], name='label_batch')
#             tf_dropout = tf.placeholder(dtype=tf.float32, name="dropout")
#
#             # ----activation selection
#             acti_func = acti_dict[acti_seg]
#
#             #----filer scaling process
#             filter_list4Seg = np.array(filter_list4Seg)
#             if seg_var.get('scaler') is not None:
#                 filter_list4Seg /= seg_var.get('scaler')
#                 filter_list4Seg = filter_list4Seg.astype(np.uint16)
#
#             if infer_method4Seg == "Seg_DifNet":
#                 logits_Seg = models_AE_Seg.Seg_DifNet(tf_input_process,tf_input_recon, kernel_list4Seg, filter_list4Seg,activation=acti_func,
#                                    pool_kernel_list=pool_kernel4Seg,pool_type_list=pool_type4Seg,
#                                    rot=rot4Seg,print_out=print_out,preprocess_dict=preprocess_dict4Seg,class_num=self.class_num)
#                 softmax_Seg = tf.nn.softmax(logits_Seg,name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'Seg_DifNet_V2':
#                 logits_Seg = models_AE_Seg.Seg_DifNet_V2(tf_input_process,tf_input_recon,seg_var['encode_dict'],seg_var['decode_dict'],
#                               class_num=self.class_num,print_out=print_out)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg.find('mit') >= 0:
#                 cfg = config_mit.config[infer_method4Seg.split("_")[-1]]
#                 model = MixVisionTransformer(
#                     embed_dims=cfg['embed_dims'],
#                     num_stages=cfg['num_stages'],
#                     num_layers=cfg['num_layers'],
#                     num_heads=cfg['num_heads'],
#                     patch_sizes=cfg['patch_sizes'],
#                     strides=cfg['strides'],
#                     sr_ratios=cfg['sr_ratios'],
#                     mlp_ratio=cfg['mlp_ratio'],
#                     drop_rate=0,
#                     attn_drop_rate=0)
#                 # model.init_weights()
#                 outs = model(tf_input_process)
#                 # print("outs shape:",outs.shape)
#                 mitDec = MiTDecoder(channels=cfg['num_heads'][-1] * cfg['embed_dims'],
#                                     dropout_ratio=0, num_classes=self.class_num)
#                 logits_Seg = mitDec(outs)
#                 logits_Seg = tf.image.resize(logits_Seg,model_shape[1:-1])
#                 print("logits_Seg shape:", logits_Seg.shape)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'Seg_pooling_net_V4':
#                 logits_Seg = models_AE_Seg.Seg_pooling_net_V4(tf_input_process,tf_input_recon, seg_var['encode_dict'], seg_var['decode_dict'],
#                                           to_reduce=seg_var.get('to_reduce'),out_channel=self.class_num,
#                                           print_out=print_out)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'Seg_pooling_net_V7':
#                 logits_Seg = models_AE_Seg.Seg_pooling_net_V7(tf_input_process,tf_input_recon, seg_var['encode_dict'], seg_var['decode_dict'],
#                                           out_channel=self.class_num,
#                                           print_out=print_out)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'Seg_pooling_net_V8':
#                 logits_Seg = models_AE_Seg.Seg_pooling_net_V8(tf_input_process, tf_input_recon, seg_var['encode_dict'],
#                                                 seg_var['decode_dict'],
#                                                 to_reduce=seg_var.get('to_reduce'), out_channel=self.class_num,
#                                                 print_out=print_out)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'Seg_pooling_net_V9':
#                 logits_Seg = models_AE_Seg.Seg_pooling_net_V9(tf_input_process, tf_input_recon, seg_var['encode_dict'],
#                                                 seg_var['decode_dict'],
#                                                 to_reduce=seg_var.get('to_reduce'), out_channel=self.class_num,
#                                                 print_out=print_out)
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
#             elif infer_method4Seg == 'AE_Seg_pooling_net':
#                 logits_Seg = Seg_out
#                 softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
#                 prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
#                 prediction_Seg = tf.cast(prediction_Seg,tf.uint8, name='predict_Seg')
#
#             #----Seg loss method selection
#             if loss_method4Seg == "cross_entropy":
#                 loss_Seg = tf.reduce_mean(v2.nn.sparse_softmax_cross_entropy_with_logits(tf_label_batch,logits_Seg),name='loss_Seg')
#
#             #----Seg optimizer selection
#             if opti_method4Seg == "adam":
#                 opt_Seg = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_Seg)
#                 # var_list = []
#                 # for var in tf.global_variables():
#                 #     if var.trainable:
#                 #         var_list.append(var)
#                 # opt_Seg = v2.optimizers.Adam(learning_rate=lr).minimize(loss_Seg,var_list)
#             # elif opti_method4Seg == 'sgd':
#
#
#             # ----appoint PB node names
#             pb_save_list.extend(['predict_Seg'])
#             pb_save_list.extend(['dummy_out'])
#
#             #----pb filename for SEG model
#             if isinstance(save_pb_name,str):
#                 filename = save_pb_name + "_seg"
#             else:
#                 filename = "pb_model_seg"
#             pb4seg_save_path = create_pb_filename(filename, pb_extension, save_dir, add_name_tail=add_name_tail)
#
#         # ----pb filename(common)
#         if isinstance(save_pb_name, str):
#             filename = save_pb_name
#         else:
#             filename = "pb_model"
#         pb_save_path = create_pb_filename(filename, pb_extension, save_dir, add_name_tail=add_name_tail)
#
#         # ----create the dir to save model weights(CKPT, PB)
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#
#         if self.to_train_seg is True and self.seg_predict_qty > 0:
#             new_dir_name = "img_seg-" + os.path.basename(save_dir)
#             #----if recon_img_dir is list format
#             if isinstance(self.predict_img_dir,list):
#                 dir_path = self.predict_img_dir[0]
#             else:
#                 dir_path = self.predict_img_dir
#             self.new_predict_dir = os.path.join(dir_path, new_dir_name)
#             if not os.path.exists(self.new_predict_dir):
#                 os.makedirs(self.new_predict_dir)
#
#         out_dir_prefix = os.path.join(save_dir, "model")
#         saver = tf.train.Saver(max_to_keep=2)
#
#         # ----PB file save filename
#         # if save_pb_name is None:
#         #     save_pb_name = 'pb_model'
#         # if add_name_tail is True:
#         #     xtime = time.localtime()
#         #     name_tailer = ''
#         #     for i in range(6):
#         #         string = str(xtime[i])
#         #         if len(string) == 1:
#         #             string = '0' + string
#         #         name_tailer += string
#         #     pb_save_path = "{}_{}.{}".format(save_pb_name, name_tailer, pb_extension)
#         # else:
#         #     pb_save_path = "{}.{}".format(save_pb_name, pb_extension)
#         # pb_save_path = os.path.join(save_dir, pb_save_path)
#
#         # ----create the log(JSON)
#         count = 0
#         for i in range(1000):
#             log_path = "{}_{}.{}".format('train_result', count, log_extension)
#             log_path = os.path.join(save_dir, log_path)
#             if not os.path.exists(log_path):
#                 break
#             count += 1
#         self.content = self.log_update(self.content, para_dict)
#         self.content['pb_save_list'] = pb_save_list
#
#         # ----local var to global
#         self.model_shape = model_shape
#         self.tf_input = tf_input
#         self.tf_keep_prob = tf_keep_prob
#         self.saver = saver
#         self.save_dir = save_dir
#         self.pb_save_path = pb_save_path
#
#
#         self.pb_save_list = pb_save_list
#         self.pb_extension = pb_extension
#         self.log_path = log_path
#         self.dtype = dtype
#         if self.to_train_ae:
#             self.avepool_out = avepool_out
#             self.recon = recon
#             self.loss_AE = loss_AE
#             self.opt_AE = opt_AE
#             self.out_dir_prefix = out_dir_prefix
#             self.loss_method = loss_method
#             self.pb4ae_save_path = pb4ae_save_path
#             self.return_ori_data = return_ori_data
#             self.infer_method = infer_method
#
#         if self.to_train_seg is True:
#             self.tf_label_batch = tf_label_batch
#             self.tf_input_recon = tf_input_recon
#             self.tf_prediction_Seg = prediction_Seg
#             self.infer_method4Seg = infer_method4Seg
#             self.logits_Seg = logits_Seg
#             self.loss_Seg = loss_Seg
#             self.opt_Seg = opt_Seg
#             self.prediction_Seg = prediction_Seg
#             self.loss_method4Seg = loss_method4Seg
#             self.pb4seg_save_path = pb4seg_save_path
#             self.tf_dropout = tf_dropout
#             self.infer_method4Seg = infer_method4Seg
#
#     def train(self,para_dict):
#         # ----use previous settings
#         if para_dict.get('use_previous_settings'):
#             if isinstance(self.para_dict, dict):
#                 para_dict = self.para_dict
#         # ----var parsing
#         epochs = para_dict['epochs']
#         GPU_ratio = para_dict.get('GPU_ratio')
#         print_out = para_dict.get('print_out')
#         encode_header = para_dict.get('encode_header')
#         encode_num = para_dict.get('encode_num')
#         encript_flag = para_dict.get('encript_flag')
#         eval_epochs = para_dict.get('eval_epochs')
#         to_fix_ae = para_dict.get('to_fix_ae')
#         to_fix_seg = para_dict.get('to_fix_seg')
#
#         #----AE
#         if self.to_train_ae:
#             ae_var = para_dict['ae_var']
#             aug_times = ae_var.get('aug_times')
#             batch_size = ae_var['batch_size']
#             ratio = ae_var.get('ratio')
#             process_dict = ae_var.get('process_dict')
#             setting_dict = ae_var.get('setting_dict')
#             save_period = ae_var.get('save_period')
#             target_dict = ae_var.get('target')
#             pause_opt_ae = ae_var.get('pause_opt_ae')
#             img_standard_path = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220901_AOI判定OK\OK(多區OK)\train\0_-16_MatchLightSet_NoFailRegion_Ok_1.jpg"
#
#
#         #----SEG
#         if self.to_train_seg:
#             seg_var = para_dict['seg_var']
#             ratio_seg = seg_var.get('ratio_seg')
#             process_seg_dict = seg_var.get('process_dict')
#             setting_seg_dict = seg_var.get('setting_dict')
#             aug_seg_times = seg_var.get('aug_times')
#             json_check = seg_var.get('json_check')
#             batch_size_seg = seg_var['batch_size']
#             pause_opt_seg = seg_var.get('pause_opt_seg')
#             self_create_label = seg_var.get('self_create_label')
#             train_with_aug_v2 = seg_var.get('train_with_aug_v2')
#
#
#         #----special_img_dir
#         if self.special_img_dir is not None:
#             special_img_ratio = para_dict.get('special_img_ratio')
#             if special_img_ratio is None:
#                 special_img_ratio = 0.04
#             elif special_img_ratio > 1.0:
#                 special_img_ratio = 1.0
#             elif special_img_ratio < 0:
#                 special_img_ratio = 0.01
#
#         if encode_header is None:
#             encode_header = [24,97,28,98]
#         if encode_num is None:
#             encode_num = 87
#         if save_period is None:
#             save_period = 1
#
#         # ----local var
#         LV = dict()
#         train_loss_list = list()
#         seg_train_loss_list = list()
#         train_acc_list = list()
#         test_loss_list = list()
#         seg_test_loss_list = list()
#         test_acc_list = list()
#         epoch_time_list = list()
#         img_quantity = 0
#         aug_enable = False
#         break_flag = False
#         error_dict = {'GPU_resource_error': False}
#         train_result_dict = {'loss': 0,"loss_method":self.loss_method}
#         test_result_dict = {'loss': 0, "loss_method": self.loss_method}
#         if self.to_train_seg:
#             seg_train_result_dict = {'loss': 0,"loss_method":self.loss_method4Seg}
#             seg_test_result_dict = {'loss': 0,"loss_method":self.loss_method4Seg}
#
#         record_type = 'loss'
#         qty_sp = 0
#         LV['pb_save_path_old'] = ''
#         LV['record_value'] = 0
#         LV['pb_seg_save_path_old'] = ''
#         LV['record_value_seg'] = 0
#         keep_prob = 0.7
#
#         #----AE hyper-parameters
#         if self.to_train_ae:
#             tl = AE_Seg_Util.tools()
#             # ----set target
#             tl.set_target(target_dict)
#             batch_size_test = batch_size
#
#             #----check if the augmentation(image processing) is enabled
#             if isinstance(process_dict, dict):
#                 if True in process_dict.values():
#                     aug_enable = True
#                     tl.set_process(process_dict, setting_dict, print_out=print_out)
#
#                     if aug_times is None:
#                         aug_times = 2
#                     batch_size = batch_size // aug_times  # the batch size must be integer!!
#
#         #----SEG hyper-parameters
#         if self.to_train_seg:
#             tl_seg = AE_Seg_Util.tools()
#             tl_seg.class_name2id = self.class_name2id
#             tl_seg.id2color = self.id2color
#             val_pipelines = [
#                 dict(type='CvtColor', to_rgb=True),
#                 dict(type='RandomCrop', height=self.model_shape[1], width=self.model_shape[2]),
#                 dict(type='Norm')
#             ]
#
#             #get seg ok paths for Aug vivid defect
#             if train_with_aug_v2:
#                 ok_img_seg_dir = seg_var.get('ok_img_seg_dir')
#                 if ok_img_seg_dir is None:
#                     train_with_aug_v2 = False
#                 else:
#                     ok_img_seg_paths,ok_img_seg_qty = tl_seg.get_paths(ok_img_seg_dir)
#                     if ok_img_seg_qty == 0:
#                         train_with_aug_v2 = False
#
#
#             if self_create_label or train_with_aug_v2:
#                 tl_seg_v2 = AE_Seg_Util.tools_v2(pipelines=seg_var['train_pipelines'],print_out=print_out)
#                 tl_seg_v2.class_name2id = self.class_name2id
#                 tl_seg_v2.id2color = self.id2color
#
#
#             titles = ['prediction', 'answer']
#             batch_size_seg_test = batch_size_seg
#             seg_p = AE_Seg_Util.Seg_performance(len(self.class_name2id),print_out=print_out)
#
#             dataloader = AE_Seg_Util.DataLoader4Seg(self.train_paths, batch_size=batch_size_seg,
#                                         pipelines=seg_var['train_pipelines'],
#                                         to_shuffle=True)
#             dataloader_val = AE_Seg_Util.DataLoader4Seg(self.test_paths, batch_size=batch_size_seg,
#                                         pipelines=val_pipelines,
#                                         to_shuffle=True)
#             dataloader.get_classname_id_color(self.id2class_name, print_out=False)
#             dataloader_val.get_classname_id_color(self.id2class_name, print_out=False)
#
#             # ----check if the augmentation(image processing) is enabled
#             # if isinstance(process_seg_dict, dict):
#             #     if True in process_seg_dict.values():
#             #         aug_seg_enable = True
#             #         tl_seg.set_process(process_seg_dict, setting_seg_dict)
#             #         if aug_seg_times is None:
#             #             aug_seg_times = 2
#             #         batch_size_seg = batch_size_seg // aug_seg_times  # the batch size must be integer!!
#             #         if batch_size_seg < 1:
#             #             batch_size_seg = 1
#
#         #----update content
#         self.content = self.log_update(self.content, para_dict)
#
#         # ----read the manual cmd
#         if para_dict.get('to_read_manual_cmd') is True:
#             j_path = os.path.join(self.save_dir, 'manual_cmd.json')
#
#         # ----calculate iterations of one epoch
#         # train_ites = math.ceil(img_quantity / batch_size)
#         # test_ites = math.ceil(len(self.test_paths) / batch_size)
#
#         t_train_start = time.time()
#         #----GPU setting
#         config = GPU_setting(GPU_ratio)
#         with tf.Session(config=config) as sess:
#             status = weights_check(sess, self.saver, self.save_dir, encript_flag=encript_flag,
#                                    encode_num=encode_num, encode_header=encode_header)
#             if status is False:
#                 error_dict['GPU_resource_error'] = True
#             elif status is True:
#                 if self.to_train_ae is True:
#                     #----AE train set quantity
#                     qty_train = self.get_train_qty(self.train_path_qty,ratio,print_out=print_out,name='AE')
#
#
#                     # qty_train = self.train_path_qty
#                     # if ratio is not None:
#                     #     if ratio <= 1.0 and ratio > 0:
#                     #         qty_train = int(self.train_path_qty * ratio)
#                     #         qty_train = np.maximum(1, qty_train)
#                     #
#                     # msg = "AE訓練集資料總共取數量{}".format(qty_train)
#                     # say_sth(msg, print_out=print_out)
#
#                     #----special set quantity
#                     if self.special_img_dir is not None:
#                         qty_sp = int(qty_train * special_img_ratio)
#                         msg = "加強學習資料總共取數量{}".format(qty_sp)
#                         say_sth(msg, print_out=print_out)
#
#                     # ----calculate iterations of one epoch
#                     img_quantity = qty_train + qty_sp
#                     train_ites = math.ceil(img_quantity / batch_size)
#                     if self.test_img_dir is not None:
#                         test_ites = math.ceil(self.test_path_qty / batch_size_test)
#
#                 #----SEG
#                 #if self.to_train_seg is True:
#                     # ----SEG train set quantity
#                     # qty_train_seg = self.get_train_qty(self.seg_train_qty, ratio_seg, print_out=print_out,
#                     #                                    name='SEG')
#
#                     # ----calculate iterations of one epoch
#                     # train_ites_seg = math.ceil(self.seg_train_qty / batch_size_seg)
#                     # if self.seg_test_qty > 0:
#                     #     test_ites_seg = math.ceil(self.seg_test_qty / batch_size_seg_test)
#                     # if self.seg_predict_qty > 0:
#                     #     predict_ites_seg = math.ceil(self.seg_predict_qty / batch_size_seg_test)
#
#                 # ----epoch training
#                 for epoch in range(epochs):
#                     # ----read manual cmd
#                     if para_dict.get('to_read_manual_cmd') is True:
#                         if os.path.exists(j_path):
#                             with open(j_path, 'r') as f:
#                                 cmd_dict = json.load(f)
#                             if cmd_dict.get('to_stop_training') is True:
#                                 break_flag = True
#                                 msg = "接收到manual cmd: stop the training!"
#                                 say_sth(msg, print_out=print_out)
#                     # ----break the training
#                     if break_flag:
#                         break_flag = False
#                         break
#
#                     # ----error check
#                     if True in list(error_dict.values()):
#                         break
#                     # ----record the start time
#                     d_t = time.time()
#
#                     train_loss = 0
#
#                     train_acc = 0
#                     test_loss = 0
#                     test_loss_2 = 0
#                     test_acc = 0
#
#                     #----AE part
#                     if self.to_train_ae is True:
#                         if to_fix_ae is True:
#                             pass
#                         else:
#                             #tf_var_AE = tf.trainable_variables(scope='AE')
#                             #----shuffle
#                             indice = np.random.permutation(self.train_path_qty)
#                             self.train_paths = self.train_paths[indice]
#                             train_paths_ori = self.train_paths[:qty_train]
#
#                             #----special img dir
#                             if self.special_img_dir is not None:
#                                 #----shuffle for special set
#                                 indice = np.random.permutation(self.sp_path_qty)
#                                 self.sp_paths = self.sp_paths[indice]
#
#                                 if self.sp_path_qty < qty_sp:
#                                     multi_ratio = math.ceil(qty_sp / self.sp_path_qty)
#                                     sp_paths = np.array(list(self.sp_paths) * multi_ratio)
#                                 else:
#                                     sp_paths = self.sp_paths
#
#                                 train_paths_ori = np.concatenate([train_paths_ori, sp_paths[:qty_sp]], axis=-1)
#
#                                 #-----shuffle for (train set + special set)
#                                 indice = np.random.permutation(img_quantity)
#                                 train_paths_ori = train_paths_ori[indice]
#
#                             if aug_enable is True:
#                                 train_paths_aug = train_paths_ori[::-1]
#                                 #train_labels_aug = train_labels_ori[::-1]
#
#                             # ----optimizations(AE train set)
#                             for index in range(train_ites):
#                                 # ----command process
#                                 if SockConnected:
#                                     # print(Sock.Message)
#                                     if len(Sock.Message):
#                                         if Sock.Message[-1][:4] == "$S00":
#                                             # Sock.send("Protocol:Ack\n")
#                                             model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)
#                                             # ----encode ckpt file
#                                             if encript_flag is True:
#                                                 file = model_save_path + '.meta'
#                                                 if os.path.exists(file):
#                                                     file_transfer(file, random_num_range=encode_num, header=encode_header)
#                                                 else:
#                                                     msg = "Warning:找不到權重檔:{}進行處理".format(file)
#                                                     say_sth(msg, print_out=print_out)
#                                                 # data_file = [file.path for file in os.scandir(self.save_dir) if
#                                                 #              file.name.split(".")[-1] == 'data-00000-of-00001']
#                                                 data_file = model_save_path + '.data-00000-of-00001'
#                                                 if os.path.exists(data_file):
#                                                     file_transfer(data_file, random_num_range=encode_num, header=encode_header)
#                                                 else:
#                                                     msg = "Warning:找不到權重檔:{}進行處理".format(data_file)
#                                                     say_sth(msg, print_out=print_out)
#
#                                             # msg = "儲存訓練權重檔至{}".format(model_save_path)
#                                             # say_sth(msg, print_out=print_out)
#                                             save_pb_file(sess, self.pb_save_list, self.pb_save_path, encode=encript_flag,
#                                                               random_num_range=encode_num, header=encode_header)
#                                             break_flag = True
#                                             break
#                                 # ----get image start and end numbers
#                                 ori_paths = tl.get_ite_data(train_paths_ori,index,batch_size=batch_size)
#                                 aug_paths = tl.get_ite_data(train_paths_aug,index,batch_size=batch_size)
#
#                                 # ----get 4-D data
#                                 if aug_enable is True:
#                                     # ----ori data
#                                     # ori_data = get_4D_data(ori_paths, self.model_shape[1:],process_dict=None)
#                                     ori_data = tl.get_4D_data(ori_paths, self.model_shape[1:], to_norm=True, to_rgb=True,
#                                                                 to_process=False,dtype=self.dtype)
#
#                                     #ori_labels = train_labels_ori[num_start:num_end]
#                                     # ----aug data
#
#                                     if self.return_ori_data:
#                                         # aug_data_no_patch,aug_data = get_4D_data(aug_paths, self.model_shape[1:],
#                                         #                         process_dict=process_dict,setting_dict=setting_dict)
#                                         aug_data_no_patch, aug_data = tl.get_4D_data(aug_paths, self.model_shape[1:],
#                                                                                      to_norm=True,
#                                                                                      to_rgb=True,
#                                                                                      to_process=True,
#                                                                                      dtype=self.dtype)
#                                         batch_data_no_patch = np.concatenate([ori_data, aug_data_no_patch], axis=0)
#                                     else:
#                                         # aug_data = get_4D_data(aug_paths, self.model_shape[1:],
#                                         #                         process_dict=process_dict,setting_dict=setting_dict)
#                                         aug_data = tl.get_4D_data(aug_paths, self.model_shape[1:],
#                                                                   to_norm=True,
#                                                                   to_rgb=True,
#                                                                   to_process=True,
#                                                                   dtype=self.dtype)
#
#                                     #aug_labels = train_labels_aug[num_start:num_end]
#                                     # ----data concat
#                                     batch_data = np.concatenate([ori_data, aug_data], axis=0)
#                                     # if process_dict.get('rdm_patch'):
#                                     #     batch_data_ori = np.concatenate([ori_data, ori_data], axis=0)
#                                     #batch_labels = np.concatenate([ori_labels, aug_labels], axis=0)
#                                 else:
#                                     # batch_data = get_4D_data(ori_paths, self.model_shape[1:])
#                                     batch_data = tl.get_4D_data(ori_paths, self.model_shape[1:],dtype=self.dtype)
#                                     #batch_labels = train_labels_ori[num_start:num_end]
#
#                                 #----put all data to tf placeholders
#                                 if self.return_ori_data:
#                                     feed_dict = {self.tf_input: batch_data, self.tf_input_ori: batch_data_no_patch,
#                                                  self.tf_keep_prob: keep_prob}
#                                 else:
#                                     feed_dict = {self.tf_input: batch_data, self.tf_keep_prob: keep_prob}
#
#                                 if self.infer_method == 'AE_pooling_net_V8':
#                                     img_standard = tl.get_4D_data([img_standard_path]*len(batch_data), self.model_shape[1:], dtype=self.dtype)
#                                     feed_dict[self.tf_input_standard] = img_standard
#
#                                 # ----optimization
#                                 try:
#                                     if pause_opt_ae is not True:
#                                         sess.run(self.opt_AE, feed_dict=feed_dict)
#                                 except:
#                                     error_dict['GPU_resource_error'] = True
#                                     msg = "Error:權重最佳化時產生錯誤，可能GPU資源不夠導致"
#                                     say_sth(msg, print_out=print_out)
#                                     break
#                                 # if self.loss_method_2 is not None:
#                                 #     sess.run(self.opt_AE_2, feed_dict=feed_dict)
#
#                                 # ----evaluation(training set)
#                                 feed_dict[self.tf_keep_prob] = 1.0
#                                 # feed_dict[self.tf_phase_train] = False
#                                 loss_temp = sess.run(self.loss_AE, feed_dict=feed_dict)
#                                 # if self.loss_method_2 is not None:
#                                 #     loss_temp_2 = sess.run(self.loss_AE_2, feed_dict=feed_dict)
#
#                                 # ----calculate the loss and accuracy
#                                 train_loss += loss_temp
#                                 # if self.loss_method_2 is not None:
#                                 #     train_loss_2 += loss_temp_2
#
#                             train_loss /= train_ites
#                             train_result_dict['loss'] = train_loss
#                             # if self.loss_method_2 is not None:
#                             #     train_loss_2 /= train_ites
#
#                             # ----break the training
#                             if break_flag:
#                                 break_flag = False
#                                 break
#                             if True in list(error_dict.values()):
#                                 break
#
#                             #----evaluation(test set)
#                             if self.test_img_dir is not None:
#                                 for index in range(test_ites):
#                                     # ----get image start and end numbers
#                                     ite_paths = tl.get_ite_data(self.test_paths, index, batch_size=batch_size_test)
#
#                                     # batch_data = get_4D_data(ite_paths, self.model_shape[1:])
#                                     batch_data = tl.get_4D_data(ite_paths, self.model_shape[1:],dtype=self.dtype)
#
#
#                                     # ----put all data to tf placeholders
#                                     if self.return_ori_data:
#                                         feed_dict[self.tf_input] = batch_data
#                                         feed_dict[self.tf_input_ori] = batch_data
#                                         # feed_dict = {self.tf_input: batch_data, self.tf_input_ori: batch_data,
#                                         #              self.tf_keep_prob: 1.0}
#                                     else:
#                                         feed_dict[self.tf_input] = batch_data
#                                         # feed_dict = {self.tf_input: batch_data, self.tf_keep_prob: 1.0}
#
#                                     if self.infer_method == 'AE_pooling_net_V8':
#                                         img_standard = tl.get_4D_data([img_standard_path] * len(batch_data),
#                                                                       self.model_shape[1:], dtype=self.dtype)
#                                         feed_dict[self.tf_input_standard] = img_standard
#
#                                     # ----session run
#                                     #sess.run(self.opt_AE, feed_dict=feed_dict)
#
#                                     # ----evaluation(training set)
#                                     # feed_dict[self.tf_phase_train] = False
#                                     try:
#                                         loss_temp = sess.run(self.loss_AE, feed_dict=feed_dict)
#                                     except:
#                                         error_dict['GPU_resource_error'] = True
#                                         msg = "Error:推論驗證集時產生錯誤"
#                                         say_sth(msg, print_out=print_out)
#                                         break
#                                     # if self.loss_method_2 is not None:
#                                     #     loss_temp_2 = sess.run(self.loss_AE_2, feed_dict=feed_dict)
#
#                                     # ----calculate the loss and accuracy
#                                     test_loss += loss_temp
#                                     # if self.loss_method_2 is not None:
#                                     #     test_loss_2 += loss_temp_2
#
#                                 test_loss /= test_ites
#                                 test_result_dict['loss'] = test_loss
#                                 # if self.loss_method_2 is not None:
#                                 #     test_loss_2 /= test_ites
#
#                             #----save ckpt, pb files
#                             # if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
#                             #     model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)
#                             #
#                             #     #----encode ckpt file
#                             #     if encript_flag is True:
#                             #         encode_CKPT(model_save_path, encode_num=encode_num, encode_header=encode_header)
#                             #
#                             #     #----save pb(normal)
#                             #     save_pb_file(sess, self.pb_save_list, self.pb_save_path, encode=encript_flag,
#                             #                  random_num_range=encode_num, header=encode_header)
#
#
#                             #----save results in the log file
#                             train_loss_list.append(float(train_loss))
#                             self.content["train_loss_list"] = train_loss_list
#                             #train_acc_list.append(float(train_acc))
#                             #self.content["train_acc_list"] = train_acc_list
#
#                             if self.test_img_dir is not None:
#                                 test_loss_list.append(float(test_loss))
#                                 #test_acc_list.append(float(test_acc))
#                                 self.content["test_loss_list"] = test_loss_list
#
#                             #----display training results
#                             msg_list = list()
#                             msg_list.append("\n----訓練週期 {} 與相關結果如下----".format(epoch))
#                             msg_list.append("AE訓練集loss:{}".format(np.round(train_loss, 4)))
#                             if self.test_img_dir is not None:
#                                 msg_list.append("AE驗證集loss:{}".format(np.round(test_loss, 4)))
#
#                             # msg_list.append("此次訓練所花時間: {}秒".format(np.round(d_t, 2)))
#                             # msg_list.append("累積訓練時間: {}秒".format(np.round(total_train_time, 2)))
#                             say_sth(msg_list, print_out=print_out, header='msg')
#
#                             #----send protocol data(for C++ to draw)
#                             msg_list, header_list = self.collect_data2ui(epoch, train_result_dict,
#                                                                          test_result_dict)
#                             say_sth(msg_list, print_out=print_out, header=header_list)
#
#                             #----find the best performance
#                             new_value = self.__get_result_value(record_type, train_result_dict, test_result_dict)
#                             if epoch == 0:
#                                 LV['record_value'] = new_value
#                             else:
#
#                                 go_flag = False
#                                 if self.loss_method == 'ssim':
#                                     if new_value > LV['record_value']:
#                                         go_flag = True
#                                 else:
#                                     if new_value < LV['record_value']:
#                                         go_flag = True
#
#                                 if go_flag is True:
#                                     # ----delete the previous pb
#                                     if os.path.exists(LV['pb_save_path_old']):
#                                         os.remove(LV['pb_save_path_old'])
#
#                                     #----save the better one
#                                     pb_save_path = "infer_{}.{}".format(new_value, self.pb_extension)
#                                     pb_save_path = os.path.join(self.save_dir, pb_save_path)
#
#                                     save_pb_file(sess, self.pb_save_list, pb_save_path, encode=encript_flag,
#                                                       random_num_range=encode_num, header=encode_header)
#                                     #----update record value
#                                     LV['record_value'] = new_value
#                                     LV['pb_save_path_old'] = pb_save_path
#
#                             #----Check if stops the training
#                             if target_dict['type'] == 'loss':
#                                 if self.test_img_dir is not None:
#                                     re = tl.target_compare(test_result_dict)
#                                     name = "驗證集"
#                                 else:
#                                     re = tl.target_compare(train_result_dict)
#                                     name = "訓練集"
#
#                                 if re is True:
#                                     msg = '模型訓練結束:\n{}{}已經達到設定目標:{}累積達{}次'.format(
#                                         name, target_dict['type'],target_dict['value'], target_dict['hit_target_times'])
#                                     say_sth(msg, print_out=print_out)
#                                     break
#
#                             # ----test image reconstruction
#                             if self.recon_flag is True:
#                                 # if (epoch + 1) % eval_epochs == 0 and train_loss > 0.80:
#                                 if (epoch + 1) % eval_epochs == 0:
#                                     for filename in self.recon_paths:
#                                         test_img = self.__img_read(filename, self.model_shape[1:],dtype=self.dtype)
#                                         if self.infer_method == 'AE_pooling_net_V8':
#                                             img_standard = tl.get_4D_data([img_standard_path] * len(test_img),
#                                                                           self.model_shape[1:], dtype=self.dtype)
#                                             feed_dict[self.tf_input_standard] = img_standard
#                                         # ----session run
#                                         feed_dict[self.tf_input] = test_img
#                                         img_sess_out = sess.run(self.recon, feed_dict=feed_dict)
#                                         # ----process of sess-out
#                                         img_sess_out = img_sess_out[0] * 255
#                                         img_sess_out = cv2.convertScaleAbs(img_sess_out)
#                                         if self.model_shape[3] == 1:
#                                             img_sess_out = np.reshape(img_sess_out, (self.model_shape[1], self.model_shape[2]))
#                                         else:
#                                             img_sess_out = cv2.cvtColor(img_sess_out, cv2.COLOR_RGB2BGR)
#
#                                         # if loss_method != 'ssim':
#                                         #     img_sess_out = cv2.cvtColor(img_sess_out, cv2.COLOR_RGB2BGR)
#
#                                         # ----save recon image
#                                         splits = filename.split("\\")[-1]
#                                         new_filename = splits.split('.')[0] + '_sess-out.' + splits.split('.')[-1]
#
#                                         new_filename = os.path.join(self.new_recon_dir, new_filename)
#                                         # cv2.imwrite(new_filename, img_sess_out)
#                                         cv2.imencode('.'+splits.split('.')[-1], img_sess_out)[1].tofile(new_filename)
#                                         # ----img diff method
#                                         img_diff = self.__img_diff_method(filename, img_sess_out, diff_th=15, cc_th=15)
#                                         img_diff = cv2.convertScaleAbs(img_diff)
#                                         new_filename = filename.split("\\")[-1]
#                                         new_filename = new_filename.split(".")[0] + '_diff.' + new_filename.split(".")[-1]
#                                         new_filename = os.path.join(self.new_recon_dir, new_filename)
#                                         # cv2.imwrite(new_filename, img_diff)
#                                         cv2.imencode('.' + splits.split('.')[-1], img_diff)[1].tofile(new_filename)
#
#                                         # ----img avepool diff method
#                                         img_diff = self.__img_patch_diff_method(filename, img_sess_out, sess, diff_th=30, cc_th=15)
#                                         img_diff = cv2.convertScaleAbs(img_diff)
#                                         new_filename = filename.split("\\")[-1]
#                                         new_filename = new_filename.split(".")[0] + '_avepool_diff.' + new_filename.split(".")[-1]
#                                         new_filename = os.path.join(self.new_recon_dir, new_filename)
#                                         # cv2.imwrite(new_filename, img_diff)
#                                         cv2.imencode('.' + splits.split('.')[-1], img_diff)[1].tofile(new_filename)
#
#                                         # ----SSIM method
#                                         # img_ssim = self.__ssim_method(filename, img_sess_out)
#                                         # img_ssim = cv2.convertScaleAbs(img_ssim)
#                                         # new_filename = filename.split("\\")[-1]
#                                         # new_filename = new_filename.split(".")[0] + '_ssim.' + new_filename.split(".")[-1]
#                                         # new_filename = os.path.join(self.new_recon_dir, new_filename)
#                                         # cv2.imwrite(new_filename, img_ssim)
#
#                             # ----save ckpt
#                             # if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
#                             #     save_pb_file(sess, self.pb_save_list, self.pb4ae_save_path,
#                             #                  encode=encript_flag,
#                             #                  random_num_range=encode_num, header=encode_header)
#
#                     #----SEG part
#                     if self.to_train_seg is True:
#                         if to_fix_seg is True:
#                             pass
#                         else:
#                             #----
#                             seg_p.reset_arg()
#                             seg_p.reset_defect_stat()
#                             train_loss_seg_list = []
#                             test_loss_seg_list = []
#
#
#                             #----
#                             # indice = np.random.permutation(self.seg_train_qty)
#                             # self.seg_train_paths = self.seg_train_paths[indice]
#                             # if json_check:
#                             #     self.seg_train_json_paths = self.seg_train_json_paths[indice]
#
#                             # seg_train_paths_ori = self.seg_train_paths[:qty_train_seg]
#                             # if json_check:
#                             #     seg_train_json_paths_ori = self.seg_train_json_paths[:qty_train_seg]
#                             #
#                             # if aug_enable is True:
#                             #     seg_train_paths_aug = seg_train_paths_ori[::-1]
#                             #     if json_check:
#                             #         seg_train_json_paths_aug = seg_train_json_paths_ori[::-1]
#
#                             #----optimizations(SEG train set)
#                             dataloader.reset()
#                             for batch_paths, batch_data, batch_label in dataloader:
#                                 print(dataloader.ite_num)
#                                 if SockConnected:
#                                     # print(Sock.Message)
#                                     if len(Sock.Message):
#                                         if Sock.Message[-1][:4] == "$S00":
#                                             # Sock.send("Protocol:Ack\n")
#                                             model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)
#                                             # ----encode ckpt file
#                                             if encript_flag is True:
#                                                 file = model_save_path + '.meta'
#                                                 if os.path.exists(file):
#                                                     file_transfer(file, random_num_range=encode_num, header=encode_header)
#                                                 else:
#                                                     msg = "Warning:找不到權重檔:{}進行處理".format(file)
#                                                     say_sth(msg, print_out=print_out)
#                                                 # data_file = [file.path for file in os.scandir(self.save_dir) if
#                                                 #              file.name.split(".")[-1] == 'data-00000-of-00001']
#                                                 data_file = model_save_path + '.data-00000-of-00001'
#                                                 if os.path.exists(data_file):
#                                                     file_transfer(data_file, random_num_range=encode_num,
#                                                                   header=encode_header)
#                                                 else:
#                                                     msg = "Warning:找不到權重檔:{}進行處理".format(data_file)
#                                                     say_sth(msg, print_out=print_out)
#
#                                             # msg = "儲存訓練權重檔至{}".format(model_save_path)
#                                             # say_sth(msg, print_out=print_out)
#                                             save_pb_file(sess, self.pb_save_list, self.pb_save_path, encode=encript_flag,
#                                                          random_num_range=encode_num, header=encode_header)
#                                             break_flag = True
#                                             break
#
#                                 # recon = sess.run(self.recon, feed_dict={self.tf_input: batch_data})
#
#                                 feed_dict = {self.tf_input: batch_data, self.tf_input_recon: batch_data,
#                                              self.tf_label_batch: batch_label,
#                                              self.tf_keep_prob: keep_prob,
#                                              self.tf_dropout: 0.3}
#                                 #----session run
#                                 # print("idx_seg:",idx_seg)
#                                 print("step 1")
#                                 try:
#                                     if pause_opt_seg is not True:
#                                         print("step 2")
#                                         sess.run(self.opt_Seg, feed_dict=feed_dict)
#                                 except:
#                                     error_dict['GPU_resource_error'] = True
#                                     msg = "Error:SEG權重最佳化時產生錯誤，可能GPU資源不夠導致"
#                                     say_sth(msg, print_out=print_out)
#                                     break
#
#                                 #----evaluation(training set)
#                                 feed_dict[self.tf_keep_prob] = 1.0
#                                 feed_dict[self.tf_dropout] = 0.0
#                                 # feed_dict[self.tf_phase_train] = False
#
#                                 loss_temp = sess.run(self.loss_Seg, feed_dict=feed_dict)
#                                 predict_label = sess.run(self.prediction_Seg, feed_dict=feed_dict)
#                                 print("step 3")
#                                 # predict_label = np.argmax(predict_label,axis=-1).astype(np.uint8)
#
#                                 #----calculate the loss and accuracy
#                                 train_loss_seg_list.append(loss_temp)
#                                 seg_p.cal_intersection_union(predict_label, batch_label)
#                                 _ = seg_p.cal_label_defect_by_acc_v2(predict_label, batch_label)
#                                 _ = seg_p.cal_predict_defect_by_acc_v2(predict_label, batch_label)
#                             print("step 4")
#                             train_loss_seg = np.mean(train_loss_seg_list)
#                             seg_train_result_dict['loss'] = train_loss_seg
#                             # ----save results in the log file
#                             seg_train_loss_list.append(float(train_loss_seg))
#                             self.content["seg_train_loss_list"] = seg_train_loss_list
#                             train_iou_seg, train_acc_seg, train_all_acc_seg = seg_p.cal_iou_acc(save_dict=self.content,
#                                                                                                 name='train')
#                             train_defect_recall = seg_p.cal_defect_recall(save_dict=self.content, name='train')
#                             train_defect_sensitivity = seg_p.cal_defect_sensitivity(save_dict=self.content,
#                                                                                     name='train')
#
#
#                             #----evaluation(test set)
#                             if self.seg_test_qty > 0:
#                                 seg_p.reset_arg()
#                                 seg_p.reset_defect_stat()
#                                 dataloader_val.reset()
#                                 for batch_paths, batch_data, batch_label in dataloader_val:
#                                     feed_dict[self.tf_input] = batch_data
#                                     feed_dict[self.tf_label_batch] = batch_label
#
#                                     loss_temp = sess.run(self.loss_Seg, feed_dict=feed_dict)
#                                     predict_label = sess.run(self.prediction_Seg, feed_dict=feed_dict)
#                                     # predict_label = np.argmax(predict_label, axis=-1).astype(np.uint8)
#
#                                     # ----calculate the loss and accuracy
#                                     test_loss_seg_list.append(loss_temp)
#                                     seg_p.cal_intersection_union(predict_label, batch_label)
#                                     _ = seg_p.cal_label_defect_by_acc_v2(predict_label, batch_label)
#                                     _ = seg_p.cal_predict_defect_by_acc_v2(predict_label, batch_label)
#
#                                 test_loss_seg = np.mean(test_loss_seg_list)
#                                 seg_test_result_dict['loss'] = test_loss_seg
#                                 # ----save results in the log file
#                                 seg_test_loss_list.append(float(test_loss_seg))
#                                 self.content["seg_test_loss_list"] = seg_test_loss_list
#                                 test_iou_seg, test_acc_seg, test_all_acc_seg = seg_p.cal_iou_acc(save_dict=self.content,
#                                                                                                     name='test')
#                                 test_defect_recall = seg_p.cal_defect_recall(save_dict=self.content, name='test')
#                                 test_defect_sensitivity = seg_p.cal_defect_sensitivity(save_dict=self.content, name='test')
#
#                                 #----find the best performance(SEG)
#                                 target_of_best = seg_var.get('target_of_best')
#                                 print("target_of_best:",target_of_best)
#                                 if target_of_best == 'defect_recall':
#                                     new_value = seg_p.sum_defect_recall()
#                                 elif target_of_best == 'defect_sensitivity':
#                                     new_value = seg_p.sum_defect_sensitivity()
#                                 elif target_of_best == 'recall+sensitivity':
#                                     new_value = seg_p.sum_defect_recall() + seg_p.sum_defect_sensitivity()
#                                 else:
#                                     new_value = seg_p.sum_iou_acc()
#
#                                 if epoch == 0:
#                                     LV['record_value_seg'] = new_value
#                                 else:
#                                     if new_value > LV['record_value_seg']:
#                                         # ----delete the previous pb
#                                         if os.path.exists(LV['pb_seg_save_path_old']):
#                                             os.remove(LV['pb_seg_save_path_old'])
#
#                                         # ----save the better one
#                                         pb_save_path = "infer_best_epoch{}.{}".format(epoch, self.pb_extension)
#                                         pb_save_path = os.path.join(self.save_dir, pb_save_path)
#
#                                         save_pb_file(sess, self.pb_save_list, pb_save_path, encode=encript_flag,
#                                                      random_num_range=encode_num, header=encode_header)
#                                         # ----update record value
#                                         LV['record_value_seg'] = new_value
#                                         LV['pb_seg_save_path_old'] = pb_save_path
#
#                             #----display training results
#                             msg_list = list()
#                             msg_list.append("\n----訓練週期 {} 與相關結果如下----".format(epoch))
#                             msg_list.append("Seg訓練集loss:{}".format(np.round(train_loss_seg, 4)))
#                             if self.test_img_dir is not None:
#                                 msg_list.append("Seg驗證集loss:{}".format(np.round(test_loss_seg, 4)))
#
#                             # msg_list.append("此次訓練所花時間: {}秒".format(np.round(d_t, 2)))
#                             # msg_list.append("累積訓練時間: {}秒".format(np.round(total_train_time, 2)))
#                             say_sth(msg_list, print_out=print_out, header='msg')
#
#                             self.display_results(self.id2class_name,print_out,'訓練集',
#                                                  iou=train_iou_seg,acc=train_acc_seg,
#                                                  defect_recall=train_defect_recall,all_acc=train_all_acc_seg,
#                                                  defect_sensitivity=train_defect_sensitivity
#                                                  )
#                             self.display_results(self.id2class_name, print_out, '驗證集',
#                                                  iou=test_iou_seg, acc=test_acc_seg,
#                                                  defect_recall=test_defect_recall, all_acc=test_all_acc_seg,
#                                                  defect_sensitivity=test_defect_sensitivity
#                                                  )
#                             # self.display_iou_acc(train_iou_seg, train_acc_seg,train_defect_recall, train_all_acc_seg,
#                             #                      self.id2class_name,name='訓練集',print_out=print_out)
#                             # self.display_iou_acc(test_iou_seg, test_acc_seg,test_defect_recall, test_all_acc_seg,
#                             #                      self.id2class_name, name='驗證集', print_out=print_out)
#
#                             #----send protocol data(for C++ to draw)
#                             msg_list, header_list = self.collect_data2ui(epoch, seg_train_result_dict,
#                                                                          seg_test_result_dict)
#                             say_sth(msg_list, print_out=print_out, header=header_list)
#
#                             #----prediction for selected images
#                             if self.seg_predict_qty > 0:
#                                 if (epoch + 1 ) % eval_epochs == 0:
#                                     for idx_seg in range(predict_ites_seg):
#                                         # ----get batch paths
#                                         seg_paths = tl_seg.get_ite_data(self.seg_predict_paths, idx_seg,
#                                                                         batch_size=batch_size_seg_test)
#
#                                         #----get batch data
#                                         batch_data, batch_label = tl_seg.get_4D_img_label_data(seg_paths,
#                                                                                                self.model_shape[1:],
#                                                                                                json_paths=None,
#                                                                                                dtype=self.dtype)
#                                         recon = sess.run(self.recon,feed_dict={self.tf_input:batch_data})
#                                         predict_label = sess.run(self.tf_prediction_Seg,
#                                                                  feed_dict={self.tf_input: batch_data,
#                                                                             self.tf_input_recon:recon})
#                                         # predict_label = np.argmax(predict_label, axis=-1).astype(np.uint8)
#
#                                         batch_data *= 255
#                                         batch_data = batch_data.astype(np.uint8)
#
#                                         for i in range(len(predict_label)):
#                                             img = batch_data[i]
#                                             # label = batch_label[i]
#                                             # predict_label = predict_label[i]
#
#                                             #----label to color
#                                             zeros = np.zeros_like(batch_data[i])
#                                             for label_num in np.unique(predict_label[i]):
#                                                 if label_num != 0:
#                                                     # print(label_num)
#                                                     coors = np.where(predict_label[i] == label_num)
#                                                     try:
#                                                         zeros[coors] = self.id2color[label_num]
#                                                     except:
#                                                         print("error")
#
#                                             predict_png = cv2.addWeighted(img, 0.5, zeros, 0.5, 0)
#                                             #----create answer png
#                                             path = self.seg_predict_paths[batch_size_seg_test * idx_seg + i]
#                                             ext = path.split(".")[-1]
#                                             json_path = path.strip(ext) + 'json'
#                                             show_imgs = [predict_png]
#                                             if os.path.exists(json_path):
#                                                 answer_png = tl_seg.get_single_label_png(path, json_path)
#                                                 show_imgs.append(answer_png)
#                                             qty_show = len(show_imgs)
#                                             plt.figure(num=1,figsize=(5*qty_show, 5*qty_show), clear=True)
#
#                                             for i, show_img in enumerate(show_imgs):
#                                                 plt.subplot(1, qty_show, i + 1)
#                                                 plt.imshow(show_img)
#                                                 plt.axis('off')
#                                                 plt.title(titles[i])
#
#
#                                             save_path = os.path.join(self.new_predict_dir, path.split("\\")[-1])
#                                             plt.savefig(save_path)
#
#                             #----save ckpt
#                             # if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
#                             #     save_pb_file(sess, self.pb_save_list, self.pb4seg_save_path,
#                             #                  encode=encript_flag,
#                             #                  random_num_range=encode_num, header=encode_header)
#
#                     #----save ckpt, pb files
#                     if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
#                             model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)
#
#                             # ----encode ckpt file
#                             if encript_flag is True:
#                                 encode_CKPT(model_save_path, encode_num=encode_num, encode_header=encode_header)
#
#                             #----save pb(normal)
#                             save_pb_file(sess, self.pb_save_list, self.pb_save_path, encode=encript_flag,
#                                          random_num_range=encode_num, header=encode_header)
#
#                     #----record the end time
#                     d_t = time.time() - d_t
#
#                     epoch_time_list.append(d_t)
#                     total_train_time = time.time() - t_train_start
#                     self.content['total_train_time'] = float(total_train_time)
#                     self.content['ave_epoch_time'] = float(np.average(epoch_time_list))
#
#                     msg_list = []
#                     msg_list.append("此次訓練所花時間: {}秒".format(np.round(d_t, 2)))
#                     msg_list.append("累積訓練時間: {}秒".format(np.round(total_train_time, 2)))
#                     say_sth(msg_list, print_out=print_out, header='msg')
#
#                     with open(self.log_path, 'w') as f:
#                         json.dump(self.content, f)
#
#                     if encript_flag is True:
#                         if os.path.exists(self.log_path):
#                             file_transfer(self.log_path, cut_num_range=30, random_num_range=10)
#                     msg = "儲存訓練結果數據至{}".format(self.log_path)
#                     say_sth(msg, print_out=print_out)
#
#             #----error messages
#             if True in list(error_dict.values()):
#                 for key, value in error_dict.items():
#                     if value is True:
#                         say_sth('', print_out=print_out, header=key)
#             else:
#                 say_sth('AI Engine結束!!期待下次再相見', print_out=print_out, header='AIE_end')
#
#     #----functions
#     def get_train_qty(self,ori_qty,ratio,print_out=False,name=''):
#         if ratio is None:
#             qty_train = ori_qty
#         else:
#             if ratio <= 1.0 and ratio > 0:
#                 qty_train = int(ori_qty * ratio)
#                 qty_train = np.maximum(1, qty_train)
#
#         msg = "{}訓練集資料總共取數量{}".format(name,qty_train)
#         say_sth(msg, print_out=print_out)
#
#         return qty_train
#
#     def config_check(self,config_dict):
#         #----var
#         must_list = ['train_img_dir', 'model_name', 'save_dir', 'epochs']
#         # must_list = ['train_img_dir', 'test_img_dir', 'save_dir', 'epochs']
#         must_flag = True
#         default_dict = {"model_shape":[None,192,192,3],
#                         'model_name':"type_1_0",
#                         'loss_method':'ssim',
#                         'activation':'relu',
#                         'save_pb_name':'inference',
#                         'opti_method':'adam',
#                         'pool_type':['max', 'ave'],
#                         'pool_kernel':[7, 2],
#                         'embed_length':144,
#                         'learning_rate':1e-4,
#                         'batch_size':8,
#                         'ratio':1.0,
#                         'aug_times':2,
#                         'hit_target_times':2,
#                         'eval_epochs':2,
#                         'save_period':2,
#                         'kernel_list':[7,5,3,3,3],
#                         'filter_list':[32,64,96,128,256],
#                         'conv_time':1,
#                         'rot':False,
#                         'scaler':1,
#                         #'preprocess_dict':{'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}
#                         }
#
#
#         #----get the must list
#         if config_dict.get('must_list') is not None:
#             must_list = config_dict.get('must_list')
#         #----collect all keys of config_dict
#         config_key_list = list(config_dict.keys())
#
#         #----check the must list
#         if config_dict.get("J_mode") is not True:
#
#             #----check of must items
#             for item in must_list:
#                 if not item in config_key_list:
#                     msg = "Error: could you plz give me parameters -> {}".format(item)
#                     say_sth(msg,print_out=print_out)
#                     if must_flag is True:
#                         must_flag = False
#
#         #----parameters parsing
#         if must_flag is True:
#             #----model name
#             if config_dict.get("J_mode") is not True:
#                 infer_num = config_dict['model_name'].split("_")[-1]
#                 if infer_num == '0':#
#                     config_dict['infer_method'] = "AE_pooling_net"
#                 elif infer_num == '1':#
#                     config_dict['infer_method'] = "AE_Unet"
#                 else:
#                     config_dict['infer_method'] = "AE_transpose_4layer"
#
#             #----optional parameters
#             for key,value in default_dict.items():
#                 if not key in config_key_list:
#                     config_dict[key] = value
#
#         return must_flag,config_dict
#
#     def __img_patch_diff_method(self,img_source_1, img_source_2, sess,diff_th=30, cc_th=30):
#
#         temp = np.array([1., 2., 3.])
#         re = None
#         # ----read img source 1
#         if isinstance(temp, type(img_source_1)):
#             img_1 = img_source_1
#         elif os.path.isfile(img_source_1):
#             # img_1 = cv2.imread(img_source_1)
#             img_1 = cv2.imdecode(np.fromfile(img_source_1, dtype=np.uint8), 1)
#             img_1 = cv2.resize(img_1,(self.model_shape[2],self.model_shape[1]))
#             # img_1 = img_1.astype('float32')
#
#         else:
#             print("The type of img_source_1 is not supported")
#
#         # ----read img source 2
#         if isinstance(temp, type(img_source_2)):
#             img_2 = img_source_2
#         elif os.path.isfile(img_source_2):
#             # img_2 = cv2.imread(img_source_2)
#             img_2 = cv2.imdecode(np.fromfile(img_source_2, dtype=np.uint8), 1)
#             # img_2 = img_2.astype('float32')
#         else:
#             print("The type of img_source_2 is not supported")
#
#         # ----subtraction
#         if img_1 is not None and img_2 is not None:
#             img_1_ave_pool = sess.run(self.avepool_out,feed_dict={self.tf_input:np.expand_dims(img_1,axis=0)})
#             img_2_ave_pool = sess.run(self.avepool_out,feed_dict={self.tf_input:np.expand_dims(img_2,axis=0)})
#             # print("img_1 shape = {}, dtype = {}".format(img_1.shape, img_1.dtype))
#             # print("img_2 shape = {}, dtype = {}".format(img_2.shape, img_2.dtype))
#             img_diff = cv2.absdiff(img_1_ave_pool[0], img_2_ave_pool[0])  # img_diff = np.abs(img - img_sess)#不能使用，因為相減若<0，會取補數
#             if img_1.shape[-1] == 3:
#                 img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)  # 利用轉成灰階，減少RGB的相加計算量
#
#             # 連通
#             img_compare = cv2.compare(img_diff, diff_th, cv2.CMP_GT)
#             retval, labels = cv2.connectedComponents(img_compare)
#             max_label_num = np.max(labels) + 1
#
#             img_1_copy = img_1.copy()
#             for i in range(0, max_label_num):  # label = 0是背景，所以從1開始
#                 y, x = np.where(labels == i)  # y,x代表類別=i的像素座標值
#                 if (y.shape[0] > cc_th) and (img_compare[y[0], x[0]] == 255):
#                     for j in range(y.shape[0]):
#                         img_1_copy.itemset((y[j], x[j], 0), 0)
#                         img_1_copy.itemset((y[j], x[j], 1), 0)
#                         img_1_copy.itemset((y[j], x[j], 2), 255)
#
#             re = img_1_copy
#             return re
#
#     def __img_diff_method(self,img_source_1, img_source_2, diff_th=30, cc_th=30):
#
#         temp = np.array([1., 2., 3.])
#         re = None
#         # ----read img source 1
#         if isinstance(temp, type(img_source_1)):
#             img_1 = img_source_1
#         elif os.path.isfile(img_source_1):
#             # img_1 = cv2.imread(img_source_1)
#             img_1 = cv2.imdecode(np.fromfile(img_source_1, dtype=np.uint8), 1)
#             img_1 = cv2.resize(img_1,(self.model_shape[2],self.model_shape[1]))
#             # img_1 = img_1.astype('float32')
#
#         else:
#             print("The type of img_source_1 is not supported")
#
#         # ----read img source 2
#         if isinstance(temp, type(img_source_2)):
#             img_2 = img_source_2
#         elif os.path.isfile(img_source_2):
#             # img_2 = cv2.imread(img_source_2)
#             img_2 = cv2.imdecode(np.fromfile(img_source_2, dtype=np.uint8), 1)
#             # img_2 = img_2.astype('float32')
#         else:
#             print("The type of img_source_2 is not supported")
#
#         # ----substraction
#         if img_1 is not None and img_2 is not None:
#             # print("img_1 shape = {}, dtype = {}".format(img_1.shape, img_1.dtype))
#             # print("img_2 shape = {}, dtype = {}".format(img_2.shape, img_2.dtype))
#             img_diff = cv2.absdiff(img_1, img_2)  # img_diff = np.abs(img - img_sess)#不能使用，因為相減若<0，會取補數
#             if img_1.shape[-1] == 3:
#                 img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)  # 利用轉成灰階，減少RGB的相加計算量
#
#             # 連通
#             img_compare = cv2.compare(img_diff, diff_th, cv2.CMP_GT)
#             retval, labels = cv2.connectedComponents(img_compare)
#             max_label_num = np.max(labels) + 1
#
#             img_1_copy = img_1.copy()
#             for i in range(0, max_label_num):  # label = 0是背景，所以從1開始
#                 y, x = np.where(labels == i)  # y,x代表類別=i的像素座標值
#                 if (y.shape[0] > cc_th) and (img_compare[y[0], x[0]] == 255):
#                     for j in range(y.shape[0]):
#                         img_1_copy.itemset((y[j], x[j], 0), 0)
#                         img_1_copy.itemset((y[j], x[j], 1), 0)
#                         img_1_copy.itemset((y[j], x[j], 2), 255)
#
#             re = img_1_copy
#             return re
#
#     def __avepool(self,input_x,k_size=3,strides=1):
#         kernel = [1,k_size,k_size,1]
#         stride_kernel = [1,strides,strides,1]
#         return tf.nn.avg_pool(input_x, ksize=kernel, strides=stride_kernel, padding='SAME')
#
#     def __Conv(self,input_x,kernel=[3,3],filter=32,conv_times=2,stride=1):
#         net = None
#         for i in range(conv_times):
#             if i == 0:
#                 net = tf.layers.conv2d(
#                     inputs=input_x,
#                     filters=filter,
#                     kernel_size=kernel,
#                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#                     strides=stride,
#                     padding="same",
#                     activation=tf.nn.relu)
#             else:
#                 net = tf.layers.conv2d(
#                     inputs=net,
#                     filters=filter,
#                     kernel_size=kernel,
#                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#                     strides=stride,
#                     padding="same",
#                     activation=tf.nn.relu)
#         return net
#
#     def say_sth(self,msg, print_out=False):
#         if print_out:
#             print(msg)
#
#     def log_update(self,content,para_dict):
#         for key, value in para_dict.items():
#             content[key] = value
#
#         return content
#
#     def dict_update(self,main_content,add_content):
#         for key, value in add_content.items():
#             main_content[key] = value
#
#     def __img_read(self, img_path, shape,dtype='float32'):
#
#         # img = cv2.imread(img_path)
#         img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
#         if img is None:
#             print("Read failed:",img_path)
#             return None
#         else:
#             img = cv2.resize(img,(shape[1],shape[0]))
#             img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#             img = img.astype(dtype)
#             img /= 255
#
#             return np.expand_dims(img,axis=0)
#
#     def __get_result_value(self,record_type,train_dict,test_dict):
#         value = None
#         if record_type == 'loss':
#             if self.test_img_dir is not None:
#                 value = test_dict['loss']
#             else:
#                 value = train_dict['loss']
#
#             if self.loss_method == 'ssim':
#                 value = np.round(value * 100, 2)
#             else:
#                 value = np.round(value, 2)
#
#         return value
#
#     def collect_data2ui(self,epoch,train_dict,test_dict):
#         #----var
#         msg_list = list()
#         header_list = list()
#         #----process
#         msg_list.append('{},{}'.format(epoch, train_dict['loss']))
#         header_list.append('train_loss')
#         # msg_list.append('{},{}'.format(epoch, train_dict['accuracy']))
#         # header_list.append('train_acc')
#         if self.test_img_dir is not None:
#             msg_list.append('{},{}'.format(epoch, test_dict['loss']))
#             header_list.append('val_loss')
#             # msg_list.append('{},{}'.format(epoch, test_dict['accuracy']))
#             # header_list.append('val_acc')
#
#
#         return msg_list,header_list
#
#     # def display_iou_acc(self,iou,acc,defect_recall,all_acc,id2name,name='',print_out=False):
#     def display_results(self,id2name,print_out,dataset_name,**kwargs):
#         msg_list = []
#         class_names = list(id2name.values())
#         #a_dict = {'iou':iou, 'acc':acc,'defect_recall':defect_recall}
#         for key,value_list in kwargs.items():
#             if key == 'all_acc':
#                 msg_list.append("Seg{}_all_acc: {}".format(dataset_name,value_list))
#             else:
#                 msg_list.append("Seg{}_{}:".format(dataset_name,key))
#                 msg_list.append("{}:".format(class_names))
#                 msg_list.append("{}:".format(value_list))
#
#             # for i,value in enumerate(value_list):
#             #     msg_list.append(" {}: {}".format(id2name[i],value))
#
#
#
#         for msg in msg_list:
#             say_sth(msg,print_out=print_out)
#
#     #----models
#
#     def __AE_transpose_4layer_test(self, input_x, kernel_list, filter_list,conv_time=1,maxpool_kernel=2):
#         #----var
#         maxpool_kernel = [1,maxpool_kernel,maxpool_kernel,1]
#         transpose_filter = [1, 1]
#
#         msg = '----AE_transpose_4layer_struct_2----'
#         self.say_sth(msg, print_out=self.print_out)
#
#         net = self.__Conv(input_x, kernel=kernel_list[0], filter=filter_list[0], conv_times=conv_time)
#         U_1_point = net
#         #net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
#         net = tf.layers.max_pooling2d(net,pool_size=[2,2],strides=2,padding='SAME')
#
#         msg = "encode_1 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#         net = self.__Conv(net, kernel=kernel_list[1], filter=filter_list[1], conv_times=conv_time)
#         U_2_point = net
#         # net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
#         net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
#         msg = "encode_2 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#
#         net = self.__Conv(net, kernel=kernel_list[2], filter=filter_list[2], conv_times=conv_time)
#         U_3_point = net
#         # net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
#         net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
#
#         msg = "encode_3 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#         net = self.__Conv(net, kernel=kernel_list[3], filter=filter_list[3], conv_times=conv_time)
#         U_4_point = net
#         # net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
#         net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
#
#         msg = "encode_4 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#
#         net = self.__Conv(net, kernel=kernel_list[4], filter=filter_list[4], conv_times=conv_time)
#
#
#         flatten = tf.layers.flatten(net)
#
#         embeddings = tf.nn.l2_normalize(flatten, 1, 1e-10, name='embeddings')
#         print("embeddings shape:",embeddings.shape)
#         # net = tf.layers.dense(inputs=prelogits, units=units, activation=None)
#         # print("net shape:",net.shape)
#         # net = tf.reshape(net,shape)
#         # -----------------------------------------------------------------------
#         # --------Decode--------
#         # -----------------------------------------------------------------------
#
#         # data= 4 x 4 x 64
#
#         net = tf.layers.conv2d_transpose(net, filter_list[3], transpose_filter, strides=2, padding='same')
#         #net = tf.concat([net, U_4_point], axis=3)
#         net = self.__Conv(net, kernel=kernel_list[3], filter=filter_list[3], conv_times=conv_time)
#         msg = "decode_1 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#         # data= 8 x 8 x 64
#         net = tf.layers.conv2d_transpose(net, filter_list[2], transpose_filter, strides=2, padding='same')
#         # net = tf.concat([net, U_3_point], axis=3)
#         net = self.__Conv(net, kernel=kernel_list[2], filter=filter_list[2], conv_times=conv_time)
#         msg = "decode_2 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#
#         net = tf.layers.conv2d_transpose(net, filter_list[1], transpose_filter, strides=2, padding='same')
#         # net = tf.concat([net, U_2_point], axis=3)
#         net = self.__Conv(net, kernel=kernel_list[1], filter=filter_list[1], conv_times=conv_time)
#         msg = "decode_3 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#         # data= 32 x 32 x 64
#
#         net = tf.layers.conv2d_transpose(net, filter_list[0], transpose_filter, strides=2, padding='same')
#         # net = tf.concat([net, U_1_point], axis=3)
#         net = self.__Conv(net, kernel=kernel_list[0], filter=filter_list[0], conv_times=conv_time)
#         msg = "decode_2 shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#
#         net = tf.layers.conv2d(
#             inputs=net,
#             filters=3,
#             kernel_size=kernel_list[0],
#             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#             padding="same",
#             activation=tf.nn.relu,
#             name='output_AE')
#         msg = "output shape = {}".format(net.shape)
#         self.say_sth(msg, print_out=self.print_out)
#         # -----------------------------------------------------------------------
#         # data= 64 x 64 x 3
#         return net




