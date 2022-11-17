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

import models_Seg
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

def create_log_path(save_dir,to_encode=False,filename='train_result'):
    count = 0
    ext = "json"

    if to_encode:
        ext = "nst"

    for i in range(1000):
        log_path = "{}_{}.{}".format(filename, count, ext)
        log_path = os.path.join(save_dir, log_path)
        if not os.path.exists(log_path):
            break
        count += 1

    return log_path

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
        if img_source is not None:
            if not isinstance(img_source, list):
                img_source = [img_source]

            if isinstance(ext,str):
                ext_list = [ext]
            else:
                ext_list = img_format

            for img_dir in img_source:
                temp = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] in ext_list]
                if len(temp) == 0:
                    say_sth("Warning:沒有找到支援的圖片檔案:{}".format(img_dir))
                else:
                    paths.extend(temp)
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
    for k,v in kwargs.items():
        if isinstance(v,list):
            msg_list.append(f"{k}:")
            msg_list.append(f"{v}")
        else:
            msg_list.append(f"{k}: {v}")

    for msg in msg_list:
        say_sth(msg,print_out=print_out)




#----main class
class Seg():
    def __init__(self,para_dict,user_dict=None):

        #----config process
        if isinstance(user_dict,dict):
            para_dict = self.update_config_dict(user_dict,para_dict)

        #----common var
        print_out = para_dict.get('print_out')
        self.print_out = print_out

        #----SEG process
        train_img_seg_dir = para_dict.get('train_img_seg_dir')
        test_img_seg_dir = para_dict.get('test_img_seg_dir')
        predict_img_dir = para_dict.get('predict_img_dir')
        # to_train_w_AE_paths = para_dict.get('to_train_w_AE_paths')
        id2class_name = para_dict.get('id2class_name')
        # select_OK_ratio = 0.2


        #----SEG train image path process
        self.seg_train_paths, self.seg_train_qty = get_paths(train_img_seg_dir)
        self.seg_test_paths, self.seg_test_qty = get_paths(test_img_seg_dir)
        # self.seg_predict_paths, self.seg_predict_qty = get_paths(predict_img_dir)

        qty_status_list = self.path_qty_process(
            dict(SEG訓練集=self.seg_train_qty,
                 SEG驗證集=self.seg_test_qty,
                 # SEG預測集=self.seg_predict_qty
                 ))
        to_train_seg = qty_status_list[0]

        #----read class names
        classname_id_color_dict = AE_Seg_Util.get_classname_id_color_v2(id2class_name,print_out=print_out)

        # ----train with AE ok images
        # self.seg_path_change_process(to_train_w_AE_paths, to_train_ae, select_OK_ratio=select_OK_ratio)

        #----log update
        log = classname_id_color_dict.copy()

        #----local var to global
        self.para_dict = para_dict
        self.status = to_train_seg
        self.log = log
        # self.train_img_seg_dir = train_img_seg_dir
        # self.test_img_seg_dir = test_img_seg_dir
        # self.predict_img_dir = predict_img_dir
        self.classname_id_color_dict = classname_id_color_dict

    def model_init(self):
        #----var parsing
        para_dict = self.para_dict
        # ----use previous settings
        if para_dict.get('use_previous_settings'):
            if isinstance(self.para_dict,dict):
                para_dict = self.para_dict

        #----common var
        model_shape = [None,para_dict['height'],para_dict['width'],3]
        lr = para_dict['learning_rate']
        save_dir = para_dict['save_dir']
        save_pb_name = para_dict.get('save_pb_name')
        encript_flag = para_dict.get('encript_flag')
        print_out = para_dict.get('print_out')
        add_name_tail = para_dict.get('add_name_tail')
        dtype = para_dict.get('dtype')
        model_type = "type_5"

        #----var
        acti_dict = {'relu': tf.nn.relu, 'mish': models_Seg.tf_mish, None: tf.nn.relu}
        pb_save_list = list()

        #----var process
        if add_name_tail is None:
            add_name_tail = True
        if dtype is None:
            dtype = 'float32'


        #----Seg inference selection
        infer_method4Seg = para_dict.get('infer_method')
        model_name4seg = para_dict.get('model_name')
        # filter_list4Seg = para_dict['filter_list']
        # loss_method4Seg = para_dict.get('loss_method')
        # opti_method4Seg = para_dict.get('opti_method')
        acti_seg = para_dict.get('activation')
        self.class_num = len(self.classname_id_color_dict['class_names'])

        #----tf_placeholder declaration
        tf_input = tf.placeholder(dtype, shape=model_shape, name='input')
        tf_keep_prob = tf.placeholder(dtype=dtype, name="keep_prob")
        tf_input_recon = tf.placeholder(dtype, shape=model_shape, name='input_recon')
        tf_label_batch = tf.placeholder(tf.int32, shape=model_shape[:-1], name='label_batch')
        tf_dropout = tf.placeholder(dtype=tf.float32, name="dropout")

        # ----activation selection
        acti_func = acti_dict[acti_seg]

        #----filer scaling process
        # filter_list4Seg = np.array(filter_list4Seg)
        # if para_dict.get('scaler') is not None:
        #     filter_list4Seg /= para_dict.get('scaler')
        #     filter_list4Seg = filter_list4Seg.astype(np.uint16)

        #----AIE model mapping
        if model_name4seg is not None:
            if model_name4seg.find(model_type) >= 0:
                infer_method4Seg = "Seg_pooling_net_V" + model_name4seg.split("_")[-1]

        #----Seg model selection
        if infer_method4Seg == 'Seg_pooling_net_V8':
            logits_Seg = models_Seg.Seg_pooling_net_V8(tf_input, para_dict['encode_dict'],
                                                       para_dict['decode_dict'],
                                                       out_channel=self.class_num,
                                                       print_out=print_out)
            softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
            prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
            prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
        elif infer_method4Seg == 'Seg_pooling_net_V9':
            logits_Seg = models_Seg.Seg_pooling_net_V9(tf_input, tf_input_recon, para_dict['encode_dict'],
                                            para_dict['decode_dict'],
                                            to_reduce=para_dict.get('to_reduce'), out_channel=self.class_num,
                                            print_out=print_out)
            softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
            prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
            prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
        elif infer_method4Seg == 'Seg_pooling_net_V10':
            logits_Seg = models_Seg.Seg_pooling_net_V10(tf_input, tf_input_recon, para_dict['encode_dict'],
                                            para_dict['decode_dict'],
                                            to_reduce=para_dict.get('to_reduce'), out_channel=self.class_num,
                                            print_out=print_out)
            softmax_Seg = tf.nn.softmax(logits_Seg, name='softmax_Seg')
            prediction_Seg = tf.argmax(softmax_Seg, axis=-1, output_type=tf.int32)
            prediction_Seg = tf.cast(prediction_Seg, tf.uint8, name='predict_Seg')
        else:
            if model_name4seg is not None:
                display_name = model_name4seg
            else:
                display_name = infer_method4Seg
            say_sth(f"Error:Seg model doesn't exist-->{display_name}", print_out=True)

        #----Seg loss method selection
        # if loss_method4Seg == "cross_entropy":
        loss_Seg = tf.reduce_mean(v2.nn.sparse_softmax_cross_entropy_with_logits(tf_label_batch,logits_Seg),name='loss_Seg')

        #----Seg optimizer selection
        # if opti_method4Seg == "adam":
        opt_Seg = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_Seg)


        # ----appoint PB node names
        pb_save_list.extend(['predict_Seg'])

        # ----pb filename(common)
        pb_save_path = create_pb_path(save_pb_name, save_dir,
                                      to_encode=encript_flag,add_name_tail=add_name_tail)

        # ----create the dir to save model weights(CKPT, PB)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #----save SEG coloer index image
        _ = AE_Seg_Util.draw_color_index(self.classname_id_color_dict['class_names'], save_dir=save_dir)

        # if self.seg_predict_qty > 0:
        #     new_dir_name = "img_seg-" + os.path.basename(save_dir)
        #     #----if recon_img_dir is list format
        #     if isinstance(self.predict_img_dir,list):
        #         dir_path = self.predict_img_dir[0]
        #     else:
        #         dir_path = self.predict_img_dir
        #     self.new_predict_dir = os.path.join(dir_path, new_dir_name)
        #     if not os.path.exists(self.new_predict_dir):
        #         os.makedirs(self.new_predict_dir)

        self.out_dir_prefix = os.path.join(save_dir, "model")
        saver = tf.train.Saver(max_to_keep=2)

        # ----create the log(JSON)
        log_path = create_log_path(save_dir,to_encode=encript_flag,filename="train_result")
        self.log['pb_save_list'] = pb_save_list
        self.log = update_dict(self.log,para_dict)

        # ----local var to global
        self.model_shape = model_shape
        self.tf_input = tf_input
        self.tf_keep_prob = tf_keep_prob
        self.saver = saver
        self.save_dir = save_dir
        self.pb_save_path = pb_save_path
        self.encript_flag = encript_flag
        self.pb_save_list = pb_save_list
        # self.pb_extension = pb_extension
        self.log_path = log_path
        self.dtype = dtype

        self.tf_label_batch = tf_label_batch
        self.tf_prediction_Seg = prediction_Seg
        self.infer_method4Seg = infer_method4Seg
        self.logits_Seg = logits_Seg
        self.loss_Seg = loss_Seg
        self.opt_Seg = opt_Seg
        self.prediction_Seg = prediction_Seg
        # self.loss_method4Seg = loss_method4Seg
        # self.pb4seg_save_path = pb4seg_save_path
        self.tf_dropout = tf_dropout
        self.infer_method4Seg = infer_method4Seg

    def train(self):
        para_dict = self.para_dict

        # ----var parsing
        epochs = para_dict['epochs']
        GPU_ratio = para_dict.get('GPU_ratio')
        print_out = para_dict.get('print_out')
        encode_header = para_dict.get('encode_header')
        encode_num = para_dict.get('encode_num')
        encript_flag = para_dict.get('encript_flag')
        eval_epochs = para_dict.get('eval_epochs')

        save_period = para_dict.get('save_period')

        if encode_header is None:
            encode_header = [24,97,28,98]
        if encode_num is None:
            encode_num = 87
        self.encode_header = encode_header
        self.encode_num = encode_num

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
        self.train_loss_list = list()
        self.seg_train_loss_list = list()
        self.test_loss_list = list()
        self.seg_test_loss_list = list()


        self.error_dict = {'GPU_resource_error': False}
        keep_prob = 0.7


        #----SEG hyper-parameters
        class_names = self.classname_id_color_dict['class_names']
        dataloader_SEG_train = AE_Seg_Util.DataLoader4Seg(self.seg_train_paths,
                                                         only_img=False,
                                                         batch_size=para_dict['batch_size'],
                                                         pipelines=para_dict.get("train_pipelines"),
                                                         to_shuffle=True,
                                                         print_out=self.print_out)

        dataloader_SEG_val = AE_Seg_Util.DataLoader4Seg(self.seg_test_paths,
                                                          only_img=False,
                                                          batch_size=para_dict['batch_size'],
                                                          pipelines=para_dict.get("val_pipelines"),
                                                          to_shuffle=False,
                                                          print_out=self.print_out)
        # if self.seg_predict_qty > 0:
        #     dataloader_SEG_predict = AE_Seg_Util.DataLoader4Seg(self.seg_predict_paths,
        #                                                     only_img=False,
        #                                                     batch_size=para_dict['batch_size'],
        #                                                     pipelines=para_dict.get("val_pipelines"),
        #                                                     to_shuffle=False,
        #                                                     print_out=self.print_out)
        #     dataloader_SEG_predict.set_classname_id_color(**self.classname_id_color_dict)

        dataloader_SEG_train.set_classname_id_color(**self.classname_id_color_dict)
        dataloader_SEG_val.set_classname_id_color(**self.classname_id_color_dict)

        #titles = ['prediction', 'answer']
        # batch_size_seg_test = batch_size_seg
        self.seg_p = AE_Seg_Util.Seg_performance(print_out=print_out)
        self.seg_p.set_classname_id_color(**self.classname_id_color_dict)

        self.set_time_start()

        #----GPU setting
        config = GPU_setting(GPU_ratio)
        self.sess = tf.Session(config=config)
        # with tf.Session(config=config) as sess:
        status = weights_check(self.sess, self.saver, self.save_dir, encript_flag=encript_flag,
                               encode_num=encode_num, encode_header=encode_header)
        if status is False:
            self.error_dict['GPU_resource_error'] = True
        else:
            # ----epoch training
            for epoch in range(epochs):
                #----read manual cmd
                # break_flag = self.read_manual_cmd(para_dict.get('to_read_manual_cmd'))

                #----record the epoch start time
                self.set_epoch_start_time()

                #----SEG part
                # say_sth("SEG opti ing", print_out=True)
                # ----optimizations(SEG train set)
                self.seg_opti_by_dataloader(dataloader_SEG_train)
                if self.break_flag:
                    break

                #----evaluation(SEG train set)
                train_loss_seg, train_iou_seg, train_acc_seg,train_defect_recall,train_defect_sensitivity = \
                    self.seg_eval_by_dataloader(dataloader_SEG_train,name='train')

                #----evaluation(SEG val set)
                test_loss_seg, test_iou_seg, test_acc_seg, test_defect_recall, test_defect_sensitivity = \
                    self.seg_eval_by_dataloader(dataloader_SEG_val, name='test')

                if self.break_flag:
                    self.save_ckpt_and_pb(epoch)
                    break
                else:
                    #----find the best performance(SEG)
                    new_value = self.get_value_from_performance(para_dict.get('target_of_best'))
                    self.performance_process_SEG(epoch, new_value, LV)

                    #----display training results

                    msg = "\n----訓練週期 {} 與相關結果如下----".format(epoch)
                    say_sth(msg, print_out=print_out)
                    display_results(None,print_out,
                                         train_SEG_loss=train_loss_seg,
                                         val_SEG_loss=test_loss_seg,
                                         )

                    display_results(class_names, print_out,
                                         train_SEG_iou=train_iou_seg,
                                         val_SEG_iou=test_iou_seg,
                                         train_SEG_accuracy=train_acc_seg,
                                         val_SEG_accuracy=test_acc_seg,
                                         train_SEG_defect_recall=train_defect_recall,
                                         val_SEG_defect_recall=test_defect_recall,
                                         train_SEG_defect_sensitivity=train_defect_sensitivity,
                                         val_SEG_defect_sensitivity=test_defect_sensitivity
                                         )

                    # ----send protocol data(for UI to draw)
                    transmit_data2ui(epoch, print_out,
                                          train_loss=train_loss_seg,
                                          train_SEG_iou=train_iou_seg,
                                          train_SEG_accuracy=train_acc_seg,
                                          train_SEG_defect_recall=train_defect_recall,
                                          train_SEG_defect_sensitivity=train_defect_sensitivity,
                                          val_loss=test_loss_seg,
                                          val_SEG_iou=test_iou_seg,
                                          val_SEG_accuracy=test_acc_seg,
                                          val_SEG_defect_recall=test_defect_recall,
                                          val_SEG_defect_sensitivity=test_defect_sensitivity
                                          )

                    # ----prediction for selected images
                    # if self.seg_predict_qty > 0:
                    #     if (epoch + 1) % eval_epochs == 0:
                    #         for batch_paths, batch_data, batch_label in dataloader_SEG_predict:
                    #             predict_labels = self.sess.run(self.tf_prediction_Seg,
                    #                                       feed_dict={self.tf_input: batch_data,
                    #                                                  })
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

                    #----save ckpt, pb files
                    if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
                        self.save_ckpt_and_pb(epoch)

                    #----record the end time
                    self.record_time()

                    #----save the log
                    self.save_log()


        #----error messages
        if True in list(self.error_dict.values()):
            for key, value in self.error_dict.items():
                if value is True:
                    say_sth('', print_out=print_out, header=key)

        #----close the session
        self.sess.close()
            # say_sth('AI Engine結束!!期待下次再相見', print_out=print_out, header='AIE_end')

    #----functions
    def update_config_dict(self,new_dict, ori_dict):

        combine_dict = update_dict(new_dict, ori_dict)
        # ----pipeline modification
        p = Pipeline_modification()
        p(config_dict=combine_dict)

        return combine_dict

    def save_log(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.log, f)

        if self.encript_flag is True:
            if os.path.exists(self.log_path):
                file_transfer(self.log_path, cut_num_range=30, random_num_range=10)

        msg = "儲存訓練結果數據至{}".format(self.log_path)
        say_sth(msg, print_out=True)

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

                feed_dict = {self.tf_input: batch_data,
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

    def seg_eval_by_dataloader(self, dataloader,name='train'):
        seg_loss = None
        iou_seg = None
        acc_seg = None
        defect_recall = None
        defect_sensitivity = None
        if self.break_flag is False:
            self.seg_p.reset_arg()
            self.seg_p.reset_defect_stat()
            seg_loss = 0
            dataloader.reset()
            for batch_paths, batch_data, batch_label in dataloader:

                feed_dict = {self.tf_input: batch_data,
                             self.tf_label_batch: batch_label,
                             self.tf_keep_prob: 1.0,
                             self.tf_dropout: 0.0}
                loss_temp = self.sess.run(self.loss_Seg, feed_dict=feed_dict)
                predict_label = self.sess.run(self.prediction_Seg, feed_dict=feed_dict)

                #----calculate the loss and accuracy
                seg_loss += loss_temp
                self.seg_p.cal_intersection_union(predict_label, batch_label)
                _ = self.seg_p.cal_label_defect_by_acc_v2(predict_label, batch_label)
                _ = self.seg_p.cal_predict_defect_by_acc_v2(predict_label, batch_label)

                #----check the break signal
                self.break_flag = break_signal_check()
                if self.break_flag:
                    break

            iou_seg, acc_seg, all_acc_seg = self.seg_p.cal_iou_acc(save_dict=self.log,name=name)
            defect_recall = self.seg_p.cal_defect_recall(save_dict=self.log, name=name)
            defect_sensitivity = self.seg_p.cal_defect_sensitivity(save_dict=self.log, name=name)

            if self.break_flag:
                seg_loss /= (dataloader.ite_num + 1)
            else:
                seg_loss /= dataloader.iterations

            #----
            if name == 'train':
                self.seg_train_loss_list.append(float(seg_loss))
                self.log["seg_train_loss_list"] = self.seg_train_loss_list
            else:
                self.seg_test_loss_list.append(float(seg_loss))
                self.log["seg_test_loss_list"] = self.seg_test_loss_list

        return seg_loss,iou_seg,acc_seg,defect_recall,defect_sensitivity

    def record_time(self,):
        now_time = time.time()
        d_t = now_time - self.epoch_start_time
        total_train_time = now_time - self.t_train_start

        self.epoch_time_list.append(d_t)
        self.log['total_train_time'] = float(total_train_time)
        self.log['ave_epoch_time'] = float(np.average(self.epoch_time_list))

        msg_list = []
        msg_list.append("此次訓練所花時間: {}秒".format(np.round(d_t, 2)))
        msg_list.append("累積訓練時間: {}秒".format(np.round(total_train_time, 2)))
        say_sth(msg_list, print_out=self.print_out, header='msg')

    def set_time_start(self):
        self.epoch_time_list = list()
        self.t_train_start = time.time()

    def set_epoch_start_time(self):
        self.epoch_start_time = time.time()

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

    def receive_msg_process(self,sess,epoch,encript_flag=True):
        break_flag = False
        if SockConnected:
            # print(Sock.Message)
            if len(Sock.Message):
                if Sock.Message[-1][:4] == "$S00":
                    # Sock.send("Protocol:Ack\n")
                    model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)
                    # ----encode ckpt file
                    if encript_flag:
                        file = model_save_path + '.meta'
                        if os.path.exists(file):
                            file_transfer(file, random_num_range=self.encode_num, header=self.encode_header)
                        else:
                            msg = "Warning:找不到權重檔:{}進行處理".format(file)
                            say_sth(msg, print_out=print_out)
                        # data_file = [file.path for file in os.scandir(self.save_dir) if
                        #              file.name.split(".")[-1] == 'data-00000-of-00001']
                        data_file = model_save_path + '.data-00000-of-00001'
                        if os.path.exists(data_file):
                            file_transfer(data_file, random_num_range=self.encode_num, header=self.encode_header)
                        else:
                            msg = "Warning:找不到權重檔:{}進行處理".format(data_file)
                            say_sth(msg, print_out=print_out)

                    # msg = "儲存訓練權重檔至{}".format(model_save_path)
                    # say_sth(msg, print_out=print_out)
                    save_pb_file(sess, self.pb_save_list, self.pb_save_path, encode=encript_flag,
                                 random_num_range=self.encode_num, header=self.encode_header)
                    break_flag = True

        return break_flag

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
                else:
                    config_dict['infer_method'] = "AE_transpose_4layer"

            #----optional parameters
            for key,value in default_dict.items():
                if not key in config_key_list:
                    config_dict[key] = value

        return must_flag,config_dict

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


class Pipeline_modification():
    # def __init__(self):
    #     print("Pipeline_modification init")

    def __call__(self, *args, **kwargs):
        # model_type = kwargs.get('model_type')
        config_dict = kwargs.get('config_dict')

        key_list = ["train_pipelines", "val_pipelines"]
        h = config_dict.get('height')
        w = config_dict.get('width')

        for key in key_list:
            if config_dict.get(key) is not None:
                self.height_width_check(config_dict[key], h, w)

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

    AE_train = Seg(para_dict)
    AE_train.model_init(para_dict)
    AE_train.train(para_dict)




