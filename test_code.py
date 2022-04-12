import time,os,tensorflow,math,cv2,shutil
import numpy as np
import matplotlib.pyplot as plt

if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile
print("Tensorflow version: ", tf.__version__)

img_format = {'tif', 'TIF', 'JPG', 'jpg', 'png', 'bmp'}

def get_paths_2(img_source):
    paths = list()
    qty_list = list()
    dirname_list = list()
    list_type = type(list())
    if type(img_source) == list_type:
        for img_dir in img_source:
            for obj in os.scandir(img_dir):
                if obj.is_dir():
                    dirname_list.append(obj.name)
                    paths_temp = [file.path for file in os.scandir(obj.path) if file.name.split(".")[-1] in img_format]
                    qty_list.append(len(paths_temp))
                    if len(paths_temp):
                        paths.extend(paths_temp)
    else:
        for obj in os.scandir(img_source):
            if obj.is_dir():
                dirname_list.append(obj.name)
                paths_temp = [file.path for file in os.scandir(obj.path) if file.name.split(".")[-1] in img_format]
                qty_list.append(len(paths_temp))
                if len(paths_temp):
                    paths.extend(paths_temp)

    return paths, qty_list, dirname_list

def file_decode_v2(file_path, random_num_range=87, save_dir=None, return_value=False, to_save=True, print_f=None):
    # ----var
    print_out = True
    header_len = 4
    decode_flag = True
    header = [24, 97, 28, 98]

    if print_f is None:
        print_f = print

    # ----read the file
    with open(file_path, 'rb') as f:
        content = f.read()

    # ----check headers
    for i in range(4):
        try:
            # print(int(content[i]))
            if int(content[i]) != header[i]:
                decode_flag = False
        except:
            decode_flag = False
    # msg = "decode_flag:{}".format(decode_flag)
    # print_f(msg)

    # ----decode process
    if decode_flag is False:
        if return_value is True:
            return None
    else:
        # print("execute decode")
        leng = len(content)
        # msg = "file length = {}".format(leng)
        # say_sth(msg, print_out=print_out)
        # print_f(msg)

        cut_num_start = random_num_range + header_len
        cut_num = int.from_bytes(content[cut_num_start:cut_num_start + 4], byteorder="little", signed=False)
        # msg = "cut_num = {}".format(cut_num)
        # say_sth(msg, print_out=print_out)
        # print_f(msg)

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
            # ----output filename
            if save_dir is None:
                new_filename = file_path
            else:
                new_filename = os.path.join(save_dir, file_path.split("\\")[-1])

            # ----save the file
            with open(new_filename, 'wb') as f:
                f.write(temp)
            # msg = "decode data is completed in {}".format(new_filename)
            # # say_sth(msg, print_out=print_out)
            # print_f(msg)

        if return_value is True:
            return temp

def model_restore_from_pb(pb_path, node_dict, GPU_ratio=None):
    tf_dict = dict()
    with tf.Graph().as_default():
        config = tf.ConfigProto(log_device_placement=False,  # 印出目前的運算是使用CPU或GPU
                                allow_soft_placement=True,  # 當設備不存在時允許tf選擇一个存在且可用的設備來繼續執行程式
                                )
        if GPU_ratio is None:
            config.gpu_options.allow_growth = True  # 依照程式執行所需要的資料來自動調整
        else:
            config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio  # 手動限制GPU資源的使用
        sess = tf.Session(config=config)
        with gfile.FastGFile(pb_path, 'rb') as f:
            ret = file_decode_v2(pb_path, return_value=True, to_save=False)
            if ret is None:
                content = f.read()
                # print("not encripted")
            else:
                content = ret
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(content)
            sess.graph.as_default()

            # ----issue solution if models with batch norm
            '''
            如果是有batch normalzition，或者残差网络层，会出现：
            ValueError: Input 0 of node InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/cond_1/AssignMovingAvg/Switch was passed 
            float from InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/moving_mean:0 incompatible with expected float_ref.
            ref:https://blog.csdn.net/dreamFlyWhere/article/details/83023256
            '''
            for node in graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']

            tf.import_graph_def(graph_def, name='')  # 匯入計算圖

        sess.run(tf.global_variables_initializer())
        for key, value in node_dict.items():
            try:
                node = sess.graph.get_tensor_by_name(value)
                tf_dict[key] = node
            except:
                print("node {} does not exist in the graph")
        return sess, tf_dict

def get_embeddings(sess, paths, tf_dict, batch_size=128):
    # ----
    len_path = len(paths)
    tf_input = tf_dict['input']
    # tf_phase_train = tf_dict['phase_train']
    tf_embeddings = tf_dict['embeddings']

    feed_dict = dict()
    if 'phase_train' in tf_dict.keys():
        tf_phase_train = tf_dict['phase_train']
        feed_dict[tf_phase_train] = False
    if 'keep_prob' in tf_dict.keys():
        tf_keep_prob = tf_dict['keep_prob']
        feed_dict[tf_keep_prob] = 1.0

    if tf_input.shape[1] is None:
        model_shape = [None, 160, 160, 3]
    else:
        model_shape = [None,tf_input.shape[1].value,tf_input.shape[2].value,tf_input.shape[3].value]
    print("tf_input shape:", model_shape)
    print("tf_embeddings shape:", tf_embeddings.shape)

    # ----
    ites = math.ceil(len_path / batch_size)
    # embeddings = np.zeros([len_path, tf_embeddings.shape[-1]], dtype=np.float32)
    embeddings = np.zeros([len_path, tf_embeddings.shape[-1]], dtype=np.float32)
    for idx in range(ites):
        num_start = idx * batch_size
        num_end = np.minimum(num_start + batch_size, len_path)
        # ----read batch data
        batch_dim = [num_end - num_start]  # [64]
        batch_dim.extend(model_shape[1:])  # [64,160, 160, 3]
        batch_data = np.zeros(batch_dim, dtype=np.float32)
        for idx_path, path in enumerate(paths[num_start:num_end]):
            # img = cv2.imread(path)
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
            if img is None:
                print("Read failed:", path)
            else:
                img = cv2.resize(img, (model_shape[2], model_shape[1]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_data[idx_path] = img
        batch_data /= 255  # norm
        feed_dict[tf_input] = batch_data
        embeddings[num_start:num_end] = sess.run(tf_embeddings, feed_dict=feed_dict)

    return embeddings

def embedding_tutorial(img_dir, databse_source, pb_path):
    #----var
    node_dict = {'input': 'input:0',
                 # 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 'keep_prob': 'keep_prob:0'
                 }
    batch_size = 128
    prediction_list = list()


    #----get all images
    paths_test = [file.path for file in os.scandir(img_dir)]
    qty_test = len(paths_test)
    if qty_test == 0:
        msg = "Error:No images in {}".format(img_dir)
        print(msg)
    else:
        #----get subdir images of face_databse_dir
        paths_ref, qty_ref_list, classname_list = get_paths_2(databse_source)
        qty_ref = len(paths_ref)
        if qty_ref == 0:
            msg = "Error:圖片資料庫無圖片_{}".format(databse_source)
            print(msg)
        else:
            #----model init
            sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=None)
            tf_embeddings = tf_dict['embeddings']

            #----tf setting for calculating distance
            with tf.Graph().as_default():
                tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
                tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
                tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
                # ----GPU setting
                config = tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                        )
                config.gpu_options.allow_growth = True
                sess_cal = tf.Session(config=config)
                sess_cal.run(tf.global_variables_initializer())
            #----calculate embeddings
            embed_ref = get_embeddings(sess, paths_ref, tf_dict, batch_size=batch_size)
            embed_tar = get_embeddings(sess, paths_test, tf_dict, batch_size=batch_size)
            print("embed_tar shape: ", embed_tar.shape)

            #----calculate distances
            feed_dict_2 = {tf_ref: embed_ref}
            distance_list = np.zeros([embed_tar.shape[0], embed_ref.shape[0]], dtype=np.float32)

            for idx in range(embed_tar.shape[0]):
                feed_dict_2[tf_tar] = embed_tar[idx]
                dis = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                distance_list[idx] = dis

            #----create dirs
            new_dir = os.path.join(img_dir, "embedding_results")
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            #----find the smallest distance
            for idx, dis in enumerate(distance_list):
                argmin = np.argmin(dis)
                prediction = paths_ref[argmin].split("\\")[-2]
                path = paths_test[idx]
                splits = path.split("\\")

                # ----save the prediction
                prediction_list.append(prediction)

                # ----save images
                output_dir = os.path.join(new_dir, prediction)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                save_path = os.path.join(output_dir, splits[-1])
                shutil.copy(path, save_path)

if __name__ == "__main__":
    from AE_Seg_prediction import tf_utility

    img_dir = r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\test'
    pb_path = r"D:\code\model_saver\AE_Seg_14\pb_model.pb"
    # pb_path_list = [
    #     # r"D:\code\model_saver\AE_Seg_1\pb_model.pb",
    #     r"D:\code\model_saver\AE_Seg_3\pb_model.pb",
    #     r"D:\code\model_saver\AE_Seg_4\pb_model.pb"
    # ]
    node_dict = {'input': 'input:0',
                 'preprocess': 'preprocess:0',
                 }

    tf_tl = tf_utility()
    tf_tl.print_out = True

    paths, qty = tf_tl.get_paths(img_dir)

    tf_tl.sess, tf_tl.tf_dict = tf_tl.model_restore_from_pb(pb_path, node_dict)
    tf_input = tf_tl.tf_dict['input']
    tf_preprocess = tf_tl.tf_dict['preprocess']

    model_shape = tf_tl.get_model_shape('input')



    for path in paths:

        # ----get batch data
        batch_data = tf_tl.get_4D_data([path],model_shape[1:])

        sessout_preprocess = tf_tl.sess.run(tf_preprocess,
                                       feed_dict={tf_input: batch_data})

        img_ori = (batch_data[0] * 255).astype(np.uint8)
        img_p = (sessout_preprocess[0] * 255).astype(np.uint8)

        plt.figure(figsize=(15,15))

        plt.subplot(1,2,1)
        plt.imshow(img_ori)
        plt.title("ori")

        plt.subplot(1,2,2)
        plt.imshow(img_p)
        plt.title("pre-process")

        plt.show()


    # img_dir = r"D:\dataset\optotech\test_img_2\train_20210924_整理\P007_L1_pad_dent_C"
    # databse_source = r"D:\dataset\optotech\CMIT_009IRC\009_0623+0807總整理資料(訓練用)\Validation"
    # pb_path = r"D:\code\model_saver\Opto_tech\CMIT_Mix_100classes\infer_acc(99.5).nst"
    # embedding_tutorial(img_dir, databse_source, pb_path)

    #----tools
    # from AE_class_2 import tools
    #
    # img_dir = r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\vali\NG"
    # process_dict = {"rdm_flip": True, 'rdm_br': True, 'rdm_blur': True,
    #                 'rdm_angle': True,
    #                 'rdm_noise': False,
    #                 'rdm_shift': True,
    #                 'rdm_patch': True,
    #                 'ave_filter':False,
    #                 'gau_filter':False
    #                 }
    # setting_dict = {'rdm_shift': 0.03, 'rdm_angle': 3, 'rdm_patch': [0.25, 0.3,10],'gau_filter':(7,7)}
    # to_norm = True
    # to_rgb = True
    # to_gray = False
    # batch_size = 16
    # shape = [192,192,3]
    #
    #
    # tl = tools()
    #
    # paths,qty = tl.get_paths(img_dir)
    # print(qty)
    #
    # t_paths = np.random.choice(paths,size=batch_size)
    #
    # #----no processing
    # batch_data = tl.get_4D_data(t_paths,shape , to_norm=to_norm, to_rgb=to_rgb,
    #                             ,to_process=False)
    # print(batch_data.shape)
    # print(np.max(batch_data))
    #
    # #----with processing
    # tl.set_process(process_dict,setting_dict)
    # #aug_data
    # aug_data_no_patch,aug_data = tl.get_4D_data(t_paths, shape,to_norm=to_norm, to_rgb=to_rgb,
    #                                           to_process=True)
    # print(aug_data.shape)
    # print(np.max(aug_data))
    #
    # print(aug_data_no_patch.shape)
    # print(np.max(aug_data_no_patch))
    #
    # for i in range(len(aug_data_no_patch)):
    #     plt.subplot(1,3,1)
    #     plt.imshow(batch_data[i])
    #
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(aug_data_no_patch[i])
    #
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(aug_data[i])
    #
    #     plt.show()


