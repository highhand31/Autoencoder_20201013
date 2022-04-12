import os, sys, cv2, json, math, shutil, time
import numpy as np
import tensorflow
# import pandas as pd
# from scipy.stats import spearmanr
# import matplotlib.pyplot as plt

if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile
print("Tensorflow version: ", tf.__version__)

img_format = {'tif', 'TIF', 'JPG', 'jpg', 'png', 'bmp'}

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
                print("node: {} does not exist in the graph".format(value))
        return sess, tf_dict

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

def get_paths_2(img_source):
    paths = list()
    qty_list = list()
    dirname_list = list()
    list_type = type(list())
    if type(img_source) == list_type:
        for img_dir in img_source:
            for obj in os.scandir(img_dir):
                if obj.is_dir():
                    paths_temp = [file.path for file in os.scandir(obj.path) if file.name.split(".")[-1] in img_format]

                    if len(paths_temp):
                        dirname_list.append(obj.name)
                        qty_list.append(len(paths_temp))
                        paths.extend(paths_temp)
    else:
        for obj in os.scandir(img_source):
            if obj.is_dir():
                paths_temp = [file.path for file in os.scandir(obj.path) if file.name.split(".")[-1] in img_format]

                if len(paths_temp):
                    dirname_list.append(obj.name)
                    qty_list.append(len(paths_temp))
                    paths.extend(paths_temp)

    return paths, qty_list, dirname_list

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

def read_json(json_path,print_out=False):
    #----var
    re_msg_list = list()
    re_dict = dict()
    status = False
    content = None

    print_f = lambda msg,print_out:print(msg) if print_out is True else None

    # ----read the JSON file
    if json_path is None:
        msg = "Warning: no infer result!!"
        print_f(msg,print_out=print_out)
        re_msg_list.append(msg)
    elif not os.path.exists(json_path):
        msg = "Error: the setting file does not exit: {}".format(json_path)
        print_f(msg, print_out=print_out)
        re_msg_list.append(msg)
    else:
        # ----decode the config file
        ret = file_decode_v2(json_path, random_num_range=10, return_value=True, to_save=False)
        if ret is None:
            msg = "The file is not secured"
            print_f(msg, print_out=print_out)
            re_msg_list.append(msg)
            try:
                with open(json_path, 'r') as f:
                    content = json.load(f)
                    status = True
            except:
                msg = "Error: read failed: {}".format(json_path)
                print_f(msg, print_out=print_out)
                re_msg_list.append(msg)
        else:
            msg = "The file is decoded"
            print_f(msg, print_out=print_out)
            re_msg_list.append(msg)
            content = json.loads(ret.decode())
            status = True

    #----return value process
    re_dict['msg_list'] = re_msg_list
    re_dict['content'] = content

    return status,re_dict

def read_latest_infer_result(dir_path):
    name_trainResult = "train_result_"
    file_num_list = list()
    file_list = list()
    status = False

    # ----classname to label dict process
    for file in os.scandir(dir_path):
        if file.name.find(name_trainResult) >= 0:
            file_num_list.append(int(file.name.split(".")[0].split("_")[-1]))
            file_list.append(file.path)
    if len(file_num_list):
        idx_max_num = int(np.argmax(file_num_list))
        path = file_list[idx_max_num]
        status, re_dict = read_json(path)
        re_dict['path'] = path
    else:
        msg = "從權重檔的資料夾找不到任何之前的訓練結果檔案"
        re_dict = {"msg_list":[msg]}

    return status,re_dict

def get_center_points(databse_source, pb_source,std_threshold=2.0):
    # ----var
    node_dict = {'input': 'input:0',
                 # 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 'keep_prob': 'keep_prob:0'
                 }
    batch_size = 128
    GPU_ratio = None
    name_trainInfer_set = {'infer_FFR', "inference_", 'test'}
    #std_threshold = 2

    # ----get subdir images of face_databse_dir
    paths_ref, qty_ref_list, classname_list = get_paths_2(databse_source)
    qty_ref = len(paths_ref)
    print(qty_ref_list)
    print(classname_list)
    if qty_ref == 0:
        msg = "Error:圖片資料庫無圖片_{}".format(databse_source)
        print(msg)
    else:
        # ----pb source
        if os.path.isdir(pb_source):
            pb_files = list()
            for pathName in name_trainInfer_set:
                temp_files = [file.path for file in os.scandir(pb_source) if file.name.find(pathName) >= 0]
                pb_files.extend(temp_files)
            if len(pb_files) == 0:
                print("No pb files in the {}".format(pb_source))
                raise ValueError
            else:
                ctime_list = list()
                for pb_path in pb_files:
                    ctime_list.append(os.path.getctime(pb_path))

                arg_max = int(np.argmax(ctime_list))
                pb_path = pb_files[arg_max]
        else:
            pb_path = pb_source

        print("pb_path:{}".format(pb_path))

        # ----model init
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=GPU_ratio)

        # ----calculate embeddings
        embed_ref = get_embeddings(sess, paths_ref, tf_dict, batch_size=batch_size)
        print("embed_ref shape: ", embed_ref.shape)

        center_embedding = np.zeros((len(classname_list),embed_ref.shape[-1]),dtype=np.float32)
        std_embedding = np.zeros_like(center_embedding)
        num_start = 0
        for i,qty in enumerate(qty_ref_list):
            num_end = num_start + qty

            if qty == 1:
                center_embedding[i] = embed_ref[num_start:num_end]
            else:
                embed_temp = embed_ref[num_start:num_end]
                mean_temp = np.mean(embed_temp,axis=0)
                std_temp = np.std(embed_temp, axis=0)
                if np.sum(std_temp) == 0:
                    center_embedding[i] = embed_temp[0:1]
                else:
                    center_embedding[i] = mean_temp
                std_embedding[i] = std_temp
                #print("class = {},mean = {},std = {}".format(classname_list[i],center_embedding[i],std_embedding[i]))
            num_start = num_end

        #----計算各類的中心與std的距離
        for i, embedding in enumerate(center_embedding):
            std = std_embedding[i]
            a = np.square((embedding - (embedding + std)))
            b = np.sum(a)
            dis_std = np.sqrt(b)

            if dis_std == 0:
                print("★類別:{},std=0，可能只有一張圖片或多張相同的圖片".format(classname_list[i]))
            else:
                a = np.square((center_embedding - embedding))
                b = np.sum(a,axis=1)
                dis_classes = np.sqrt(b)
                # print("★類別:{},std:{},與各類別中心點的距離:{}".format(classname_list[i],dis_std, dis_classes))

                index_sort = np.argsort(dis_classes)
                dis_times = dis_classes/ dis_std
                under_thre_list = np.where(dis_times <= std_threshold)[0]
                if len(under_thre_list) > 1:#至少會有1個，自己
                    print("★類別:{},與本身std間的距離:{}".format(classname_list[i], dis_std))
                    for class_idx in under_thre_list:
                        classname = classname_list[class_idx]
                        dis = round(dis_classes[class_idx],2)
                        times = round(dis_times[class_idx],2)
                        if dis != 0:
                            print("     ◆鄰近的的類別:{}，距離:{},標準差倍數:{}".format(classname,str(dis),str(times)))





        #-----計算各類中心的距離
        # for embedding in center_embedding:
        #
        #     a = np.square((center_embedding - embedding))
        #     b = np.sum(a,axis=1)
        #     c = np.sqrt(b)
        #     print("c = ",c)
        #     print("c = ",c)

        return center_embedding, std_embedding, classname_list

def classification_by_center_embed_comparison_opto(img_source, databse_source, pb_source, coeff_list=[2],
                                            GPU_ratio=None,save_dir=None,with_answers=True):
    # ----var
    node_dict = {'input': 'input:0',
                 # 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 'keep_prob': 'keep_prob:0'
                 }
    batch_size = 128
    name_trainInfer_set = {'infer_FFR', "inference_", 'test'}
    content = dict()
    json_path = None
    paths_underkill = list()
    fp, tp, tn, fn = 0, 0, 0, 0
    coeff = 4.0
    save_dir_list = list()

    d_t = time.time()

    # ----get all images
    if with_answers is False:
        paths_test = list()
        if isinstance(img_source,list):
            for img_dir in img_source:
                temp = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] in img_format]
                paths_test.extend(temp)
        else:
            paths_test = [file.path for file in os.scandir()]
    else:
        paths_test, qty_test_list, classname_test_list = get_paths_2(img_source)
    qty_test = len(paths_test)

    if qty_test == 0:
        msg = "Error:No images in {}".format(img_source)
        print(msg)
    else:
        # ----get subdir images of face_databse_dir
        paths_ref, qty_ref_list, classname_list = get_paths_2(databse_source)
        qty_ref = len(paths_ref)
        if qty_ref == 0:
            msg = "Error:圖片資料庫無圖片_{}".format(databse_source)
            print(msg)
        else:
            # ----pb source
            if os.path.isdir(pb_source):
                pb_files = list()
                for pathName in name_trainInfer_set:
                    temp_files = [file.path for file in os.scandir(pb_source) if file.name.find(pathName) >= 0]
                    pb_files.extend(temp_files)
                if len(pb_files) == 0:
                    print("No pb files in the {}".format(pb_source))
                    raise ValueError
                else:
                    ctime_list = list()
                    for pb_path in pb_files:
                        ctime_list.append(os.path.getctime(pb_path))

                    arg_max = int(np.argmax(ctime_list))
                    pb_path = pb_files[arg_max]
            else:
                pb_path = pb_source

            print("pb_path:{}".format(pb_path))

            # ----model init
            sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=GPU_ratio)
            tf_embeddings = tf_dict['embeddings']

            # ----tf setting for calculating distance
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

            # ----calculate embeddings
            #embed_ref = get_embeddings(sess, paths_ref, tf_dict, batch_size=batch_size)
            center_embedding, std_embedding, classname_list = get_center_points(databse_source,pb_path)
            std_dis_list = np.zeros(center_embedding.shape[0],dtype=np.float32)
            for i,std in enumerate(std_embedding):
                std_dis_list[i] = np.sqrt(np.sum(np.square(std)))
            # print("std_dis_list:",std_dis_list)

            embed_tar = get_embeddings(sess, paths_test, tf_dict, batch_size=batch_size)
            print("embed_tar shape: ", embed_tar.shape)

            feed_dict_2 = {tf_ref: center_embedding}
            distance_list = np.zeros([embed_tar.shape[0], center_embedding.shape[0]], dtype=np.float32)

            for idx in range(embed_tar.shape[0]):
                feed_dict_2[tf_tar] = embed_tar[idx]
                dis = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                distance_list[idx] = dis

            #----create dirs
            if save_dir is not None:

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                for coeff in coeff_list:
                    new_dir = os.path.join(save_dir,"std_{}".format(coeff))
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    save_dir_list.append(new_dir)

            print("pb_path:{}".format(pb_path))
            for idx_coeff,coeff in enumerate(coeff_list):
                if with_answers is False:
                    for idx, dis in enumerate(distance_list):
                        argmin = np.argmin(dis)
                        path = paths_test[idx]
                        splits = path.split("\\")
                        prediction = classname_list[argmin]
                else:
                    count_o = 0
                    count_o_under_threshold = 0
                    count_o_over_std = 0
                    count_x_over_std = 0
                    paths_under_threshold = list()
                    paths_x = list()
                    count_x = 0
                    # count_unknown = 0
                    count_class_o = np.zeros_like(classname_test_list,dtype=np.int32)
                    for idx,dis in enumerate(distance_list):
                        argmin = np.argmin(dis)
                        path = paths_test[idx]
                        splits = path.split("\\")
                        answer = splits[-2]
                        prediction = classname_list[argmin]
                        if prediction == answer:
                            if dis[argmin] > coeff * std_dis_list[argmin]:
                                count_o_over_std += 1
                                save_type = 'correct_outside_std'
                            else:
                                count_o += 1
                                idx_class = classname_test_list.index(prediction)
                                count_class_o[idx_class] += 1
                                save_type = None

                        else:
                            if dis[argmin] > coeff * std_dis_list[argmin]:
                                count_x_over_std += 1
                                save_type = 'wrong_outside_std'
                            else:
                                count_x += 1
                                save_type = 'wrong_inside_std'

                        #----save wrong images
                        if len(save_dir_list) > 0 and save_type is not None:
                            ori_filename = splits[-1].split(".")[0]
                            file_type = splits[-1].split(".")[-1]
                            save_path = "{}_{}_{}.{}".format(ori_filename, prediction, answer, file_type)
                            output_dir = os.path.join(save_dir_list[idx_coeff], save_type)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            save_path = os.path.join(output_dir, save_path)
                            shutil.copy(path, save_path)

                    # ----statistics 1

                    print("\nstd coefficient:{}".format(coeff))
                    print("quantity of correct CLS:{}".format(count_o))
                    print("quantity of incorrect CLS:{}".format(count_x))
                    print("quantity of correct CLS but outside the std distance:{}".format(
                        count_o_over_std))
                    print("quantity of incorrect CLS but outside the std distance:{}".format(
                        count_x_over_std))
                    # print("quantity of unknow CLS:{}".format(count_unknown))
                    print("accuracy:{}".format(count_o / (count_o + count_x  + count_o_over_std + count_x_over_std)))

                    #----statistics 2
                    qty_class_x = 0
                    for i, qty_class_o in enumerate(count_class_o):
                        qty_class_x += qty_test_list[i] - qty_class_o
                        print("類別{}共{}張，分類正確{}張，準確率{}".format(classname_list[i],qty_test_list[i],qty_class_o,qty_class_o/qty_test_list[i]))

        d_t = time.time() - d_t
        print("ave process time = ",d_t / qty_test )

def classification_by_embed_comparison_seg(img_source, databse_source, pb_source, coeff_list=[2.2],
                                            GPU_ratio=None,save_dir=None):
    # ----var
    node_dict = {'input': 'input:0',
                 # 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 'keep_prob': 'keep_prob:0'
                 }
    batch_size = 128
    name_trainInfer_set = {'infer_FFR', "inference_", 'test'}
    content = dict()
    json_path = None
    paths_underkill = list()
    fp, tp, tn, fn = 0, 0, 0, 0
    coeff = 4.0
    save_dir_list = list()

    d_t = time.time()

    # ----get all images
    paths_test, qty_test_list, classname_test_list = get_paths_2(img_source)
    qty_test = len(paths_test)

    if qty_test == 0:
        msg = "Error:No images in {}".format(img_source)
        print(msg)
    else:
        # ----get subdir images of face_databse_dir
        paths_ref, qty_ref_list, classname_list = get_paths_2(databse_source)
        qty_ref = len(paths_ref)
        if qty_ref == 0:
            msg = "Error:圖片資料庫無圖片_{}".format(databse_source)
            print(msg)
        else:
            # ----pb source
            if os.path.isdir(pb_source):
                pb_files = list()
                for pathName in name_trainInfer_set:
                    temp_files = [file.path for file in os.scandir(pb_source) if file.name.find(pathName) >= 0]
                    pb_files.extend(temp_files)
                if len(pb_files) == 0:
                    print("No pb files in the {}".format(pb_source))
                    raise ValueError
                else:
                    ctime_list = list()
                    for pb_path in pb_files:
                        ctime_list.append(os.path.getctime(pb_path))

                    arg_max = int(np.argmax(ctime_list))
                    pb_path = pb_files[arg_max]
            else:
                pb_path = pb_source

            print("pb_path:{}".format(pb_path))

            # ----model init
            sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=GPU_ratio)
            tf_embeddings = tf_dict['embeddings']

            # ----tf setting for calculating distance
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

            # ----calculate embeddings

            # center_embedding, std_embedding, classname_list = get_center_points(databse_source,pb_path)
            # std_dis_list = np.zeros(center_embedding.shape[0],dtype=np.float32)
            # for i,std in enumerate(std_embedding):
            #     std_dis_list[i] = np.sqrt(np.sum(np.square(std)))
            # print("std_dis_list:",std_dis_list)
            embed_ref = get_embeddings(sess, paths_ref, tf_dict, batch_size=batch_size)
            embed_tar = get_embeddings(sess, paths_test, tf_dict, batch_size=batch_size)
            print("embed_tar shape: ", embed_tar.shape)

            feed_dict_2 = {tf_ref: embed_ref}
            distance_list = np.zeros([len(embed_tar), len(embed_ref)], dtype=np.float32)

            for idx in range(embed_tar.shape[0]):
                feed_dict_2[tf_tar] = embed_tar[idx]
                dis = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                distance_list[idx] = dis

            #----create dirs
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                for coeff in coeff_list:
                    new_dir = os.path.join(save_dir,"dis_threshold_{}".format(coeff))
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    save_dir_list.append(new_dir)

            print("pb_path:{}".format(pb_path))
            for idx_coeff,coeff in enumerate(coeff_list):
                count_o = 0
                count_under_threshold = 0
                count_over_threshold = 0
                count_o_over_std = 0
                count_x_over_std = 0
                paths_under_threshold = list()
                paths_x = list()
                count_x = 0
                # count_unknown = 0
                count_class_o = np.zeros_like(classname_test_list,dtype=np.int32)
                for idx,dis in enumerate(distance_list):
                    argmin = np.argmin(dis)
                    path = paths_test[idx]
                    splits = path.split("\\")
                    answer = splits[-2]
                    # prediction = classname_list[argmin]
                    if dis[argmin] <= coeff:
                        count_under_threshold += 1
                        save_type = None
                    else:
                        count_over_threshold += 1
                        save_type = 'outside_the_threshold'

                    # if prediction == answer:
                    #     if dis[argmin] > coeff * std_dis_list[argmin]:
                    #         count_o_over_std += 1
                    #         save_type = 'correct_outside_std'
                    #     else:
                    #         count_o += 1
                    #         idx_class = classname_test_list.index(prediction)
                    #         count_class_o[idx_class] += 1
                    #         save_type = None
                    # else:
                    #     if dis[argmin] > coeff * std_dis_list[argmin]:
                    #         count_x_over_std += 1
                    #         save_type = 'wrong_outside_std'
                    #     else:
                    #         count_x += 1
                    #         save_type = 'wrong_inside_std'

                    #----save wrong images
                    if len(save_dir_list) > 0 and save_type is not None:
                        ori_filename = splits[-1].split(".")[0]
                        file_type = splits[-1].split(".")[-1]
                        # save_path = "{}_{}_{}.{}".format(ori_filename, prediction, answer, file_type)
                        save_path = "{}.{}".format(ori_filename, file_type)
                        output_dir = os.path.join(save_dir_list[idx_coeff], save_type)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        save_path = os.path.join(output_dir, save_path)
                        shutil.copy(path, save_path)

                # ----statistics 1

                print("\ndistance_threshold:{}".format(coeff))
                print("quantity of insider:{}".format(count_under_threshold))
                print("quantity of outsider:{}".format(count_over_threshold))
                # print("quantity of incorrect CLS:{}".format(count_x))
                # print("quantity of correct CLS but outside the std distance:{}".format(
                #     count_o_over_std))
                # print("quantity of incorrect CLS but outside the std distance:{}".format(
                #     count_x_over_std))
                # # print("quantity of unknow CLS:{}".format(count_unknown))
                # print("accuracy:{}".format(count_o / (count_o + count_x  + count_o_over_std + count_x_over_std)))

                #----statistics 2
                # qty_class_x = 0
                # for i, qty_class_o in enumerate(count_class_o):
                #     qty_class_x += qty_test_list[i] - qty_class_o
                #     print("類別{}共{}張，分類正確{}張，準確率{}".format(classname_list[i],qty_test_list[i],qty_class_o,qty_class_o/qty_test_list[i]))

        d_t = time.time() - d_t
        print("ave process time = ",d_t / qty_test )

def center_embed_infer(img_source, databse_source, pb_source, coeff_list=[2], GPU_ratio=None,save_dir=None,
                       embed_type='center',save_image=False,method='mse',prob_threshold=0.9):

    # ----var
    node_dict = {'input': 'input:0',
                 # 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 'keep_prob': 'keep_prob:0'
                 }
    batch_size = 128
    name_trainInfer_set = {'infer_FFR', "inference_", 'test'}
    content = dict()
    json_path = None
    paths_underkill = list()
    fp, tp, tn, fn = 0, 0, 0, 0
    coeff = 4.0
    save_dir_list = list()
    paths_test = list()
    prediction_list = list()

    d_t = time.time()

    # ----get all images
    if isinstance(img_source,list):
        for img_dir in img_source:
            temp = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] in img_format]
            paths_test.extend(temp)
    else:
        paths_test = [file.path for file in os.scandir()]

    qty_test = len(paths_test)

    if qty_test == 0:
        msg = "Error:No images in {}".format(img_source)
        print(msg)
    else:
        # ----get subdir images of face_databse_dir
        paths_ref, qty_ref_list, classname_list = get_paths_2(databse_source)
        qty_ref = len(paths_ref)
        if qty_ref == 0:
            msg = "Error:圖片資料庫無圖片_{}".format(databse_source)
            print(msg)
        else:
            # ----pb source
            if os.path.isdir(pb_source):
                pb_files = list()
                for pathName in name_trainInfer_set:
                    temp_files = [file.path for file in os.scandir(pb_source) if file.name.find(pathName) >= 0]
                    pb_files.extend(temp_files)
                if len(pb_files) == 0:
                    print("No pb files in the {}".format(pb_source))
                    raise ValueError
                else:
                    ctime_list = list()
                    for pb_path in pb_files:
                        ctime_list.append(os.path.getctime(pb_path))

                    arg_max = int(np.argmax(ctime_list))
                    pb_path = pb_files[arg_max]

                    #----find train_result files
                    # ret, re_dict = read_latest_infer_result(pb_source)
                    # if ret is False:
                    #     print("No train_result files")
                    #     raise ValueError
                    # elif ret is True:
                    #     print("train_result file:",re_dict['path'])
                    #     classname_list = re_dict['content']['label_dict']
                    #     print(classname_list)
            else:
                pb_path = pb_source
                # # ----find train_result files
                # ret, re_dict = read_latest_infer_result(os.path.dirname(pb_source))
                # if ret is False:
                #     print("No train_result files")
                #     raise ValueError
                # elif ret is True:
                #     print("train_result file:", re_dict['path'])
                #     classname_list = re_dict['content']['label_dict']
                #     print(classname_list)

            print("pb_path:{}".format(pb_path))

            # ----model init
            sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=GPU_ratio)
            tf_embeddings = tf_dict['embeddings']

            # ----tf setting for calculating distance
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

            # ----calculate embeddings
            if embed_type == 'onebyone':
                embed_ref = get_embeddings(sess, paths_ref, tf_dict, batch_size=batch_size)
            else:
                embed_ref, std_embedding, classname_list = get_center_points(databse_source,pb_path)
                std_dis_list = np.zeros(embed_ref.shape[0],dtype=np.float32)
                for i,std in enumerate(std_embedding):
                    std_dis_list[i] = np.sqrt(np.sum(np.square(std)))
                # print("std_dis_list:",std_dis_list)

            embed_tar = get_embeddings(sess, paths_test, tf_dict, batch_size=batch_size)
            print("embed_tar shape: ", embed_tar.shape)

            feed_dict_2 = {tf_ref: embed_ref}
            distance_list = np.zeros([embed_tar.shape[0], embed_ref.shape[0]], dtype=np.float32)
            if method == 'correlation':
                corre_list = np.zeros([embed_tar.shape[0], embed_ref.shape[0]], dtype=np.float32)
                embed_total = np.concatenate([embed_tar,embed_ref])
                t_1 = time.time()
                corres = np.corrcoef(embed_total)
                t_1 = time.time() - t_1
                print("correlation calculation time:",t_1)
            elif method == 'spearmanr':
                corre_list = np.zeros([embed_tar.shape[0], embed_ref.shape[0]], dtype=np.float32)
                embed_total = np.concatenate([embed_tar, embed_ref])
                t_1 = time.time()
                sp_corres = spearmanr(embed_total,axis=1)
                t_1 = time.time() - t_1
                print("spearmanr calculation time:", t_1)

            for idx in range(embed_tar.shape[0]):
                if method == 'correlation':
                    #embed_temp = np.concatenate([embed_tar[idx:idx+1],embed_ref])
                    #corres = np.corrcoef(embed_temp)[0]
                    corre = corres[idx][len(embed_tar):]
                    corre = (corre + 1) / 2
                    corre_list[idx] = corre
                elif method == 'spearmanr':
                    corre = sp_corres.correlation[idx][len(embed_tar):]
                    corre = (corre + 1) / 2
                    corre_list[idx] = corre
                else:
                    feed_dict_2[tf_tar] = embed_tar[idx]
                    dis = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                    distance_list[idx] = dis

            #----create dirs
            if save_dir is not None:
                save_dir = "{}_{}".format(save_dir,embed_type)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if save_image is True:
                    for coeff in coeff_list:
                        new_dir = os.path.join(save_dir,"std_{}".format(coeff))
                        if not os.path.exists(new_dir):
                            os.makedirs(new_dir)
                        save_dir_list.append(new_dir)

            print("pb_path:{}".format(pb_path))
            if embed_type == 'onebyone':
                for idx in range(len(paths_test)):
                # for idx, dis in enumerate(distance_list):
                    if method == 'correlation':
                        corre = corre_list[idx]
                        argmax = np.argmax(corre)
                        prediction = paths_ref[argmax].split("\\")[-2]
                        if corre[argmax] >= prob_threshold:
                            prediction += '_inside'
                        else:
                            prediction += '_outside'
                    elif method == 'spearmanr':
                        corre = corre_list[idx]
                        argmax = np.argmax(corre)
                        prediction = paths_ref[argmax].split("\\")[-2]
                        if corre[argmax] >= prob_threshold:
                            prediction += '_inside'
                        else:
                            prediction += '_outside'
                    else:
                        dis = distance_list[idx]
                        argmin = np.argmin(dis)
                        prediction = paths_ref[argmin].split("\\")[-2]
                        if dis[argmin] < 0.2:
                            prediction += '_inside'
                        else:
                            prediction += '_outside'


                    path = paths_test[idx]
                    splits = path.split("\\")

                    #----save the prediction
                    prediction_list.append(prediction)

                    # ----save images
                    if len(save_dir_list) > 0:
                        # ori_filename = splits[-1].split(".")[0]
                        # file_type = splits[-1].split(".")[-1]
                        # save_path = "{}_{}_{}.{}".format(ori_filename, prediction, answer, file_type)
                        output_dir = os.path.join(save_dir_list[0], prediction)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        save_path = os.path.join(output_dir, splits[-1])
                        shutil.copy(path, save_path)
            else:
                for idx_coeff,coeff in enumerate(coeff_list):

                    for idx, dis in enumerate(distance_list):
                        argmin = np.argmin(dis)
                        path = paths_test[idx]
                        splits = path.split("\\")
                        prediction = classname_list[argmin]
                        if dis[argmin] > coeff * std_dis_list[argmin]:
                            #count_x_over_std += 1
                            save_type = 'outside_std'
                        else:
                            #count_x += 1
                            save_type = 'inside_std'

                        #----save the prediction
                        prediction_list.append(prediction + '_' + save_type)

                        # ----save images
                        if len(save_dir_list) > 0 and save_type is not None:
                            # ori_filename = splits[-1].split(".")[0]
                            # file_type = splits[-1].split(".")[-1]
                            # save_path = "{}_{}_{}.{}".format(ori_filename, prediction, answer, file_type)
                            output_dir = os.path.join(save_dir_list[idx_coeff], save_type,prediction)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            save_path = os.path.join(output_dir, splits[-1])
                            shutil.copy(path, save_path)
                    # else:
                    #     count_o = 0
                    #     count_o_under_threshold = 0
                    #     count_o_over_std = 0
                    #     count_x_over_std = 0
                    #     paths_under_threshold = list()
                    #     paths_x = list()
                    #     count_x = 0
                    #     # count_unknown = 0
                    #     count_class_o = np.zeros_like(classname_test_list,dtype=np.int32)
                    #     for idx,dis in enumerate(distance_list):
                    #         argmin = np.argmin(dis)
                    #         path = paths_test[idx]
                    #         splits = path.split("\\")
                    #         answer = splits[-2]
                    #         prediction = classname_list[argmin]
                    #         if prediction == answer:
                    #             if dis[argmin] > coeff * std_dis_list[argmin]:
                    #                 count_o_over_std += 1
                    #                 save_type = 'correct_outside_std'
                    #             else:
                    #                 count_o += 1
                    #                 idx_class = classname_test_list.index(prediction)
                    #                 count_class_o[idx_class] += 1
                    #                 save_type = None
                    #
                    #         else:
                    #             if dis[argmin] > coeff * std_dis_list[argmin]:
                    #                 count_x_over_std += 1
                    #                 save_type = 'wrong_outside_std'
                    #             else:
                    #                 count_x += 1
                    #                 save_type = 'wrong_inside_std'
                    #
                    #         #----save wrong images
                    #         if len(save_dir_list) > 0 and save_type is not None:
                    #             ori_filename = splits[-1].split(".")[0]
                    #             file_type = splits[-1].split(".")[-1]
                    #             save_path = "{}_{}_{}.{}".format(ori_filename, prediction, answer, file_type)
                    #             output_dir = os.path.join(save_dir_list[idx_coeff], save_type)
                    #             if not os.path.exists(output_dir):
                    #                 os.makedirs(output_dir)
                    #             save_path = os.path.join(output_dir, save_path)
                    #             shutil.copy(path, save_path)
                    #
                    #     # ----statistics 1
                    #
                    #     print("\nstd coefficient:{}".format(coeff))
                    #     print("quantity of correct CLS:{}".format(count_o))
                    #     print("quantity of incorrect CLS:{}".format(count_x))
                    #     print("quantity of correct CLS but outside the std distance:{}".format(
                    #         count_o_over_std))
                    #     print("quantity of incorrect CLS but outside the std distance:{}".format(
                    #         count_x_over_std))
                    #     # print("quantity of unknow CLS:{}".format(count_unknown))
                    #     print("accuracy:{}".format(count_o / (count_o + count_x  + count_o_over_std + count_x_over_std)))
                    #
                    #     #----statistics 2
                    #     qty_class_x = 0
                    #     for i, qty_class_o in enumerate(count_class_o):
                    #         qty_class_x += qty_test_list[i] - qty_class_o
                    #         print("類別{}共{}張，分類正確{}張，準確率{}".format(classname_list[i],qty_test_list[i],qty_class_o,qty_class_o/qty_test_list[i]))

            #----save results in JSON
            # ====save path
            xtime = time.localtime()
            tailer = ''
            for i in range(6):
                tailer += str(xtime[i])
            # if to_encode is True:
            #     extension = 'nst'
            save_path = "{}_{}.{}".format('inferResult', tailer, 'json')
            save_path = os.path.join(save_dir, save_path)
            print("JSON save path:",save_path)

            content = dict()
            content['img_source'] = img_source
            content['databse_source'] = databse_source
            content['pb_path'] = pb_path
            content['path_list'] = paths_test
            content['embed_type'] = embed_type
            content['std_threshold'] = coeff_list
            content['prediction_list'] = prediction_list
            content['classname_list'] = list(set(classname_list))
            # if method == 'correlation':
            #     corre_list = corre_list.astype(float)
            #     content['corre_list'] = corre_list.tolist()

            with open(save_path, 'w') as f:
                json.dump(content, f)

        d_t = time.time() - d_t
        print("ave process time = ",d_t / qty_test )

def get_tsv_files(img_source,pb_source,node_dict,save_dir,add_center_point=False):
    # ----var
    # node_dict = {'input': 'input:0',
    #              # 'phase_train': 'phase_train:0',
    #              'embeddings': 'embeddings:0',
    #              'keep_prob': 'keep_prob:0'
    #              }
    batch_size = 128
    name_trainInfer_set = {'infer_FFR', "inference_", 'test'}
    paths = list()
    labels = list()

    #----get paths
    # dirs = [obj.path for obj in os.scandir(root_img_dir) if obj.is_dir()]
    paths, qty_list, classname_list = get_paths_2(img_source)



    qty = len(paths)

    if qty == 0:
        msg = "Error:No images in {}".format(img_source)
        print(msg)
    else:
        #----get labels for dir name
        for path in paths:
            label = path.split("\\")[-2]
            labels.append(label)
        labels = np.array(labels)

        #----pb source
        if os.path.isdir(pb_source):
            pb_files = list()
            for pathName in name_trainInfer_set:
                temp_files = [file.path for file in os.scandir(pb_source) if file.name.find(pathName) >= 0]
                pb_files.extend(temp_files)
            if len(pb_files) == 0:
                print("No pb files in the {}".format(pb_source))
                raise ValueError
            else:
                ctime_list = list()
                for pb_path in pb_files:
                    ctime_list.append(os.path.getctime(pb_path))

                arg_max = int(np.argmax(ctime_list))
                pb_path = pb_files[arg_max]
        else:
            pb_path = pb_source
        print("pb_path:{}".format(pb_path))

        #----model init
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=None)

        #----get embeddings
        embeds = get_embeddings(sess, paths, tf_dict, batch_size=batch_size)
        embeds = embeds.astype(float)
        print(labels.shape)
        print(embeds.shape)

        #----get center points
        if add_center_point is True:
            center_embedding = np.zeros((len(classname_list), embeds.shape[-1]), dtype=float)
            num_start = 0
            for i, qty in enumerate(qty_list):
                num_end = num_start + qty

                embed_temp = embeds[num_start:num_end]
                mean_temp = np.mean(embed_temp, axis=0)
                center_embedding[i] = mean_temp

                num_start = num_end

            #----concat data
            embeds = np.concatenate([center_embedding,embeds])
            center_names = list()
            for name in classname_list:
                center_names.append(name + "_center")
            labels = np.concatenate([center_names,labels])



        #----save as files
        filenames = ['data.tsv','label.tsv']
        # filenames = ['label.tsv']
        for i,content in enumerate([embeds,labels]):
            df = pd.DataFrame(content)
            save_path = os.path.join(save_dir,filenames[i])
            with open(save_path,'w') as obj_w:
                obj_w.write(df.to_csv(sep='\t',index=False,header=False))
                print("The {} is saved:{}".format(filenames[i].split(".")[0],save_path))

if __name__ == "__main__":
    #----check classname and qty
    # img_source = r"D:\dataset\optotech\CMIT_009IRC\009_0623+0807總整理資料(訓練用)\Training"
    # paths, qty_list, dirname_list = get_paths_2(img_source)
    #
    # for classname,qty in zip(dirname_list,qty_list):
    #     print("classname:{},qty:{}".format(classname,qty))

    #----get center points of image group
    # databse_source = r"D:\dataset\optotech\CMIT_009IRC\009_0623+0807總整理資料(訓練用)\Training"
    # databse_source = r"D:\dataset\optotech\test_img_2\train_20210907"
    databse_source = r"D:\dataset\optotech\test_img_2\train_20210924_整理"
    # databse_source = r"C:\Users\User\Desktop\file_test\database\only_OK"
    # databse_source = r"C:\Users\User\Desktop\file_test\database\only_OK_2"

    # pb_path = r"D:\code\model_saver\Opto_tech\CMIT_009_data0623_data0807\inference_20210914085714.nst"
    pb_path = r"D:\code\model_saver\Opto_tech\CMIT_009_53classes_192x192_noPool_concatPreprocess_rot_3\infer_acc(99.8).nst"
    # pb_path = r"D:\code\model_saver\Opto_tech\CMIT_Mix_100classes\infer_acc(99.5).nst"
    # pb_path = r"D:\code\model_saver\Opto_tech\CMIT_Mix_100classes\infer_acc(100.0).nst"
    # center_embedding, std_embedding, classname_list = get_center_points(databse_source, pb_path,std_threshold=1.0)
    # for classname,center_series in zip(classname_list,center_embedding):
    #     print("classname{},center_series:{}".format(classname,center_series))

    #----get tsv files
    img_source = [r"D:\dataset\optotech\CMIT_009IRC\009_0623+0807總整理資料(訓練用)\Training"]
    # root_dir = r"D:\dataset\optotech\test_img_2\train_20210924_整理"
    # root_dir = r"D:\dataset\fashion_mnist\train"
    # img_source = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]
    # pb_path = r"D:\code\model_saver\fashionM_exp25\test_20216281592.nst"
    # pb_path = r"D:\code\model_saver\fashionM_exp48\test_202172182424.nst"
    pb_path = r"D:\code\model_saver\Opto_tech\CMIT_009_53classes_192x192_noPool_concatPreprocess_rot_3\infer_acc(99.8).nst"
    node_dict = {'input': 'input:0',
                 # 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 # 'embeddings': 'prediction:0',
                 'keep_prob': 'keep_prob:0'
                 }
    save_dir = r"C:\Users\User\Desktop\file_test"
    # get_tsv_files(img_source, pb_path,node_dict,save_dir,add_center_point=True)


    #----Embedding comparison
    root_dir = r"D:\dataset\optotech\CMIT_009IRC\7_Data\total\training"
    # root_dir = r"D:\dataset\optotech\test_img_2\train_20210924_整理"
    img_source = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]
    # img_source = [
    #     r"D:\dataset\optotech\CMIT_009IRC\7_Data\total\training\L2_OK",
    #     r"D:\dataset\optotech\CMIT_009IRC\7_Data\total\training\L2_NG",
    #     # r"D:\dataset\optotech\test_img_2\test\scratch",
    #     # r"D:\dataset\optotech\test_img_2\test\scratch_2",
    #     # r"D:\dataset\optotech\test_img_2\test\pollution",
    #     # r"D:\dataset\optotech\test_img_2\test\lost_Al"
    # ]

    #----比對的根圖片資料夾
    # databse_source = r"D:\dataset\optotech\CMIT_009IRC\009_0623+0807總整理資料(訓練用)\Training"
    # databse_source = r"D:\dataset\optotech\test_img_2\train_20210907"
    # databse_source = r"D:\dataset\optotech\test_img_2\train_20210924_整理"
    # databse_source = r"D:\dataset\optotech\CMIT_009IRC\009_0623+0807總整理資料(訓練用)\only_OK"
    databse_source = r"C:\Users\User\Desktop\file_test\database\only_OK\ori"
    # databse_source = r"C:\Users\User\Desktop\file_test\database\only_OK_2"

    # pb_path = r"D:\code\model_saver\st_2118_chip_16crops"#可以填入檔案路徑或資料夾，若為資料夾會自動找最好的-->最新的pb file
    pb_path = r"D:\code\model_saver\Opto_tech\CMIT_009_data0623_data0807\inference_20210914085714.nst"
    # pb_path = r"D:\code\model_saver\Opto_tech\CMIT_Mix_100classes\infer_acc(99.5).nst"
    # pb_path = r"D:\code\model_saver\Opto_tech\CMIT_Mix_100classes\infer_acc(100.0).nst"
    # pb_path = r"D:\code\model_saver\Opto_tech\CMIT_OK_3classes\infer_acc(100.0).pb"
    # pb_path = r"D:\code\model_saver\Opto_tech\CMIT_009_53classes_192x192_noPool_concatPreprocess_rot_3\infer_acc(99.8).nst"
    pb_path = r"D:\code\model_saver\AE_Rot_11\pb_model.pb"
    # pb_path = r"D:\code\model_saver\Opto_tech\CMIT_009_41classes_embed256_20211029\infer_acc(91.3).nst"
    #----標準差倍數
    coeff_list = [0.5]
    dis_threshold_list = [0.035,0.04,0.05]
    # embed_type = 'center'
    embed_type = 'onebyone'

    #----儲存資料夾
    #若不儲存，設定None
    #若儲存，只會儲存分類錯誤及分類正確但距離大於std的圖片
    save_dir = r"C:\Users\User\Desktop\file_test\embed_results"
    # save_dir = r"D:\dataset\optotech\test_img_2\embed_results_192x192_IRV1_totalNoPool_concatProcessed_rot_embed128_53classes"
    # save_dir = None
    save_image = False
    method = "spearmanr"#"none"#'correlation'#spearmanr
    prob_threshold = 0.99

    center_embed_infer(img_source, databse_source, pb_path, coeff_list=coeff_list,
                                                   GPU_ratio=None,save_dir=save_dir,embed_type=embed_type,
                       save_image=save_image,method=method,prob_threshold=prob_threshold)

    # classification_by_center_embed_comparison_opto(img_source, databse_source, pb_path, coeff_list=coeff_list,
    #                                                GPU_ratio=None,save_dir=save_dir,with_answers=False)

    # classification_by_embed_comparison_seg(img_source, databse_source, pb_path, coeff_list=dis_threshold_list,
    #                                                GPU_ratio=None,save_dir=save_dir)
