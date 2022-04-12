import os, sys, cv2, json, math, shutil, time
import numpy as np
import tensorflow
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

def augmentation(img_dir, process_dict=None, aug_num=10, setting_dict=None, output_dir=None):
    # ----var
    processing_enable = False
    crop_range = (3, 3)  # [x_range, y_range]
    shift_range = (3, 3)  # [x_shift, y_shift]
    flip_list = (1, 0, -1, 2)
    br_range = (0.88, 1.12)
    # count = 0
    kernel_list = (1, 3, 5)
    angle_range = (-5, 5)

    # ----collect paths
    paths = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] in img_format]
    qty = len(paths)
    if qty == 0:
        print('No images')
        raise ValueError
    else:
        print("image quantity:", qty)

    # ----setting dict
    if setting_dict is not None:
        if 'rdm_crop' in setting_dict.keys():
            crop_range = setting_dict['rdm_crop']
            # print("crop_range[x_range, y_range] is changed to ",crop_range)
        if 'rdm_shift' in setting_dict.keys():
            shift_range = setting_dict['rdm_shift']
        if 'rdm_br' in setting_dict.keys():
            br_range = setting_dict['rdm_br']
            # print("br_range is changed to ",br_range)
        if 'rdm_flip' in setting_dict.keys():
            flip_list = setting_dict['rdm_flip']
            # print("flip_list is changed to ",flip_list)
        if 'rdm_blur' in setting_dict.keys():
            kernel_list = setting_dict['rdm_blur']
            # print("kernel_list is changed to ",kernel_list)
        if 'rdm_angle' in setting_dict.keys():
            angle_range = setting_dict['rdm_angle']
            # print("angle_range is changed to ",angle_range)

    # ----create default np array
    # batch_dim = [len_path]
    # batch_dim.extend(output_shape)
    # batch_data = np.zeros(batch_dim, dtype=np.uint8)

    # ----check process_dict
    if isinstance(process_dict, dict):
        if len(process_dict) > 0:
            processing_enable = True  # image processing is enabled
            # if 'rdm_crop' in process_dict.keys():
            #     x_start = np.random.randint(x_range,size=len_path)
            #     y_start = np.random.randint(y_range,size=len_path)
    else:
        print("No aug method!!")
        raise ValueError

    # ----create output dir
    if output_dir is None:
        output_dir = os.path.join(img_dir, 'Aug')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, path in enumerate(paths):
        # img_ori = cv2.imread(path)
        img_ori = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        if img_ori is None:
            print("read failed:", path)
        else:
            for i in range(aug_num):
                splits = path.split("\\")[-1].split(".")
                extension = splits[-1]
                if extension in ('jpg', 'JPG', 'JPEG'):
                    extension = 'png'
                img = img_ori.copy()

                # ----image processing
                if processing_enable is True:
                    if process_dict.get('rdm_crop') is True:
                        # ----From the random point, crop the image
                        x = np.random.randint(crop_range[0])
                        y = np.random.randint(crop_range[1])
                        img = img[y:-(y + 1), x:-(x + 1), :]
                    if process_dict.get('rdm_shift') is True:
                        x = np.random.randint(shift_range[0])
                        y = np.random.randint(shift_range[1])
                        c = np.random.randint(4)
                        if c == 0:
                            img = img[y:, x:, :]
                        elif c == 1:
                            img = img[y:, :-(x + 1), :]
                        elif c == 2:
                            img = img[:-(y + 1), x:, :]
                        elif c == 3:
                            img = img[:-(y + 1), :-(x + 1), :]
                        else:
                            img = img[y:, x:, :]
                    if process_dict.get('rdm_br') is True:
                        mean_br = np.mean(img)
                        try:
                            br_factor = np.random.randint(math.floor(mean_br * br_range[0]),
                                                          math.ceil(mean_br * br_range[1]))
                            img = np.clip(img / mean_br * br_factor, 0, 255)
                            img = img.astype(np.uint8)
                        except:
                            msg = "Error:rdm_br value"
                            print(msg)
                    if process_dict.get('rdm_flip') is True:
                        flip_type = np.random.choice(flip_list)
                        img = cv2.flip(img, flip_type)
                    if process_dict.get('rdm_blur') is True:
                        kernel = tuple(np.random.choice(kernel_list, size=2))
                        # print("kernel:", kernel)
                        img = cv2.GaussianBlur(img, kernel, 0, 0)
                    if process_dict.get('rdm_noise') is True:
                        uniform_noise = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
                        cv2.randu(uniform_noise, 0, 255)
                        ret, impulse_noise = cv2.threshold(uniform_noise, 230, 255, cv2.THRESH_BINARY_INV)
                        img = cv2.bitwise_and(img, img, mask=impulse_noise)
                    if process_dict.get('rdm_angle') is True:
                        angle = np.random.randint(angle_range[0], angle_range[1])
                        h, w = img.shape[:2]
                        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                        img = cv2.warpAffine(img, M, (w, h))

                # ----resize and change the color format
                # img = cv2.resize(img,(output_shape[1],output_shape[0]))
                # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                # batch_data[idx] = img

                # ----save images

                new_path = "{}_aug{}.{}".format(splits[0], i + 1, extension)
                # new_path = "Aug_{}_{}.png".format(str(idx),count)
                new_path = os.path.join(output_dir, new_path)
                cv2.imwrite(new_path, img)
                # count += 1

def change_filename(img_dir):
    paths = [file.path for file in os.scandir(img_dir) if file.name[-3:] in img_format]

    for idx, path in enumerate(paths):
        splits = path.split("\\")
        class_name = splits[-2]
        new_path = "{}_{}.{}".format(class_name, str(idx), path[-3:])
        new_path = os.path.join(img_dir, new_path)
        print(new_path)

        # ----replace filename
        os.replace(path, new_path)

def create_label_dict(root_dir, to_replace=False):
    name2label = dict()
    label2name = dict()
    count = 0

    for obj in os.scandir(root_dir):
        if obj.is_dir():
            name2label[obj.name] = count
            label2name[str(count)] = obj.name

            # ----modify the dir name
            new_dirname = os.path.join(root_dir, str(count))
            if to_replace is True:
                os.replace(obj.path, new_dirname)

            count += 1

    content = {"name2label": name2label, "label2name": label2name}
    save_path = os.path.join(root_dir, "label_dict.json")
    with open(save_path, 'w')as f:
        json.dump(content, f)

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

def get_tf_outputs(sess, paths, tf_dict, batch_size=128):
    # ----
    len_path = len(paths)
    tf_input = tf_dict['input']
    # tf_phase_train = tf_dict['phase_train']
    tf_output = tf_dict['output']

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
    print("tf_output shape:", tf_output.shape)

    # ----
    ites = math.ceil(len_path / batch_size)
    # embeddings = np.zeros([len_path, tf_embeddings.shape[-1]], dtype=np.float32)
    #embeddings = np.zeros([len_path, tf_embeddings.shape[-1]], dtype=np.float32)
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
        sess.run(tf_output, feed_dict=feed_dict)
        #embeddings[num_start:num_end] = sess.run(tf_embeddings, feed_dict=feed_dict)

    #return embeddings

def get_sess_out(sess, paths, tf_dict, output_name):
    # ----
    # len_path = len(paths)
    tf_input = tf_dict['input']
    # tf_phase_train = tf_dict['phase_train']
    tf_output = tf_dict[output_name]

    feed_dict = dict()
    if 'phase_train' in tf_dict.keys():
        tf_phase_train = tf_dict['phase_train']
        feed_dict[tf_phase_train] = False
    if 'keep_prob' in tf_dict.keys():
        tf_keep_prob = tf_dict['keep_prob']
        feed_dict[tf_keep_prob] = 1.0

    model_shape = tf_input.shape
    # model_shape = [None,160,160,3]
    # print("tf_input shape:", model_shape)
    # print("tf_output shape:", tf_output.shape)

    # ----
    # ----read batch data
    batch_data = list()
    for idx_path, path in enumerate(paths):
        # img = cv2.imread(path)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        if img is None:
            print("Read failed:", path)
        else:
            img = cv2.resize(img, (model_shape[2], model_shape[1]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch_data.append(img)
    batch_data = np.array(batch_data)
    batch_data = batch_data.astype(np.float32)
    batch_data /= 255  # norm
    feed_dict[tf_input] = batch_data
    sess_out = sess.run(tf_output, feed_dict=feed_dict)

    return sess_out

def face_matching_evaluation(img_dir, face_databse_dir, pb_path, test_num=None, threshold=0.7, GPU_ratio=None):
    # ----var
    paths = list()
    paths_ref = list()
    node_dict = {'input': 'input:0',
                 # 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 'keep_prob': 'keep_prob:0'
                 }
    batch_size = 128
    content = dict()

    # ----get all images
    # for dir_name, subdir_names, filenames in os.walk(root_dir):
    #     if len(filenames):
    #         for file in filenames:
    #             if file[-3:] in img_format:
    #                 paths.append(os.path.join(dir_name,file))
    paths = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] in img_format]

    if len(paths) == 0:
        print("No images in ", img_dir)
    else:
        # ----create output dir
        under_th_dir = os.path.join(img_dir, 'under_threshold')
        over_th_dir = os.path.join(img_dir, 'over_threshold')
        dir_list = [under_th_dir, over_th_dir]
        for dir_path in dir_list:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # ----test_num process
        if test_num is not None:
            # paths = np.random.choice(paths,test_num)
            test_num = np.minimum(test_num, len(paths))
            paths = paths[:test_num]

        # ----get images of face_databse_dir
        for dir_name, subdir_names, filenames in os.walk(face_databse_dir):
            if len(filenames):
                for file in filenames:
                    if file[-3:] in img_format:
                        paths_ref.append(os.path.join(dir_name, file))
        len_path_ref = len(paths_ref)
        if len_path_ref == 0:
            print("No images in ", face_databse_dir)
        else:
            # ----model init
            sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=GPU_ratio)
            tf_embeddings = tf_dict['embeddings']

            # ----tf setting for calculating distance
            with tf.Graph().as_default():
                tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
                tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
                tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
                # ----GPU setting
                config = tf.ConfigProto(log_device_placement=True,
                                        allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                        )
                config.gpu_options.allow_growth = True
                sess_cal = tf.Session(config=config)
                sess_cal.run(tf.global_variables_initializer())

            # ----calculate embeddings
            embed_ref = get_embeddings(sess, paths_ref, tf_dict, batch_size=batch_size)
            embed_tar = get_embeddings(sess, paths, tf_dict, batch_size=batch_size)
            print("embed_ref shape: ", embed_ref.shape)
            print("embed_tar shape: ", embed_tar.shape)

            # ----calculate distance and get the minimum
            arg_dis = list()
            dis_list = list()
            count_o = 0
            count_unknown = 0
            feed_dict_2 = {tf_ref: embed_ref}
            for idx, embedding in enumerate(embed_tar):
                feed_dict_2[tf_tar] = embedding
                distance = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                arg_temp = np.argsort(distance)[:1]
                arg_dis.append(arg_temp)
                dis_list.append(distance[arg_temp])

            for path, arg_list, dises in zip(paths, arg_dis, dis_list):
                # answer = path.split("\\")[-2]
                # answer = path.split("\\")[-1].split('_')[0]
                splits = path.split("\\")[-1].split(".")
                for arg, dis in zip(arg_list, dises):
                    save_path = "{}_{}.{}".format(splits[0], dis, splits[-1])
                    if dis <= threshold:
                        save_path = os.path.join(under_th_dir, save_path)
                        shutil.copy(path, save_path)
                    else:
                        save_path = os.path.join(over_th_dir, save_path)
                        shutil.copy(path, save_path)
                    content[path] = float(dis)
                    # prediction = paths_ref[arg].split("\\")[-1].split("_")[0]
                    # prediction = paths_ref[arg].split("\\")[-2]
                    # if prediction == answer:
                    #     count_o += 1
                    # else:
                    #     print("\n{},label:{},prediction:{},prob:{}".format(path, answer, prediction,None))
                    #     print("similar image:",paths_ref[arg])
                    # else:
                    #     count_unknown += 1

            json_path = 'embedding_evaluation.json'
            with open(json_path, 'w') as f:
                json.dump(content, f)

            # ----statistics
            # print("accuracy: ",count_o /len(paths) )
            # print("unknown: ",count_unknown /len(paths) )

def embedding_comparison(img_source, databse_source, pb_path, record_qty=10, GPU_ratio=None, compare_type='under',
                         to_save=False):
    # ----var
    node_dict = {'input': 'input:0',
                 # 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 'keep_prob': 'keep_prob:0'
                 }
    batch_size = 128
    content = dict()
    json_path = None

    # ----get all images
    paths = get_paths(img_source)

    if len(paths) == 0:
        print("No images in ", img_dir)
    else:
        # ----create output dir
        # under_th_dir = os.path.join(img_dir,'under_threshold')
        # over_th_dir = os.path.join(img_dir,'over_threshold')
        # dir_list = [under_th_dir,over_th_dir]
        # for dir_path in dir_list:
        #     if not os.path.exists(dir_path):
        #         os.makedirs(dir_path)

        # ----test_num process
        # if test_num is not None:
        #     # paths = np.random.choice(paths,test_num)
        #     test_num = np.minimum(test_num,len(paths))
        #     paths = paths[:test_num]

        # ----get images of face_databse_dir
        paths_ref = get_paths(databse_source)
        # for dir_name, subdir_names, filenames in os.walk(databse_dir):
        #     if len(filenames):
        #         for file in filenames:
        #             if file[-3:] in img_format:
        #                 paths_ref.append(os.path.join(dir_name, file))
        len_path_ref = len(paths_ref)
        if len_path_ref == 0:
            print("No images in ", databse_source)
        else:
            # ----model init
            sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=GPU_ratio)
            tf_embeddings = tf_dict['embeddings']

            # ----tf setting for calculating distance
            with tf.Graph().as_default():
                tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
                tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
                tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
                # ----GPU setting
                config = tf.ConfigProto(log_device_placement=True,
                                        allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                        )
                config.gpu_options.allow_growth = True
                sess_cal = tf.Session(config=config)
                sess_cal.run(tf.global_variables_initializer())

            # ----calculate embeddings
            embed_ref = get_embeddings(sess, paths_ref, tf_dict, batch_size=batch_size)
            embed_tar = get_embeddings(sess, paths, tf_dict, batch_size=batch_size)
            print("embed_ref shape: ", embed_ref.shape)
            print("embed_tar shape: ", embed_tar.shape)

            # ----calculate distance and get the minimum
            arg_dis = list()
            dis_list = list()
            count_o = 0
            count_unknown = 0
            feed_dict_2 = {tf_ref: embed_ref}
            distance_list = np.zeros([embed_tar.shape[0], embed_ref.shape[0]], dtype=float)
            d_t = time.time()
            # for idx, embedding in enumerate(embed_tar):
            for idx in range(embed_tar.shape[0]):
                feed_dict_2[tf_tar] = embed_tar[idx]
                # distance = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                # arg_dis = np.argsort(distance)
                # distance_list.append(sess_cal.run(tf_dis, feed_dict=feed_dict_2))
                distance_list[idx] = sess_cal.run(tf_dis, feed_dict=feed_dict_2)

            arg_dis_list = np.argsort(distance_list)
            for idx, arg_dis in enumerate(arg_dis_list):

                if compare_type == 'over':
                    arg_dis = arg_dis[::-1]
                    # identical_paths = np.where(arg_dis >= threshold)
                # else:
                # identical_paths = np.where(distance <= threshold)

                # print("ori path:",paths[idx])
                temp_path = list()
                temp_dis = list()
                distance = distance_list[idx]
                for i in arg_dis[:record_qty]:
                    # if distance_list[idx][i] <= threshold:
                    # if not distance_list[idx][i] <= 1e-5:#距離不等於0
                    # if i != idx:
                    # print("距離0的圖:",paths[i])
                    temp_path.append(paths_ref[i])
                    temp_dis.append((distance[i]))  # 前面的distance定義時就已經將dtype=float，這邊就不用再轉換

                content[paths[idx]] = {'paths': temp_path, 'distance': temp_dis}
                # arg_temp = np.argsort(distance)[:1]
                # arg_dis.append(arg_temp)
                # dis_list.append(distance[arg_temp])
            d_t = time.time() - d_t
            print("process time: ", d_t)
            # for path, arg_list,dises in zip(paths,arg_dis,dis_list):
            #     #answer = path.split("\\")[-2]
            #     # answer = path.split("\\")[-1].split('_')[0]
            #     for arg,dis in zip(arg_list,dises):
            #         if dis <= threshold:
            #             save_path = os.path.join(under_th_dir,path.split("\\")[-1])
            #             shutil.copy(path,save_path)
            #         else:
            #             save_path = os.path.join(over_th_dir, path.split("\\")[-1])
            #             shutil.copy(path, save_path)
            #         content[path] = float(dis)
            #             # prediction = paths_ref[arg].split("\\")[-1].split("_")[0]
            #             # prediction = paths_ref[arg].split("\\")[-2]
            #             # if prediction == answer:
            #             #     count_o += 1
            #             # else:
            #             #     print("\n{},label:{},prediction:{},prob:{}".format(path, answer, prediction,None))
            #             #     print("similar image:",paths_ref[arg])
            #         # else:
            #         #     count_unknown += 1
            if to_save is True:
                json_path = 'embedding_evaluation.json'
                with open(json_path, 'w') as f:
                    json.dump(content, f)

            # ----statistics
            # print("accuracy: ",count_o /len(paths) )
            # print("unknown: ",count_unknown /len(paths) )

    return content, json_path

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

def get_paths(img_source, append=False):
    paths = list()
    list_type = type(list())
    if append is True:
        qty = list()
    else:
        qty = 0

    if type(img_source) == list_type:
        for img_dir in img_source:
            temp_paths = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] in img_format]
            if len(temp_paths) > 0:
                if append is True:
                    paths.append(temp_paths)
                    qty.append(len(temp_paths))
                else:
                    paths.extend(temp_paths)
                    qty += len(temp_paths)
    else:
        temp_paths = [file.path for file in os.scandir(img_source) if file.name.split(".")[-1] in img_format]
        if len(temp_paths) > 0:
            qty = len(temp_paths)
            paths.extend(temp_paths)

    return paths

def show_embed_results(source, show_qty=1):
    # ----var
    qty_result = 0
    content = None

    if isinstance(source, dict) is True:
        content = source
    else:
        with open(source, 'r') as f:
            content = json.load(f)

    if content is not None:
        for path, result_dict in content.items():
            dis_list = result_dict['distance']
            path_ref_list = result_dict['paths']

            # ----read the ori image
            # img_ori = cv2.imread(path)
            img_ori = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
            if img_ori is None:
                print("Read failed:", path)
            else:
                img_ori = img_ori[:, :, ::-1]
                img_list = [img_ori]
                title = "ori:{}".format(path.split("\\")[-1])
                title_list = [title]

                for i, path_ref in enumerate(path_ref_list):
                    distance = dis_list[i]

                    # img_ref = cv2.imread(path_ref)
                    img_ref = cv2.imdecode(np.fromfile(path_ref, dtype=np.uint8), 1)
                    if img_ref is None:
                        print("Read failed:", path_ref)
                    else:
                        img_ref = img_ref[:, :, ::-1]
                        img_list.append(img_ref)
                        title = "{}\nD:{}".format(path_ref.split("\\")[-1], round(distance, 4))
                        title_list.append(title)

                    if i + 1 >= show_qty:
                        break

                img_plot(img_list, title_list=title_list, figsize=(7, 7))

def get_ave_embed_distance(source):
    # ----var
    ave_dis = None

    if isinstance(source, dict) is True:
        content = source
    else:
        with open(source, 'r') as f:
            content = json.load(f)

    if content is not None:
        dis_list = list()
        for result_dict in content.values():
            for distance in result_dict['distance']:
                if not distance <= 1e-5:
                    dis_list.append(distance)
                    break

        ave_dis = np.average(dis_list)
        std_dis = np.std(dis_list)

    return ave_dis, std_dis

def classification_by_embed_comparison(source, save_dir, save_dict=None, threshold=0.2):
    if save_dict is None:
        save_dict = {'under': True, 'over': True}

    if isinstance(source, dict) is True:
        content = source
    else:
        with open(source, 'r') as f:
            content = json.load(f)

    if content is not None:
        # ----create save dir
        for dir_name, status in save_dict.items():
            if status is True:
                temp_dir = os.path.join(save_dir, dir_name)
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

        for path, result_dict in content.items():
            dis_list = result_dict['distance']

            if dis_list[0] <= threshold:
                dir_name = 'under'
            else:
                dir_name = 'over'

            if save_dict.get(dir_name) is True:
                new_path = os.path.join(save_dir, dir_name, path.split("\\")[-1])
                shutil.copy(path, new_path)


def get_center_points(databse_source, pb_source):
    # ----var
    node_dict = {'input': 'input:0',
                 # 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 'keep_prob': 'keep_prob:0'
                 }
    batch_size = 128
    GPU_ratio = None
    name_trainInfer_set = {'infer_FFR', "inference_", 'test'}

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
            embed_temp = embed_ref[num_start:num_end]
            center_embedding[i] = np.mean(embed_temp,axis=0)
            std_embedding[i] = np.std(embed_temp,axis=0)
            #print("class = {},mean = {},std = {}".format(classname_list[i],center_embedding[i],std_embedding[i]))
            num_start = num_end

        #----計算各類的中心與std的距離
        for i, embedding in enumerate(center_embedding):
            std = std_embedding[i]
            a = np.square((embedding - (embedding + std)))
            b = np.sum(a)
            c = np.sqrt(b)
            print("class:{},與std間的距離:{}".format(classname_list[i],c))
            a = np.square((center_embedding - embedding))
            b = np.sum(a,axis=1)
            c = np.sqrt(b)
            print("class:{},與各類別的距離:{}".format(classname_list[i], c))

        #-----計算各類中心的距離
        # for embedding in center_embedding:
        #
        #     a = np.square((center_embedding - embedding))
        #     b = np.sum(a,axis=1)
        #     c = np.sqrt(b)
        #     print("c = ",c)
        #     print("c = ",c)

        return center_embedding, std_embedding, classname_list

def classification_by_embed_comparison_opto(img_source, databse_source, pb_source, threshold_list=[0.2],
                                            prob_threshold_list=[0.7],
                                            GPU_ratio=None, binary_check=False):
    # ----var
    node_dict = {'input': 'input:0',
                 # 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 'keep_prob': 'keep_prob:0'
                 }
    batch_size = 128
    name_trainInfer_set = {'infer_FFR', "inference_",'test'}
    content = dict()
    json_path = None
    paths_underkill = list()
    fp,tp,tn,fn = 0,0,0,0

    #----record the time
    d_t = time.time()

    # ----get all images
    paths_test, qty_test_list, _ = get_paths_2(img_source)
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

                #----GPU setting
                config = tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                        )
                config.gpu_options.allow_growth = True
                sess_cal = tf.Session(config=config)
                sess_cal.run(tf.global_variables_initializer())

            # ----calculate embeddings
            embed_ref = get_embeddings(sess, paths_ref, tf_dict, batch_size=batch_size)
            embed_tar = get_embeddings(sess, paths_test, tf_dict, batch_size=batch_size)
            print("embed_ref shape: ", embed_ref.shape)
            print("embed_tar shape: ", embed_tar.shape)

            # ----calculate distance and get the minimum
            arg_dis = list()
            dis_list = list()
            count_NG = 0
            count_OK = 0

            count_underkill = 0
            count_overkill = 0
            feed_dict_2 = {tf_ref: embed_ref}
            distance_list = np.zeros([embed_tar.shape[0], embed_ref.shape[0]], dtype=np.float32)

            # for idx, embedding in enumerate(embed_tar):
            for idx in range(embed_tar.shape[0]):
                feed_dict_2[tf_tar] = embed_tar[idx]
                # distance = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                # arg_dis = np.argsort(distance)
                # distance_list.append(sess_cal.run(tf_dis, feed_dict=feed_dict_2))
                dis = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                distance_list[idx] = dis

            for threshold in threshold_list:
                binary_dis_list = np.where(distance_list > threshold, 0, 1)  # 1表示小於threshold
                for prob_threshold in prob_threshold_list:
                    #----reset all values
                    count_o = 0
                    count_o_under_threshold = 0
                    paths_under_threshold = list()
                    paths_x = list()
                    count_x = 0
                    count_unknown = 0

                    for idx, binary_dis in enumerate(binary_dis_list):
                        num_start = 0
                        ratio_list = list()
                        count_list = list()
                        for qty in qty_ref_list:
                            num_end = num_start + qty
                            class_dis_list = binary_dis[num_start:num_end]
                            count = np.where(class_dis_list == 1)
                            count = len(count[0])
                            if len(class_dis_list) != 0:
                                ratio = count / len(class_dis_list)
                            else:
                                ratio = 0
                            count_list.append(count)
                            ratio_list.append(ratio)

                            # ----
                            num_start = num_end

                        arg_max = np.argmax(ratio_list)
                        prediction = classname_list[arg_max]
                        answer = paths_test[idx].split("\\")[-2]
                        if binary_check is True:
                            if prediction.find("NG") >= 0:# 不管大於或小於機率門檻值都等於判NG，所以NG的不用跟機率門檻值進行比較
                                if answer.find("NG") >= 0:
                                    tn += 1
                                else:
                                    fn += 1
                            elif prediction.find("OK") >= 0:
                                if ratio_list[arg_max] >= prob_threshold:
                                    if answer.find("OK") >= 0:
                                        tp += 1
                                    else:
                                        fp += 1
                                else:#小於機率門檻值判NG
                                    if answer.find("OK") >= 0:
                                        fn += 1

                                    else:
                                        tn += 1
                        else:
                            if prediction == answer:
                                if ratio_list[arg_max] >= prob_threshold:
                                    count_o += 1
                                else:
                                    count_o_under_threshold += 1
                                    paths_under_threshold.append(paths_test[idx])
                            else:
                                count_x += 1
                                paths_x.append(paths_test[idx])

                    # ----statistics
                    print("pb_path:{}".format(pb_path))
                    print("threshold:{}, prob_threshold:{}".format(threshold, prob_threshold))
                    if binary_check is True:
                        # print("total OK:{}, NG:{}".format(count_OK,count_NG))
                        print("total OK:{}, NG:{}".format(tp + fn,tn + fp))
                        print('tp:{}, fn:{}, tn:{}, fp:{}'.format(tp,fn,tn,fp))
                        print('underkill qty:{},ratio:{}'.format(fp,fp / (tn + fp)))
                        print('overkill qty:{}, ratio:{}'.format(fn,fn / (tp + fn)))
                        print("count_unknown:", count_unknown)

                        print("accuracy: ", (tn + tp) / (tn + tp + fn + fp))
                        print("accuracy: ", (tn + tp) / (tn + tp + fn + fp))
                    else:
                        print("quantity of correct CLS:{}".format(count_o))
                        print("quantity of correct CLS but under the threshold:{}".format(count_o_under_threshold))
                        print("quantity of incorrect CLS:{}".format(count_x))
                        print("accuracy:{}".format(count_o/(count_o + count_x + count_o_under_threshold)))

            d_t = time.time() - d_t
            print("ave process time = ",d_t/qty_test)

            # print("accuracy: ", count_o / len(paths_test))
            # print("unknown: ",count_unknown /len(paths) )

            # arg_dis_list = np.argsort(distance_list)
            # for idx, arg_dis in enumerate(arg_dis_list):
            #
            #     if compare_type == 'over':
            #         arg_dis = arg_dis[::-1]
            #         # identical_paths = np.where(arg_dis >= threshold)
            #     # else:
            #     # identical_paths = np.where(distance <= threshold)
            #
            #     # print("ori path:",paths[idx])
            #     temp_path = list()
            #     temp_dis = list()
            #     distance = distance_list[idx]
            #     for i in arg_dis[:record_qty]:
            #         # if distance_list[idx][i] <= threshold:
            #         # if not distance_list[idx][i] <= 1e-5:#距離不等於0
            #         # if i != idx:
            #         # print("距離0的圖:",paths[i])
            #         temp_path.append(paths_ref[i])
            #         temp_dis.append((distance[i]))  # 前面的distance定義時就已經將dtype=float，這邊就不用再轉換
            #
            #     content[paths[idx]] = {'paths': temp_path, 'distance': temp_dis}
            #     # arg_temp = np.argsort(distance)[:1]
            #     # arg_dis.append(arg_temp)
            #     # dis_list.append(distance[arg_temp])
            # d_t = time.time() - d_t
            # print("process time: ", d_t)
            # # for path, arg_list,dises in zip(paths,arg_dis,dis_list):
            # #     #answer = path.split("\\")[-2]
            # #     # answer = path.split("\\")[-1].split('_')[0]
            # #     for arg,dis in zip(arg_list,dises):
            # #         if dis <= threshold:
            # #             save_path = os.path.join(under_th_dir,path.split("\\")[-1])
            # #             shutil.copy(path,save_path)
            # #         else:
            # #             save_path = os.path.join(over_th_dir, path.split("\\")[-1])
            # #             shutil.copy(path, save_path)
            # #         content[path] = float(dis)
            # #             # prediction = paths_ref[arg].split("\\")[-1].split("_")[0]
            # #             # prediction = paths_ref[arg].split("\\")[-2]
            # #             # if prediction == answer:
            # #             #     count_o += 1
            # #             # else:
            # #             #     print("\n{},label:{},prediction:{},prob:{}".format(path, answer, prediction,None))
            # #             #     print("similar image:",paths_ref[arg])
            # #         # else:
            # #         #     count_unknown += 1
            # if to_save is True:
            #     json_path = 'embedding_evaluation.json'
            #     with open(json_path, 'w') as f:
            #         json.dump(content, f)

            # ----statistics
            # print("accuracy: ",count_o /len(paths) )
            # print("unknown: ",count_unknown /len(paths) )

    # return content, json_path
def classification_by_center_embed_comparison_opto(img_source, databse_source, pb_source, coeff_list=[2],
                                            GPU_ratio=None):
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

    d_t = time.time()

    # ----get all images
    paths_test, qty_test_list, _ = get_paths_2(img_source)
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
            print("std_dis_list:",std_dis_list)

            embed_tar = get_embeddings(sess, paths_test, tf_dict, batch_size=batch_size)
            print("embed_tar shape: ", embed_tar.shape)

            feed_dict_2 = {tf_ref: center_embedding}
            distance_list = np.zeros([embed_tar.shape[0], center_embedding.shape[0]], dtype=np.float32)

            for idx in range(embed_tar.shape[0]):
                feed_dict_2[tf_tar] = embed_tar[idx]
                dis = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                distance_list[idx] = dis

            for coeff in coeff_list:
                count_o = 0
                count_o_under_threshold = 0
                count_o_over_std = 0
                count_x_over_std = 0
                paths_under_threshold = list()
                paths_x = list()
                count_x = 0
                count_unknown = 0
                for idx,dis in enumerate(distance_list):
                    argmin = np.argmin(dis)

                    answer = paths_test[idx].split("\\")[-2]
                    prediction = classname_list[argmin]
                    if prediction == answer:
                        if dis[argmin] > coeff * std_dis_list[argmin]:
                            count_o_over_std += 1
                        else:
                            count_o += 1
                    else:
                        if dis[argmin] > coeff * std_dis_list[argmin]:
                            count_x_over_std += 1
                        else:
                            count_x += 1

                # ----statistics
                print("pb_path:{}".format(pb_path))
                print("std coefficient:{}".format(coeff))
                print("quantity of correct CLS:{}".format(count_o))
                print("quantity of incorrect CLS:{}".format(count_x))
                print("quantity of correct CLS but over the std distance:{}".format(
                    count_o_over_std))
                print("quantity of incorrect CLS but over the std distance:{}".format(
                    count_x_over_std))
                print("quantity of unknow CLS:{}".format(count_unknown))
                print("accuracy:{}".format(count_o / (count_o + count_x + count_unknown + count_o_over_std + count_x_over_std)))
        d_t = time.time() - d_t
        print("ave process time = ",d_t / qty_test )

def classification(img_source, pb_path, prob_threshold=0.7, ):
    # ----var
    node_dict = {'input': 'input:0',
                 # 'phase_train': 'phase_train:0',
                 'keep_prob': 'keep_prob:0',
                 'embeddings': 'embeddings:0',
                 'prediction': 'prediction:0'
                 }
    count_o = 0
    count_x = 0
    count_unknown = 0
    count_underkill = 0
    count_overkill = 0

    # ----get classname list
    ret, value = get_train_result(pb_path, key='label_dict')
    if ret is True:
        print(value)
        classname_list = value.copy()
        if isinstance(classname_list, dict):
            classname_list = list(classname_list.keys())
        # content = re_dict['content']
        # print(content)
    else:
        for msg in value['msg_list']:
            print(msg)

    # ----get all images
    paths_test, qty_test_list, _ = get_paths_2(img_source)
    qty_test = len(paths_test)

    if qty_test == 0:
        msg = "Error:No images in {}".format(img_source)
        print(msg)
    else:
        # ----model init
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=None)
        # tf_embeddings = tf_dict['embeddings']
        d_t = time.time()
        predictions = get_predictions(sess, paths_test, tf_dict, batch_size=128)
        print(predictions.shape)

        for idx, prediction in enumerate(predictions):
            path = paths_test[idx]
            answer = path.split("\\")[-2]
            argmax = np.argmax(prediction)
            predict_class = classname_list[argmax]

            if prediction[argmax] < prob_threshold:
                count_unknown += 1
            elif predict_class == answer:
                count_o += 1
            else:
                count_x += 1
                # ----under kill
                if answer.find('NG') >= 0 and predict_class.find('OK') >= 0:
                    count_underkill += 1
                else:
                    count_overkill += 1

        # ----statistics
        print("prob_threshold:{}".format(prob_threshold))
        print("count_x:", count_x)
        print('underkill:', count_underkill)
        print('overkill:', count_overkill)
        print("count_unknown:", count_unknown)

        print("accuracy: ", count_o / len(paths_test))
        print("accuracy: ", count_o / len(paths_test))

def get_predictions(sess, paths, tf_dict, batch_size=128):
    # ----
    len_path = len(paths)
    tf_input = tf_dict['input']

    tf_predictions = tf_dict['prediction']

    feed_dict = dict()
    if 'keep_prob' in tf_dict.keys():
        tf_keep_prob = tf_dict['keep_prob']
        feed_dict[tf_keep_prob] = 1.0
    if 'phase_train' in tf_dict.keys():
        tf_phase_train = tf_dict['phase_train']
        feed_dict[tf_phase_train] = False

    model_shape = tf_input.shape
    # model_shape = [None,160,160,3]
    print("tf_input shape:", model_shape)

    # ----
    ites = math.ceil(len_path / batch_size)
    predictions = np.zeros([len_path, tf_predictions.shape[-1]], dtype=np.float32)
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
                img = cv2.resize(img, (model_shape[2].value, model_shape[1].value))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_data[idx_path] = img
        batch_data /= 255  # norm
        feed_dict[tf_input] = batch_data
        predictions[num_start:num_end] = sess.run(tf_predictions, feed_dict=feed_dict)

    return predictions

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

def oneClass_detection(img_dir, pb_path, node_dict, method_dict, batch_size=32, output_dir=None, save_dict=None,
                       to_save_diff=False, save_type='copy', to_show=False):
    # ----var
    k_size = 3
    # node_dict = {'input': 'input:0',
    #              # 'phase_train': 'phase_train:0',
    #              'embeddings': 'embeddings:0',
    #              # 'keep_prob':'keep_prob:0'
    #              }
    # batch_size = 128
    # content = dict()
    # json_path = None
    result = {'OK': 0, 'NG': 0}

    # ----get all images
    paths = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] in img_format]
    qty_path = len(paths)

    if qty_path == 0:
        print("No images in ", img_dir)
    else:
        # ----model init
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=None)
        model_shape = tf_dict['input'].shape
        print("model_shape:", model_shape)
        # tf_embeddings = tf_dict['embeddings']

        # ----method
        if method_dict.get('avepool') is not None:
            if method_dict.get('k_size') is not None:
                k_size = method_dict['k_size']
            # ----tf setting for calculating distance
            with tf.Graph().as_default():
                tf_input = tf.placeholder(tf.float32, shape=model_shape, name='input_avepool')
                avepool_out = avepool(tf_input, k_size=k_size, strides=1)
                # ----GPU setting
                config = tf.ConfigProto(log_device_placement=True,
                                        allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                        )
                config.gpu_options.allow_growth = True
                sess_cal = tf.Session(config=config)
                sess_cal.run(tf.global_variables_initializer())

        # ----create output dir
        if save_dict is not None:
            for key, value in save_dict.items():
                if value is True:
                    dir_path = os.path.join(output_dir, key)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
        if to_save_diff is True:
            recon_dir = os.path.join(output_dir, 'recon_diff_output')
            if not os.path.exists(recon_dir):
                os.makedirs(recon_dir)

        ites = math.ceil(qty_path / batch_size)
        for idx in range(ites):
            # ----get image start and end numbers
            num_start = idx * batch_size
            num_end = np.minimum(num_start + batch_size, qty_path)

            batch_paths = paths[num_start:num_end]
            sess_out_array = get_sess_out(sess, batch_paths, tf_dict, 'output')

            for i, path in enumerate(batch_paths):
                # ----process of sess-out
                img_sess_out = sess_out_array[i] * 255
                img_sess_out = cv2.convertScaleAbs(img_sess_out)
                if model_shape[3] == 1:
                    img_sess_out = np.reshape(img_sess_out, (model_shape[1], model_shape[2]))
                else:
                    img_sess_out = cv2.cvtColor(img_sess_out, cv2.COLOR_RGB2BGR)

                # ----save recon image
                if to_save_diff is True:
                    splits = path.split("\\")[-1]
                    new_filename = splits.split('.')[0] + '_sess-out.' + splits.split('.')[-1]
                    new_filename = os.path.join(recon_dir, new_filename)
                    cv2.imwrite(new_filename, img_sess_out)

                # ----method selection
                if method_dict.get('diff') is not None:
                    # ----parsing var
                    diff_th = method_dict['diff']['diff_th']
                    cc_th = method_dict['diff']['cc_th']
                    edge = method_dict['diff']['edge']

                    img_diff, ret = img_diff_method(path, img_sess_out, model_shape, diff_th=diff_th, cc_th=cc_th,
                                                    edge=edge)
                    if ret is True:
                        result['NG'] += 1
                    else:
                        result['OK'] += 1
                    if to_save_diff is True:
                        img_diff = cv2.convertScaleAbs(img_diff)
                        new_filename = path.split("\\")[-1]
                        new_filename = new_filename.split(".")[0] + '_diff.' + new_filename.split(".")[-1]
                        new_filename = os.path.join(recon_dir, new_filename)
                        cv2.imwrite(new_filename, img_diff)
                    if save_dict is not None:
                        if ret is True:  # NG
                            if save_dict.get('NG') is True:
                                new_filename = os.path.join(output_dir, 'NG', path.split("\\")[-1])
                                if save_type == 'copy':
                                    shutil.copy(path, new_filename)
                                elif save_type == 'move':
                                    shutil.move(path, new_filename)
                        else:  # OK
                            if save_dict.get('OK') is True:
                                new_filename = os.path.join(output_dir, 'OK', path.split("\\")[-1])
                                if save_type == 'copy':
                                    shutil.copy(path, new_filename)
                                elif save_type == 'move':
                                    shutil.move(path, new_filename)

                    if to_show is True:
                        # img_ori = cv2.imread(path)
                        img_ori = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
                        img_plot([img_ori[:, :, ::-1], img_sess_out[:, :, ::-1],
                                  img_diff[:, :, ::-1]], ['ori_{}'.format(path.split("\\")[-1]), 'sess_out', 'diff'],
                                 # figsize=(7,7)
                                 )
                elif method_dict.get('avepool') is not None:
                    # ----parsing var
                    diff_th = method_dict['avepool']['diff_th']
                    cc_th = method_dict['avepool']['cc_th']
                    edge = method_dict['avepool']['edge']

                    img_diff, ret = img_avepool_method(path, img_sess_out, sess_cal, tf_input, avepool_out, model_shape,
                                                       diff_th=diff_th, cc_th=cc_th, edge=edge)
                    if ret is True:
                        result['NG'] += 1
                        # print("NG:",path.split("\\")[-1])
                    else:
                        result['OK'] += 1
                    if to_save_diff is True:
                        img_diff = cv2.convertScaleAbs(img_diff)
                        new_filename = path.split("\\")[-1]
                        new_filename = new_filename.split(".")[0] + '_avepool_diff.' + new_filename.split(".")[-1]
                        new_filename = os.path.join(recon_dir, new_filename)
                        cv2.imwrite(new_filename, img_diff)
                    if save_dict is not None:
                        if ret is True:  # NG
                            if save_dict.get('NG') is True:
                                new_filename = os.path.join(output_dir, 'NG', path.split("\\")[-1])
                                if save_type == 'copy':
                                    shutil.copy(path, new_filename)
                                elif save_type == 'move':
                                    shutil.move(path, new_filename)
                        else:  # OK
                            if save_dict.get('OK') is True:
                                new_filename = os.path.join(output_dir, 'OK', path.split("\\")[-1])
                                if save_type == 'copy':
                                    shutil.copy(path, new_filename)
                                elif save_type == 'move':
                                    shutil.move(path, new_filename)
                    if to_show is True:
                        # img_ori = cv2.imread(path)
                        img_ori = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
                        img_plot([img_ori[:, :, ::-1], img_sess_out[:, :, ::-1],
                                  img_diff[:, :, ::-1]],
                                 ['ori_{}'.format(path.split("\\")[-1]), 'sess_out', 'diff_avepool'],  # figsize=(7,7)
                                 )

        # ----display the result
        print(result)

def img_diff_method(img_source_1, img_source_2, model_shape, diff_th=30, cc_th=30, edge=6):
    temp = np.array([1., 2., 3.])
    re = None
    status = False
    # ----read img source 1
    if isinstance(temp, type(img_source_1)):
        img_1 = img_source_1
    elif os.path.isfile(img_source_1):
        # img_1 = cv2.imread(img_source_1)
        img_1 = cv2.imdecode(np.fromfile(img_source_1, dtype=np.uint8), 1)
        ori_h, ori_w, _ = img_1.shape
        img_1 = cv2.resize(img_1, (model_shape[2], model_shape[1]))
        # img_1 = img_1.astype('float32')
    else:
        print("The type of img_source_1 is not supported")

    # ----read img source 2
    if isinstance(temp, type(img_source_2)):
        img_2 = img_source_2
        h, w, _ = img_2.shape
    elif os.path.isfile(img_source_2):
        # img_2 = cv2.imread(img_source_2)
        img_2 = cv2.imdecode(np.fromfile(img_source_2, dtype=np.uint8), 1)
        h, w, _ = img_2.shape
        # img_2 = img_2.astype('float32')
    else:
        print("The type of img_source_2 is not supported")

    # ----subtraction
    if img_1 is not None and img_2 is not None:
        # print("img_1 shape = {}, dtype = {}".format(img_1.shape, img_1.dtype))
        # print("img_2 shape = {}, dtype = {}".format(img_2.shape, img_2.dtype))
        img_diff = cv2.absdiff(img_1, img_2)  # img_diff = np.abs(img - img_sess)#不能使用，因為相減若<0，會取補數
        if img_1.shape[-1] == 3:
            img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)  # 利用轉成灰階，減少RGB的相加計算量

        # ----cut edges
        if edge >= 0:
            edge_model = int(edge * w / ori_w)
            # print("before cutting:", img_diff.shape)
            img_diff = img_diff[edge_model:-edge_model, edge_model:-edge_model]
            img_1 = img_1[edge_model:-edge_model, edge_model:-edge_model, :]
            # print("after cutting:",img_diff.shape)

        # ----gray average value threshold
        img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        ave = np.average(img_1_gray)
        # print("img_1 max:{},ave:{},std:{}".format(np.max(img_1_gray), ave, np.std(img_1_gray)))
        if ave < 20:  # 當img_1(原圖)平均畫素值過小，表示整張圖是接近黑色的
            status = True  # NG
            re = img_1
        else:
            # ----連通
            img_compare = cv2.compare(img_diff, diff_th, cv2.CMP_GT)  # 當img_diff每一點都大於diff_th就會填入255，此外填0
            # retval, labels = cv2.connectedComponents(img_compare)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_compare, connectivity=4)
            max_label_num = np.max(labels) + 1

            img_1_copy = img_1.copy()
            for i in range(0, num_labels):  # label = 0是背景，所以從1開始
                y, x = np.where(labels == i)  # y,x代表類別=i的像素座標值
                if (y.shape[0] > cc_th) and (img_compare[y[0], x[0]] == 255):  # 數值=255表示img_diff有超過門檻值
                    status = True
                    for j in range(y.shape[0]):
                        img_1_copy.itemset((y[j], x[j], 0), 0)
                        img_1_copy.itemset((y[j], x[j], 1), 0)
                        img_1_copy.itemset((y[j], x[j], 2), 255)

            re = img_1_copy
        return re, status

def img_avepool_method(img_source_1, img_source_2, sess, tf_input, avepool_out, model_shape, diff_th=30, cc_th=30,
                       edge=6):
    temp = np.array([1., 2., 3.])
    re = None
    status = False
    # ----read img source 1
    if isinstance(temp, type(img_source_1)):
        img_1 = img_source_1
    elif os.path.isfile(img_source_1):
        # img_1 = cv2.imread(img_source_1)
        img_1 = cv2.imdecode(np.fromfile(img_source_1, dtype=np.uint8), 1)
        ori_h, ori_w, _ = img_1.shape
        img_1 = cv2.resize(img_1, (model_shape[2], model_shape[1]))
        # img_1 = img_1.astype('float32')

    else:
        print("The type of img_source_1 is not supported")

    # ----read img source 2
    if isinstance(temp, type(img_source_2)):
        img_2 = img_source_2
        h, w, _ = img_2.shape
    elif os.path.isfile(img_source_2):
        # img_2 = cv2.imread(img_source_2)
        img_2 = cv2.imdecode(np.fromfile(img_source_2, dtype=np.uint8), 1)
        h, w, _ = img_2.shape
        # img_2 = img_2.astype('float32')
    else:
        print("The type of img_source_2 is not supported")

    # ----subtraction
    if img_1 is not None and img_2 is not None:
        img_diff = cv2.absdiff(img_1, img_2)
        img_diff_ave_pool = sess.run(avepool_out, feed_dict={tf_input: np.expand_dims(img_diff, axis=0)})
        img_diff = img_diff_ave_pool[0]

        # img_1_ave_pool = sess.run(avepool_out,feed_dict={tf_input:np.expand_dims(img_1,axis=0)})
        # img_2_ave_pool = sess.run(avepool_out,feed_dict={tf_input:np.expand_dims(img_2,axis=0)})
        # img_diff = cv2.absdiff(img_1_ave_pool[0], img_2_ave_pool[0])  # img_diff = np.abs(img - img_sess)#不能使用，因為相減若<0，會取補數
        if img_1.shape[-1] == 3:
            img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)  # 利用轉成灰階，減少RGB的相加計算量

        # ----cut edges
        if edge > 0:
            # 要按比例縮小
            edge_model = int(edge * w / ori_w)
            # print("before cutting:", img_diff.shape)
            img_diff = img_diff[edge_model:-edge_model, edge_model:-edge_model]
            img_1 = img_1[edge_model:-edge_model, edge_model:-edge_model, :]

            # print("after cutting:",img_diff.shape)

        # ----gray average value threshold
        img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        # print("img_1 max:{},ave:{},std:{}".format(np.max(img_1_gray), np.average(img_1_gray), np.std(img_1_gray)))
        ave = np.average(img_1_gray)
        if ave < 20:  # 當img_1(原圖)平均畫素值過小，表示整張圖是接近黑色的
            status = True  # NG
            re = img_1
        else:
            # ----連通
            img_compare = cv2.compare(img_diff, diff_th, cv2.CMP_GT)
            # retval, labels = cv2.connectedComponents(img_compare)
            # max_label_num = np.max(labels) + 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_compare, connectivity=4)

            img_1_copy = img_1.copy()
            for i in range(0, num_labels):  # label = 0是背景，所以從1開始
                y, x = np.where(labels == i)  # y,x代表類別=i的像素座標值
                if (y.shape[0] > cc_th) and (img_compare[y[0], x[0]] == 255):
                    status = True
                    for j in range(y.shape[0]):
                        img_1_copy.itemset((y[j], x[j], 0), 0)
                        img_1_copy.itemset((y[j], x[j], 1), 0)
                        img_1_copy.itemset((y[j], x[j], 2), 255)

            re = img_1_copy
        return re, status

def avepool(input_x, k_size=3, strides=1):
    kernel = [1, k_size, k_size, 1]
    stride_kernel = [1, strides, strides, 1]
    return tf.nn.avg_pool(input_x, ksize=kernel, strides=stride_kernel, padding='SAME')

def img_plot(img_list, title_list=None, figsize=None):
    data_type = type(tuple())
    if type(figsize) == data_type:
        plt.figure(figsize=figsize)
    img_qty = len(img_list)
    for i, img_data in enumerate(img_list):
        plt.subplot(1, img_qty, i + 1)
        plt.imshow(img_data)
        plt.axis("off")
        if title_list is not None and len(title_list) == img_qty:
            plt.title(title_list[i])
    plt.show()

def get_train_result(pb_path, key=None, random_num_range=10):
    # ----var
    msg_list = list()
    re_dict = dict
    ret = False
    dir_path = os.path.dirname(pb_path)
    files = [file.path for file in os.scandir(dir_path) if file.name.find('train_result') >= 0]

    if len(files) == 0:
        msg_list.append('Error: no train result files ')
        re_dict['msg_list'] = msg_list
    else:
        ctime_list = list()
        for file in files:
            ctime_list.append(os.path.getctime(file))

        arg_max = int(np.argmax(ctime_list))
        file = files[arg_max]

        # ----
        ret, re_dict = file_decode_v2_1(file, random_num_range=random_num_range,
                                        save_dir=None, to_save=False,
                                        print_f=None, print_out=False)
        if ret is False:
            # error_msg_list = re_dict['msg_list']
            # for msg in error_msg_list:
            #     # print_f(msg)
            #     msg_list.append(msg)
            # print_f("The file is not secured")
            try:
                with open(file, 'r') as f:
                    re_dict['content'] = json.load(f)
                    ret = True
            except:
                msg = "Error: read failed: {}".format(file)
                # print_f(msg)
                re_dict['msg_list'].append(msg)
        elif ret is True:
            # re_msg_list.append("The file is decoded successfully")
            content = re_dict['content']
            content = json.loads(content.decode())
            re_dict['content'] = content

    # ----key process
    if key is not None:
        value = re_dict['content'].get(key)
    else:
        value = re_dict

    return ret, value

def file_decode_v2_1(file_path, random_num_range=87, save_dir=None, to_save=True,
                     print_f=None, print_out=True):
    # ----var
    status = False
    re_dict = dict()
    content = None
    re_msg_list = list()
    header_len = 4
    decode_flag = True
    header = [24, 97, 28, 98]

    # ----print out lambda func
    if print_f is None:
        print_f = print

    display = lambda msg, print_out: print_f(msg) if print_out is True else None

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
    # msg = "Error:headers don't match"
    # re_msg_list.append(msg)
    # print_f(msg)
    # display(msg,print_out)

    # ----decode process
    if decode_flag is True:
        # print("execute decode")
        leng = len(content)
        msg = "file length = {}".format(str(leng))
        # say_sth(msg, print_out=print_out)
        # print_f(msg)
        display(msg, print_out)

        cut_num_start = random_num_range + header_len
        cut_num = int.from_bytes(content[cut_num_start:cut_num_start + 4], byteorder="little", signed=False)
        msg = "cut_num = {}".format(str(cut_num))
        # say_sth(msg, print_out=print_out)
        # print_f(msg)
        display(msg, print_out)

        compliment_num_start = cut_num_start + 4
        compliment_num = int.from_bytes(content[compliment_num_start:compliment_num_start + 4], byteorder="little",
                                        signed=False)
        msg = "compliment_num = {}".format(str(compliment_num))
        # say_sth(msg, print_out=print_out)
        # print_f(msg)
        display(msg, print_out)

        seq_num_start = compliment_num_start + 4
        seq = content[seq_num_start:seq_num_start + cut_num]

        pb_fragment = content[seq_num_start + cut_num:]
        leng = len(pb_fragment)
        # msg = "pb_fragment size = {}".format(pb_fragment)
        # say_sth(msg, print_out=print_out)

        slice_num = math.ceil(leng / cut_num)
        msg = "slice_num = {}".format(str(slice_num))
        # say_sth(msg, print_out=print_out)
        # print_f(msg)
        display(msg, print_out)

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

        status = True
        content = temp

        if to_save is True:
            # ----output filename
            if save_dir is None:
                new_filename = file_path
            else:
                new_filename = os.path.join(save_dir, file_path.split("\\")[-1])

            # ----save the file
            try:
                with open(new_filename, 'wb') as f:
                    f.write(temp)
                msg = "decode data is completed in {}".format(new_filename)
                # say_sth(msg, print_out=print_out)
                # print_f(msg)
                display(msg, print_out)
            except:
                status = True
                msg = "Error: fail to save in {}".format(new_filename)
                display(msg, print_out)
                re_msg_list.append(msg)

    # ----return value process
    re_dict['msg_list'] = re_msg_list
    re_dict['content'] = content

    return status, re_dict




if __name__ == "__main__":
    #----get center points
    # img_source = [
    #     r"D:\dataset\fashion_mnist\train",
    # ]
    #pb_path = r"D:\code\model_saver\fashionM_exp47"
    # get_center_points(img_source,pb_path)

    # ----classification_by_embed_comparison_opto
    img_source = [
                    # r"F:\dataset\optoTech\CMIT\009_dataset\validation",
                    r"D:\dataset\fashion_mnist\val",
                    # r"C:\Users\User\Desktop\test",
                    # r"F:\dataset\optoTech\CMIT\009_dataset\training",
                    # r"F:\dataset\optoTech\CMIT\014_2_dataset\train_L1"
                   ]
    # databse_source =  r"F:\model_saver\fashionM_exp7\test_202162294611.nst"
    databse_source =  r"D:\dataset\fashion_mnist\train"
    # databse_source =  r"C:\Users\User\Desktop\test"
    # databse_source =  r"F:\dataset\optoTech\CMIT\014_2_dataset\train_L1"
    # pb_path_list = [r"F:\model_saver\CMIT_014_exp5\test_2021692383.nst",
    #                 r"F:\model_saver\CMIT_014_exp4\test_202169152447.nst",
    #                 r"F:\model_saver\CMIT_014_exp3\test_20216915857.nst",
    #                 r"F:\model_saver\CMIT_014_exp2\test_20216914340.nst",
    #                 r"F:\model_saver\CMIT_014_exp1\test_202169134925.nst",
    #                 r"F:\model_saver\CMIT_014_exp0\test_20216813165.nst"
    #                 ]
    # pb_path = r"F:\model_saver\CMIT_009_exp11\infer_acc(99.6).nst"
    # pb_path = r"F:\model_saver\CMIT_009_exp11\inference_2021617113644.nst"
    # pb_path = r"F:\model_saver\fashionM_exp7\test_202162294611.nst"
    # pb_path = r"D:\code\model_saver\st-2118_chip\infer_FRR(1.89)_FAR(0.0).nst"
    # pb_path = r"D:\code\model_saver\st-2118_chip"
    pb_path = r"D:\code\model_saver\fashionM_exp7"
    # pb_path = r"F:\model_saver\AE_CMIT_014_exp1\pb_model.pb"
    # threshold_list = [0.1, 0.25, 0.5, 0.75]
    dis_list = [0.9]
    prob_list = [0.1]
    coeff_list = [2,4,6]
    binary_check = False
    # for threshold in threshold_list:
    # for pb_path in pb_path_list:
    # classification_by_embed_comparison_opto(img_source, databse_source, pb_path, threshold_list=dis_list,
    #                                         prob_threshold_list=prob_list,
    #                                         GPU_ratio=None, binary_check=binary_check)


    classification_by_center_embed_comparison_opto(img_source, databse_source, pb_path, coeff_list=coeff_list,
                                                       GPU_ratio=None)




    # ----classification method
    # classification(img_source, pb_path, prob_threshold=prob)

    # ----get train results
    # pb_path = r"F:\model_saver\CMIT_007IRA_exp3\inference_202163113146.nst"
    # ret,value = get_train_result(pb_path,key='label_dict')
    # if ret is True:
    #     print(value)
    #     # content = re_dict['content']
    #     # print(content)
    # else:
    #     for msg in re_dict['msg_list']:
    #         print(msg)

    # ----data aug
    # img_dir = r"D:\dataset\optotech\009IRC\CMIT_20210421\ggg"
    # process_dict = {'rdm_crop': True, 'rdm_shift': True, 'rdm_br': True, 'rdm_flip': True,
    #                 'rdm_blur': True, 'rdm_noise': False, 'rdm_angle': True}
    # setting_dict = {'rdm_crop': (5, 5), 'rdm_shift': (30, 30)}
    # aug_num = 15
    # augmentation(img_dir, aug_num=aug_num,process_dict=process_dict,setting_dict=setting_dict,output_dir=None)

    # ----change filenames
    # img_dir = r"D:\dataset\optotech\107YGG-B-HAU\others"
    # change_filename(img_dir)

    # ----image quantity display
    # img_dir = r"D:\dataset\optotech\107YGG-B-HAU\ori"
    #
    # for obj in os.scandir(img_dir):
    #     if obj.is_dir():
    #         leng = len([file.path for file in os.scandir(obj.path) if file.name[-3:] in img_format])
    #         print(obj.name)
    #         print(leng)

    # ----create lable dict and replace dirnames
    # root_dir = r"D:\dataset\optotech\ED-012IRA-BF-HAU\test"
    # to_replace = True
    # create_label_dict(root_dir, to_replace=to_replace)

    # ----embedding evaluation
    # img_dir = r"D:\dataset\optotech\014\014_1_Crop"
    # face_databse_dir = r"D:\dataset\optotech\014\manual_selection\NG"
    # # face_databse_dir = r"D:\dataset\optotech\new_light_20210122_007IRA\black"
    #
    # pb_path = r"D:/code/model_saver/Opto_tech/CLS_014_AI_selection/inference_2021426181728.nst"
    # threshold = 0.2
    # face_matching_evaluation(img_dir, face_databse_dir, pb_path, threshold=threshold,test_num=None, GPU_ratio=None)

    # ----embedding_comparison
    # img_dir = r"D:\dataset\optotech\014\014_1_Crop"
    # # img_dir = r"D:\dataset\optotech\014\test_img\NG"
    # databse_dir = r"D:\dataset\optotech\014\manual_selection\all"
    # # databse_dir = r"D:\dataset\optotech\014\manual_selection\NG"
    # pb_path = r"D:/code/model_saver/Opto_tech/CLS_014_AI_selection/inference_2021426181728.nst"
    # threshold = 0.05
    # compare_type = 'under'
    # content,_ = embedding_comparison(img_dir, databse_dir, pb_path,
    #                                   GPU_ratio=None,compare_type=compare_type)

    # ----oneClass_detection
    # # img_dir = r"D:\dataset\optotech\014\014_1_Crop"
    # # img_dir = r"D:\dataset\optotech\014\014_1_Crop\Binary_CLS\NG\OK"
    # img_dir = r"D:\dataset\optotech\014\014_3_Crop\014_3_Binary_CLS\OK"
    # # img_dir = r"D:\dataset\optotech\014\test_img\t5"
    # pb_path = r"D:\code\model_saver\Opto_tech\AE_014\pb_model.pb"
    # node_dict =  {'input': 'input:0',
    #              # 'phase_train': 'phase_train:0',
    #              # 'output': 'output_AE/mul:0',
    #              'output': 'output_AE/Relu:0',
    #              # 'output': 'embeddings:0',
    #              # 'keep_prob':'keep_prob:0'
    #              }
    # method_dict = {
    #             'diff':{'diff_th':20, 'cc_th':40,'edge':60},
    #           # 'avepool':{'diff_th':15, 'cc_th':30,'edge':60,'k_size':3}
    #           }
    # output_dir = img_dir
    #
    # save_dict = {"OK":False, "NG":True}
    # to_save_diff = False
    # save_type = 'copy'
    # to_show = False
    #
    # oneClass_detection(img_dir, pb_path, node_dict, method_dict,batch_size=32,
    #                    output_dir=output_dir,to_save_diff=to_save_diff,save_dict=save_dict,
    #                    save_type=save_type,to_show=to_show)
    #
