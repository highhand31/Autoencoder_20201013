import numpy as np
import cv2,os,shutil,math,time,json,re
import matplotlib.pyplot as plt
from Utility import tools
from Embedding_center_comparison import model_restore_from_pb

import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
print("Tensorflow version: ", tf.__version__)

img_format = {'png','PNG','jpg','JPG','JPEG','bmp','BMP'}


def pixel_substraction(data_ori, data_recon, **kwargs):
    #----var
    predict_list = list()

    #----
    diff_th = kwargs['diff_th'] / 255
    cc_th = kwargs['cc_th']
    blur_type = kwargs.get('blur_type')
    kernel = kwargs.get('kernel')
    zoom_in_value = kwargs.get('zoom_in_value')
    mask_json_path = kwargs.get('mask_json_path')
    to_mask = kwargs.get('to_mask')
    height = data_recon.shape[1]
    width = data_recon.shape[2]

    #----防呆機制
    # data_ori qty == data_recon
    if len(data_ori) != len(data_recon):
        print("data len qty errors")
        raise ValueError
    else:
        # ----get rectanle coordinance by reading the json file
        if mask_json_path is not None:
            rec_coor_list = get_rec_coor_list(mask_json_path, height, width)
        elif to_mask is True:
            paths = kwargs.get('paths')

        # ----
        index = 0
        for img_ori, img_recon in zip(data_ori, data_recon):

            if blur_type == 'average':
                # ----average filtering
                img_ori = cv2.blur(img_ori, kernel)
                img_recon = cv2.blur(img_recon, kernel)
            elif blur_type == "gaussian":
                # ----GaussianBlur
                img_ori = cv2.GaussianBlur(img_ori, kernel, 0, 0)
                img_recon = cv2.GaussianBlur(img_recon, kernel, 0, 0)

            #----相減取絕對值
            img_subs = np.abs(img_ori - img_recon)

            #----轉換成單色圖片
            img_subs = cv2.cvtColor(img_subs, cv2.COLOR_BGR2GRAY)

            #----比較像素差異是否大於門檻值(diff_th)
            img_compare = cv2.compare(img_subs, diff_th, cv2.CMP_GT)

            #----內縮
            if zoom_in_value is not None:
                v = zoom_in_value
                zeros = np.zeros_like(img_compare)
                if isinstance(zoom_in_value, list):
                    zeros[v[0]:-v[1], v[2]:-v[3]] = img_compare[v[0]:-v[1], v[2]:-v[3]]
                else:
                    zeros[v:-v, v:-v] = img_compare[v:-v, v:-v]
                img_compare = zeros

            # ----rectangle mask
            elif mask_json_path is not None:
                for pts in rec_coor_list:
                    cv2.rectangle(img_compare, pts[0], pts[1], (0), -1)
            #----segmentation + connected components
            elif to_mask is True:
                splits = paths[index].split('\\')

                path_mask = os.path.join(os.path.dirname(paths[index]), 'get_masks_mask', splits[-1])
                if os.path.exists(path_mask):
                    img_mask = np.fromfile(path_mask, dtype=np.uint8)
                    img_mask = cv2.imdecode(img_mask, 0)
                    img_mask = cv2.resize(img_mask, (width, height))
                    img_compare = cv2.bitwise_and(img_compare, img_compare, mask=img_mask)
                else:
                    print("不存在:", path_mask)

            #----reset values
            defect_count = 0
            predict = 'NG'

            # ----connect components
            label_num, label_map, stats, centroids = cv2.connectedComponentsWithStats(img_compare, connectivity=4)
            for label in range(1, label_num):  # 0是背景
                s = stats[label]
                #----比較連通數目是否大於門檻值(cc_th)
                if s[-1] > cc_th:
                    defect_count += 1
                    break
            if defect_count == 0:
                predict = 'OK'

            predict_list.append(predict)
            index +=1

        return predict_list

def np_data2img_data(np_data):
    img_data = np_data.copy()
    img_data *= 255
    img_data = np.clip(img_data,0,255).astype(np.uint8)
    return img_data

def zoom_in(data_in, ratio=None, value=None):
    # ----統一使用4d data來處理，return時再做轉換
    imgs = data_in.copy()
    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, axis=0)

    if ratio is not None:
        values = np.array(imgs.shape[1:3]) * ratio
        values = values.astype(np.int16)
        imgs = imgs[:, values[0]:-values[0], values[1]:-values[1], :]
    elif value is not None:
        imgs = imgs[:, value:-value, value:-value, :]

    # ----return process
    if data_in.ndim == 3:
        return imgs[0]
    else:
        return imgs

def get_process_data(paths, output_shape, process_dict=None, setting_dict=None):
    # ----var
    len_path = len(paths)
    processing_enable = False
    name_list = ['ave_filter', 'gau_filter']

    # ----check process_dict
    if process_dict is not None:
        for name in name_list:
            if process_dict.get(name) is True:
                processing_enable = True
                kernel = (5, 5)
                if setting_dict is not None and setting_dict.get(name):
                    kernel = setting_dict[name]

    # ----create default np array
    batch_dim = [len_path]
    batch_dim.extend(output_shape)
    batch_data = np.zeros(batch_dim, dtype=np.float32)

    # ----
    for idx, path in enumerate(paths):
        # img = cv2.imread(path)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        if img is None:
            msg = "read failed:".format(path)
            print(msg)
        else:
            ori_h, ori_w, _ = img.shape
            # print("img shape=",img.shape)
            # ----image processing
            if processing_enable is True:
                if process_dict.get('ave_filter'):
                    img = cv2.blur(img, kernel)
                if process_dict.get('gau_filter'):
                    img = cv2.GaussianBlur(img, kernel, 0, 0)

            # ----resize and change the color format
            img = cv2.resize(img, (output_shape[1], output_shape[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch_data[idx] = img

    # ----zoom in
    if process_dict.get('zoom_in') is True:
        ratio = setting_dict['zoom_in'][0]
        value = setting_dict['zoom_in'][1]
        batch_data = zoom_in(batch_data, ratio=ratio, value=value)

    return batch_data / 255

def recon_pixel_comparison(img_dir, pb_path, diff_th, cc_th,**kwargs):
    '''
    :param img_dir:
    :param pb_path:
    :param diff_th:
    :param cc_th:
    :param kwargs:
                    zoom_in_value=None
                    node_dict = None
                    process_dict = None
                    setting_dict = None
                    save_type = [False, False, False, False]
                    save_dir = None
    :return:
    '''

    exam_array = np.zeros([4], dtype=np.uint32)
    #     exam_array = np.zeros([4],dtype=np.uint32)
    names = ['pred_ok_ans_ok', 'pred_ok_ans_ng', 'pred_ng_ans_ok', 'pred_ng_ans_ng']
    #pred_ans_path_list = [[], [], [], []]
    pred_ans_path_dict = {
        'pred_ok_ans_ok':[],
        'pred_ok_ans_ng':[],
        'pred_ng_ans_ok':[],
        'pred_ng_ans_ng':[],
    }

    '''
    exam_array positions:
    0: pred_ok_ans_ok_count
    1: pred_ok_ans_ng_count
    2: pred_ng_ans_ok_count
    3: pred_ng_ans_ng_count
    '''
    ans_list = list()
    ans_ok_count = 0
    ans_ng_count = 0

    t = tools()
    #----var parsing
    zoom_in_value = kwargs.get('zoom_in_value')
    node_dict = kwargs.get('node_dict')
    process_dict = kwargs.get('process_dict')
    setting_dict = kwargs.get('setting_dict')
    save_type = kwargs.get('save_type')
    save_dir = kwargs.get('save_dir')
    mask_json_path = kwargs.get('mask_json_path')
    to_mask = kwargs.get('to_mask')

    d_t = time.time()

    #----get paths
    paths, qty = t.get_subdir_paths(img_dir)

    #----model restoration
    if node_dict is None:
        print("node_dict is None")
        raise ValueError
    else:
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=None)
    # tf_embeddings = tf_dict['embeddings']
    tf_recon = tf_dict['output']
    height = tf_dict['input'].shape[1].value
    width = tf_dict['input'].shape[2].value
    model_shape = (height, width, 3)

    # ----create save dir
    if True in save_type:
        if save_dir is None:
            raise ValueError
        else:
            save_dir = os.path.join(os.path.dirname(save_dir),"AE_results_{}x{}".format(height,width))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    ites = math.ceil(len(paths) / batch_size)

    #----ground truth extraction
    for path in paths:
        ground = path.split("\\")[-2]
        # if ground.find("OK") >= 0:
        if len(re.findall('ok',ground,re.I)) > 0:
            ans_list.append("OK")
        else:
            ans_list.append("NG")
    ans_list = np.array(ans_list)
    ans_ok_count = len(np.where(ans_list == 'OK')[0])
    ans_ng_count = len(np.where(ans_list == 'NG')[0])

    print("ans_ok_count:{},ans_ng_count:{}".format(ans_ok_count, ans_ng_count))


    for i in range(ites):
        num_start = batch_size * i
        num_end = num_start + batch_size
        if num_end > len(paths):
            num_end = len(paths)

        batch_paths = paths[num_start:num_end]

        # ----create recon
        batch_data = get_process_data(batch_paths, model_shape, process_dict=process_dict,
                                 setting_dict=setting_dict)
        # print(batch_data.shape)
        batch_recon = sess.run(tf_recon, feed_dict={tf_dict['input']: batch_data})
        #batch_recon = sess.run(tf_recon,feed_dict={tf_dict['input']:batch_recon})

        predict_list = pixel_substraction(batch_data, batch_recon,
                                          diff_th=diff_th, cc_th=cc_th,
                                          zoom_in_value=zoom_in_value,
                                          mask_json_path=mask_json_path,
                                          to_mask=to_mask,paths=paths[num_start:num_end])

        for j in range(num_end - num_start):
            abs_num = num_start + j
            path = paths[abs_num]

            if predict_list[j] == "OK":
                if ans_list[abs_num] == 'OK':
                    # predict OK, ans OK
                    exam_array[0] += 1
                    pred_ans_path_dict['pred_ok_ans_ok'].append(path)
                    # if save_type[0] is True:
                    #     # pred_ans_path_list[0].append(path)
                    #     pred_ans_path_dict['pred_ok_ans_ok'].append(path)
                else:
                    # predict OK, ans NG
                    exam_array[1] += 1
                    pred_ans_path_dict['pred_ok_ans_ng'].append(path)
                    # if save_type[1] is True:
                    #     # pred_ans_path_list[1].append(path)
                    #     pred_ans_path_dict['pred_ok_ans_ng'].append(path)
            else:  # predict NG
                if ans_list[abs_num] == 'OK':
                    # predict NG, ans OK
                    exam_array[2] += 1
                    pred_ans_path_dict['pred_ng_ans_ok'].append(path)
                    # if save_type[2] is True:
                    #     # pred_ans_path_list[2].append(path)
                    #     pred_ans_path_dict['pred_ng_ans_ok'].append(path)
                else:
                    # predict NG, ans NG
                    exam_array[3] += 1
                    pred_ans_path_dict['pred_ng_ans_ng'].append(path)
                    # if save_type[3] is True:
                    #     # pred_ans_path_list[3].append(path)
                    #     pred_ans_path_dict['pred_ng_ans_ng'].append(path)

    # ----info display
    print("\ndiff_th:{},cc_th:{}".format(diff_th, cc_th))
    for key,value in pred_ans_path_dict.items():
        print("{} count:{}".format(key,len(value)))
    underkill = len(pred_ans_path_dict['pred_ok_ans_ng']) / (ans_ng_count+1e-8)
    overkill = len(pred_ans_path_dict['pred_ng_ans_ok']) / (ans_ok_count+1e-8)
    print("underkill: ", underkill)
    print("overkill: ", overkill)

    # for k, name in enumerate(names):
    #     print("{}:{}".format(name, exam_array[k]))
    # underkill = exam_array[1] / ans_ng_count
    # overkill = exam_array[2] / ans_ok_count
    # print("underkill: ", underkill)
    # print("overkill: ", overkill)

    #----save images
    for i, to_save in enumerate(save_type):
        if to_save is True:
            output_dir = os.path.join(save_dir, names[i])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # print(len(pred_ans_path_list[i]))
            for path in pred_ans_path_dict[names[i]]:
                new_path = os.path.join(output_dir, path.split("\\")[-1])
                shutil.copy(path, new_path)

    #----time calculation
    d_t = time.time() - d_t
    ave_time = d_t / len(paths)
    print("image qty:{}, average process time:{}".format(len(paths),ave_time))

    #----save record
    kwargs['img_dir'] = img_dir
    kwargs['pb_path'] = pb_path
    kwargs['diff_th'] = diff_th
    kwargs['cc_th'] = cc_th
    kwargs['pred_ans_path_dict'] = pred_ans_path_dict
    kwargs['img_qty'] = len(paths)
    kwargs['ave_time'] = ave_time

    xtime = time.localtime()
    name_tailer = ''
    for i in range(6):
        string = str(xtime[i])
        if len(string) == 1:
            string = '0' + string
        name_tailer += string
    record_filename = recon_pixel_comparison.__name__ + "_" +name_tailer + '.json'
    record_filename = os.path.join(img_dir,record_filename)
    with open(record_filename,'w') as f:
        json.dump(kwargs,f)

    print("The result is saved in ",record_filename)

def get_rec_coor_list(json_path,height,width):
    rec_coor_list = list()

    with open(json_path, 'r') as f:
        content = json.load(f)
        points_list = content['shapes']
        json_height = content["imageHeight"]
        json_width = content["imageWidth"]
        h_ratio = height / json_height
        w_ratio = width / json_width

    for points_dict in points_list:
        if points_dict['shape_type'] == 'rectangle':
            points = np.array(points_dict['points'])
            points *= [[h_ratio, w_ratio], [h_ratio, w_ratio]]
            points = points.astype(np.int16)
            rec_coor_list.append([tuple(points[0]), tuple(points[1])])

    return rec_coor_list

def AE_find_defects(img_dir, pb_path, diff_th, cc_th, batch_size=32, zoom_in_value=None,to_mask=False,
                           node_dict=None, process_dict=None, setting_dict=None,cc_type="rec",
                           save_type='',read_subdir=False,mask_json_path=None,save_recon=False):
    save_rec = True
    rec_coor_list = list()
    color = (1, 0, 0)
    # ----norm
    diff_th /= 255
    # ----zoom in value
    if zoom_in_value is not None:
        v = zoom_in_value


    #----model restoration
    if node_dict is None:
        print("node_dict is None")
        raise ValueError
    else:
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=None)

    tf_recon = tf_dict['output']
    tf_loss = tf_dict['loss']
    height = tf_dict['input'].shape[1].value
    width = tf_dict['input'].shape[2].value
    model_shape = (height, width, 3)

    if read_subdir is True:
        sub_dirs = [obj.path for obj in os.scandir(img_dir) if obj.is_dir()]
    else:
        sub_dirs = [img_dir]

    #----get rectanle coordinance by reading the json file
    if mask_json_path is not None:
        with open(mask_json_path, 'r') as f:
            content = json.load(f)
            points_list = content['shapes']
            json_height = content["imageHeight"]
            json_width = content["imageWidth"]
            h_ratio = height / json_height
            w_ratio = width / json_width

        for points_dict in points_list:
            if points_dict['shape_type'] == 'rectangle':
                points = np.array(points_dict['points'])
                points *= [[h_ratio, w_ratio], [h_ratio, w_ratio]]
                points = points.astype(np.int16)
                rec_coor_list.append([tuple(points[0]),tuple(points[1])])



    for dir_path in sub_dirs:
        paths = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]

        if len(paths):
            save_dir = os.path.join(dir_path,'find_defects')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if save_recon:
                save_recon_dir = os.path.join(save_dir,'recon')
                if not os.path.exists(save_recon_dir):
                    os.makedirs(save_recon_dir)


            ites = math.ceil(len(paths) / batch_size)
            for index in range(ites):
                num_start = batch_size * index
                num_end = num_start + batch_size
                if num_end > len(paths):
                    num_end = len(paths)

                batch_paths = paths[num_start:num_end]

                # ----create recon
                batch_data = get_process_data(batch_paths, model_shape, process_dict=process_dict,
                                              setting_dict=setting_dict)
                # print(batch_data.shape)
                batch_recon = sess.run(tf_recon, feed_dict={tf_dict['input']: batch_data})
                # batch_loss = sess.run(tf_loss, feed_dict={tf_dict['input_ori']: batch_data})
                # print("batch_loss:",batch_loss)


                #----save recon images
                if save_recon:
                    for i,recon in enumerate(batch_recon):
                        recon = cv2.convertScaleAbs(recon * 255)
                        recon = cv2.cvtColor(recon,cv2.COLOR_RGB2BGR)
                        save_path = os.path.join(save_recon_dir, paths[num_start + i].split("\\")[-1])
                        cv2.imencode('.jpg', recon)[1].tofile(save_path)



                #----相減取絕對值
                img_subs = np.abs(batch_data - batch_recon)

                img_grays = np.zeros(img_subs.shape[:3])
                # Gray = R*0.299 + G*0.587 + B*0.114
                img_grays[:, :, :] = img_subs[:, :, :, 0] * 0.299 + \
                                     img_subs[:, :, :, 1] * 0.587 + \
                                     img_subs[:, :, :, 2] * 0.114

                img_compare_array = np.where(img_grays >= diff_th, 255, 0)
                img_compare_array = img_compare_array.astype(np.uint8)
                if zoom_in_value is not None:
                    zeros = np.zeros_like(img_compare_array)
                    if isinstance(zoom_in_value,list):
                        zeros[:,v[0]:-v[1], v[2]:-v[3]] = img_compare_array[:,v[0]:-v[1], v[2]:-v[3]]
                    else:
                        zeros[:,v:-v, v:-v] = img_compare_array[:,v:-v, v:-v]
                    img_compare_array = zeros

                #----rectangle mask
                elif mask_json_path is not None:
                    for i in range(img_compare_array.shape[0]):
                        for pts in rec_coor_list:
                            cv2.rectangle(img_compare_array[i], pts[0], pts[1], (0), -1)
                # ----segmentation + connected components
                elif to_mask is True:
                    for i, img_compare in enumerate(img_compare_array):
                        splits = batch_paths[i].split('\\')

                        path_mask = os.path.join(os.path.dirname(paths[index]), 'get_masks_mask', splits[-1])
                        if os.path.exists(path_mask):
                            img_mask = np.fromfile(path_mask, dtype=np.uint8)
                            img_mask = cv2.imdecode(img_mask, 0)
                            img_mask = cv2.resize(img_mask, (width, height))
                            img_compare_array[i] = cv2.bitwise_and(img_compare, img_compare, mask=img_mask)
                        else:
                            print("不存在:", path_mask)

                for j, img_compare in enumerate(img_compare_array):
                    # img_compare = img_compare_array[j]
                    abs_num = num_start + j
                    path = paths[abs_num]
                    img_ori = batch_data[j]
                    img_copy = batch_data[j].copy()

                    # img_subs = img_subs_array[j]

                    # ----轉換成單色圖片
                    #             img_subs = cv2.cvtColor(img_subs,cv2.COLOR_BGR2GRAY)
                    #         print(img_subs.dtype)

                    #             img_compare = cv2.compare(img_subs, diff_th, cv2.CMP_GT)#使用cv2的函數較快

                    # ----room in
                    # if zoom_in_value is not None:
                    #     zeros = np.zeros_like(img_compare, dtype=np.uint8)
                    #     zeros[v:-v, v:-v] = img_compare[v:-v, v:-v]
                    #     img_compare = zeros

                    # ----
                    defect_count = 0
                    predict = 'NG'

                    # ----connected components(method 1)使用方框標註出瑕疵
                    label_num, label_map, stats, centroids = cv2.connectedComponentsWithStats(img_compare,
                                                                                              connectivity=4)
                    for label in range(1, label_num):  # 0是背景
                        s = stats[label]
                        if s[-1] > cc_th:
                            defect_count += 1

                            if cc_type == "rec":
                            #     if save_rec is True:
                            #         crop = img_copy[s[1]:s[1] + s[3],s[0]:s[0] + s[2],:]
                            #         crop = cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)
                            #         crop = np_data2img_data(crop)
                            #         save_path = path.split("\\")[-1].split(".")[0] + "_defect_" +str(label) + '.bmp'
                            #         save_path = os.path.join(save_dir, save_path)
                            #         cv2.imencode('.bmp', crop)[1].tofile(save_path)
                                cv2.rectangle(img_copy, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), color, 1)




                            else:
                                mask = np.where(label_map == label)
                                img_copy[mask] = color

                    # if defect_count == 0:
                    #     predict = 'OK'

                    # ----save images
                    if save_type == 'compare':
                        show_list = [img_ori, img_copy]
                        show_qty = len(show_list)
                        plot = plt.figure(figsize=(15, 15),clear=True)
                        for i in range(show_qty):
                            show_img = np_data2img_data(show_list[i])
                            plt.subplot(1, show_qty, i + 1)
                            plt.imshow(show_img)
                            plt.axis('off')

                        # ----create the save path
                        #save_sub_dir = "pred_{}_ans_{}".format(predict, ans_list[abs_num])
                        save_path = os.path.join(save_dir, path.split("\\")[-1].split(".")[0] + '.jpg')
                        plt.savefig(save_path)
                        del plot
                    else:
                        show_img = np_data2img_data(img_copy)
                        # ----create the save path
                        #save_sub_dir = "pred_{}_ans_{}".format(predict, ans_list[abs_num])
                        save_path = os.path.join(save_dir, path.split("\\")[-1])
                        cv2.imencode('.jpg', show_img[:, :, ::-1])[1].tofile(save_path)

def AE_crop_defects(img_dir, pb_path, diff_th, cc_th, batch_size=32, zoom_in_value=None,
                           node_dict=None, process_dict=None, setting_dict=None,
                          read_subdir=False,save_dir=None):
    save_rec = True
    color = (1, 0, 0)
    # ----norm
    diff_th /= 255
    # ----zoom in value
    if zoom_in_value is not None:
        v = zoom_in_value


    #----model restoration
    if node_dict is None:
        print("node_dict is None")
        raise ValueError
    else:
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=None)

    tf_recon = tf_dict['output']
    height = tf_dict['input'].shape[1].value
    width = tf_dict['input'].shape[2].value
    model_shape = (height, width, 3)

    if read_subdir is True:
        sub_dirs = [obj.path for obj in os.scandir(img_dir) if obj.is_dir()]
    else:
        sub_dirs = [img_dir]

    #----create dirs
    if save_dir is None:
        save_dir = os.path.dirname(img_dir)
    save_dir = os.path.join(save_dir, 'crop_defects')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



    for dir_path in sub_dirs:
        paths = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]

        if len(paths):
            #----create save sub dir
            save_sub_dir = dir_path.split("\\")[-1]
            save_sub_dir = os.path.join(save_dir,save_sub_dir)
            if not os.path.exists(save_sub_dir):
                os.makedirs(save_sub_dir)

            #----
            ites = math.ceil(len(paths) / batch_size)
            for index in range(ites):
                num_start = batch_size * index
                num_end = num_start + batch_size
                if num_end > len(paths):
                    num_end = len(paths)

                batch_paths = paths[num_start:num_end]

                # ----create recon
                batch_data = get_process_data(batch_paths, model_shape, process_dict=process_dict,
                                              setting_dict=setting_dict)
                # print(batch_data.shape)
                batch_recon = sess.run(tf_recon, feed_dict={tf_dict['input']: batch_data})

                #----相減取絕對值
                img_subs = np.abs(batch_data - batch_recon)

                img_grays = np.zeros(img_subs.shape[:3])
                # Gray = R*0.299 + G*0.587 + B*0.114
                img_grays[:, :, :] = img_subs[:, :, :, 0] * 0.299 + \
                                     img_subs[:, :, :, 1] * 0.587 + \
                                     img_subs[:, :, :, 2] * 0.114

                img_compare_array = np.where(img_grays >= diff_th, 255, 0)
                img_compare_array = img_compare_array.astype(np.uint8)
                if zoom_in_value is not None:
                    zeros = np.zeros_like(img_compare_array)
                    zeros[:,v:-v, v:-v] = img_compare_array[:,v:-v, v:-v]
                    img_compare_array = zeros

                for j, img_compare in enumerate(img_compare_array):
                    # img_compare = img_compare_array[j]
                    abs_num = num_start + j
                    path = paths[abs_num]
                    img_ori = batch_data[j]
                    img_copy = batch_data[j].copy()


                    # ----
                    defect_count = 0

                    # ----connected components(method 1)使用方框標註出瑕疵
                    label_num, label_map, stats, centroids = cv2.connectedComponentsWithStats(img_compare,
                                                                                              connectivity=4)
                    for label in range(1, label_num):  # 0是背景
                        s = stats[label]
                        if s[-1] > cc_th:
                            crop = img_copy[s[1]:s[1] + s[3],s[0]:s[0] + s[2],:]
                            crop = cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)
                            crop = np_data2img_data(crop)
                            #----set the new filename
                            save_path = path.split("\\")[-1].split(".")[0] + "_defect" +str(defect_count) + "_cc" + str(s[4]) + '.bmp'
                            save_path = os.path.join(save_sub_dir, save_path)
                            cv2.imencode('.bmp', crop)[1].tofile(save_path)
                            defect_count += 1




                    # if defect_count == 0:
                    #     predict = 'OK'

                    # ----save images
                    # if save_type == 'compare':
                    #     show_list = [img_ori, img_copy]
                    #     show_qty = len(show_list)
                    #     plt.figure(figsize=(15, 15))
                    #     for i in range(show_qty):
                    #         show_img = np_data2img_data(show_list[i])
                    #         plt.subplot(1, show_qty, i + 1)
                    #         plt.imshow(show_img)
                    #         plt.axis('off')
                    #
                    #     # ----create the save path
                    #     #save_sub_dir = "pred_{}_ans_{}".format(predict, ans_list[abs_num])
                    #     save_path = os.path.join(save_dir, path.split("\\")[-1].split(".")[0] + '.jpg')
                    #     plt.savefig(save_path)
                    # else:
                    #     show_img = np_data2img_data(img_copy)
                    #     # ----create the save path
                    #     #save_sub_dir = "pred_{}_ans_{}".format(predict, ans_list[abs_num])
                    #     save_path = os.path.join(save_dir, path.split("\\")[-1])
                    #     cv2.imencode('.jpg', show_img[:, :, ::-1])[1].tofile(save_path)


if __name__ == "__main__":
    #----recon_pixel_comparison
    '''
    目的:使用AE的PB檔，統計OK與NG的個數，計算出過殺與漏檢，不會產出瑕疵標註圖
    答案來源:使用圖片所屬的資料夾名稱，要使用大寫，例如 OK_tiny_defect，程式會解析成OK類別；NG_ALlost會解析成NG類別
    diff_th:原圖與生成圖的像素差異值
    cc_th:連通數目值
    zoom_in_value是設定內縮像素，為了不考慮圖片邊緣產生的瑕疵
    process_dict是用來決定圖片是否要在推論前進行filter處理
    setting_dict是用來決定filter處理的kernel
    save_type:分類完後有4種圖片可以選擇儲存，分別是pred_ok_ans_ok、pred_ok_ans_ng、pred_ng_ans_ok、pred_ng_ans_ng
    '''
    img_source = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\test"
    # img_source = r"D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\predict_img"
    # pb_path = r"D:\code\model_saver\Opto_tech\AE_PDAP_top_20220208_3\infer_99.65.nst"
    # pb_path = r"D:\code\model_saver\AE_Seg_20\pb_model_seg.pb"
    pb_path = r"D:\code\model_saver\AE_Seg_21\infer_best_epoch288.pb"
    # save_dir = r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E04\02\results_544x832"
    save_dir = img_source

    diff_th = 20
    cc_th = 10

    zoom_in_value = None#[75,77,88,88]#5 #[75,77,88,88]
    mask_json_path = None#r"D:\dataset\optotech\silicon_division\PDAP\top_view\PD-55077GR-AP AI Training_2022.01.26\AE_results_544x832_diff10cc60_more_filters\pred_ng_ans_ok\t0_12_-20_MatchLightSet_作用區_Ng_1.json"
    to_mask = False

    batch_size = 16

    node_dict = {'input': 'input:0',
                 'input_ori': 'input_ori:0',
                 'loss': "loss_AE:0",
                 'embeddings': 'embeddings:0',
                 'output': "output_AE:0"
                 }

    process_dict = {'ave_filter': False, 'gau_filter': False}
    setting_dict = {'ave_filter': (3, 3), 'gau_filter': (3, 3)}

    save_type = [False,# pred_ok_ans_ok_count
                 False,#pred_ok_ans_ng_count
                 False,#pred_ng_ans_ok_count
                 True #pred_ng_ans_ng_count
                 ]


    # recon_pixel_comparison(img_source, pb_path, diff_th, cc_th, batch_size=batch_size,
    #                        zoom_in_value=zoom_in_value,
    #                        mask_json_path = mask_json_path,
    #                        to_mask=to_mask,
    #                        node_dict=node_dict,
    #                        process_dict=process_dict, setting_dict=setting_dict,
    #                        save_type=save_type, save_dir=save_dir)

    #----AE_find_defects
    '''
    目的:將次資料夾內的圖片標註上瑕疵
    cc_type:標註瑕疵的方式，可選擇方框(rec)或上色(dye)
    save_type:預設值是''，僅儲存瑕疵圖，亦可輸入compare，會儲存原圖與瑕疵圖的比較
    read_subdir:預設值是False，僅讀取資料夾內的圖片，若True，則會讀取'次'資料夾的圖片
    '''
    # save_dir = r"D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\test\AE_results"
    # zoom_in_value = [75,77,88,88]#5 #[75,77,88,88]
    mask_json_path = None#r"D:\dataset\optotech\silicon_division\PDAP\top_view\PD-55077GR-AP AI Training_2022.01.26\AE_results_544x832_diff10cc60_more_filters\pred_ng_ans_ok\t0_12_-20_MatchLightSet_作用區_Ng_1.json"
    AE_find_defects(save_dir, pb_path, diff_th, cc_th, batch_size=batch_size, zoom_in_value=zoom_in_value,to_mask=to_mask,
                    node_dict=node_dict, process_dict=process_dict, setting_dict=setting_dict, cc_type="",
                    save_type='no',save_recon=True,read_subdir=False,mask_json_path=mask_json_path)








