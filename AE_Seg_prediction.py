import os,math,cv2,json,imgviz,time,re
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from Utility import tools,Seg_performance,get_classname_id_color,tf_utility

#----tensorflow version check
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile
    # import tensorflow_addons as tfa
    # from tensorflow.keras.layers import Activation
    # from tensorflow.keras.utils import get_custom_objects
print("Tensorflow version of {}: {}".format(__file__,tf.__version__))
print_out = True
TCPConnected = False

img_format = {'png','PNG','jpg','JPG','JPEG','bmp','BMP'}
#
# def AE_find_defects(img_dir, pb_path, diff_th, cc_th, batch_size=32, zoom_in_value=None,to_mask=False,
#                            node_dict=None, process_dict=None, setting_dict=None,cc_type="rectangle",
#                            save_type='',read_subdir=False,mask_json_path=None):

def say_sth(msg, print_out=False,header=None):
    if print_out:
        print(msg)
    # if TCPConnected:
    #     TCPClient.send(msg + "\n")

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

def AE_Seg_comparison(img_dir,classnames_path,pb_path_list,node_dict,to_save=True):
    tf_tl = tf_utility()
    tf_tl.print_out = True
    batch_size = 12
    predict_png_list = list()
    titles = []


    paths, qty = tf_tl.get_paths(img_dir)
    msg = "SEG圖片數量:{}".format(qty)
    say_sth(msg, print_out=True)

    if qty == 0:
        print("Error:no image files")
    else:
        #----titles
        for pb_path in pb_path_list:
            titles.append(pb_path.split("\\")[-2])

        # ----create save dir
        # ----name tailer
        xtime = time.localtime()
        name_tailer = ''
        for i in range(6):
            string = str(xtime[i])
            if len(string) == 1:
                string = '0' + string
            name_tailer += string
        save_dir = os.path.join(img_dir,'AE_Seg_comparison_'+name_tailer)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # ----get classname2id, id2color
        class_names, class_name_to_id, id_to_color = tf_tl.get_classname_id_color(classnames_path)
        tf_tl.class_name2id = class_name_to_id
        tf_tl.id2color = id_to_color
        colormap = imgviz.label_colormap()

        ites = math.ceil(qty / batch_size)

        #----choose pb file
        for pb_path in pb_path_list:
            tf_tl.sess, tf_tl.tf_dict = tf_tl.model_restore_from_pb(pb_path, node_dict)
            tf_input = tf_tl.tf_dict['input']
            tf_prediction = tf_tl.tf_dict['prediction']
            model_shape = tf_tl.get_model_shape('input')
            temp_list = []

            for idx_seg in range(ites):
                # ----get batch paths
                seg_paths = tf_tl.get_ite_data(paths, idx_seg,
                                               batch_size=batch_size)

                # ----get batch data
                batch_data = tf_tl.get_4D_data(seg_paths,model_shape[1:])
                # batch_data, batch_label = tf_tl.get_4D_img_label_data(seg_paths,
                #                                                       model_shape[1:],
                #                                                       json_paths=None,
                #                                                       dtype=dtype)
                predict_label = tf_tl.sess.run(tf_prediction,feed_dict={tf_input: batch_data})
                predict_label = np.argmax(predict_label, axis=-1).astype(np.uint8)

                batch_data *= 255
                batch_data = batch_data.astype(np.uint8)

                for i in range(len(predict_label)):
                    img = batch_data[i]

                    #----label to color
                    zeros = np.zeros_like(batch_data[i])
                    for label_num in np.unique(predict_label[i]):
                        if label_num != 0:
                            # print(label_num)
                            coors = np.where(predict_label[i] == label_num)
                            zeros[coors] = tf_tl.id2color[label_num]

                    predict_png = cv2.addWeighted(img, 0.5, zeros, 0.5, 0)
                    temp_list.append(predict_png)

            predict_png_list.append(temp_list)


        #----comparison
        qty_show = len(pb_path_list)
        plt.figure(num=1,figsize=(5 * qty_show, 5 * qty_show), clear=True)

        for idx_path,path in enumerate(paths):
            for i,img_list in enumerate(predict_png_list):
                plt.subplot(1, qty_show, i + 1)
                plt.imshow(img_list[idx_path])
                plt.axis('off')
                plt.title(titles[i])

            if to_save is False:
                plt.show()
            else:
                save_path = os.path.join(save_dir, path.split("\\")[-1])
                plt.savefig(save_path)

        # ----save the log
        content = {'img_dir':img_dir,
                   'pb_path_list':pb_path_list,
                   'node_dict':node_dict,
                   'classnames_path':classnames_path,
                   'to_save':to_save,
                   'time':time.asctime()}

        save_path = os.path.join(save_dir,'log.json')
        with open(save_path,'w') as f:
            json.dump(content,f)

def cal_intersection_union(predict,label,num_classes):

    bins_num = num_classes + 1
    intersection = predict[predict == label]

    area_intersection, _ = np.histogram(intersection, bins=np.arange(bins_num))
    area_predict, _ = np.histogram(predict, bins=np.arange(bins_num))
    area_label, _ = np.histogram(label, bins=np.arange(bins_num))
    area_union = area_predict + area_label - area_intersection

    # print("area_intersection:", area_intersection)
    # print("area_predict:", area_predict)
    # print("area_label:", area_label)
    # print("union:", area_predict + area_label - area_intersection)

    return area_predict,area_label,area_intersection,area_union

def dict_transform(ori_dict,set_key=False,set_value=False):
    new_dict = dict()
    for key,value in ori_dict.items():
        if set_key:
            key = int(key)
        if set_value:
            value = int(value)
        new_dict[key] = value

    return new_dict

def get_latest_json_content(dir_path):
    #----var
    content = None
    file_name = "train_result_"

    #----
    file_nums = [int(file.name.split(".")[0].split("_")[-1]) for file in os.scandir(dir_path) if
                 file.name.find(file_name) >= 0]

    if len(file_nums) == 0:
        print("No train results file found")
        #raise ValueError
    else:
        extensions = [file.name.split(".")[-1] for file in os.scandir(dir_path) if
                      file.name.find(file_name) >= 0]
        seq = np.argsort(file_nums)

        json_path = os.path.join(dir_path, file_name + str(file_nums[seq[-1]]) + '.' + extensions[seq[-1]])
        print("The latest json file: ", json_path)

        # ----read the file
        ret = file_decode_v2(json_path, random_num_range=10, return_value=True, to_save=False)
        # ret is None or bytes

        if ret is None:
            print("ret is None. The file is not secured")
            with open(json_path, 'r') as f:
                content = json.load(f)
        else:
            print("ret is not None. The file is decoded")
            content = json.loads(ret.decode())

    return content

def image_similarity(img_dir,pb_path,node_dict,to_save_recon=False):
    #----var
    tf_tl = tf_utility()
    tf_tl.print_out = True
    batch_size = 1
    loss_list = []
    select_num = 10

    #----get image paths
    if isinstance(img_dir,list):
        paths = list()
        qty = 0
        for dir_path in img_dir:
            paths_temp, qty_temp = tf_tl.get_paths(dir_path)
            if qty_temp > 0:
                paths.extend(paths_temp.tolist())
                qty += qty_temp
    else:
        paths, qty = tf_tl.get_paths(img_dir)

    msg = "圖片數量:{}".format(qty)
    say_sth(msg, print_out=True)

    if qty == 0:
        msg = "No images are found!"
        say_sth(msg, print_out=True)
    else:
        #----create save dir
        if isinstance(img_dir,list):
            save_dir = os.path.join(img_dir[0], pb_path.split("\\")[-2])
        else:
            save_dir = os.path.join(img_dir, pb_path.split("\\")[-2])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if to_save_recon:
            save_dir4recon = os.path.join(save_dir, 'reconstruction images')
            if not os.path.exists(save_dir4recon):
                os.makedirs(save_dir4recon)

        #----start time
        d_t = time.time()

        tf_tl.sess, tf_tl.tf_dict = tf_tl.model_restore_from_pb(pb_path, node_dict)
        tf_input = tf_tl.tf_dict['input']
        tf_input_ori = tf_tl.tf_dict['input_ori']
        tf_recon = tf_tl.tf_dict['recon']
        tf_loss = tf_tl.tf_dict['loss']

        model_shape = tf_tl.get_model_shape('input')

        ites = math.ceil(qty / batch_size)

        #----iteration
        for idx in range(ites):
            #----get batch paths
            batch_paths = tf_tl.get_ite_data(paths, idx, batch_size=batch_size)

            #----get batch data
            batch_data = tf_tl.get_4D_data(batch_paths,model_shape[1:])

            feed_dict = {tf_input:batch_data,tf_input_ori:batch_data}

            loss_t = tf_tl.sess.run(tf_loss, feed_dict=feed_dict)
            if to_save_recon:
                recons = tf_tl.sess.run(tf_recon,feed_dict)

                for i,path in enumerate(batch_paths):
                    splits = path.split("\\")[-1].split(".")
                    recon_ori = recons[i] * 255
                    # recon = cv2.convertScaleAbs(recon_ori)
                    recon = recon_ori.astype(np.uint8)
                    # print("相減:",np.sum(np.abs(recon_2-recon)))

                    loss = np.around(loss_t,decimals=4)

                    new_filename = splits[0] + "_ssim_{}.".format(str(loss)) + splits[-1]
                    # print(new_filename)
                    save_path = os.path.join(save_dir4recon,new_filename)
                    cv2.imencode('.{}'.format(splits[-1]), recon[:,:,::-1])[1].tofile(save_path)


            loss_list.append(loss_t)

        d_t = time.time() - d_t
        print("ave time:", d_t / qty)

        #----get previous training results
        msg_list = []
        c_dict = get_latest_json_content(os.path.dirname(pb_path))
        if c_dict is not None:
            train_loss_list = c_dict.get('train_loss_list')
            if isinstance(train_loss_list,list):
                if len(train_loss_list) > select_num:
                    ave = np.average(train_loss_list[-select_num:])
                    std = np.std(train_loss_list[-select_num:])
                    lower_limit = ave-2*std
                    msg = "之前訓練的最後{} epoch的SSIM平均:{},標準差:{}".format(select_num,ave,std)
                    msg_list.append(msg)
                    msg = "2倍標準差範圍: {} ~ {}".format(ave-2*std,ave+2*std)
                    msg_list.append(msg)
                else:
                    ave = np.average(train_loss_list)
                    std = np.std(train_loss_list)
                    lower_limit = ave - 2 * std
                    msg = "之前訓練的SSIM平均:{},標準差:{}".format(ave,std)
                    msg_list.append(msg)



        #----statistics
        ave_loss = np.average(loss_list)
        std_loss = np.std(loss_list)
        msg = "目前資料集的SSIM 平均:{},標準差:{}".format(ave_loss,std_loss)
        msg_list.append(msg)
        msg = "2倍標準差範圍: {} ~ {}".format(ave_loss - 2 * std_loss, ave_loss + 2 * std_loss)
        msg_list.append(msg)
        coors = np.where(loss_list < lower_limit)
        msg = "小於當時訓練的lower limit個數:{},檢出比例:{}".format(len(coors[0]),len(coors[0])/len(loss_list))
        msg_list.append(msg)
        for msg in msg_list:
            say_sth(msg, print_out=True)

def recon_seg_prediction(img_dir,pb_path,node_dict,to_save_predict_image=False,compare_with_label=False,
                         to_save_defect_undetected=False,to_save_false_detected=False,
                         acc_threshold=0.3,plt_arange='',prob_threshold=None,cc_th=None,
                         id2class_name_path=None):
    # ----var
    tf_tl = tf_utility()
    tf_tl.print_out = True
    titles = ['prediction', 'answer']
    batch_size = 1
    undetected_list = []
    falseDetected_list = []
    contour_cc_differ = list()
    ok_count = 0
    ng_count = 0


    # paths, json_paths, qty = tf_tl.get_subdir_paths_withJsonCheck(img_dir)
    if isinstance(img_dir,list):
        paths = list()
        json_paths = list()
        qty = 0
        for dir_path in img_dir:
            paths_temp, qty_temp = tf_tl.get_paths(dir_path)
            json_paths_temp = tf_tl.get_relative_json_files(paths_temp)
            if qty_temp > 0:
                paths.extend(paths_temp.tolist())
                json_paths.extend(json_paths_temp.tolist())
                qty += qty_temp
    else:
        paths, qty = tf_tl.get_paths(img_dir)
        json_paths = tf_tl.get_relative_json_files(paths)

    msg = "SEG圖片數量:{}".format(qty)
    say_sth(msg, print_out=True)

    if qty == 0:
        msg = "No images are found!"
        say_sth(msg, print_out=True)
    else:
        #----create save dir
        make_dir_list = []
        if isinstance(img_dir,list):
            save_dir = os.path.join(img_dir[0], pb_path.split("\\")[-2])
        else:
            save_dir = os.path.join(img_dir, pb_path.split("\\")[-2])
        make_dir_list.append(save_dir)

        if to_save_predict_image:
            save_dir4prediction = os.path.join(save_dir, 'prediction')
            make_dir_list.append(save_dir4prediction)
        if to_save_false_detected:
            save_dir4false_detected = os.path.join(save_dir, 'false_detected')
            make_dir_list.append(save_dir4false_detected)
        if to_save_defect_undetected:
            save_dir4defect_undetected = os.path.join(save_dir, 'defect_undetected')
            make_dir_list.append(save_dir4defect_undetected)

        makedirs(make_dir_list)

        #----get classname2id, id2color(from train_results)
        if id2class_name_path is None:
            content = get_latest_json_content(os.path.dirname(pb_path))
            if content is None:
                msg = "couldn't find train_result json files"
                say_sth(msg, print_out=True)
                raise ValueError
            # content取出來的id都是str，但是實際上使用是int，所以要再經過轉換
            class_names = content['class_names']
            class_name2id = dict_transform(content['class_name2id'], set_value=True)
            id2color = dict_transform(content['id2color'], set_key=True)
        else:
            class_names, class_name2id, id2class_name, id2color = get_classname_id_color(id2class_name_path, print_out=True,
                                                                                         save_dir=None)


        tf_tl.class_name2id = class_name2id
        tf_tl.id2color = id2color

        #----Seg performance
        seg_p = Seg_performance(len(class_names), print_out=True)
        if to_save_defect_undetected:
            seg_p.save_dir = save_dir4defect_undetected
            seg_p.to_save_img_undetected = True
        if to_save_false_detected:
            seg_p.save_dir4falseDetected = save_dir4false_detected
            seg_p.to_save_img_falseDetected = True

        #----start time
        d_t = time.time()

        tf_tl.sess, tf_tl.tf_dict = tf_tl.model_restore_from_pb(pb_path, node_dict)
        tf_input = tf_tl.tf_dict['input']
        tf_prediction = tf_tl.tf_dict['prediction']
        tf_input_recon = tf_tl.tf_dict.get('input_recon')
        tf_recon = tf_tl.tf_dict['recon']
        tf_softmax = tf_tl.tf_dict['softmax_Seg']

        model_shape = tf_tl.get_model_shape('input')

        predict_ites_seg = math.ceil(qty / batch_size)
        seg_p.reset_arg()
        seg_p.reset_defect_stat(acc_threshold=acc_threshold)

        for idx_seg in range(predict_ites_seg):
            # ----get batch paths
            batch_paths = tf_tl.get_ite_data(paths, idx_seg, batch_size=batch_size)
            batch_json_paths = tf_tl.get_ite_data(json_paths, idx_seg, batch_size=batch_size)

            # ----get batch data
            batch_data, batch_label = tf_tl.get_4D_img_label_data(batch_paths,
                                                                  model_shape[1:],
                                                                  json_paths=batch_json_paths,
                                                                  )
            # batch_data = tf_tl.get_4D_data(seg_paths,model_shape[1:])
            feed_dict = {tf_input: batch_data}

            batch_recon = tf_tl.sess.run(tf_recon, feed_dict=feed_dict)
            if tf_input_recon is not None:
                feed_dict[tf_input_recon] = batch_recon
            predict_label = tf_tl.sess.run(tf_prediction,
                                           feed_dict=feed_dict)  # ,tf_input_recon:batch_recon

            predict_softmax = tf_tl.sess.run(tf_softmax,feed_dict=feed_dict)

            #----method 1(prob_threshold)
            if prob_threshold is not None and prob_threshold <= 1:
                max_probs_batch = np.max(predict_softmax,axis=-1)
                coors_prob = np.where(max_probs_batch < prob_threshold)
                predict_label[coors_prob] = 0
                # print("{} pixels are set 0".format(len(coors_prob[0])))

            # predict_label = np.argmax(predict_label, axis=-1).astype(np.uint8)

            #----calculate the intersection and onion
            # seg_p.cal_intersection_union(predict_label, batch_label)

            #----calculate defects from label aspect by accuracy
            # t_list = seg_p.cal_label_defect_by_acc(predict_label, batch_label, paths=batch_paths, id2color=id2color)
            # if len(t_list):
            #     undetected_list.extend(t_list)

            #----calculate defects from prediction aspect by accuracy
            # t_list = seg_p.cal_predict_defect_by_acc(predict_label, batch_label, paths=batch_paths, id2color=id2color)
            # if len(t_list):
            #     falseDetected_list.extend(t_list)

            batch_data *= 255
            batch_data = batch_data.astype(np.uint8)

            for i in range(len(predict_label)):
                img = batch_data[i]
                # print(paths[batch_size * idx_seg + i])

                # ----label to color
                zeros = np.zeros_like(batch_data[i])
                ng_flag = False
                for label_num in np.unique(predict_label[i]):
                    if label_num != 0:
                        coors = np.where(predict_label[i] == label_num)
                        predict_name = id2class_name[label_num]
                        text_list = re.findall('ng', predict_name, re.I)
                        #print("predict_name:",predict_name)
                        if cc_th is None:
                            zeros[coors] = tf_tl.id2color[label_num]
                            if len(text_list):
                                if ng_flag is False:
                                    ng_flag = True
                        else:
                            if len(coors[0]) >= cc_th:
                                zeros[coors] = tf_tl.id2color[label_num]
                                if len(text_list):
                                    if ng_flag is False:
                                        ng_flag = True


                        # print(label_num)




                        #----method 2
                        # max_probs = np.max(predict_softmax[i][coors], axis=-1)
                        #
                        # n = len(np.where(max_probs > 0.9)[0])
                        # ratio = n / len(max_probs)
                        # print("label_num:{},ratio:{}".format(label_num,ratio))
                        #
                        # if ratio > 0.9:
                        #     zeros[coors] = tf_tl.id2color[label_num]

                #----ok,ng count
                if ng_flag:
                    ng_count += 1
                else:
                    ok_count += 1




                predict_png = cv2.addWeighted(img, 1, zeros, 0.5, 0)
                # ----create answer png
                show_imgs = [predict_png]
                path = paths[batch_size * idx_seg + i]
                if compare_with_label is True:
                    ext = path.split(".")[-1]
                    json_path = path.strip(ext) + 'json'

                    if os.path.exists(json_path):
                        answer_png = tf_tl.get_single_label_png(path, json_path)
                        show_imgs.append(answer_png)
                    else:
                        img_ori = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
                        show_imgs.append(img_ori[:, :, ::-1])
                        titles[-1] = 'original'
                qty_show = len(show_imgs)
                if qty_show == 1:
                    if to_save_predict_image is False:
                        pass
                    else:
                        save_path = os.path.join(save_dir4prediction, path.split("\\")[-1])
                        cv2.imwrite(save_path, show_imgs[0][:, :, ::-1])
                else:
                    plt.figure(num=1,figsize=(10 * qty_show, 5 * qty_show), clear=True)

                    for j, show_img in enumerate(show_imgs):
                        if plt_arange == 'vertical':
                            plt.subplot(qty_show, 1, j + 1)
                        else:
                            plt.subplot(1, qty_show, j + 1)
                        plt.imshow(show_img)
                        plt.axis('off')
                        plt.title(titles[j])

                    if to_save_predict_image:
                        save_path = os.path.join(save_dir4prediction, path.split("\\")[-1])
                        plt.savefig(save_path)

        d_t = time.time() - d_t
        print("ave time:", d_t / qty)

        #----statistics
        # iou, acc, all_acc = seg_p.cal_iou_acc()
        # defect_recall = seg_p.cal_defect_recall()
        # predict_sensitivity = seg_p.cal_defect_sensitivity()
        # contour_cc_differ = np.array(contour_cc_differ)
        print("ok count:{}, ng count:{}".format(ok_count,ng_count))

        #----save the log
        save_log(save_dir,'log.json',img_dir=img_dir,pb_path=pb_path,node_dict=node_dict,
                 to_save_predict_image=to_save_predict_image,compare_with_label=compare_with_label,
                 to_save_defect_undetected=to_save_defect_undetected,iou=iou.tolist(),acc=acc.tolist(),
                 all_acc=all_acc.tolist(),acc_threshold=acc_threshold,
                 label_defect_stat=seg_p.label_defect_stat.tolist(),
                 defect_recall=defect_recall.tolist(),undetected_list=undetected_list,
                 predict_defect_stat=seg_p.predict_defect_stat.tolist(),
                 predict_sensitivity=predict_sensitivity.tolist(),falseDetected_list=falseDetected_list,
                 time=time.asctime())


        # print(class_names)
        # print("iou:", iou)
        # print("acc:", acc)
        # print("all_acc:", all_acc)
        # print("label defect stat:", seg_p.label_defect_stat)
        # print("defect recall:", defect_recall)
        # print("prediction defect stat:",seg_p.predict_defect_stat)
        # print("defect sensitivity:",seg_p.defect_sensitivity)


def recon_seg_prediction_v2(img_dir,pb_path,node_dict,img_save_dict,threshold_dict,compare_with_answers=False,
                         plt_arange='',
                         id2class_name_path=None):
    # ----var
    tf_tl = tf_utility()
    tf_tl.print_out = True
    titles = ['prediction', 'answer']
    batch_size = 1
    undetected_list = []
    falseDetected_list = []
    ok_count = 0
    ng_count = 0
    to_save_predict_image = img_save_dict.get('to_save_predict_image')
    to_save_defect_undetected = img_save_dict.get('to_save_defect_undetected')
    to_save_false_detected = img_save_dict.get('to_save_false_detected')
    img_compare_with_label = img_save_dict.get('img_compare_with_label')
    img_compare_with_ori = img_save_dict.get('img_compare_with_ori')
    acc_threshold = threshold_dict.get('acc_threshold')
    prob_threshold = threshold_dict.get('prob_threshold')
    cc_th = threshold_dict.get('cc_th')

    if acc_threshold is None:
        acc_threshold = 0.3
        threshold_dict['acc_threshold'] = acc_threshold


    # paths, json_paths, qty = tf_tl.get_subdir_paths_withJsonCheck(img_dir)
    if isinstance(img_dir,list):
        paths = list()
        json_paths = list()
        qty = 0
        for dir_path in img_dir:
            paths_temp, qty_temp = tf_tl.get_paths(dir_path)

            if qty_temp > 0:
                paths.extend(paths_temp.tolist())
                qty += qty_temp
                if compare_with_answers:
                    json_paths_temp = tf_tl.get_relative_json_files(paths_temp)
                    if len(json_paths_temp) > 0:
                        json_paths.extend(json_paths_temp.tolist())

    else:
        paths, qty = tf_tl.get_paths(img_dir)
        if compare_with_answers:
            json_paths = tf_tl.get_relative_json_files(paths)

    msg = "SEG圖片數量:{}".format(qty)
    say_sth(msg, print_out=True)

    if qty == 0:
        msg = "No images are found!"
        say_sth(msg, print_out=True)
    else:
        #----create save dir
        make_dir_list = []
        if isinstance(img_dir,list):
            save_dir = os.path.join(img_dir[0], pb_path.split("\\")[-2])
        else:
            save_dir = os.path.join(img_dir, pb_path.split("\\")[-2])
        make_dir_list.append(save_dir)

        if to_save_predict_image:
            save_dir4prediction_ok = os.path.join(save_dir, 'prediction','ok')
            save_dir4prediction_ng = os.path.join(save_dir, 'prediction','ng')
            make_dir_list.append(save_dir4prediction_ok)
            make_dir_list.append(save_dir4prediction_ng)
        if to_save_false_detected:
            save_dir4false_detected = os.path.join(save_dir, 'false_detected')
            make_dir_list.append(save_dir4false_detected)
        if to_save_defect_undetected:
            save_dir4defect_undetected = os.path.join(save_dir, 'defect_undetected')
            make_dir_list.append(save_dir4defect_undetected)

        makedirs(make_dir_list)

        #----get classname2id, id2color(from train_results)
        if id2class_name_path is None:
            content = get_latest_json_content(os.path.dirname(pb_path))
            if content is None:
                msg = "couldn't find train_result json files"
                say_sth(msg, print_out=True)
                raise ValueError
            # content取出來的id都是str，但是實際上使用是int，所以要再經過轉換
            class_names = content['class_names']
            class_name2id = dict_transform(content['class_name2id'], set_value=True)
            id2color = dict_transform(content['id2color'], set_key=True)
        else:
            class_names, class_name2id, id2class_name, id2color = get_classname_id_color(id2class_name_path, print_out=True,
                                                                                         save_dir=None)


        tf_tl.class_name2id = class_name2id
        tf_tl.id2color = id2color

        #----Seg performance
        if compare_with_answers:
            seg_p = Seg_performance(len(class_names), print_out=True)
            if to_save_defect_undetected:
                seg_p.save_dir = save_dir4defect_undetected
                seg_p.to_save_img_undetected = True
            if to_save_false_detected:
                seg_p.save_dir4falseDetected = save_dir4false_detected
                seg_p.to_save_img_falseDetected = True
            seg_p.reset_arg()
            seg_p.reset_defect_stat(acc_threshold=acc_threshold)

        #----start time
        d_t = time.time()

        tf_tl.sess, tf_tl.tf_dict = tf_tl.model_restore_from_pb(pb_path, node_dict)
        tf_input = tf_tl.tf_dict['input']
        tf_prediction = tf_tl.tf_dict['prediction']
        tf_input_recon = tf_tl.tf_dict.get('input_recon')
        tf_recon = tf_tl.tf_dict['recon']
        tf_softmax = tf_tl.tf_dict['softmax_Seg']

        model_shape = tf_tl.get_model_shape('input')

        predict_ites_seg = math.ceil(qty / batch_size)


        for idx_seg in range(predict_ites_seg):
            # ----get batch paths
            batch_paths = tf_tl.get_ite_data(paths, idx_seg, batch_size=batch_size)
            batch_json_paths = tf_tl.get_ite_data(json_paths, idx_seg, batch_size=batch_size)

            # ----get batch data
            if compare_with_answers:
                batch_data, batch_label = tf_tl.get_4D_img_label_data(batch_paths,
                                                                      model_shape[1:],
                                                                      json_paths=batch_json_paths,
                                                                      )
            else:
                batch_data = tf_tl.get_4D_data(batch_paths, model_shape[1:])
            # batch_data = tf_tl.get_4D_data(seg_paths,model_shape[1:])
            feed_dict = {tf_input: batch_data}

            batch_recon = tf_tl.sess.run(tf_recon, feed_dict=feed_dict)
            if tf_input_recon is not None:
                feed_dict[tf_input_recon] = batch_recon
            predict_label = tf_tl.sess.run(tf_prediction,
                                           feed_dict=feed_dict)  # ,tf_input_recon:batch_recon

            predict_softmax = tf_tl.sess.run(tf_softmax,feed_dict=feed_dict)

            #----method 1(prob_threshold)
            if prob_threshold is not None and prob_threshold <= 1:
                max_probs_batch = np.max(predict_softmax,axis=-1)
                coors_prob = np.where(max_probs_batch < prob_threshold)
                predict_label[coors_prob] = 0
                # print("{} pixels are set 0".format(len(coors_prob[0])))

            # predict_label = np.argmax(predict_label, axis=-1).astype(np.uint8)

            if compare_with_answers:
                #----calculate the intersection and onion
                seg_p.cal_intersection_union(predict_label, batch_label)

                #----calculate defects from label aspect by accuracy
                t_list = seg_p.cal_label_defect_by_acc(predict_label, batch_label, paths=batch_paths, id2color=id2color)
                if len(t_list):
                    undetected_list.extend(t_list)

                #----calculate defects from prediction aspect by accuracy
                t_list = seg_p.cal_predict_defect_by_acc(predict_label, batch_label, paths=batch_paths, id2color=id2color)
                if len(t_list):
                    falseDetected_list.extend(t_list)

            batch_data *= 255
            batch_data = batch_data.astype(np.uint8)

            for i in range(len(predict_label)):
                img = batch_data[i]
                # print(paths[batch_size * idx_seg + i])

                # ----label to color
                # zeros = np.zeros_like(batch_data[i])
                zeros = img.copy()

                ng_flag = False
                map_sum = np.sum(predict_label[i])
                if map_sum > 0:
                    cc_nums, cc_map, stats, centroids = cv2.connectedComponentsWithStats(predict_label[i], connectivity=8)
                    for cc_num in range(1, cc_nums):
                        s = stats[cc_num]
                        coors = np.where(cc_map == cc_num)
                        predict_class = predict_label[i][coors][0]
                        predict_name = id2class_name[predict_class]
                        text_list = re.findall('ng', predict_name, re.I)
                        if cc_th is None:
                            zeros[coors] = tf_tl.id2color[predict_class]
                            if len(text_list):
                                if ng_flag is False:
                                    ng_flag = True
                        else:
                            if s[-1] >= cc_th:
                                zeros[coors] = tf_tl.id2color[predict_class]
                                if len(text_list):
                                    if ng_flag is False:
                                        ng_flag = True
            #----use unique numbers to handle
            # for label_num in np.unique(predict_label[i]):
            #     if label_num != 0:
            #         coors = np.where(predict_label[i] == label_num)
            #         predict_name = id2class_name[label_num]
            #         text_list = re.findall('ng', predict_name, re.I)
            #         if cc_th is None:
            #             zeros[coors] = tf_tl.id2color[label_num]
            #             if len(text_list):
            #                 if ng_flag is False:
            #                     ng_flag = True
            #         else:
            #             if len(coors[0]) >= cc_th:
            #                 zeros[coors] = tf_tl.id2color[label_num]
            #                 if len(text_list):
            #                     if ng_flag is False:
            #                         ng_flag = True
            #
            #
            #         # print(label_num)
            #
            #
            #
            #
            #         #----method 2
            #         # max_probs = np.max(predict_softmax[i][coors], axis=-1)
            #         #
            #         # n = len(np.where(max_probs > 0.9)[0])
            #         # ratio = n / len(max_probs)
            #         # print("label_num:{},ratio:{}".format(label_num,ratio))
            #         #
            #         # if ratio > 0.9:
            #         #     zeros[coors] = tf_tl.id2color[label_num]

            #----ok,ng count
                if ng_flag:
                    ng_count += 1
                else:
                    ok_count += 1



                #----set save dir for prediction images
                if to_save_predict_image:
                    # predict_png = cv2.addWeighted(img, 1, zeros, 0.5, 0)
                    predict_png = zeros
                    # ----create answer png
                    show_imgs = [predict_png]
                    path = paths[batch_size * idx_seg + i]
                    if img_compare_with_ori:
                        img_ori = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
                        show_imgs.append(img_ori[:, :, ::-1])
                        titles[-1] = 'original'
                    if compare_with_answers:
                        if img_compare_with_label is True:
                            ext = path.split(".")[-1]
                            json_path = path.strip(ext) + 'json'

                            if os.path.exists(json_path):
                                answer_png = tf_tl.get_single_label_png(path, json_path)
                                show_imgs.append(answer_png)
                            else:
                                img_ori = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
                                show_imgs.append(img_ori[:, :, ::-1])
                                titles[-1] = 'original'

                    if ng_flag:
                        dir_path = save_dir4prediction_ng
                    else:
                        dir_path = save_dir4prediction_ok

                    qty_show = len(show_imgs)
                    if qty_show == 1:
                        save_path = os.path.join(dir_path, path.split("\\")[-1])
                        ext = "." + path.split("\\")[-1].split('.')[-1]
                        # cv2.imwrite(save_path, show_imgs[0][:, :, ::-1])
                        cv2.imencode(ext, show_imgs[0][:, :, ::-1])[1].tofile(save_path)
                    else:
                        plt.figure(num=1,figsize=(10 * qty_show, 5 * qty_show), clear=True)

                        for j, show_img in enumerate(show_imgs):
                            if plt_arange == 'vertical':
                                plt.subplot(qty_show, 1, j + 1)
                            else:
                                plt.subplot(1, qty_show, j + 1)
                            plt.imshow(show_img)
                            plt.axis('off')
                            plt.title(titles[j])

                        save_path = os.path.join(dir_path, path.split("\\")[-1].split(".")[0] + '.jpg')
                        plt.savefig(save_path)

        d_t = time.time() - d_t
        print("ave time:", d_t / qty)

        #----statistics
        if compare_with_answers:
            iou, acc, all_acc = seg_p.cal_iou_acc()
            defect_recall = seg_p.cal_defect_recall()
            predict_sensitivity = seg_p.cal_defect_sensitivity()
            print("classnames:",class_names)
            print("iou:", iou)
            print("acc:", acc)
            print("all_acc:", all_acc)
            print("label defect stat:", seg_p.label_defect_stat)
            print("defect recall:", defect_recall)
            print("prediction defect stat:", seg_p.predict_defect_stat)
            print("defect sensitivity:", seg_p.defect_sensitivity)

        print("ok count:{}, ng count:{}".format(ok_count,ng_count))

        #----save the log
        if compare_with_answers:
            save_log(save_dir,'log.json',img_dir=img_dir,pb_path=pb_path,node_dict=node_dict,
                     img_save_dict=img_save_dict, threshold_dict=threshold_dict,
                     compare_with_answers=compare_with_answers, id2class_name_path=id2class_name_path,
                     iou=iou.tolist(),acc=acc.tolist(), all_acc=all_acc.tolist(),
                     label_defect_stat=seg_p.label_defect_stat.tolist(),
                     defect_recall=defect_recall.tolist(),undetected_list=undetected_list,
                     predict_defect_stat=seg_p.predict_defect_stat.tolist(),
                     predict_sensitivity=predict_sensitivity.tolist(),falseDetected_list=falseDetected_list,
                     time=time.asctime())
        else:
            save_log(save_dir, 'log.json', img_dir=img_dir, pb_path=pb_path, node_dict=node_dict,
                     img_save_dict=img_save_dict,threshold_dict=threshold_dict,compare_with_answers=compare_with_answers,
                     id2class_name_path=id2class_name_path,
                     time=time.asctime())


def infer_speed_test(img_dir,pb_path,node_dict,batch_size=1,set_img_num=1000):
    # ----var
    tf_tl = tf_utility()
    tf_tl.print_out = True
    # titles = ['prediction', 'answer']
    # batch_size = 1
    undetected_list = []
    contour_cc_differ = list()


    # paths, json_paths, qty = tf_tl.get_subdir_paths_withJsonCheck(img_dir)
    if isinstance(img_dir,list):
        paths = list()
        json_paths = list()
        qty = 0
        for dir_path in img_dir:
            paths_temp, qty_temp = tf_tl.get_paths(dir_path)
            # json_paths_temp = tf_tl.get_relative_json_files(paths_temp)
            if qty_temp > 0:
                paths.extend(paths_temp.tolist())
                # json_paths.extend(json_paths_temp.tolist())
                qty += qty_temp
    else:
        paths, qty = tf_tl.get_paths(img_dir)
        # json_paths = tf_tl.get_relative_json_files(paths)

    if qty == 0:
        msg = "No images are found!"
        say_sth(msg, print_out=True)
    else:
        if qty < set_img_num:
            paths = paths * math.ceil(set_img_num / qty)

        paths = paths[:set_img_num]
        qty = len(paths)

        msg = "SEG圖片數量:{}".format(qty)
        say_sth(msg, print_out=True)

        #----create save dir
        # if isinstance(img_dir,list):
        #     save_dir = os.path.join(img_dir[0], pb_path.split("\\")[-2])
        # else:
        #     save_dir = os.path.join(img_dir, pb_path.split("\\")[-2])
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_dir4defect_undetected = os.path.join(save_dir, 'defect_undetected')
        # if not os.path.exists(save_dir4defect_undetected):
        #     os.makedirs(save_dir4defect_undetected)

        #----get classname2id, id2color(from train_results)
        # content = get_latest_json_content(os.path.dirname(pb_path))
        # if content is None:
        #     msg = "couldn't find train_result json files"
        #     say_sth(msg, print_out=True)
        #     raise ValueError
        # # content取出來的id都是str，但是實際上使用是int，所以要再經過轉換
        # class_names = content['class_names']
        # class_name2id = dict_transform(content['class_name2id'], set_value=True)
        # id2color = dict_transform(content['id2color'], set_key=True)
        #
        # tf_tl.class_name2id = class_name2id
        # tf_tl.id2color = id2color

        #----Seg performance
        # seg_p = Seg_performance(len(class_names), print_out=True)
        # if to_save_defect_undetected:
        #     seg_p.save_dir = save_dir4defect_undetected
        #     seg_p.to_save_img_undetected = to_save_defect_undetected

        #----start time
        d_t = time.time()

        tf_tl.sess, tf_tl.tf_dict = tf_tl.model_restore_from_pb(pb_path, node_dict)
        tf_input = tf_tl.tf_dict['input']
        tf_prediction = tf_tl.tf_dict['prediction']
        tf_input_recon = tf_tl.tf_dict.get('input_recon')
        tf_recon = tf_tl.tf_dict['recon']
        tf_softmax = tf_tl.tf_dict['softmax_Seg']

        model_shape = tf_tl.get_model_shape('input')

        predict_ites_seg = math.ceil(qty / batch_size)
        print("model init time:",time.time()-d_t)

        d_t = time.time()
        for idx_seg in range(predict_ites_seg):
            # ----get batch paths
            batch_paths = tf_tl.get_ite_data(paths, idx_seg, batch_size=batch_size)
            # batch_json_paths = tf_tl.get_ite_data(json_paths, idx_seg, batch_size=batch_size)

            # ----get batch data
            # batch_data, batch_label = tf_tl.get_4D_img_label_data(batch_paths,
            #                                                       model_shape[1:],
            #                                                       json_paths=batch_json_paths,
            #                                                       )
            batch_data = tf_tl.get_4D_data(batch_paths,model_shape[1:])
            # batch_data = tf_tl.get_4D_data(seg_paths,model_shape[1:])
            feed_dict = {tf_input: batch_data}

            batch_recon = tf_tl.sess.run(tf_recon, feed_dict=feed_dict)
            if tf_input_recon is not None:
                feed_dict[tf_input_recon] = batch_recon
            # predict_label = tf_tl.sess.run(tf_prediction,
            #                                feed_dict=feed_dict)  # ,tf_input_recon:batch_recon

            predict_softmax = tf_tl.sess.run(tf_softmax,feed_dict=feed_dict)


        d_t = time.time() - d_t
        print("qty:{}, total time:{}, ave time:{}".format(qty,d_t,d_t / qty))

def save_log(save_dir,save_filename,**kwargs):
    save_path = os.path.join(save_dir, save_filename)
    with open(save_path, 'w') as f:
        json.dump(kwargs, f)

def makedirs(make_dir_list):
    msg_list = list()
    for dir_path in make_dir_list:
        if os.path.exists(dir_path):
            msg = "The folder is existed: {}".format(dir_path)
        else:
            os.makedirs(dir_path)
            msg = "The folder is created: {}".format(dir_path)
        msg_list.append(msg)

    for msg in msg_list:
        print(msg)

if __name__ == "__main__":
    # img_dir = r'D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\train'
    # img_dir = r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\NG(多區NG-輕嚴重)_20220504\selected'
    img_dir = [
        # r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train',
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220408新增破洞+金顆粒 資料\1",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220408新增破洞+金顆粒 資料\2",
        # r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\test',
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\All_defects\髒污(屬OK)"
        # r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\only_L2\L2_OK_無分類"
        r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\L2_NG_all"
        # r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\L2_OK_晶紋"
        # r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\AE_Seg\Seg\test"
    ]
    # img_dir = [
    #     r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\PD-55092G-AP_0.0.1_dataset\Seg_data\gold_particle\test",
    #     r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\PD-55092G-AP_0.0.1_dataset\Seg_data\hole\test",
    #     r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\PD-55092G-AP_0.0.1_dataset\Seg_data\Particle\test"
    # ]
    #
    # pb_path = r"D:\code\model_saver\AE_Seg_105\infer_best_epoch39.pb"
    # pb_path = r"C:\Users\User\Desktop\train_result\infer_best_epoch4_0.0.3_.pb"
    # pb_path = r"D:\code\model_saver\AE_Seg_009IRC-FB_Jane_20220627\infer_best_epoch171.nst"
    # pb_path = r"D:\code\model_saver\AE_Seg_132\infer_best_epoch178.pb"
    # pb_path = r"D:\code\model_saver\AE_Seg_134\infer_20220706181231.pb"
    # pb_path = r"D:\code\model_saver\AE_Seg_136\infer_best_epoch183.pb"
    pb_path = r"D:\code\model_saver\AE_Seg_136\infer_best_epoch174.pb"
    # pb_path = r"D:\code\model_saver\AE_Seg_109\infer_best_epoch3.nst"
    # pb_path = r"D:\code\model_saver\AE_Seg_106\infer_90.76.nst"
    # pb_path = r"D:\code\model_saver\AE_Seg_33\infer_best_epoch240.pb"
    # pb_path = r"D:\code\model_saver\AE_Seg_103\infer_best_epoch218.pb"

    id2class_name_path = r"D:\dataset\optotech\009IRC-FB\classnames.txt"
    node_dict = {'input': 'input:0',
                 'recon':'output_AE:0',
                 'input_recon':'input_recon:0',
                 'prediction': 'predict_Seg:0',
                 'softmax_Seg': 'softmax_Seg:0'
                 }
    # prob_threshold = 0.9
    # cc_th = None
    # recon_seg_prediction(img_dir, pb_path, node_dict,
    #                      id2class_name_path=id2class_name_path,
    #                      to_save_predict_image=False,
    #                      to_save_false_detected=False,
    #                      to_save_defect_undetected=False,
    #                      compare_with_label=False,
    #                      acc_threshold=0.3,
    #                      prob_threshold=prob_threshold,
    #                      cc_th=cc_th
    #                      )
    # infer_speed_test(img_dir, pb_path, node_dict,batch_size=1,set_img_num=1000)

    compare_with_answers = False
    img_save_dict = dict()
    img_save_dict['to_save_predict_image'] = False
    img_save_dict['to_save_false_detected'] = False
    img_save_dict['to_save_defect_undetected'] = False
    img_save_dict['img_compare_with_label'] = False
    img_save_dict['img_compare_with_ori'] = False

    threshold_dict = dict()
    threshold_dict['prob_threshold'] = None
    threshold_dict['cc_th'] = None
    threshold_dict['acc_threshold'] = 0.3

    recon_seg_prediction_v2(img_dir, pb_path, node_dict, img_save_dict, threshold_dict,
                            compare_with_answers=compare_with_answers,
                            id2class_name_path=id2class_name_path)

    #----for loop
    # prob_thresholds = [None,0.6,0.9]
    # cc_ths = [30,50,70]
    # for prob_threshold in prob_thresholds:
    #     for cc_th in cc_ths:
    #         threshold_dict['prob_threshold'] = prob_threshold
    #         threshold_dict['cc_th'] = cc_th
    #         recon_seg_prediction_v2(img_dir, pb_path, node_dict, img_save_dict, threshold_dict,
    #                                 compare_with_answers=compare_with_answers,
    #                                 id2class_name_path=id2class_name_path)
    #         print("Above: results of prob_threshold:{},cc_th:{} ".format(prob_threshold,cc_th))






