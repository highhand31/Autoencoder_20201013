import os,math,cv2,json,imgviz,time
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from Utility import tools,Seg_performance,get_classname_id_color

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
        plt.figure(figsize=(5 * qty_show, 5 * qty_show), clear=True)

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

def recon_seg_prediction(img_dir,pb_path,node_dict,to_save_predict_image=False,compare_with_label=False,
                         to_save_defect_undetected=False,acc_threshold=0.3,plt_arange=''):
    # ----var
    tf_tl = tf_utility()
    tf_tl.print_out = True
    titles = ['prediction', 'answer']
    batch_size = 1
    undetected_list = []
    contour_cc_differ = list()


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
        if isinstance(img_dir,list):
            save_dir = os.path.join(img_dir[0], pb_path.split("\\")[-2])
        else:
            save_dir = os.path.join(img_dir, pb_path.split("\\")[-2])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir4defect_undetected = os.path.join(save_dir, 'defect_undetected')
        if not os.path.exists(save_dir4defect_undetected):
            os.makedirs(save_dir4defect_undetected)

        #----get classname2id, id2color(from train_results)
        content = get_latest_json_content(os.path.dirname(pb_path))
        if content is None:
            msg = "could't find train_result json files"
            say_sth(msg, print_out=True)
            raise ValueError
        # content取出來的id都是str，但是實際上使用是int，所以要再經過轉換
        class_names = content['class_names']
        class_name2id = dict_transform(content['class_name2id'], set_value=True)
        id2color = dict_transform(content['id2color'], set_key=True)

        tf_tl.class_name2id = class_name2id
        tf_tl.id2color = id2color

        #----Seg performance
        seg_p = Seg_performance(len(class_names), print_out=True)
        if to_save_defect_undetected:
            seg_p.save_dir = save_dir4defect_undetected
            seg_p.to_save_img_undetected = to_save_defect_undetected

        #----start time
        d_t = time.time()

        tf_tl.sess, tf_tl.tf_dict = tf_tl.model_restore_from_pb(pb_path, node_dict)
        tf_input = tf_tl.tf_dict['input']
        tf_prediction = tf_tl.tf_dict['prediction']
        tf_input_recon = tf_tl.tf_dict['input_recon']
        tf_recon = tf_tl.tf_dict['recon']

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

            batch_recon = tf_tl.sess.run(tf_recon, feed_dict={tf_input: batch_data})
            predict_label = tf_tl.sess.run(tf_prediction,
                                           feed_dict={tf_input: batch_data,
                                                      tf_input_recon: batch_recon})  # ,tf_input_recon:batch_recon
            # predict_label = np.argmax(predict_label, axis=-1).astype(np.uint8)

            #----calculate the intersection and onion
            seg_p.cal_intersection_union(predict_label, batch_label)

            #----calculate defect by accuracy
            t_list, contour_cc_differ_temp= seg_p.cal_defect_by_acc(predict_label, batch_label, paths=batch_paths, id2color=id2color)
            contour_cc_differ.extend(contour_cc_differ_temp)
            if len(t_list):
                undetected_list.extend(t_list)


            batch_data *= 255
            batch_data = batch_data.astype(np.uint8)

            for i in range(len(predict_label)):
                img = batch_data[i]

                # ----label to color
                zeros = np.zeros_like(batch_data[i])
                for label_num in np.unique(predict_label[i]):
                    if label_num != 0:
                        # print(label_num)
                        coors = np.where(predict_label[i] == label_num)
                        zeros[coors] = tf_tl.id2color[label_num]

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
                        save_path = os.path.join(save_dir, path.split("\\")[-1])
                        cv2.imwrite(save_path, show_imgs[0][:, :, ::-1])
                else:
                    plt.figure(figsize=(10 * qty_show, 5 * qty_show), clear=True)

                    for j, show_img in enumerate(show_imgs):
                        if plt_arange == 'vertical':
                            plt.subplot(qty_show, 1, j + 1)
                        else:
                            plt.subplot(1, qty_show, j + 1)
                        plt.imshow(show_img)
                        plt.axis('off')
                        plt.title(titles[j])

                    if to_save_predict_image:
                        save_path = os.path.join(save_dir, path.split("\\")[-1])
                        plt.savefig(save_path)

        d_t = time.time() - d_t
        print("ave time:", d_t / qty)

        #----statistics
        iou, acc, all_acc = seg_p.cal_iou_acc()
        defect_recall = seg_p.cal_defect_recall()
        # contour_cc_differ = np.array(contour_cc_differ)

        #----save the log
        save_log(save_dir,'log.json',img_dir=img_dir,pb_path=pb_path,node_dict=node_dict,
                 to_save_predict_image=to_save_predict_image,compare_with_label=compare_with_label,
                 to_save_defect_undetected=to_save_defect_undetected,iou=iou.tolist(),acc=acc.tolist(),
                 all_acc=all_acc.tolist(),acc_threshold=acc_threshold,defect_stat=seg_p.defect_stat.tolist(),
                 defect_recall=defect_recall.tolist(),undetected_list=undetected_list,time=time.asctime())


        print(class_names)
        print("iou:", iou)
        print("acc:", acc)
        print("all_acc:", all_acc)
        print("defect stat:", seg_p.defect_stat)
        print("defect recall:", defect_recall)

        # print("len of contour_cc_differ:", len(contour_cc_differ))
        # print('average error pixels:', np.average(contour_cc_differ))
        # print("std of average error pixels:", np.std(contour_cc_differ))
def save_log(save_dir,save_filename,**kwargs):
    save_path = os.path.join(save_dir, save_filename)
    with open(save_path, 'w') as f:
        json.dump(kwargs, f)

class tf_utility(tools):
    def __init__(self):
        super().__init__()
        self.paths = list()
        self.print_out = False
        self.sess = None
        self.tf_dict = None
        print('tf version:', tf.__version__)

    def model_restore_from_pb(self,pb_path, node_dict, GPU_ratio=None):
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

    def get_model_shape(self,input_name):
        model_shape = None
        if isinstance(self.tf_dict,dict) is False:
            say_sth("The tf_dict is not a Dict format",print_out=self.print_out)
        else:
            tf_input = self.tf_dict.get(input_name)
            if tf_input is None:
                say_sth("{} is not existed in the tf_dict".format(input_name), print_out=self.print_out)
            else:
                shape = tf_input.shape
                model_shape = [None,shape[1].value,shape[2].value,shape[3].value]
                say_sth("model_shape:{}".format(model_shape), print_out=self.print_out)

        return model_shape

    def get_single_label_png(self,path,json_path):
        save_img = None
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        except:
            img = None

        if img is None:
            msg = "read failed:".format(path)
            say_sth(msg)
        else:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            label_shapes = self.get_label_shapes(json_path)
            if label_shapes is None:
                say_sth("label_shapes is None")
            lbl = self.shapes_to_label(
                img_shape=img.shape,
                shapes=label_shapes,
                label_name_to_value=self.class_name2id,
            )
            # print(np.unique(lbl))
            #----contour test
            #contours, hierarchy = cv2.findContours(lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #----
            zeros = np.zeros_like(img)
            for label_num in np.unique(lbl):
                if label_num != 0:
                    # print(label_num)
                    coors = np.where(lbl == label_num)
                    zeros[coors] = self.id2color[label_num]
            save_img = cv2.addWeighted(img, 1.0, zeros, 0.5, 0)

            #save_img = cv2.drawContours(save_img,contours,-1,(0,0,255),1)

        return save_img



if __name__ == "__main__":
    # img_dir = r'D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\test'
    # img_dir = r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\test'
    img_dir = [
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220408新增破洞+金顆粒 資料\1",
        r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220408新增破洞+金顆粒 資料\2",
    ]
    # pb_path = r"D:\code\model_saver\AE_Seg_20\pb_model.pb"
    pb_path = r"D:\code\model_saver\AE_Seg_21\infer_best_epoch288.pb"
    node_dict = {'input': 'input:0',
                 'recon':'output_AE:0',
                 'input_recon':'input_recon:0',
                 'prediction': 'predict_Seg:0',
                 }

    recon_seg_prediction(img_dir, pb_path, node_dict,
                         to_save_predict_image=False,
                         compare_with_label=False,
                         to_save_defect_undetected=True,
                         acc_threshold=0.3
                         )






