import os,math,cv2,json,imgviz,shutil,uuid,PIL,time,re
import numpy as np
import tensorflow
import matplotlib.pyplot as plt

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

class Seg_performance():
    def __init__(self,num_classes,print_out=False):
        self.num_classes = num_classes
        self.reset_arg()
        self.bins_num = num_classes + 1
        self.hist = np.histogram
        self.bins = np.arange(self.bins_num)
        self.print_out = print_out
        self.epsilon = 1e-8
        self.iou = 0
        self.acc = 0
        self.all_acc = 0
        self.to_save_img_undetected = False
        self.to_save_img_falseDetected = False
        self.save_dir = None
        self.save_dir4falseDetected = None

    def cal_intersection_union(self,predict, label):

        intersect = predict[predict == label]

        area_intersect, _ = self.hist(intersect, bins=self.bins)
        area_predict, _ = self.hist(predict, bins=self.bins)
        area_label, _ = self.hist(label, bins=self.bins)
        area_union = area_predict + area_label - area_intersect

        self.total_area_predict += area_predict
        self.total_area_label += area_label
        self.total_area_intersect += area_intersect
        self.total_area_union += area_union


        # if self.print_out:
        #     msg_list = []
        #     msg_list.append("area_intersection: {}".format(area_intersect))
        #     msg_list.append("area_predict: {}".format(area_predict))
        #     msg_list.append("area_label: {}".format(area_label))
        #     msg_list.append("union: {}".format(area_union))
        #     for msg in msg_list:
        #         say_sth(msg,print_out=True)

        #return area_predict, area_label, area_intersection, area_union

    def cal_iou_acc(self,save_dict=None,name=''):
        self.all_acc = self.total_area_intersect.sum() / (self.total_area_label.sum() + self.epsilon)
        self.iou = self.total_area_intersect / (self.total_area_union + self.epsilon)
        self.acc = self.total_area_intersect / (self.total_area_label + self.epsilon)

        #----save in the dict
        if save_dict is not None:
            for arg_name,value in zip(['iou','acc','all_acc'],[self.iou,self.acc,self.all_acc]):
                key = "seg_{}_{}_list".format(name,arg_name)
                if save_dict.get(key) is None:
                    save_dict[key] = []
                    save_dict[key].append(value.tolist())
                else:
                    save_dict[key].append(value.tolist())

        # if self.print_out:
        #     msg_list = []
        #     msg_list.append("all_acc: {}".format(all_acc))
        #     msg_list.append( "iou: {}".format(iou))
        #     msg_list.append("acc: {}".format(acc))
        #     for msg in msg_list:
        #         say_sth(msg,print_out=True)

        return self.iou,self.acc,self.all_acc

    def __cal_defect_by_acc(self,batch_predict,batch_label,paths=None,id2color=None):
        #----var
        count = 0
        h,w = batch_label.shape[1:]
        undetected_list = list()
        contour_cc_differ = list()


        for predict,label in zip(batch_predict,batch_label):

            #----connected components
            # cc_t = time.time()
            label_nums, label_map, stats, centroids = cv2.connectedComponentsWithStats(label, connectivity=8)

            #----contour test(contour???????????????????????????????????????)
            # contours, hierarchy = cv2.findContours(label,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            # contour_qty = 0
            # for i in range(len(contours)):
            #     qty_temp = cv2.contourArea(contours[i]) + len(contours[i])
            #     contour_qty += qty_temp
                # print("contour area:",qty_temp)

            # print("label_nums:{},contour number:{}".format(label_nums,len(contours)))
            intersect = (predict == label)
            zeros = np.zeros_like(label).astype('bool')
            cc_qty = 0
            for label_num in range(0, label_nums):
                s = stats[label_num]
                # if label_num != 0:
                coors = np.where(label_map == label_num)
                zeros[coors] = True
                logi_and = np.logical_and(zeros, intersect)
                sum_logi_and = np.sum(logi_and)
                acc_t = sum_logi_and / s[-1]
                if label_num > 0:
                    cc_qty += s[-1]
                    # print("CC area:",s[-1])
                defect_class = label[coors][0]
                self.label_defect_stat[defect_class][0] += 1
                if acc_t >= self.acc_threshold:
                    self.label_defect_stat[defect_class][1] += 1
                else:#save images
                    if self.to_save_img_undetected and self.save_dir is not None:
                        if paths is not None and id2color is not None:
                            img = cv2.imdecode(np.fromfile(paths[count],dtype=np.uint8),1)
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img,(w,h))
                            #----label range(rectangle)
                            color = id2color[defect_class]
                            # cv2.rectangle(img, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), color, 1)
                            img[coors] = color
                            #----predict range(dye)
                            if sum_logi_and > 0:
                                coors_predict = np.where(logi_and == True)
                                img[coors_predict] = 255 - np.array(color)
                            undetected_list.append([paths[count],int(sum_logi_and),int(s[-1])])
                            #print("path:{}\npredict / label pixels: {}/{}\n".format(paths[count],sum_logi_and,s[-1]))



                            #???????????????????????????????????????????????????????????????????????????if else???????????????
                            # cv2.putText(img, "defect number:{}".format(defect_class), (s[0] + s[2], s[1] + s[3]),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,#FONT_HERSHEY_SIMPLEX #FONT_HERSHEY_PLAIN
                            #             (255, 255, 255), 1)
                            splits = paths[count].split("\\")[-1].split(".")
                            save_path = os.path.join(self.save_dir,splits[0] + "_ratio_{}_{}.".format(sum_logi_and,s[-1]) + splits[-1])
                            # cv2.imwrite(save_path,img[:,:,::-1])
                            cv2.imencode('.{}'.format(splits[-1]), img[:, :, ::-1])[1].tofile(save_path)

                # ----reset
                zeros[coors] = False

            count += 1
            # contour_cc_differ.append(np.abs(cc_qty - contour_qty))

        #contour_cc_differ = np.array(contour_cc_differ)
        # print("len of contour_cc_differ:",len(contour_cc_differ))
        # print('average error rate:',np.average(contour_cc_differ))
        # print("std of average error rate:",np.std(contour_cc_differ))

        return undetected_list,contour_cc_differ

    def cal_label_defect_by_acc(self,batch_predict,batch_label,paths=None,id2color=None):
        #----var
        count = 0
        h,w = batch_label.shape[1:]
        undetected_list = list()

        for predict, label in zip(batch_predict,batch_label):
            # ----find out unique numbers
            p_defect_classes = np.unique(predict)
            l_defect_classes = np.unique(label)

            #----connected components
            # cc_t = time.time()
            label_nums, label_map, stats, centroids = cv2.connectedComponentsWithStats(label, connectivity=8)
            # print("label_nums:{},contour number:{}".format(label_nums,len(contours)))

            intersect = (predict == label)
            zeros = np.zeros_like(label).astype('bool')
            # cc_qty = 0
            for label_num in range(0, label_nums):
                s = stats[label_num]
                # if label_num != 0:
                coors = np.where(label_map == label_num)
                zeros[coors] = True
                logi_and = np.logical_and(zeros, intersect)
                sum_logi_and = np.sum(logi_and)
                acc_t = sum_logi_and / s[-1]
                # if label_num > 0:
                #     cc_qty += s[-1]
                    # print("CC area:",s[-1])
                defect_class = label[coors][0]
                self.label_defect_stat[defect_class][0] += 1
                if acc_t >= self.acc_threshold:
                    self.label_defect_stat[defect_class][1] += 1
                    # print("label_defect acc_t:",acc_t)
                    #----show image
                    # img_ans = np.where(label_map == label_num,255,0).astype(np.uint8)
                    # logi_and = logi_and.astype(np.uint8)
                    # logi_and *= 255
                    #
                    # plt.subplot(1,2,1)
                    # plt.imshow(img_ans, cmap='gray')
                    # plt.title('answer')
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(logi_and,cmap='gray')
                    # plt.title(acc_t)
                    #
                    # plt.show()
                else:#save images
                    if self.to_save_img_undetected and self.save_dir is not None:
                        if paths is not None and id2color is not None:
                            img = cv2.imdecode(np.fromfile(paths[count],dtype=np.uint8),1)
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img,(w,h))
                            #----label range(rectangle)
                            color = id2color[defect_class]
                            # cv2.rectangle(img, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), color, 1)
                            img[coors] = color
                            #----predict range(dye)
                            if sum_logi_and > 0:
                                coors_predict = np.where(logi_and == True)
                                img[coors_predict] = 255 - np.array(color)
                            undetected_list.append([paths[count],int(sum_logi_and),int(s[-1])])
                            #print("path:{}\npredict / label pixels: {}/{}\n".format(paths[count],sum_logi_and,s[-1]))


                            #???????????????????????????????????????????????????????????????????????????if else???????????????
                            # cv2.putText(img, "defect number:{}".format(defect_class), (s[0] + s[2], s[1] + s[3]),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,#FONT_HERSHEY_SIMPLEX #FONT_HERSHEY_PLAIN
                            #             (255, 255, 255), 1)
                            splits = paths[count].split("\\")[-1].split(".")
                            save_path = os.path.join(self.save_dir,splits[0] + "_ratio_{}_{}.".format(sum_logi_and,s[-1]) + splits[-1])
                            # cv2.imwrite(save_path,img[:,:,::-1])
                            cv2.imencode('.{}'.format(splits[-1]), img[:, :, ::-1])[1].tofile(save_path)

                # ----reset
                zeros[coors] = False

            count += 1

        return undetected_list

    def cal_label_defect_by_acc_v2(self,batch_predict,batch_label,paths=None,id2color=None):
        # ----var
        count = 0
        h, w = batch_label.shape[1:]
        undetected_list = list()

        for predict, label in zip(batch_predict, batch_label):
            # ----find out unique numbers
            # p_defect_classes = np.unique(predict)
            l_defect_classes = np.unique(label)

            intersect = (predict == label)

            for l_defect_class in l_defect_classes:
                #----
                if l_defect_class == 0:#the background  -->no need to cc
                    self.label_defect_stat[l_defect_class][0] += 1
                    zeros_p = np.where(predict == l_defect_class,True,False)
                    zeros_l = np.where(label == l_defect_class,True,False)
                    logi_and = np.logical_and(zeros_p, zeros_l)
                    sum_logi_and = np.sum(logi_and)
                    acc_t = sum_logi_and / np.sum(zeros_l)
                    if acc_t >= self.acc_threshold:
                        self.label_defect_stat[l_defect_class][1] += 1
                        # print("label_defect:{},acc_t:{}".format(l_defect_class,acc_t))
                else:
                    label_t = np.where(label == l_defect_class, 1, 0).astype(np.uint8)
                    l_cc_nums, l_cc_map, l_stats, l_centroids = cv2.connectedComponentsWithStats(label_t,
                                                                                                 connectivity=8)
                    self.label_defect_stat[l_defect_class][0] += (l_cc_nums - 1)
                    for l_cc_num in range(1, l_cc_nums):
                        s = l_stats[l_cc_num]
                        coors = np.where(l_cc_map == l_cc_num)
                        zeros_l = np.where(l_cc_map == l_cc_num, True, False)
                        logi_and = np.logical_and(zeros_l, intersect)
                        sum_logi_and = np.sum(logi_and)
                        acc_t = sum_logi_and / s[-1]
                        if acc_t >= self.acc_threshold:
                            self.label_defect_stat[l_defect_class][1] += 1
                            # print("label_defect:{},acc_t:{}".format(l_defect_class,acc_t))
                            # ----show image
                            # img_ans = np.where(label_map == label_num,255,0).astype(np.uint8)
                            # logi_and = logi_and.astype(np.uint8)
                            # logi_and *= 255
                            #
                            # plt.subplot(1,2,1)
                            # plt.imshow(img_ans, cmap='gray')
                            # plt.title('answer')
                            # plt.subplot(1, 2, 2)
                            # plt.imshow(logi_and,cmap='gray')
                            # plt.title(acc_t)
                            #
                            # plt.show()
                        else:  # save images
                            if self.to_save_img_undetected and self.save_dir is not None:
                                if paths is not None and id2color is not None:
                                    img = cv2.imdecode(np.fromfile(paths[count], dtype=np.uint8), 1)
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    img = cv2.resize(img, (w, h))
                                    # ----label range(rectangle)
                                    color = id2color[l_defect_class]
                                    # cv2.rectangle(img, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), color, 1)
                                    img[coors] = color
                                    # ----predict range(dye)
                                    if sum_logi_and > 0:
                                        coors_predict = np.where(logi_and == True)
                                        img[coors_predict] = 255 - np.array(color)
                                    undetected_list.append([paths[count], int(sum_logi_and), int(s[-1])])
                                    # print("path:{}\npredict / label pixels: {}/{}\n".format(paths[count],sum_logi_and,s[-1]))

                                    # ???????????????????????????????????????????????????????????????????????????if else???????????????
                                    # cv2.putText(img, "defect number:{}".format(defect_class), (s[0] + s[2], s[1] + s[3]),
                                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,#FONT_HERSHEY_SIMPLEX #FONT_HERSHEY_PLAIN
                                    #             (255, 255, 255), 1)
                                    splits = paths[count].split("\\")[-1].split(".")
                                    save_path = os.path.join(self.save_dir,
                                                             splits[0] + "_ratio_{}_{}.".format(sum_logi_and, s[-1]) +
                                                             splits[
                                                                 -1])
                                    # cv2.imwrite(save_path,img[:,:,::-1])
                                    cv2.imencode('.{}'.format(splits[-1]), img[:, :, ::-1])[1].tofile(save_path)

        return undetected_list

    def cal_label_defect_by_acc_vTest2(self,batch_predict,batch_label,paths=None,id2color=None):
        #----var
        count = 0
        h,w = batch_label.shape[1:]
        undetected_list = list()

        for predict, label in zip(batch_predict,batch_label):

            for defect_class in np.unique(label):
                if defect_class != 0:
                    label_t = np.where(label == defect_class,1,0).astype(np.uint8)

                    #----connected components
                    # cc_t = time.time()
                    label_nums, label_map, stats, centroids = cv2.connectedComponentsWithStats(label_t, connectivity=8)
                    # print("label_nums:{},contour number:{}".format(label_nums,len(contours)))

                    intersect = (predict == label)
                    zeros = np.zeros_like(label).astype('bool')
                    # cc_qty = 0
                    for label_num in range(1, label_nums):
                        s = stats[label_num]
                        # if label_num != 0:
                        coors = np.where(label_map == label_num)
                        zeros[coors] = True
                        logi_and = np.logical_and(zeros, intersect)
                        sum_logi_and = np.sum(logi_and)
                        acc_t = sum_logi_and / s[-1]
                        # if label_num > 0:
                        #     cc_qty += s[-1]
                            # print("CC area:",s[-1])
                        # defect_class = label[coors][0]
                        self.label_defect_stat[defect_class][0] += 1
                        if acc_t >= self.acc_threshold:
                            self.label_defect_stat[defect_class][1] += 1
                        else:#save images
                            if self.to_save_img_undetected and self.save_dir is not None:
                                if paths is not None and id2color is not None:
                                    img = cv2.imdecode(np.fromfile(paths[count],dtype=np.uint8),1)
                                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                                    img = cv2.resize(img,(w,h))
                                    #----label range(rectangle)
                                    color = id2color[defect_class]
                                    # cv2.rectangle(img, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), color, 1)
                                    img[coors] = color
                                    #----predict range(dye)
                                    if sum_logi_and > 0:
                                        coors_predict = np.where(logi_and == True)
                                        img[coors_predict] = 255 - np.array(color)
                                    undetected_list.append([paths[count],int(sum_logi_and),int(s[-1])])
                                    #print("path:{}\npredict / label pixels: {}/{}\n".format(paths[count],sum_logi_and,s[-1]))


                                    #???????????????????????????????????????????????????????????????????????????if else???????????????
                                    # cv2.putText(img, "defect number:{}".format(defect_class), (s[0] + s[2], s[1] + s[3]),
                                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,#FONT_HERSHEY_SIMPLEX #FONT_HERSHEY_PLAIN
                                    #             (255, 255, 255), 1)
                                    splits = paths[count].split("\\")[-1].split(".")
                                    save_path = os.path.join(self.save_dir,splits[0] + "_ratio_{}_{}.".format(sum_logi_and,s[-1]) + splits[-1])
                                    # cv2.imwrite(save_path,img[:,:,::-1])
                                    cv2.imencode('.{}'.format(splits[-1]), img[:, :, ::-1])[1].tofile(save_path)

                        # ----reset
                        zeros[coors] = False

                    count += 1

        return undetected_list

    def cal_predict_defect_by_acc(self, batch_predict, batch_label, paths=None, id2color=None):
        # ----var
        count = 0
        h, w = batch_label.shape[1:]
        falseDetected_list = list()

        for predict, label in zip(batch_predict, batch_label):

            #----show images
            # img_p = np.where(predict>0,255,0).astype(np.uint8)
            # img_l = np.where(label>0,255,0).astype(np.uint8)
            # plt.subplot(1,2,1)
            # plt.imshow(img_p,cmap='gray')
            # plt.title('prediction')
            # plt.subplot(1,2,2)
            # plt.imshow(img_l,cmap='gray')
            # plt.title('label')
            # plt.show()
            # ----connected components
            # cc_t = time.time()
            p_label_nums, p_label_map, p_stats, p_centroids = cv2.connectedComponentsWithStats(predict, connectivity=8)
            # print("label_nums:{},contour number:{}".format(label_nums,len(contours)))

            #intersect = (predict == label)
            # predict_map = np.zeros_like(predict).astype('bool')

            for p_label_num in range(0, p_label_nums):
                #s = stats[p_label_num]
                # if label_num != 0:
                p_coors = np.where(p_label_map == p_label_num)
                # predict_map = np.zeros_like(predict).astype('bool')
                # predict_map[p_coors] = True
                predict_map = np.where(p_label_map == p_label_num,True,False)
                # zeros[coors] = True
                # logi_and = np.logical_and(zeros, intersect)
                # sum_logi_and = np.sum(logi_and)
                # acc_t = sum_logi_and / s[-1]

                predict_defect_class = predict[p_coors][0]
                #----test
                # label_t2 = np.where(label == predict_defect_class, 1, 0).astype(bool)
                # logi_and = np.logical_and(label_t2, predict_map)
                # sum_logi_and = np.sum(logi_and)
                # acc_t = sum_logi_and / s[-1]
                #----
                # label_t = np.where(label == predict_defect_class,predict_defect_class,0).astype(np.uint8)
                label_t = np.where(label == predict_defect_class,1,0).astype(np.uint8)
                l_label_nums, l_label_map, l_stats, l_centroids = cv2.connectedComponentsWithStats(label_t, connectivity=8)

                # label_map = np.zeros_like(label).astype('bool')
                for l_label_num in range(1, l_label_nums):
                    s = l_stats[l_label_num]
                    # label_map = np.zeros_like(label).astype('bool')
                    # l_coors = np.where(l_label_map == l_label_num)
                    # label_map[l_coors] = True
                    label_map = np.where(l_label_map == l_label_num,True,False)
                    logi_and = np.logical_and(label_map, predict_map)
                    sum_logi_and = np.sum(logi_and)
                    acc_t = sum_logi_and / s[-1]

                    # ----reset
                    # label_map[l_coors] = False

                    if acc_t >= self.acc_threshold:
                        self.predict_defect_stat[predict_defect_class][1] += 1
                        print("predict_defect acc_t:", acc_t)
                        break



                self.predict_defect_stat[predict_defect_class][0] += 1
                # if acc_t >= self.acc_threshold:
                #     self.predict_defect_stat[predict_defect_class][1] += 1
                # else:  # save false detected images
                # if self.to_save_img_falseDetected and self.save_dir4falseDetected is not None:
                #     if paths is not None and id2color is not None:
                #         img = cv2.imdecode(np.fromfile(paths[count], dtype=np.uint8), 1)
                #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #         img = cv2.resize(img, (w, h))
                #         # ----label range(rectangle)
                #         color = id2color[predict_defect_class]
                #         # cv2.rectangle(img, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), color, 1)
                #         img[coors] = color
                #         # ----predict range(dye)
                #         if sum_logi_and > 0:
                #             coors_label = np.where(logi_and == True)
                #             img[coors_label] = 255 - np.array(color)
                #         falseDetected_list.append([paths[count], int(sum_logi_and), int(s[-1])])
                #         # print("path:{}\npredict / label pixels: {}/{}\n".format(paths[count],sum_logi_and,s[-1]))
                #
                #         # ???????????????????????????????????????????????????????????????????????????if else???????????????
                #         # cv2.putText(img, "defect number:{}".format(defect_class), (s[0] + s[2], s[1] + s[3]),
                #         #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,#FONT_HERSHEY_SIMPLEX #FONT_HERSHEY_PLAIN
                #         #             (255, 255, 255), 1)
                #         splits = paths[count].split("\\")[-1].split(".")
                #         save_path = os.path.join(self.save_dir4falseDetected,
                #                                  splits[0] + "_ratio_{}_{}.".format(sum_logi_and, s[-1]) + splits[
                #                                      -1])
                #         # cv2.imwrite(save_path,img[:,:,::-1])
                #         cv2.imencode('.{}'.format(splits[-1]), img[:, :, ::-1])[1].tofile(save_path)

                # ----reset
                # predict_map[p_coors] = False

            count += 1

        return falseDetected_list

    def cal_predict_defect_by_acc_v2(self, batch_predict, batch_label, paths=None, id2color=None):
        # ----var
        h, w = batch_label.shape[1:]
        falseDetected_list = list()

        for predict, label in zip(batch_predict, batch_label):

            #----find out unique numbers
            p_defect_classes = np.unique(predict)
            l_defect_classes = np.unique(label)

            for p_defect_class in p_defect_classes:
                if p_defect_class == 0:#the background  -->no need to cc
                    self.predict_defect_stat[p_defect_class][0] += 1
                    zeros_p = np.where(predict == p_defect_class,True,False)
                    zeros_l = np.where(label == p_defect_class,True,False)
                    logi_and = np.logical_and(zeros_p, zeros_l)
                    sum_logi_and = np.sum(logi_and)
                    acc_t = sum_logi_and / np.sum(zeros_l)
                    if acc_t >= self.acc_threshold:
                        self.predict_defect_stat[p_defect_class][1] += 1
                        # print("predict_defect:{},acc_t:{}".format(p_defect_class,acc_t))
                else:
                    predict_t = np.where(predict == p_defect_class, 1, 0).astype(np.uint8)
                    p_cc_nums, p_cc_map, p_stats, p_centroids = cv2.connectedComponentsWithStats(predict_t, connectivity=8)

                    self.predict_defect_stat[p_defect_class][0] += (p_cc_nums - 1)

                    if p_defect_class in l_defect_classes:

                        label_t = np.where(label == p_defect_class,1,0).astype(np.uint8)

                        l_cc_nums, l_cc_map, l_stats, l_centroids = cv2.connectedComponentsWithStats(label_t, connectivity=8)

                        for p_cc_num in range(1, p_cc_nums):
                            zeros_p = np.where(p_cc_map == p_cc_num,True,False)

                            for l_cc_num in range(1, l_cc_nums):
                                zeros_l = np.where(l_cc_map == l_cc_num,True,False)

                                logi_and = np.logical_and(zeros_p,zeros_l)
                                sum_logi_and = np.sum(logi_and)
                                acc_t = sum_logi_and / l_stats[l_cc_num][-1]
                                if acc_t >= self.acc_threshold:
                                    self.predict_defect_stat[p_defect_class][1] += 1
                                    # print("predict_defect:{},acc_t:{}".format(p_defect_class, acc_t))
                                    break

        return falseDetected_list

    def cal_defect_by_acc(self,batch_predict,batch_label,paths=None,id2color=None):
        #----var
        count = 0
        h,w = batch_label.shape[1:]
        undetected_list = list()

        for predict,label in zip(batch_predict,batch_label):

            #----connected components
            # cc_t = time.time()
            label_nums, label_map, stats, centroids = cv2.connectedComponentsWithStats(label, connectivity=8)
            # print("label_nums:{},contour number:{}".format(label_nums,len(contours)))

            intersect = (predict == label)
            zeros = np.zeros_like(label).astype('bool')
            # cc_qty = 0
            for label_num in range(0, label_nums):
                s = stats[label_num]
                # if label_num != 0:
                coors = np.where(label_map == label_num)
                zeros[coors] = True
                logi_and = np.logical_and(zeros, intersect)
                sum_logi_and = np.sum(logi_and)
                acc_t = sum_logi_and / s[-1]
                # if label_num > 0:
                #     cc_qty += s[-1]
                    # print("CC area:",s[-1])
                defect_class = label[coors][0]
                self.label_defect_stat[defect_class][0] += 1
                if acc_t >= self.acc_threshold:
                    self.label_defect_stat[defect_class][1] += 1
                else:#save images
                    if self.to_save_img_undetected and self.save_dir is not None:
                        if paths is not None and id2color is not None:
                            img = cv2.imdecode(np.fromfile(paths[count],dtype=np.uint8),1)
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img,(w,h))
                            #----label range(rectangle)
                            color = id2color[defect_class]
                            # cv2.rectangle(img, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), color, 1)
                            img[coors] = color
                            #----predict range(dye)
                            if sum_logi_and > 0:
                                coors_predict = np.where(logi_and == True)
                                img[coors_predict] = 255 - np.array(color)
                            undetected_list.append([paths[count],int(sum_logi_and),int(s[-1])])
                            #print("path:{}\npredict / label pixels: {}/{}\n".format(paths[count],sum_logi_and,s[-1]))


                            #???????????????????????????????????????????????????????????????????????????if else???????????????
                            # cv2.putText(img, "defect number:{}".format(defect_class), (s[0] + s[2], s[1] + s[3]),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,#FONT_HERSHEY_SIMPLEX #FONT_HERSHEY_PLAIN
                            #             (255, 255, 255), 1)
                            splits = paths[count].split("\\")[-1].split(".")
                            save_path = os.path.join(self.save_dir,splits[0] + "_ratio_{}_{}.".format(sum_logi_and,s[-1]) + splits[-1])
                            # cv2.imwrite(save_path,img[:,:,::-1])
                            cv2.imencode('.{}'.format(splits[-1]), img[:, :, ::-1])[1].tofile(save_path)

                # ----reset
                zeros[coors] = False

            count += 1

        return undetected_list

    def reset_arg(self,):
        a = np.zeros((self.num_classes,), dtype=float)
        self.total_area_predict = a.copy()
        self.total_area_label = a.copy()
        self.total_area_intersect = a.copy()
        self.total_area_union = a.copy()

    def reset_defect_stat(self,acc_threshold=0.3):
        # self.defect_stat = np.zeros([self.num_classes,2],dtype=float)
        self.label_defect_stat = np.zeros([self.num_classes,2],dtype=float)
        self.predict_defect_stat = np.zeros([self.num_classes,2],dtype=float)
        self.acc_threshold = acc_threshold

    def cal_defect_recall(self,save_dict=None,name=''):
        self.defect_recall = self.label_defect_stat.T[1] / (self.label_defect_stat.T[0] + 1e-8)

        # ----save in the dict
        if save_dict is not None:
            for arg_name, value in zip(["defect_stat","defect_recall"], [self.label_defect_stat,self.defect_recall]):
                key = "seg_{}_{}_list".format(name, arg_name)
                if save_dict.get(key) is None:
                    save_dict[key] = []
                    save_dict[key].append(value.tolist())
                else:
                    save_dict[key].append(value.tolist())



        return self.defect_recall

    def cal_defect_sensitivity(self,save_dict=None,name=''):
        self.defect_sensitivity = self.predict_defect_stat.T[1] / (self.predict_defect_stat.T[0] + 1e-8)

        # ----save in the dict
        if save_dict is not None:
            for arg_name, value in zip(["predict_defect_stat","defect_sensitivity"], [self.predict_defect_stat,self.defect_sensitivity]):
                key = "seg_{}_{}_list".format(name, arg_name)
                if save_dict.get(key) is None:
                    save_dict[key] = []
                    save_dict[key].append(value.tolist())
                else:
                    save_dict[key].append(value.tolist())



        return self.defect_sensitivity

    def sum_iou_acc(self):
        return np.sum(self.iou+self.acc)

    def sum_defect_recall(self):

        return np.sum(self.defect_recall)

class tools():
    def __init__(self,print_out=False):
        self.p_dict = dict()
        self.s_dict = dict()
        self.class_name2id = dict()
        self.id2color = dict()
        self.p_name_list = list()
        self.shape_keys = [
            "label",
            "points",
            "group_id",
            "shape_type",
            "flags",
        ]
        self.print_out = print_out

        #----
        img_dir = r"D:\dataset\nature\natural_images\flower"
        self.paths_flower = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] in img_format]

    def get_paths(self,img_source):
        # ----var
        paths = list()
        if not isinstance(img_source, list):
            img_source = [img_source]

        for img_dir in img_source:
            temp = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] in img_format]
            if len(temp) == 0:
                say_sth("Warning:?????????????????????????????????:{}".format(img_dir))
            else:
                paths.extend(temp)

        return np.array(paths),len(paths)

    def get_subdir_paths(self,img_source):
        # ----var
        dirs = list()
        paths = list()
        if not isinstance(img_source, list):
            img_source = [img_source]

        #----collect all subdirs
        for img_dir in img_source:
            dirs_temp = [obj.path for obj in os.scandir(img_dir) if obj.is_dir()]
            dirs.extend(dirs_temp)

        for dir_path in dirs:
            temp = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
            if len(temp) == 0:
                say_sth("Warning:?????????????????????????????????????????????:{}".format(dir_path),print_out=self.print_out)
            else:
                paths.extend(temp)

        return np.array(paths), len(paths)

    def create_label_png(self,img_dir,save_type=''):
        batch_size = 32

        paths, json_paths, qty = self.get_subdir_paths_withJsonCheck([img_dir])

        if qty == 0:
            say_sth("Error: no matched files:{}".format(img_dir))
        else:

            for i in range(qty):
                try:
                    img = cv2.imdecode(np.fromfile(paths[i], dtype=np.uint8), 1)
                except:
                    img = None

                if img is None:
                    msg = "read failed:".format(paths[i])
                    say_sth(msg)
                else:
                    label_shapes = self.get_label_shapes(json_paths[i])
                    if label_shapes is None:
                        continue
                    lbl = self.shapes_to_label(
                        img_shape=img.shape,
                        shapes=label_shapes,
                        label_name_to_value=self.class_name2id,
                    )

                    #----
                    zeros = np.zeros_like(img)
                    for label_num in np.unique(lbl):
                        if label_num != 0:
                            #print(label_num)
                            coors = np.where(lbl == label_num)
                            zeros[coors] = self.id2color[label_num]

                    #----save type
                    if save_type == 'ori+label':
                        save_img = cv2.addWeighted(img, 0.5, zeros, 0.5, 0)
                    else:
                        save_img = zeros


                    #----save image
                    ext = paths[i].split(".")[-1]
                    save_path = paths[i].strip(ext) + 'png'
                    cv2.imencode('.png', save_img)[1].tofile(save_path)

    def get_subdir_paths_withJsonCheck(self,img_source):
        # ----var
        dirs = list()
        paths = list()
        json_paths = list()
        if not isinstance(img_source, list):
            img_source = [img_source]

        # ----collect all subdirs
        # for img_dir in img_source:
        #     dirs_temp = [obj.path for obj in os.scandir(img_dir) if obj.is_dir()]
        #     dirs.extend(dirs_temp)

        for dir_path in img_source:
            names = [file.name for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
            if len(names) == 0:
                say_sth("Warning:?????????????????????????????????????????????:{}".format(dir_path),print_out=self.print_out)
            else:
                for name in names:
                    json_path = os.path.join(dir_path,name.split(".")[0] + '.json')
                    if os.path.exists(json_path):
                        paths.append(os.path.join(dir_path,name))
                        json_paths.append(json_path)
                    else:
                        say_sth("Warning:json????????????:{}".format(json_path),print_out=self.print_out)

        return np.array(paths),np.array(json_paths),len(paths)

    def get_relative_json_files(self,paths):
        json_paths = []
        msg_list = []

        for path in paths:
            ext = path.split(".")[-1]
            json_path = path.strip(ext) + 'json'
            if os.path.exists(json_path):
                json_paths.append(json_path)
            else:
                json_paths.append(None)
                msg_list.append("json file?????????:{}".format(json_path))

        if self.print_out:
            for msg in msg_list:
                say_sth(msg,print_out=True)


        return np.array(json_paths)

    def get_ite_data(self,paths,ite_num,batch_size=16,labels=None):
        num_start = batch_size * ite_num
        num_end = np.minimum(num_start + batch_size, len(paths))

        if labels is None:
            return paths[num_start:num_end]
        else:
            return paths[num_start:num_end], labels[num_start:num_end]

    def set_target(self,target_dict):
        t_compare = False
        name_list = ['loss']

        if target_dict is not None:
            t_type = target_dict.get('type')
            t_value = target_dict.get('value')
            t_times = target_dict.get('hit_target_times')

            if t_type in name_list:
                if t_value is not None:
                    if t_times is not None:
                        try:
                            t_compare = True
                            self.t_type = t_type
                            self.t_value = float(t_value)
                            self.t_times = int(t_times)
                            self.t_count = 0
                        except:
                            print("set_target failed!!")

        self.t_compare = t_compare

    def target_compare(self,data_dict):
        re = False
        if self.t_compare is True:
            if data_dict['loss_method'] == 'ssim':
                if data_dict[self.t_type] > self.t_value:
                    self.t_count += 1
            else:
                if data_dict[self.t_type] < self.t_value:
                    self.t_count += 1

            #----
            if self.t_count >= self.t_times:
                re = True

        return re

    def set_process(self,process_dict,ori_setting_dict,print_out=False):
        noise_img_dir = r"D:\dataset\perlin_noise"
        color_img_dir = r"D:\dataset\nature\natural_images\car"
        setting_dict = ori_setting_dict.copy()
        default_set_dict = {
            'ave_filter':(3,3),'gau_filter':(3,3),'rdm_shift':0.1,'rdm_br':np.array([0.88,1.12]),
            'rdm_flip':[1, 0, -1, 2],'rdm_blur':[1,1,1,3,3,3,5,5],'rdm_angle':[-5,5],
            'rdm_patch':[0.25,0.3,10],'rdm_perlin':[noise_img_dir,color_img_dir,30,1000,5]}
        '''
        rdm_patch:
            margin_ratio = 0.25
            patch_ratio = 0.3
            size_min = 10
        rdm_perlin:
            pixel range lower limit = 30
            pixel range upper limit = 1000
            defect number = 5
        '''
        p_name_list = list(default_set_dict.keys())
        if setting_dict is None:
            setting_dict = dict()

        for key,value in process_dict.items():
            if value is True:
                if setting_dict.get(key) is None:
                    setting_dict[key] = default_set_dict.get(key)#????????????process???True??????????????????????????????
                elif key == 'rdm_br':
                    br_range = setting_dict[key]
                    if br_range > 0 and br_range <= 1:
                        setting_dict[key] = np.array([1 - br_range, 1 + br_range])
                elif key == 'rdm_angle':
                    angle_range = setting_dict[key]
                    if angle_range > 0 and angle_range <= 90:
                        setting_dict[key] = [-angle_range, angle_range]
                #----perlin image process
                if key == 'rdm_perlin':
                    self.rdm_perlin = dict()

                    paths_noise = [file.path for file in os.scandir(setting_dict[key][0]) if file.name.split(".")[-1] == 'png']
                    qty_n = len(paths_noise)
                    msg = "Perlin noise image qty:{}".format(qty_n)
                    say_sth(msg,print_out=print_out)

                    paths_color = [file.path for file in os.scandir(setting_dict[key][1]) if
                                    file.name.split(".")[-1] in img_format]
                    qty_c = len(paths_color)
                    msg = "Color image qty:{}".format(qty_c)
                    say_sth(msg, print_out=print_out)

                    self.rdm_perlin['paths_noise'] = paths_noise
                    self.rdm_perlin['paths_color'] = paths_color
                    self.rdm_perlin['pixel_range'] = setting_dict[key][2:4]
                    self.rdm_perlin['defect_num'] = setting_dict[key][-1]


        self.p_dict = process_dict
        self.s_dict = setting_dict
        self.p_name_list = p_name_list

    def get_label_shapes(self,json_path):

        # keys = [
        #     "version",
        #     "imageData",
        #     "imagePath",
        #     "shapes",  # polygonal annotations
        #     "flags",  # image level flags
        #     "imageHeight",
        #     "imageWidth",
        # ]

        try:
            with open(json_path, 'r',encoding='utf-8') as f:
                data = json.load(f)
                shapes = [
                    dict(
                        label=s["label"],
                        points=s["points"],
                        shape_type=s.get("shape_type", "polygon"),
                        flags=s.get("flags", {}),
                        group_id=s.get("group_id"),
                        other_data={
                            k: v for k, v in s.items() if k not in self.shape_keys
                        },
                    )
                    for s in data["shapes"]
                ]
            return shapes
        except:
            print("Warning: read failed {}".format(json_path))
            return None

    def shapes_to_label(self,img_shape, shapes, label_name_to_value):
        dtype = np.uint8

        # if len(label_name_to_value) > 128:
        #     dtype = np.int16

        cls = np.zeros(img_shape[:2], dtype=dtype)
        # ins = np.zeros_like(cls)
        # instances = []
        for shape in shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            if group_id is None:
                group_id = uuid.uuid1()
            shape_type = shape.get("shape_type", None)

            # cls_name = label
            # instance = (cls_name, group_id)
            #
            # if instance not in instances:
            #     instances.append(instance)
            # ins_id = instances.index(instance) + 1
            cls_id = label_name_to_value[label]


            mask = self.shape_to_mask(img_shape[:2], points, shape_type)
            cls[mask] = cls_id
            # ins[mask] = ins_id

        return cls

    def shape_to_mask(self,
            img_shape, points, shape_type=None, line_width=10, point_size=5
    ):
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        draw = PIL.ImageDraw.Draw(mask)
        xy = [tuple(point) for point in points]
        if shape_type == "circle":
            assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
            (cx, cy), (px, py) = xy
            d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
        elif shape_type == "rectangle":
            assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
            draw.rectangle(xy, outline=1, fill=1)
        elif shape_type == "line":
            assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == "linestrip":
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == "point":
            assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
            cx, cy = xy[0]
            r = point_size
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
        else:
            assert len(xy) > 2, "Polygon must have points more than 2"
            draw.polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def resize_label(self,lbl, resize,to_process=False,rdm_angle=False,M=None):
        #resize format: [w,h]
        coor_dict = dict()
        h,w = lbl.shape
        for label_num in np.unique(lbl):
            if label_num != 0:
                # ----???????????????label number?????????
                coor_dict[label_num] = np.where(lbl == label_num)

        for label_num in coor_dict.keys():
            # ----??????zeros??????(shape??????resize???label map)
            z_temp = np.zeros_like(lbl)
            # ----?????????label number?????????????????????1
            z_temp[coor_dict[label_num]] = 1
            #----??????
            if to_process is True and rdm_angle is True:
                # M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                z_temp = cv2.warpAffine(z_temp, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # ----???z_temp??????resize(??????????????????1???resize???????????????????????????)
            z_temp = cv2.resize(z_temp, resize)
            # ----??????resize????????????label number????????????
            coor_dict[label_num] = np.where(z_temp == 1)

        z_temp = np.zeros([resize[1], resize[0]], dtype=np.uint8)
        # print("z_temp shape:",z_temp.shape)
        for label_num in coor_dict.keys():
            z_temp[coor_dict[label_num]] = label_num
        return z_temp

    def batch_resize_label(self,lbl_array, resize):
        shape = [lbl_array.shape[0],resize[1],resize[0]]
        re_array = np.zeros(shape,dtype=lbl_array.dtype)

        for i, lbl in enumerate(lbl_array):
            lbl = self.resize_label(lbl,resize)
            re_array[i] = lbl

        return re_array

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
            #----
            zeros = np.zeros_like(img)
            for label_num in np.unique(lbl):
                if label_num != 0:
                    # print(label_num)
                    coors = np.where(lbl == label_num)
                    zeros[coors] = self.id2color[label_num]
            save_img = cv2.addWeighted(img, 0.5, zeros, 0.5, 0)

        return save_img

    def get_4D_img_label_data(self,paths,output_shape,json_paths=None,to_norm=True,to_rgb=True,to_process=False,
                              dtype='float32',to_save_label=False):
        len_path = len(paths)
        to_gray = False
        M = None

        #----create default np array
        batch_dim = [len_path]
        batch_dim.extend(output_shape)
        if batch_dim[-1] == 1:
            to_gray = True

        batch_data = np.zeros(batch_dim, dtype=dtype)

        #----label data: gray images
        batch_data_label = np.zeros(batch_dim[:-1], dtype=np.uint8)

        #----setting dict process
        if to_process is True:
            if self.p_dict.get('rdm_shift'):
                corner = np.random.randint(4, size=len_path)
            if self.p_dict.get('rdm_br'):
                #----??????1(??????)
                # set_range = self.s_dict['rdm_br'] * 100
                # print("set_range:",set_range)
                # br_factors = np.random.randint(set_range[0],set_range[1],size=len_path)
                # br_factors = br_factors.astype(np.float16)
                # br_factors /= 100
                #----??????2(??????)
                br_factors = np.random.random(size=len_path)
                br_factors *= (self.s_dict['rdm_br'][1] - self.s_dict['rdm_br'][0])
                br_factors += self.s_dict['rdm_br'][0]
                # br_factors = br_factors.astype(np.float16)#???float32?????????float16?????????????????????
                # print("br_factors:",br_factors)


            # if self.p_dict.get('rdm_patch'):
            #     batch_data_no_patch = np.zeros_like(batch_data)
            # if self.p_dict.get('rdm_patch'):
            #     margin_ratio = self.s_dict['rdm_patch'][0]
            #     patch_ratio = self.s_dict['rdm_patch'][1]
            #     size_min = self.s_dict['rdm_patch'][2]

        #----read images and do processing
        for idx, path in enumerate(paths):
            try:
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
            except:
                img = None

            if img is None:
                msg = "read failed:".format(path)
                say_sth(msg)
            else:
                # d_t = time.time()
                #----get the NPY path
                ext = path.split(".")[-1]
                npy_path = path.strip(ext) + 'npy'

                if os.path.exists(npy_path):
                    lbl = np.load(npy_path)
                else:
                    #----decide the json path
                    if json_paths is None:
                        json_path = path.strip(ext) + 'json'
                    else:
                        json_path = json_paths[idx]
                        if json_path is None or json_path == '':
                            json_path = path.strip(ext) + 'json'

                    #----read the json file
                    if not os.path.exists(json_path):
                        lbl = np.zeros(img.shape[:-1], dtype=np.uint8)
                    else:
                        label_shapes = self.get_label_shapes(json_path)
                        if label_shapes is None:
                            continue
                        try:
                            lbl = self.shapes_to_label(
                                img_shape=img.shape,
                                shapes=label_shapes,
                                label_name_to_value=self.class_name2id,
                            )
                        except:
                            print("label Error:",json_path)


                    if to_save_label is True:
                        np.save(npy_path, lbl)
                # d_t = time.time() - d_t
                # print("read label time:",d_t)
                # print(np.unique(lbl))


                if to_process is True:

                    ori_h, ori_w, _ = img.shape
                    if self.p_dict.get('ave_filter'):
                        img = cv2.blur(img, self.s_dict['ave_filter'])
                    if self.p_dict.get('gau_filter'):
                        img = cv2.GaussianBlur(img, self.s_dict['gau_filter'], 0, 0)
                    if self.p_dict.get('rdm_shift'):
                        c = corner[idx]
                        y = np.random.randint(ori_h * self.s_dict['rdm_shift'])
                        x = np.random.randint(ori_w * self.s_dict['rdm_shift'])
                        p = int(round((y + x) / 2))
                        # print("x={},y={},p={}".format(x,y,p))
                        if c == 0:
                            img = img[p:, p:, :]
                            lbl = lbl[p:, p:]
                        elif c == 1:
                            img = img[p:, :-(p + 1), :]
                            lbl = lbl[p:, :-(p + 1)]
                        elif c == 2:
                            img = img[:-(p + 1), p:, :]
                            lbl = lbl[:-(p + 1), p:]
                        elif c == 3:
                            img = img[:-(p + 1), :-(p + 1), :]
                            lbl = lbl[:-(p + 1), :-(p + 1)]
                        else:
                            img = img[p:, p:, :]
                            lbl = lbl[p:, p:]
                    if self.p_dict.get('rdm_br'):
                        # mean_br = np.mean(img)
                        try:
                            # br_factor = np.random.randint(math.floor(mean_br * self.s_dict['rdm_br'][0]),
                            #                               math.ceil(mean_br * self.s_dict['rdm_br'][1]))
                            # print("br_factor:",br_factors[idx])
                            img = np.clip(img * br_factors[idx] ,0, 255)
                            img = img.astype(np.uint8)
                        except:
                            msg = "Error:rdm_br value"
                            say_sth(msg)
                    if self.p_dict.get('rdm_flip'):
                        flip_type = np.random.choice(self.s_dict['rdm_flip'])
                        if flip_type != 2:
                            img = cv2.flip(img, flip_type)
                            lbl = cv2.flip(lbl, flip_type)
                    if self.p_dict.get('rdm_blur'):
                        kernel = tuple(np.random.choice(self.s_dict['rdm_blur'], size=2))
                        # print("kernel:", kernel)
                        if np.random.randint(0, 2) == 0:
                            img = cv2.blur(img, kernel)
                        else:
                            img = cv2.GaussianBlur(img, kernel, 0, 0)
                    # if self.p_dict.get('rdm_noise'):
                    #     uniform_noise = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
                    #     cv2.randu(uniform_noise, 0, 255)
                    #     ret, impulse_noise = cv2.threshold(uniform_noise, 230, 255, cv2.THRESH_BINARY_INV)
                    #     img = cv2.bitwise_and(img, img, mask=impulse_noise)
                    if self.p_dict.get('rdm_angle'):
                        angle = np.random.randint(self.s_dict['rdm_angle'][0], self.s_dict['rdm_angle'][1])
                        h, w = img.shape[:2]#??????????????????shape?????????????????????shift??????????????????
                        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                        lbl = cv2.warpAffine(lbl, M, (w, h), borderMode=cv2.BORDER_REPLICATE,flags=cv2.INTER_NEAREST)
                    # if self.p_dict.get('rdm_patch'):
                    #     #----put no patch image
                    #     # img_no_patch = img.copy()
                    #     img_no_patch = cv2.resize(img.copy(), (output_shape[1], output_shape[0]))
                    #     img_no_patch = self.img_transform(img_no_patch, to_rgb=to_rgb, to_gray=to_gray)
                    #     batch_data_no_patch[idx] = img_no_patch
                    #     # ----
                    #     maxi_len = int(sum(img.shape[:2]) / 2 * patch_ratio)
                    #     patch_times = np.random.randint(1, 4)
                    #     patch_types = np.random.randint(0, 4, patch_times)
                    #     margin_x = margin_ratio * img.shape[1]
                    #     margin_y = margin_ratio * img.shape[0]
                    #     for patch_type in patch_types:
                    #         center_x = np.random.randint(margin_x, img.shape[1] - margin_x)
                    #         center_y = np.random.randint(margin_y, img.shape[0] - margin_y)
                    #         if np.random.randint(2) == 1:
                    #             color = (0, 0, 0)
                    #         else:
                    #             color = np.random.randint(0, 255, 3).tolist()
                    #         if patch_type == 0:
                    #             end_x = np.random.randint(size_min, maxi_len / 2)
                    #             # end_y = np.random.randint(center_y,img.shape[0])
                    #             cv2.rectangle(img, (center_x, center_y), (center_x + end_x, center_y + end_x), color,
                    #                           -1)
                    #             # print("square,center_x:{},center_y:{},len:{}".format(center_x, center_y, end_x))
                    #         elif patch_type == 1:
                    #             radius = np.random.randint(size_min, maxi_len / 2)
                    #             cv2.circle(img, (center_x, center_y), radius, color, -1)
                    #             # print("circle,center_x:{},center_y:{},radius:{}".format(center_x, center_y, radius))
                    #         elif patch_type == 2:  # rectangle_1
                    #             lens = np.random.randint(size_min, maxi_len / 2, 2)
                    #             cv2.rectangle(img, (center_x, center_y), (center_x + lens[0], center_y + lens[1]),
                    #                           color, -1)
                    #             # print("rectangle,center_x:{},center_y:{},len:{}".format(center_x, center_y, lens))
                    #         elif patch_type == 3:  # rectangle_2
                    #             ave_size = sum(img.shape[:2]) / 2
                    #             long_len = np.random.randint(ave_size // 2, ave_size * 0.9)
                    #             short_len = np.random.randint(3, ave_size // 4)
                    #             if center_x >= ave_size // 2:
                    #                 if center_y >= ave_size // 2:
                    #                     pass
                    #                 else:
                    #                     cv2.rectangle(img, (center_x, center_y),
                    #                                   (center_x + short_len, center_y + long_len),
                    #                                   color, -1)
                    #             else:
                    #                 if center_y >= ave_size // 2:
                    #                     cv2.rectangle(img, (center_x, center_y),
                    #                                   (center_x + long_len, center_y + short_len),
                    #                                   color, -1)
                    #                 else:
                    #                     lens = [long_len, short_len]
                    #                     np.random.shuffle(lens)
                    #                     cv2.rectangle(img, (center_x, center_y),
                    #                                   (center_x + lens[0], center_y + lens[1]),
                    #                                   color, -1)

                #----resize and change the color format
                img = cv2.resize(img, (output_shape[1], output_shape[0]))
                lbl = cv2.resize(lbl, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_NEAREST)
                #----show the lbl image(before and after resize)
                # img_lbl = np.where(lbl > 0,255,0).astype(np.uint8)
                # lbl = self.resize_label(lbl,(output_shape[1], output_shape[0]), to_process=to_process,
                #                         rdm_angle=self.p_dict.get('rdm_angle'),M=M)



                # img_lbl_r = np.where(lbl > 0,255,0).astype(np.uint8)
                # plt.subplot(1,2,1)
                # plt.imshow(img_lbl, cmap='gray')
                # plt.title('label before resize')
                # plt.subplot(1, 2, 2)
                # plt.imshow(img_lbl_r, cmap='gray')
                # plt.title('label after resize')
                # plt.show()

                if to_gray is True:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(img,axis=-1)
                elif to_rgb is True:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_data[idx] = img
                batch_data_label[idx] = lbl

        if to_norm is True:
            batch_data /= 255

        #----return value process
        return batch_data, batch_data_label
        # if to_process is True:
        #     if self.p_dict.get('rdm_patch'):
        #         if to_norm is True:
        #             batch_data_no_patch /= 255
        #         return batch_data_no_patch, batch_data
        #     else:
        #         return batch_data
        # else:
        #     return batch_data

    def get_4D_data(self,paths, output_shape,to_norm=True,to_rgb=True,to_process=False,dtype='float32'):
        len_path = len(paths)
        to_gray = False
        key_list = ['rdm_shift','rdm_patch']

        #----create default np array
        batch_dim = [len_path]
        batch_dim.extend(output_shape)
        if batch_dim[-1] == 1:
            to_gray = True

        batch_data = np.zeros(batch_dim, dtype=dtype)

        #----setting dict process
        if to_process is True:
            for key in key_list:
                if self.p_dict.get(key):
                    if key == 'rdm_shift':
                        corner = np.random.randint(4, size=len_path)
                    if key == 'rdm_patch':
                        batch_data_no_patch = np.zeros_like(batch_data)
                        margin_ratio = self.s_dict[key][0]
                        patch_ratio = self.s_dict[key][1]
                        size_min = self.s_dict[key][2]


        #----read images and do processing
        for idx, path in enumerate(paths):
            try:
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
            except:
                img = None

            if img is None:
                msg = "read failed:".format(path)
                say_sth(msg)
            else:
                if to_process is True:
                    ori_h, ori_w, _ = img.shape
                    if self.p_dict.get('ave_filter'):
                        img = cv2.blur(img, self.s_dict['ave_filter'])
                    if self.p_dict.get('gau_filter'):
                        img = cv2.GaussianBlur(img, self.s_dict['gau_filter'], 0, 0)
                    if self.p_dict.get('rdm_shift'):
                        c = corner[idx]
                        y = np.random.randint(ori_h * self.s_dict['rdm_shift'])
                        x = np.random.randint(ori_w * self.s_dict['rdm_shift'])
                        p = int(round((y + x) / 2))
                        # print("x={},y={},p={}".format(x,y,p))
                        if c == 0:
                            img = img[p:, p:, :]
                        elif c == 1:
                            img = img[p:, :-(p + 1), :]
                        elif c == 2:
                            img = img[:-(p + 1), p:, :]
                        elif c == 3:
                            img = img[:-(p + 1), :-(p + 1), :]
                        else:
                            img = img[p:, p:, :]
                    if self.p_dict.get('rdm_br'):
                        mean_br = np.mean(img)
                        try:
                            br_factor = np.random.randint(math.floor(mean_br * self.s_dict['rdm_br'][0]),
                                                          math.ceil(mean_br * self.s_dict['rdm_br'][1]))
                            img = np.clip(img / mean_br * br_factor, 0, 255)
                            img = img.astype(np.uint8)
                        except:
                            msg = "Error:rdm_br value"
                            say_sth(msg)
                    if self.p_dict.get('rdm_flip'):
                        flip_type = np.random.choice(self.s_dict['rdm_flip'])
                        if flip_type != 2:
                            img = cv2.flip(img, flip_type)
                    if self.p_dict.get('rdm_blur'):
                        kernel = tuple(np.random.choice(self.s_dict['rdm_blur'], size=2))
                        # print("kernel:", kernel)
                        if np.random.randint(0, 2) == 0:
                            img = cv2.blur(img, kernel)
                        else:
                            img = cv2.GaussianBlur(img, kernel, 0, 0)
                    if self.p_dict.get('rdm_noise'):
                        uniform_noise = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
                        cv2.randu(uniform_noise, 0, 255)
                        ret, impulse_noise = cv2.threshold(uniform_noise, 230, 255, cv2.THRESH_BINARY_INV)
                        img = cv2.bitwise_and(img, img, mask=impulse_noise)
                    if self.p_dict.get('rdm_angle'):
                        angle = np.random.randint(self.s_dict['rdm_angle'][0], self.s_dict['rdm_angle'][1])
                        h, w = img.shape[:2]
                        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                    if self.p_dict.get('rdm_patch'):
                        #----put no patch image
                        # img_no_patch = img.copy()
                        img_no_patch = cv2.resize(img.copy(), (output_shape[1], output_shape[0]))
                        img_no_patch = self.img_transform(img_no_patch, to_rgb=to_rgb, to_gray=to_gray)
                        batch_data_no_patch[idx] = img_no_patch
                        # ----
                        maxi_len = int(sum(img.shape[:2]) / 2 * patch_ratio)
                        patch_times = np.random.randint(1, 4)
                        patch_types = np.random.randint(0, 4, patch_times)
                        margin_x = margin_ratio * img.shape[1]
                        margin_y = margin_ratio * img.shape[0]
                        mean = np.mean(img)
                        for patch_type in patch_types:
                            center_x = np.random.randint(margin_x, img.shape[1] - margin_x)
                            center_y = np.random.randint(margin_y, img.shape[0] - margin_y)
                            #----choose colors
                            if mean > 127:
                                color = np.random.randint(224, 255, 1).tolist() * 3
                            else:
                                color = np.random.randint(0, 32, 1).tolist() * 3

                            # if np.random.randint(2) == 1:
                            #     color = (0, 0, 0)
                            # else:
                            #     color = np.random.randint(0, 255, 3).tolist()
                            if patch_type == 0:
                                end_x = np.random.randint(size_min, maxi_len / 2)
                                # end_y = np.random.randint(center_y,img.shape[0])
                                cv2.rectangle(img, (center_x, center_y), (center_x + end_x, center_y + end_x), color,
                                              -1)
                                # print("square,center_x:{},center_y:{},len:{}".format(center_x, center_y, end_x))
                            elif patch_type == 1:
                                radius = np.random.randint(size_min, maxi_len / 2)
                                cv2.circle(img, (center_x, center_y), radius, color, -1)
                                # print("circle,center_x:{},center_y:{},radius:{}".format(center_x, center_y, radius))
                            elif patch_type == 2:  # rectangle_1
                                lens = np.random.randint(size_min, maxi_len / 2, 2)
                                cv2.rectangle(img, (center_x, center_y), (center_x + lens[0], center_y + lens[1]),
                                              color, -1)
                                # print("rectangle,center_x:{},center_y:{},len:{}".format(center_x, center_y, lens))
                            elif patch_type == 3:  # rectangle_2
                                ave_size = sum(img.shape[:2]) / 2
                                long_len = np.random.randint(ave_size // 2, ave_size * 0.9)
                                short_len = np.random.randint(3, ave_size // 4)
                                if center_x >= ave_size // 2:
                                    if center_y >= ave_size // 2:
                                        pass
                                    else:
                                        cv2.rectangle(img, (center_x, center_y),
                                                      (center_x + short_len, center_y + long_len),
                                                      color, -1)
                                else:
                                    if center_y >= ave_size // 2:
                                        cv2.rectangle(img, (center_x, center_y),
                                                      (center_x + long_len, center_y + short_len),
                                                      color, -1)
                                    else:
                                        lens = [long_len, short_len]
                                        np.random.shuffle(lens)
                                        cv2.rectangle(img, (center_x, center_y),
                                                      (center_x + lens[0], center_y + lens[1]),
                                                      color, -1)
                    if self.p_dict.get('rdm_perlin'):
                        #img = cv2.resize(img, (output_shape[1], output_shape[0]))
                        # mask,img_back = self.get_perlin_noise(output_shape[:-1],res=(16,16))
                        img = self.add_noise_by_perlin(img,self.rdm_perlin)
                        # img = self.add_noise_by_perlin_w_mask(img,self.rdm_perlin)


                #----resize and change the color format
                img = cv2.resize(img, (output_shape[1], output_shape[0]))

                if to_gray is True:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(img,axis=-1)
                elif to_rgb is True:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_data[idx] = img

        if to_norm is True:
            batch_data /= 255

        #----return value process
        if to_process is True:
            if self.p_dict.get('rdm_patch'):
                if to_norm is True:
                    batch_data_no_patch /= 255
                return batch_data_no_patch, batch_data
            else:
                return batch_data
        else:
            return batch_data

    def get_4D_data_create_mask(self,paths, output_shape,to_norm=True,to_rgb=True,to_process=False,dtype='float32'):
        len_path = len(paths)
        to_gray = False
        key_list = ['rdm_shift','rdm_patch']

        #----create default np array
        batch_dim = [len_path]
        batch_dim.extend(output_shape)
        if batch_dim[-1] == 1:
            to_gray = True

        batch_data = np.zeros(batch_dim, dtype=dtype)
        batch_label = np.zeros(batch_dim[:-1], dtype=dtype)

        #----setting dict process
        if to_process is True:
            for key in key_list:
                if self.p_dict.get(key):
                    if key == 'rdm_shift':
                        corner = np.random.randint(4, size=len_path)
                    if key == 'rdm_patch':
                        batch_data_no_patch = np.zeros_like(batch_data)
                        margin_ratio = self.s_dict[key][0]
                        patch_ratio = self.s_dict[key][1]
                        size_min = self.s_dict[key][2]


        #----read images and do processing
        for idx, path in enumerate(paths):
            try:
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
            except:
                img = None

            if img is None:
                msg = "read failed:".format(path)
                say_sth(msg)
            else:
                label = None
                if to_process is True:
                    ori_h, ori_w, _ = img.shape
                    if self.p_dict.get('ave_filter'):
                        img = cv2.blur(img, self.s_dict['ave_filter'])
                    if self.p_dict.get('gau_filter'):
                        img = cv2.GaussianBlur(img, self.s_dict['gau_filter'], 0, 0)
                    if self.p_dict.get('rdm_shift'):
                        c = corner[idx]
                        y = np.random.randint(ori_h * self.s_dict['rdm_shift'])
                        x = np.random.randint(ori_w * self.s_dict['rdm_shift'])
                        p = int(round((y + x) / 2))
                        # print("x={},y={},p={}".format(x,y,p))
                        if c == 0:
                            img = img[p:, p:, :]
                        elif c == 1:
                            img = img[p:, :-(p + 1), :]
                        elif c == 2:
                            img = img[:-(p + 1), p:, :]
                        elif c == 3:
                            img = img[:-(p + 1), :-(p + 1), :]
                        else:
                            img = img[p:, p:, :]
                    if self.p_dict.get('rdm_br'):
                        mean_br = np.mean(img)
                        try:
                            br_factor = np.random.randint(math.floor(mean_br * self.s_dict['rdm_br'][0]),
                                                          math.ceil(mean_br * self.s_dict['rdm_br'][1]))
                            img = np.clip(img / mean_br * br_factor, 0, 255)
                            img = img.astype(np.uint8)
                        except:
                            msg = "Error:rdm_br value"
                            say_sth(msg)
                    if self.p_dict.get('rdm_flip'):
                        flip_type = np.random.choice(self.s_dict['rdm_flip'])
                        if flip_type != 2:
                            img = cv2.flip(img, flip_type)
                    if self.p_dict.get('rdm_blur'):
                        kernel = tuple(np.random.choice(self.s_dict['rdm_blur'], size=2))
                        # print("kernel:", kernel)
                        if np.random.randint(0, 2) == 0:
                            img = cv2.blur(img, kernel)
                        else:
                            img = cv2.GaussianBlur(img, kernel, 0, 0)
                    if self.p_dict.get('rdm_noise'):
                        uniform_noise = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
                        cv2.randu(uniform_noise, 0, 255)
                        ret, impulse_noise = cv2.threshold(uniform_noise, 230, 255, cv2.THRESH_BINARY_INV)
                        img = cv2.bitwise_and(img, img, mask=impulse_noise)
                    if self.p_dict.get('rdm_angle'):
                        angle = np.random.randint(self.s_dict['rdm_angle'][0], self.s_dict['rdm_angle'][1])
                        h, w = img.shape[:2]
                        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                    if self.p_dict.get('rdm_patch'):
                        #----put no patch image
                        # img_no_patch = img.copy()
                        img_no_patch = cv2.resize(img.copy(), (output_shape[1], output_shape[0]))
                        img_no_patch = self.img_transform(img_no_patch, to_rgb=to_rgb, to_gray=to_gray)
                        batch_data_no_patch[idx] = img_no_patch
                        # ----
                        maxi_len = int(sum(img.shape[:2]) / 2 * patch_ratio)
                        patch_times = np.random.randint(1, 4)
                        patch_types = np.random.randint(0, 4, patch_times)
                        margin_x = margin_ratio * img.shape[1]
                        margin_y = margin_ratio * img.shape[0]
                        mean = np.mean(img)
                        for patch_type in patch_types:
                            center_x = np.random.randint(margin_x, img.shape[1] - margin_x)
                            center_y = np.random.randint(margin_y, img.shape[0] - margin_y)
                            #----choose colors
                            if mean > 127:
                                color = np.random.randint(224, 255, 1).tolist() * 3
                            else:
                                color = np.random.randint(0, 32, 1).tolist() * 3

                            # if np.random.randint(2) == 1:
                            #     color = (0, 0, 0)
                            # else:
                            #     color = np.random.randint(0, 255, 3).tolist()
                            if patch_type == 0:
                                end_x = np.random.randint(size_min, maxi_len / 2)
                                # end_y = np.random.randint(center_y,img.shape[0])
                                cv2.rectangle(img, (center_x, center_y), (center_x + end_x, center_y + end_x), color,
                                              -1)
                                # print("square,center_x:{},center_y:{},len:{}".format(center_x, center_y, end_x))
                            elif patch_type == 1:
                                radius = np.random.randint(size_min, maxi_len / 2)
                                cv2.circle(img, (center_x, center_y), radius, color, -1)
                                # print("circle,center_x:{},center_y:{},radius:{}".format(center_x, center_y, radius))
                            elif patch_type == 2:  # rectangle_1
                                lens = np.random.randint(size_min, maxi_len / 2, 2)
                                cv2.rectangle(img, (center_x, center_y), (center_x + lens[0], center_y + lens[1]),
                                              color, -1)
                                # print("rectangle,center_x:{},center_y:{},len:{}".format(center_x, center_y, lens))
                            elif patch_type == 3:  # rectangle_2
                                ave_size = sum(img.shape[:2]) / 2
                                long_len = np.random.randint(ave_size // 2, ave_size * 0.9)
                                short_len = np.random.randint(3, ave_size // 4)
                                if center_x >= ave_size // 2:
                                    if center_y >= ave_size // 2:
                                        pass
                                    else:
                                        cv2.rectangle(img, (center_x, center_y),
                                                      (center_x + short_len, center_y + long_len),
                                                      color, -1)
                                else:
                                    if center_y >= ave_size // 2:
                                        cv2.rectangle(img, (center_x, center_y),
                                                      (center_x + long_len, center_y + short_len),
                                                      color, -1)
                                    else:
                                        lens = [long_len, short_len]
                                        np.random.shuffle(lens)
                                        cv2.rectangle(img, (center_x, center_y),
                                                      (center_x + lens[0], center_y + lens[1]),
                                                      color, -1)
                    if self.p_dict.get('rdm_perlin'):
                        #img = cv2.resize(img, (output_shape[1], output_shape[0]))
                        # mask,img_back = self.get_perlin_noise(output_shape[:-1],res=(16,16))
                        # img = self.add_noise_by_perlin(img,self.rdm_perlin)
                        img,label = self.add_noise_by_perlin_w_mask(img,self.rdm_perlin)


                #----image resize and change the color format
                img = cv2.resize(img, (output_shape[1], output_shape[0]))

                if to_gray is True:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(img,axis=-1)
                elif to_rgb is True:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_data[idx] = img

                #----label resize
                if label is not None:
                    label = cv2.resize(label, (output_shape[1], output_shape[0]))
                    batch_label[idx] = label
        #----norm
        batch_label /= 255
        batch_label = batch_label.astype(np.uint8)
        if to_norm is True:
            batch_data /= 255

        #----return value process
        return batch_data, batch_label

    def img_transform(self,img,to_rgb=True,to_gray=False):
        if to_gray is True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)
        elif to_rgb is True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def add_noise_by_perlin(self,img,set_dict):
        #img format: BGR
        #size format = [H,W]
        size = img.shape[:2]
        pixel_range = set_dict['pixel_range']
        defect_num = set_dict['defect_num']
        img_nature = np.fromfile(np.random.choice(set_dict['paths_color']), dtype=np.uint8)
        img_nature = cv2.imdecode(img_nature, 1)#BGR format
        img_nature = cv2.resize(img_nature, size[::-1])

        img_perlin = np.fromfile(np.random.choice(set_dict['paths_noise']), dtype=np.uint8)
        img_perlin = cv2.imdecode(img_perlin, 0)
        img_perlin = cv2.resize(img_perlin, size[::-1])

        label_num, label_map, stats, centroids = cv2.connectedComponentsWithStats(img_perlin, connectivity=8)
        pixel_num = stats.T[-1]
        b = np.where(pixel_num >= pixel_range[0], pixel_num, pixel_range[1] + 3)
        index_list = np.where(b < pixel_range[1])  # index_list????????????pixel_range???index
        zeros = np.zeros_like(img_perlin)
        defect_num = np.minimum(len(index_list[0]), defect_num)
        # print("??????????????????defect_num:", defect_num)
        # print("??????pixel_range?????????:", len(index_list[0]))
        for idx in np.random.choice(index_list[0], defect_num, replace=False):
            # print("idx:{},pixel num:{}".format(idx, pixel_num[idx]))

            coors = np.where(label_map == idx)  # coors??????tuple?????????????????????list extend??????????????????
            zeros[coors] = 255

        defect = cv2.bitwise_and(img_nature, img_nature, mask=zeros)
        img_lack = cv2.bitwise_and(img, img, mask=255 - zeros)

        img_result = img_lack + defect
        # plt.imshow(img_result[:,:,::-1])
        # plt.show()

        return img_result

    def add_noise_by_perlin_w_mask(self,img,set_dict):
        #img format: BGR
        #size format = [H,W]
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        size = img_gray.shape
        pixel_range = set_dict['pixel_range']
        defect_num = set_dict['defect_num']
        # img_nature = np.fromfile(np.random.choice(set_dict['paths_color']), dtype=np.uint8)
        # img_nature = cv2.imdecode(img_nature, 1)#BGR format
        # img_nature = cv2.resize(img_nature, size[::-1])

        img_perlin = np.fromfile(np.random.choice(set_dict['paths_noise']), dtype=np.uint8)
        img_perlin = cv2.imdecode(img_perlin, 0)
        img_perlin = cv2.resize(img_perlin, size[::-1])

        label_num, label_map, stats, centroids = cv2.connectedComponentsWithStats(img_perlin, connectivity=8)
        pixel_num = stats.T[-1]
        b = np.where(pixel_num >= pixel_range[0], pixel_num, pixel_range[1] + 3)
        index_list = np.where(b < pixel_range[1])  # index_list????????????pixel_range???index
        zeros = np.zeros_like(img_perlin)
        defect_num = np.minimum(len(index_list[0]), defect_num)
        # print("??????????????????defect_num:", defect_num)
        # print("??????pixel_range?????????:", len(index_list[0]))
        img_result = img_gray.copy()
        for idx in np.random.choice(index_list[0], defect_num, replace=False):
            # print("idx:{},pixel num:{}".format(idx, pixel_num[idx]))

            coors = np.where(label_map == idx)  # coors??????tuple?????????????????????list extend??????????????????
            pixel_list = img_gray[coors]
            img_result[coors] = 255 - pixel_list
            zeros[coors] = 255

        # defect = cv2.bitwise_and(img_nature, img_nature, mask=zeros)
        # img_result = cv2.bitwise_and(img,img,mask=255-zeros)
        # img_lack = cv2.bitwise_and(img, img, mask=255 - zeros)

        img_result = cv2.cvtColor(img_result,cv2.COLOR_GRAY2BGR)

        #----display
        to_show = False
        if to_show:
            plt.figure(num=1,figsize=(10,10))
            plt.subplot(1,2,1)
            plt.imshow(img_result[:,:,::-1])
            plt.subplot(1,2,2)
            plt.imshow(zeros,cmap='gray')
            plt.show()

        return img_result,zeros

    def get_perlin_noise(self,shape,res=(16,16)):
        a = generate_perlin_noise_2d(
            shape, res, tileable=(False, False), interpolant=interpolant)
        ave = np.average(a)
        std = np.std(a)
        b = np.where(a > ave + 1 * std, 255, 0).astype(np.uint8)
        # ----random natural image
        img_back = np.fromfile(np.random.choice(self.paths_flower), dtype=np.uint8)
        img_back = cv2.imdecode(img_back, 1)
        img_back = cv2.resize(img_back, b.shape[::-1])

        return b,img_back

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
            config = tf.ConfigProto(log_device_placement=False,  # ??????????????????????????????CPU???GPU
                                    allow_soft_placement=True,  # ???????????????????????????tf?????????????????????????????????????????????????????????
                                    )
            if GPU_ratio is None:
                config.gpu_options.allow_growth = True  # ???????????????????????????????????????????????????
            else:
                config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio  # ????????????GPU???????????????
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
                ????????????batch normalzition???????????????????????????????????????
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

                tf.import_graph_def(graph_def, name='')  # ???????????????

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

class DataLoader():
    def __init__(self,paths,batch_size=32,shuffle=True,labels=None):
        qty = len(paths)
        ites = math.ceil(qty / batch_size)
        if isinstance(paths,np.ndarray) is False:
            paths = np.array(paths)
        if labels is not None:
            if isinstance(labels, np.ndarray) is False:
                labels = np.array(labels)
        self.qty = qty
        self.batch_size = batch_size
        self.paths = paths
        self.labels = labels
        self.iterations = ites
        self.ite_num = 0

        if shuffle:
            self.shuffle()
    def shuffle(self):
        indices = np.random.permutation(self.qty)
        self.paths = self.paths[indices]
        if self.labels is not None:
            self.labels = self.labels[indices]
    def __next__(self):
        num_start = self.batch_size * self.ite_num
        num_end = np.minimum(num_start + self.batch_size, self.qty)
        self.ite_num += 1

        if self.labels is not None:
            return self.paths[num_start:num_end], self.labels[num_start:num_end]
        else:
            return self.paths[num_start:num_end]






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

def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)

def generate_perlin_noise_2d(shape,res,tileable=(False, False),interpolant=interpolant):
    """Generate a 2D numpy array of perlin noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of shape shape with the generated noise.
    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5,
        lacunarity=2, tileable=(False, False),
        interpolant=interpolant
):
    """Generate a 2D numpy array of fractal noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.
    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency*res[0], frequency*res[1]), tileable, interpolant
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise

def modify_contrast_and_brightness(img, br=0, ct=100,standard=127.5):
    # brightness = br
    # contrast = ct
    # img_p = img.copy()

    B = br / 255.0  # brightness
    c = ct / 255.0  # contrast
    k = math.tan((45 + 44 * c) / 180 * math.pi)
    print("B = ", B)
    print("K = ", k)

    img_p = (img - standard * (1 - B)) * k + standard * (1 + B)

    # ????????????????????? 0~255 ???????????????255 = 255????????? 0 = 0

    return np.clip(img_p, 0, 255).astype(np.uint8)

def contrast_and_brightness_compare(paths, num, br=0, ct=10,standard=127.5):
    # np.random.shuffle(paths)


    for i, path in enumerate(paths[:num]):
        print(path)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        if img is None:
            print("Read failed:", path)
        else:
            #img_p = img.copy()
            img_p = modify_contrast_and_brightness(img.astype(np.float32), br=br, ct=ct,standard=standard)
            # plt.subplot(num, 2, i * 2 + 1)
            # plt.subplot(1, 2, i * 2 + 1)
            plt.figure(num=1,figsize=(15, 15))
            plt.subplot(1, 2, 1)
            plt.imshow(img[:, :, ::-1])
            plt.title("ori")
            # plt.subplot(num, 2, i * 2 + 2)
            plt.subplot(1, 2, 2)
            plt.imshow(img_p[:, :, ::-1])
            plt.title('processed')

            plt.show()

def say_sth(msg, print_out=False,header=None):
    if print_out:
        print(msg)
    # if TCPConnected:
    #     TCPClient.send(msg + "\n")

def log_update(content,para_dict):
    for key, value in para_dict.items():
        content[key] = value

    return content

def get_label_dict(img_dir):
    #----var
    label_dict = dict()

    #----get all dir names
    dirs = [obj.name for obj in os.scandir(img_dir) if obj.is_dir()]
    len_dir = len(dirs)

    if len_dir == 0:
        msg = "No dir in the {}".format(img_dir)
        say_sth(msg,print_out=print_out)
        raise ValueError
    elif len(dirs) == 2:
        if 'ok' in dirs and 'ng' in dirs:
            label_dict = {'ng':0, 'ok':1}
        elif 'OK' in dirs and 'NG' in dirs:
            label_dict = {'NG':0, 'OK':1}
        else:
            for idx,dir_name in enumerate(dirs):
                label_dict[dir_name] = idx
    else:
        for idx, dir_name in enumerate(dirs):
            label_dict[dir_name] = idx

    return label_dict

def get_paths_labels(img_dir ,label_dict ,type='extend'):
    # ----var
    img_format = {'png', 'jpg', 'bmp'}
    re_paths = list()
    re_labels = list()
    classname_list = list(label_dict.keys())
    stat_dict = dict()

    #----init stat dict
    for classname in classname_list:
        stat_dict[classname] = 0

    # ----read dirs
    dirs = [obj.path for obj in os.scandir(img_dir) if obj.is_dir()]
    if len(dirs) == 0:
        # print("No dirs in the ",img_dir)
        msg = "No dirs in the {}".format(img_dir)
        say_sth(msg ,print_out=print_out)
    else:
        # -----read paths of each dir
        for dir_path in dirs:
            dir_name = dir_path.split("\\")[-1]
            #----dir name must in classname_list
            if dir_name in classname_list:
                path_temp = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
                qty_path = len(path_temp)
                if qty_path == 0:
                    # print("No images in the ",dir_path)
                    msg = "??????????????????:{}".format(dir_path)
                    say_sth(msg, print_out=print_out)
                else:
                    # ----get the label number from class name
                    #label_num = dir_path.split("\\")[-1]
                    label_num = label_dict[dir_name]
                    # ----create the label array
                    label_temp = np.ones(len(path_temp) ,dtype=np.int16) * label_num

                    # ----collect paths and labels
                    if type == 'append':
                        re_paths.append(path_temp)
                        re_labels.append(label_temp)
                    elif type == 'extend':
                        re_paths.extend(path_temp)
                        re_labels.extend(label_temp)

                    #----record the qty in stat dict
                    stat_dict[dir_name] += qty_path
            else:
                msg = "?????????:{}?????????????????????????????????????????????".format(dir_path)
                say_sth(msg, print_out=print_out)


        # ----list to numpy array
        # re_paths = np.array(re_paths)
        # re_labels = np.array(re_labels)

        # ----shuffle
        # indice = np.random.permutation(re_paths.shape[0])
        # re_paths = re_paths[indice]
        # re_labels = re_labels[indice]

    return re_paths, re_labels, stat_dict

def get_paths_labels_quantity(img_dir ,name2label_dict ):
    # ----var
    img_format = {'png', 'jpg', 'bmp'}
    re_paths = list()
    re_labels = list()
    path_dict = dict()
    label_dict = dict()
    qty_dict = dict()

    # ----read dirs
    dirs = [obj.path for obj in os.scandir(img_dir) if obj.is_dir()]
    if len(dirs) == 0:
        # print("No dirs in the ",img_dir)
        msg = "No dirs in the {}".format(img_dir)
        say_sth(msg ,print_out=print_out)
    else:
        # -----read paths of each dir
        for dir_path in dirs:
            path_temp = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
            qty_path = len(path_temp)
            #print("qty_path:",qty_path)
            if qty_path == 0:
                # print("No images in the ",dir_path)
                msg = "No images in the {}".format(dir_path)
                say_sth(msg, print_out=print_out)
            else:
                # ----get the label number from class name
                class_name = dir_path.split("\\")[-1]
                if class_name not in name2label_dict.keys():
                    msg = "Warning: the class name:{} is not included in the label_dict".format(class_name)
                    say_sth(msg, print_out=print_out)
                else:
                    label_num = name2label_dict[class_name]
                    # ----create the label array
                    label_temp = np.ones(qty_path,dtype=np.int16) * label_num

                    path_dict[class_name] = path_temp
                    label_dict[class_name] = label_temp
                    qty_dict[class_name] = qty_path
                # ----collect paths and labels
                # if type == 'append':
                #     re_paths.append(path_temp)
                #     re_labels.append(label_temp)
                # elif type == 'extend':
                #     re_paths.extend(path_temp)
                #     re_labels.extend(label_temp)

        # ----list to numpy array
        # re_paths = np.array(re_paths)
        # re_labels = np.array(re_labels)

        # ----shuffle
        # indice = np.random.permutation(re_paths.shape[0])
        # re_paths = re_paths[indice]
        # re_labels = re_labels[indice]

    return path_dict, label_dict, qty_dict

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
    random_num = np.random.randint(0, 128, random_num_range, dtype=np.uint8)  # ???????????????????????????
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

def evaluation_binary(predictions,labels,result_array=None,prob_threshold=0.51):
    #----var
    if result_array is None:
        result_array = np.zeros([4],dtype=np.int16)#{'tp':0,'fn':0,'tn':0,'fp':0}

    for idx, prediction in enumerate(predictions):
        argmax = np.argmax(prediction)
        if argmax == 1 and prediction[argmax] >= prob_threshold:
            if labels[idx] == 1:
                #True positive
                result_array[0] += 1
            else:
                #False positive
                result_array[3] += 1
        else:
            if labels[idx] == 0:
                #True negative
                result_array[2] += 1
            else:
                #False negative
                result_array[1] += 1
    return result_array

def evaluation(predictions,labels,prob_threshold=0.51):
    count = 0
    for i in range(predictions.shape[0]):
        pred_label = np.argmax(predictions[i])
        if pred_label == labels[i] and predictions[i][pred_label] >= prob_threshold:
            count += 1

    return count

def get_4D_data(paths, output_shape, process_dict=None,setting_dict=None):
    # ----var
    len_path = len(paths)
    processing_enable = False

    # ----check process_dict
    if isinstance(process_dict, dict):
        if len(process_dict) > 0:
            processing_enable = True  # image processing is enabled
            if process_dict.get('ave_filter') is True:
                kernel = (5,5)
            if process_dict.get('gau_filter') is True:
                kernel = (5,5)
            if process_dict.get('rdm_crop') is True:
                crop_range = [3, 3]  # [x_range, y_range]
                if setting_dict is not None:
                    if setting_dict.get('rdm_crop') is not None:
                        crop_range = setting_dict.get('rdm_crop')

                x_start = np.random.randint(crop_range[0], size=len_path)
                y_start = np.random.randint(crop_range[1], size=len_path)
            if process_dict.get('rdm_shift') is True:
                shift_ratio = 0.1
#                 shift_range = [3, 3]  # [x_shift, y_shift]
                #print("shift_range:",shift_range)
                if setting_dict is not None:
                    if setting_dict.get('rdm_shift') is not None:
                        shift_ratio = setting_dict.get('rdm_shift')
#                 print("shift_range:",shift_range)
#                 x_s2 = np.random.randint(shift_range[0], size=len_path)
#                 y_s2 = np.random.randint(shift_range[1], size=len_path)
                corner = np.random.randint(4, size=len_path)

    #----setting dict
    if processing_enable is True:
        # ----resized size
        h, w, _ = output_shape
        #----setting parsing
        if setting_dict is not None:
            if process_dict.get('ave_filter') is True:
                if setting_dict.get('ave_filter') is not None:
                    kernel = setting_dict['ave_filter']
            if process_dict.get('gau_filter') is True:
                if setting_dict.get('gau_filter') is not None:
                    kernel = setting_dict['gau_filter']
            if process_dict.get('rdm_br'):
                br_range = [0.88,1.12]
                if setting_dict.get('rdm_br') is not None:
                    br_range = setting_dict['rdm_br']
                    if br_range > 0 and br_range <= 1:
                        br_range = [1 - br_range, 1 + br_range]
            if process_dict.get('rdm_flip'):
                flip_list = [1, 0, -1, 2]
                if setting_dict.get('rdm_flip') is not None:
                    flip_list = setting_dict['rdm_flip']
            if process_dict.get('rdm_blur'):
                kernel_list = [1,1,1,3,3,3,5,5]
                if setting_dict.get('rdm_blur') is not None:
                    kernel_list = setting_dict['rdm_blur']
            if process_dict.get('rdm_angle'):
                angle_range = [-5,5]
                if setting_dict.get('rdm_angle') is not None:
                    angle_range = setting_dict['rdm_angle']
                    if angle_range >0 and angle_range <= 90:
                        angle_range = [-angle_range,angle_range]
            if process_dict.get('rdm_patch'):
                margin_ratio = 0.25
                patch_ratio = 0.3
                size_min = 10

                if setting_dict.get('rdm_patch') is not None:
                    margin_ratio = setting_dict['rdm_patch'][0]
                    patch_ratio = setting_dict['rdm_patch'][1]

    # ----create default np array
    batch_dim = [len_path]
    batch_dim.extend(output_shape)
    batch_data = np.zeros(batch_dim, dtype=np.float32)
    if processing_enable is True:
        if process_dict.get('rdm_patch') is True:
            batch_data_no_patch = np.zeros_like(batch_data)

    #----
    for idx, path in enumerate(paths):
        # img = cv2.imread(path)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        if img is None:
            msg = "read failed:".format(path)
            say_sth(msg)
        else:
            ori_h,ori_w,_ = img.shape
            #print("img shape=",img.shape)
            # ----image processing
            if processing_enable is True:
                if process_dict.get('ave_filter'):
                    img = cv2.blur(img,kernel)
                if process_dict.get('gau_filter'):
                    img = cv2.GaussianBlur(img,kernel,0,0)
                if process_dict.get('rdm_crop'):
                    # ----From the random point, crop the image
                    y = y_start[idx] * h / ori_h
                    x = x_start[idx] * w / ori_w
                    p = int(round(y + x))
                    img = img[p:-(p+1), p:-(p+1), :]
                if process_dict.get('rdm_shift'):
                    c = corner[idx]
                    y = np.random.randint(ori_h * shift_ratio)
                    x = np.random.randint(ori_w * shift_ratio)
                    p = int(round((y + x) / 2))
                    #print("x={},y={},p={}".format(x,y,p))
                    if c == 0:
                        img = img[p:, p:, :]
                    elif c == 1:
                        img = img[p:, :-(p + 1), :]
                    elif c == 2:
                        img = img[:-(p + 1), p:, :]
                    elif c == 3:
                        img = img[:-(p + 1), :-(p + 1), :]
                    else:
                        img = img[p:, p:, :]
                if process_dict.get('rdm_br'):
                    mean_br = np.mean(img)
                    try:
                        br_factor = np.random.randint(math.floor(mean_br * br_range[0]), math.ceil(mean_br * br_range[1]))
                        img = np.clip(img / mean_br * br_factor, 0, 255)
                        img = img.astype(np.uint8)
                    except:
                        msg = "Error:rdm_br value"
                        say_sth(msg)
                if process_dict.get('rdm_flip'):
                    flip_type = np.random.choice(flip_list)
                    if flip_type != 2:
                        img = cv2.flip(img, flip_type)
                if process_dict.get('rdm_blur'):
                    kernel = tuple(np.random.choice(kernel_list, size=2))
                    # print("kernel:", kernel)
                    if np.random.randint(0,2) == 0:
                        img = cv2.blur(img,kernel)
                    else:
                        img = cv2.GaussianBlur(img, kernel, 0, 0)
                if process_dict.get('rdm_noise'):
                    uniform_noise = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
                    cv2.randu(uniform_noise, 0, 255)
                    ret, impulse_noise = cv2.threshold(uniform_noise, 230, 255, cv2.THRESH_BINARY_INV)
                    img = cv2.bitwise_and(img, img, mask=impulse_noise)
                if process_dict.get('rdm_angle'):
                    angle = np.random.randint(angle_range[0], angle_range[1])
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h),borderMode=cv2.BORDER_REPLICATE)
                if process_dict.get('rdm_patch'):
                    #----put no patch image
                    img_no_patch = img.copy()
                    img_no_patch = cv2.resize(img_no_patch, (output_shape[1], output_shape[0]))
                    img_no_patch = cv2.cvtColor(img_no_patch, cv2.COLOR_BGR2RGB)
                    batch_data_no_patch[idx] = img_no_patch
                    #----
                    maxi_len = int(sum(img.shape[:2]) / 2 * patch_ratio)
                    patch_times = np.random.randint(1,4)
                    patch_types = np.random.randint(0, 4, patch_times)
                    margin_x = margin_ratio * img.shape[1]
                    margin_y = margin_ratio * img.shape[0]
                    for patch_type in patch_types:
                        center_x = np.random.randint(margin_x, img.shape[1] - margin_x)
                        center_y = np.random.randint(margin_y, img.shape[0] - margin_y)
                        if np.random.randint(2) == 1:
                            color = (0, 0, 0)
                        else:
                            color = np.random.randint(0, 255, 3).tolist()
                        if patch_type == 0:
                            end_x = np.random.randint(size_min, maxi_len / 2)
                            # end_y = np.random.randint(center_y,img.shape[0])
                            cv2.rectangle(img, (center_x, center_y), (center_x + end_x, center_y + end_x), color, -1)
                            #print("square,center_x:{},center_y:{},len:{}".format(center_x, center_y, end_x))
                        elif patch_type == 1:
                            radius = np.random.randint(size_min, maxi_len / 2)
                            cv2.circle(img, (center_x, center_y), radius, color, -1)
                            #print("circle,center_x:{},center_y:{},radius:{}".format(center_x, center_y, radius))
                        elif patch_type == 2:#rectangle_1
                            lens = np.random.randint(size_min, maxi_len / 2, 2)
                            cv2.rectangle(img, (center_x, center_y), (center_x + lens[0], center_y + lens[1]), color, -1)
                            #print("rectangle,center_x:{},center_y:{},len:{}".format(center_x, center_y, lens))
                        elif patch_type == 3:  # rectangle_2
                            ave_size = sum(img.shape[:2]) / 2
                            long_len = np.random.randint(ave_size // 2, ave_size * 0.9)
                            short_len = np.random.randint(3, ave_size // 4)
                            if center_x >= ave_size // 2:
                                if center_y >= ave_size // 2:
                                    pass
                                else:
                                    cv2.rectangle(img, (center_x, center_y), (center_x + short_len, center_y + long_len),
                                                  color, -1)
                            else:
                                if center_y >= ave_size // 2:
                                    cv2.rectangle(img, (center_x, center_y), (center_x + long_len, center_y + short_len),
                                                  color, -1)
                                else:
                                    lens = [long_len, short_len]
                                    np.random.shuffle(lens)
                                    cv2.rectangle(img, (center_x, center_y), (center_x + lens[0], center_y + lens[1]),
                                                  color, -1)

            # ----resize and change the color format
            img = cv2.resize(img, (output_shape[1], output_shape[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch_data[idx] = img

    if processing_enable is True:
        if process_dict.get('rdm_patch'):
            return batch_data_no_patch/255,batch_data/255
    else:
        return batch_data/255

def data_process(img_array, output_shape, process_dict=None):
    # ----var
    len_path = img_array.shape[0]
    processing_enable = False
    y_range = 3
    x_range = 3
    flip_list = [1, 0, -1, 2]
    kernel_list = [1]

    # print("img_array ahspe: ",img_array.shape)
    if np.max(img_array) <= 1:
        img_array *= 255

    # ----create default np array
    # batch_dim = [len_path]
    # batch_dim.extend(output_shape)
    # batch_data = np.zeros(batch_dim, dtype=np.float32)

    # ----check process_dict
    if isinstance(process_dict, dict):
        if len(process_dict) > 0:
            processing_enable = True  # image processing is enabled
            if 'rdm_crop' in process_dict.keys():
                x_start = np.random.randint(x_range, size=len_path)
                y_start = np.random.randint(y_range, size=len_path)

    for idx in range(len_path):
        img = img_array[idx]
        # ----image processing
        if processing_enable is True:
            if 'rdm_crop' in process_dict.keys():
                if process_dict['rdm_crop'] is True:
                    # ----From the random point, crop the image
                    img = img[y_start[idx]:-(y_start[idx]+1), x_start[idx]:-(x_start[idx]+1), :]
            if 'rdm_br' in process_dict.keys():
                if process_dict['rdm_br'] is True:
                    mean_br = np.mean(img)
                    br_factor = np.random.randint(mean_br * 0.92, mean_br * 1.08)
                    img = np.clip(img / mean_br * br_factor, 0, 255)
                    # img = img.astype(np.uint8)
            if 'rdm_flip' in process_dict.keys():
                if process_dict['rdm_flip'] is True:
                    flip_type = np.random.choice(flip_list)
                    if flip_type != 2:
                        img = cv2.flip(img, flip_type)
            if 'rdm_blur' in process_dict.keys():
                if process_dict['rdm_blur'] is True:
                    kernel = tuple(np.random.choice(kernel_list, size=2))
                    #print("kernel:", kernel)
                    img = cv2.GaussianBlur(img, kernel, 0, 0)
            if 'rdm_noise' in process_dict.keys():
                if process_dict['rdm_noise'] is True:
                    uniform_noise = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
                    cv2.randu(uniform_noise, 0, 255)
                    ret, impulse_noise = cv2.threshold(uniform_noise, 230, 255, cv2.THRESH_BINARY_INV)
                    img = cv2.bitwise_and(img, img, mask=impulse_noise)
            if 'rdm_angle' in process_dict.keys():
                if process_dict['rdm_angle'] is True:
                    angle = np.random.randint(-10, 10)
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h))

        # ----resize and change the color format
        img = cv2.resize(img, (output_shape[1], output_shape[0]))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array[idx] = img

    return img_array / 255

def get_4D_data_2(paths, img_shape, process_dict=None):
        # ----var
        random_flip = False
        random_brightness = False
        random_crop = False
        random_angle = False
        random_noise = False
        flip_list = [1, 0]

        # ----create default np array
        batch_dim = [len(paths)]
        batch_dim.extend(img_shape)
        batch_data = np.zeros(batch_dim, dtype=np.float32)

        # ----update var
        if isinstance(process_dict, dict):
            if 'random_flip' in process_dict.keys():
                random_flip = process_dict['random_flip']
            if 'random_brightness' in process_dict.keys():
                random_brightness = process_dict['random_brightness']
            if 'random_crop' in process_dict.keys():
                random_crop = process_dict['random_crop']
            if 'random_angle' in process_dict.keys():
                random_angle = process_dict['random_angle']
            if 'random_noise' in process_dict.keys():
                random_noise = process_dict['random_noise']

        for idx, path in enumerate(paths):
            img = cv2.imread(path)
            if img is None:
                print("read failed:", path)
            else:
                img = cv2.resize(img, (img_shape[1], img_shape[0]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # ----random brightness
                if random_brightness is True:
                    mean_br = np.mean(img)
                    br_factor = np.random.randint(mean_br * 0.7, mean_br * 1.3)
                    img = np.clip(img / mean_br * br_factor, 0, 255)
                    img = img.astype(np.uint8)

                # ----random crop
                if random_crop is True:
                    # ----resize the image 1.15 times
                    img = cv2.resize(img, None, fx=1.15, fy=1.15)

                    # ----Find a random point
                    y_range = img.shape[0] - img_shape[0]
                    x_range = img.shape[1] - img_shape[1]
                    x_start = np.random.randint(x_range)
                    y_start = np.random.randint(y_range)

                    # ----From the random point, crop the image
                    img = img[y_start:y_start + img_shape[0], x_start:x_start + img_shape[1], :]

                # ----random flip
                if random_flip is True:
                    flip_type = np.random.choice(flip_list)
                    if flip_type == 1:
                        img = cv2.flip(img, flip_type)

                # ----random angle
                if random_angle is True:
                    angle = np.random.randint(-60, 60)
                    height, width = img.shape[:2]
                    M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (width, height))

                # ----random noise
                if random_noise is True:
                    uniform_noise = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
                    cv2.randu(uniform_noise, 0, 255)
                    ret, impulse_noise = cv2.threshold(uniform_noise, 240, 255, cv2.THRESH_BINARY_INV)
                    img = cv2.bitwise_and(img, img, mask=impulse_noise)

                batch_data[idx] = img

        batch_data /= 255
        return batch_data

def check_results(dir_path,encript_flag=False,epoch_range=None,only2see=None):

    #----var
    # json_path = os.path.join(dir_path,'train_result.json')
    file_name = "train_result_"
    plot_type = []
    # train_loss_list = list()
    # test_loss_list = list()
    # train_acc_list = list()
    # test_acc_list = list()
    # test_overKill_list = list()
    # test_undet_list = list()
    # total_train_time = 0
    exclude_list = ['train_loss_list','train_acc_list','test_loss_list','test_acc_list','save_dir']
    data_name_list = ['train_loss_list','test_loss_list',
                      'train_acc_list','test_acc_list',
                      'seg_train_loss_list','seg_test_loss_list',
                      'seg_train_iou_list','seg_test_iou_list',
                      'seg_train_acc_list','seg_test_acc_list',
                      'seg_train_defect_recall_list','seg_test_defect_recall_list',
                      'seg_train_defect_sensitivity_list','seg_test_defect_sensitivity_list']
    tailer = '.json'
    qty_plot = 2
    class_names = None

    if encript_flag is True:
        tailer = '.nst'

    if isinstance(only2see,list):
        data_name_list = only2see

    #----plot data init
    data_dict = dict()
    for key in data_name_list:
        data_dict[key] = list()

    file_nums = [int(file.name.split(".")[0].split("_")[-1]) for file in os.scandir(dir_path) if
                 file.name.find(file_name) >= 0]
    seq = np.argsort(file_nums)

    #----decode files(save and overlap the original files)
    # if encript_flag is True:
    #     for idx in seq:
    #         json_path = os.path.join(dir_path, file_name + str(file_nums[idx]) + tailer)
    #         if os.path.exists(json_path):
    #             file_decode_v2(json_path,random_num_range=10)
    #             time.sleep(0.1)

    for idx in seq:
        json_path = os.path.join(dir_path,file_name + str(file_nums[idx]) + tailer)

        #----read the json file
        if os.path.exists(json_path):
            print(json_path)
            #----decode the file
            # if encript_flag is True:
            #     file_decode_v2(json_path)
            #     time.sleep(2)

            #----read the file
            ret = file_decode_v2(json_path, random_num_range=10,return_value=True,to_save=False)#ret is None or bytes

            if ret is None:
                print("ret is None. The file is not secured")
                with open(json_path, 'r') as f:
                    content = json.load(f)
            else:
                print("ret is not None. The file is decoded")
                content = json.loads(ret.decode())
            key_list = list(content.keys())
            #----encode the file
            #file_transfer(json_path)


            print("--------{} parameters--------".format(file_name + str(file_nums[idx])))
            # ----display info except loss,acc
            for key, value in content.items():
                if key not in exclude_list:
                    print(key, ": ", value)

            #----var parsing
            for data_name in data_name_list:
                if content.get(data_name) is not None:
                    print("data_name:",data_name)
                    if isinstance(content[data_name],list):
                        if len(content[data_name]) > 0:
                            data_dict[data_name].extend(content[data_name])
                    else:
                        data_dict[data_name].append(content[data_name])
            # if content.get('train_loss_list'):
            #     train_loss_list.extend(content['train_loss_list'])
            # train_acc_list.extend(content['train_acc_list'])
            # if content["test_img_dir"] is not None:
            #     test_loss_list.extend(content['test_loss_list'])
            #     test_acc_list.extend(content['test_acc_list'])


            # if content.get('total_train_time') is not None:
            #     total_train_time += content['total_train_time']

            # if "test_overKill_list" in key_list:
            #             #     if content['test_overKill_list'] is not None:
            #             #         test_overKill_list.extend(content['test_overKill_list'])
            #             # if "test_undet_list" in key_list:
            #             #     if content['test_undet_list'] is not None:
            #             #         test_undet_list.extend(content['test_undet_list'])
            #             #         qty_plot += 1

    patterns = 'seg|loss|acc'
    # ----statistics
    print("--------Statistics--------")
    for data_name, data_list in data_dict.items():
        if len(data_list) > 0:
            if data_name.find('seg') >= 0:
                if data_name.find('loss') >= 0:
                    arg = np.argmin(data_list)
                    plot_type.append('seg|loss')
                elif data_name.find('train_iou') >= 0:
                    data_dict[data_name] = np.array(data_list).astype(np.float16).T
                    plot_type.append('seg|train_iou')
                elif data_name.find('test_iou') >= 0:
                    data_dict[data_name] = np.array(data_list).astype(np.float16).T
                    plot_type.append('seg|test_iou')
                elif data_name.find('train_acc') >= 0:
                    data_dict[data_name] = np.array(data_list).astype(np.float16).T
                    plot_type.append('seg|train_acc')
                elif data_name.find('test_acc') >= 0:
                    data_dict[data_name] = np.array(data_list).astype(np.float16).T
                    plot_type.append('seg|test_acc')
                elif data_name.find('train_defect_recall') >= 0:
                    data_dict[data_name] = np.array(data_list).astype(np.float16).T
                    plot_type.append('seg|train_defect_recall')
                elif data_name.find('test_defect_recall') >= 0:
                    data_dict[data_name] = np.array(data_list).astype(np.float16).T
                    plot_type.append('seg|test_defect_recall')
                elif data_name.find('train_defect_sensitivity') >= 0:
                    data_dict[data_name] = np.array(data_list).astype(np.float16).T
                    plot_type.append('seg|train_defect_sensitivity')
                elif data_name.find('test_defect_sensitivity') >= 0:
                    data_dict[data_name] = np.array(data_list).astype(np.float16).T
                    plot_type.append('seg|test_defect_sensitivity')

            else:
                if data_name.find('loss') >= 0:
                    #----check if this training is AE_SEG
                    ae_var = content.get('ae_var')
                    if ae_var is None:
                        loss_method = content.get('loss_method')
                    else:
                        loss_method = ae_var.get('loss_method')
                    if loss_method == 'ssim':
                        arg = np.argmax(data_list)
                    else:
                        arg = np.argmin(data_list)

                    plot_type.append('loss')
                elif data_name.find('acc') >= 0:
                    arg = np.argmax(data_list)
                    plot_type.append('acc')

            print("Epochs executed:", len(data_list))
            #print("The best of {} is {} at epoch {}".format(data_name,data_list[arg], arg+1))
        else:
            print("{} has no data".format(data_name))


    #----plot loss results
    plot_type = set(plot_type)
    qty_plot = len(plot_type)
    y_qty = math.ceil(qty_plot / 2)
    if content.get('class_names') is not None:
        class_names = content['class_names']
        select_name = "_background_"
        #----??????_background_??????(???????????????????????????plt?????????)
        if select_name in class_names:
            idx = class_names.index(select_name)
            class_names[idx] = "background"


    plt.figure(num=1,figsize=(int(qty_plot * 10),int(qty_plot * 3)))
    for idx,show_name in enumerate(plot_type):
        if qty_plot > 1:
            plt.subplot(y_qty, 2, idx+1)
        #----
        for data_name, data_list in data_dict.items():
            if len(data_list) > 0:
                if np.array(data_list).ndim > 1:
                    x_num_ori = [i + 1 for i in range(len(data_list[-1]))]
                else:
                    x_num_ori = [i + 1 for i in range(len(data_list))]


                if epoch_range is None:
                    x_num = x_num_ori
                    data_show = data_list
                else:
                    s = epoch_range
                    if len(epoch_range) >= 2:
                        x_num = x_num_ori[s[0]:s[1]]
                        if np.array(data_list).ndim > 1:
                            data_show = data_list[:,s[0]:s[1]]
                        else:
                            data_show = data_list[s[0]:s[1]]
                    else:
                        x_num = x_num_ori[s[0]:]
                        if np.array(data_list).ndim > 1:
                            data_show = data_list[:,s[0]:]
                        else:
                            data_show = data_list[s[0]:]

                if data_name.find('seg') >= 0:
                    if len(re.findall(show_name, data_name, re.I)) > 1:
                        print(data_name)
                        if len(re.findall("iou|acc|defect_recall|defect_sensitivity", data_name, re.I)) == 1:
                            for i, value in enumerate(data_show):
                                plt.plot(x_num, value, label=class_names[i])
                        # if data_name.find('iou') >= 0:
                        #     for i, iou in enumerate(data_show):
                        #         plt.plot(x_num, iou, label=class_names[i])
                        # elif data_name.find('acc') >= 0:
                        #     for i, acc in enumerate(data_show):
                        #         plt.plot(x_num, acc, label=class_names[i])
                        # elif data_name.find('defect_recall') >= 0:
                        #     for i, acc in enumerate(data_show):
                        #         plt.plot(x_num, acc, label=class_names[i])
                        else:
                            plt.plot(x_num, data_show, label=data_name)
                else:
                    if data_name.find(show_name) >= 0:
                        plt.plot(x_num, data_show, label=data_name)





        plt.legend()
        plt.ylabel(show_name)
        plt.xlabel("epoch")

    plt.show()


    # x_num = [i+1 for i in range(len(data_dict['train_loss_list']))]
    #
    # plt.subplot(1, qty_plot, 1)
    # for key in show_plot_list:
    #     if key.find('loss') >= 0:
    #         if len(data_dict[key]) > 0:
    #             plt.plot(x_num, data_dict[key], label=key)
    #
    #
    #
    # # plt.plot(x_num,train_loss_list,label="train_loss")
    # # if content["test_img_dir"] is not None:
    # #     plt.plot(x_num, test_loss_list, label="test_loss")
    #
    # plt.legend()
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    #
    # #----plot acc results
    # plt.subplot(1,qty_plot,2)
    # for key in show_plot_list:
    #     if key.find('acc') >= 0:
    #         if len(data_dict[key]) > 0:
    #             plt.plot(x_num, data_dict[key], label=key)
    # plt.plot(x_num, train_acc_list, label="train_acc")
    # if content["test_img_dir"] is not None:
    #     plt.plot(x_num, test_acc_list, label="test_acc")
    # plt.legend()
    # # plt.ylim((0.9, 0.98))  # ??????y???????????????
    # plt.ylabel("accuracy")
    # plt.xlabel("epoch")

    # if qty_plot > 2:
    #     # ----plot overKill and undet results
    #     plt.subplot(1, qty_plot, 3)
    #     plt.plot(x_num, test_overKill_list, label="test_overKill")
    #     plt.plot(x_num, test_undet_list, label="test_underKill")
    #     plt.legend()
    #     # plt.ylim((0.9, 0.98))  # ??????y???????????????
    #     plt.ylabel("%")
    #     plt.xlabel("epoch")


    #----show plots
    # plt.show()

    # ----encode files
    # if encript_flag is True:
    #     for idx in seq:
    #         json_path = os.path.join(dir_path, file_name + str(file_nums[idx]) + tailer)
    #         if os.path.exists(json_path):
    #             file_transfer(json_path)

def result_comparison(result_dict,extract_name_list,y_axis_list,file_name = "train_result_",encript_flag=False,
                      set_x=None):
    #----var
    data_dict = dict()
    # len_vector = np.zeros(len(extract_name_list),dtype=np.int32)

    tailer = '.json'

    if encript_flag is True:
        tailer = '.nst'

    for count,name in enumerate(result_dict.keys()):
        dir_path = result_dict[name]

        file_nums = [int(file.name.split(".")[0].split("_")[-1]) for file in os.scandir(dir_path) if
                     file.name.find(file_name) >= 0]
        if len(file_nums) == 0:
            print("No files with ({})".format(file_name))
        else:
            seq = np.argsort(file_nums)

            #----reset lists
            temp_dict = dict()
            for extract_name in extract_name_list:
                temp_dict[extract_name] = list()

            for idx in seq:
                json_path = os.path.join(dir_path, file_name + str(file_nums[idx]) + tailer)

                # ----read the file
                ret = file_decode_v2(json_path, random_num_range=10, return_value=True,
                                     to_save=False)  # ret is None or bytes

                if ret is None:
                    print("ret is None. The file is not secured")
                    with open(json_path, 'r') as f:
                        content = json.load(f)
                else:
                    print("ret is not None. The file is decoded")
                    content = json.loads(ret.decode())

                # ----var parsing
                for extract_name in extract_name_list:
                    if extract_name in content.keys():
                        temp_dict[extract_name].extend(content[extract_name])

                #----read the json file
                # if os.path.exists(json_path):
                #     with open(json_path, 'r') as f:
                #         content = json.load(f)
                #
                #     #----var parsing
                #     for extract_name in extract_name_list:
                #         if extract_name in content.keys():
                #             temp_dict[extract_name].extend(content[extract_name])


            #----combine all data to a dict
            #temp_dict = {'train_loss_list':train_loss_list,'train_acc_list':train_acc_list,'test_acc_list':test_acc_list}
            data_dict[name] = temp_dict

            #----calculate data length
            len_temp = np.zeros(len(extract_name_list),dtype=np.int32)
            for idx,extract_name in enumerate(extract_name_list):
                if extract_name in content.keys():
                    len_temp[idx] = len(temp_dict[extract_name])


            #----compare the data length(choose smaller length)
            if count == 0:
                len_vector = len_temp
            else:
                len_vector = np.minimum(len_vector,len_temp)

    #----set the range
    if set_x is not None:
        set_num = np.ones_like(len_vector) * int(set_x)
        len_vector = np.minimum(len_vector,set_num)

    # ----plot loss results
    plt.figure(figsize=(39, 5))

    for idx, extract_name in enumerate(extract_name_list):
        x_num = [i + 1 for i in range(len_vector[idx])]

        plt.subplot(1, len(extract_name_list), idx+1)
        for name in data_dict.keys():
            content = data_dict[name][extract_name]
            plt.plot(x_num,content[:len_vector[idx]] , label=name)
            # plt.plot(x_num[:25],content[:25] , label=name)
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel(y_axis_list[idx])
            plt.title(extract_name_list[idx])
    # ----show plots
    plt.show()

    # plt.legend()
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    #
    # # ----plot acc results
    # plt.subplot(1, 2, 2)
    # # plt.plot(x_num, train_acc_list, label="train_acc")
    # if "test_img_dir" in content.keys():
    #     plt.plot(x_num, test_acc_list, label="test_acc")
    # plt.legend()
    # plt.ylim((0.9, 0.98))  # ??????y???????????????
    # plt.ylabel("accuracy")
    # plt.xlabel("epoch")

def img_mask(img_source, json_path, img_type='path',zoom_in_value=None):

    if img_type == 'path':
        img_ori = np.fromfile(img_source,dtype=np.uint8)
        img_ori = cv2.imdecode(img_ori,1)
        img = img_ori.copy()
    else:
        img_ori = img_source
        img = img_source.copy()

    if zoom_in_value is not None:
        z = zoom_in_value
        img = np.zeros_like(img_ori)
        img[z[0]:-z[1],z[2]:-z[3],] = img_ori[z[0]:-z[1],z[2]:-z[3],]


    #----get rectanle coordinance by reading the json file
    with open(json_path,'r') as f:
        content = json.load(f)
        points_list = content['shapes']
        height = content["imageHeight"]
        weight = content["imageWidth"]
        h_ratio = img_ori.shape[0] / height
        w_ratio = img_ori.shape[1] / weight

    for points_dict in points_list:
        if points_dict['shape_type'] == 'rectangle':
            points = np.array(points_dict['points'])
            points /= [[h_ratio,w_ratio],[h_ratio,w_ratio]]
            points = points.astype(np.int16)
            # points = tuple(points)
            print(points)

            cv2.rectangle(img, tuple(points[0]), tuple(points[1]), (0, 0, 0), -1)

    plt.figure(num=1,figsize=(10,10))

    plt.subplot(1,2,1)
    plt.imshow(img_ori[:,:,::-1])
    plt.title("Original")

    plt.subplot(1,2,2)
    plt.imshow(img[:, :, ::-1])
    plt.title("Masked")

    plt.show()




    #cv2.rectangle(img_copy, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), color, 1)

def data_distribution(img_list,save_dir_list,json_list=None,ratio=0.8,select_num=None):
    qty = len(img_list)

    #----shuffle
    indice = np.random.permutation(qty)
    img_list = img_list[indice]
    if json_list is not None:
        json_list = json_list[indice]

    if select_num is None:
        cut_num = int(qty * ratio)
    else:
        cut_num = select_num
    print("??????{}??????{}".format(cut_num,save_dir_list[0]))
    print("??????{}??????{}".format(qty - cut_num,save_dir_list[1]))

    sep_list = [img_list[:cut_num], img_list[cut_num:]]

    #----create dirs
    for save_dir in save_dir_list:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if json_list is None:
        for sep_img_list, save_dir in zip(sep_list,save_dir_list):
            for img_path in sep_img_list:
                new_path = os.path.join(save_dir, img_path.split('\\')[-1])
                shutil.copy(img_path, new_path)
    else:
        sep_list_2 = [json_list[:cut_num], json_list[cut_num:]]
        for sep_img_list,sep_json_list, save_dir in zip(sep_list,sep_list_2,save_dir_list):
            for img_path,json_path in zip(sep_img_list,sep_json_list):
                for path in [img_path,json_path]:
                    new_path = os.path.join(save_dir,path.split('\\')[-1])
                    shutil.copy(path,new_path)

        # for img_path,json_path in zip(img_list[cut_num:],json_list[cut_num:]):
        #     for path in [img_path,json_path]:
        #         new_path = os.path.join(save_dir_list[1],path.split('\\')[-1])
        #         shutil.copy(path,new_path)

def get_classname_id_color(source,print_out=False,save_dir=None):
    class_names = []
    class_name2id = dict()
    id2class_name = dict()
    id2color = dict()
    msg_list = []
    status = 0

    if isinstance(source,dict):
        status = 1
    elif isinstance(source,list):
        source_list = source
        status = 2
    else:
        source_list = [source]
        status = 2

    if status > 0:
        colormap = imgviz.label_colormap()

    if status == 1:
        id2class_name = source.copy()
        for class_id, class_name in source.items():
            class_names.append(class_name)
            class_name2id[class_name] = class_id
            id2color[class_id] = colormap[class_id].tolist()
    elif status == 2:
        label_path = None
        for source in source_list:
            if os.path.isdir(source):
                label_path = os.path.join(source, 'classnames.txt')
            elif os.path.isfile(source):
                label_path = source

            if os.path.exists(label_path):
                break
            else:
                label_path = None

        if label_path is None:
            say_sth("????????????????????????????????????")
        else:
            if save_dir is None:
                save_dir = os.path.dirname(label_path)
            for i, line in enumerate(open(label_path).readlines()):
                class_id = i - 1  # starts with -1
                class_name = line.strip()

                if class_id == -1:
                    assert class_name == "__ignore__"
                    continue
                elif class_id == 0:
                    assert class_name == "_background_"
                    # continue
                class_names.append(class_name)
                class_name2id[class_name] = class_id

            for key, value in class_name2id.items():
                id2class_name[value] = key
                id2color[value] = colormap[value].tolist()
        # label_path = os.path.join(root_dir, 'classnames.txt')
        # if not os.path.exists(label_path):
        #     say_sth("???????????????:{}".format(label_path),print_out=print_out)
        # else:
        #     for i, line in enumerate(open(label_path).readlines()):
        #         class_id = i - 1  # starts with -1
        #         class_name = line.strip()
        #
        #         if class_id == -1:
        #             assert class_name == "__ignore__"
        #             continue
        #         elif class_id == 0:
        #             assert class_name == "_background_"
        #             # continue
        #         class_names.append(class_name)
        #         class_name2id[class_name] = class_id
        #
        #     for key, value in class_name2id.items():
        #         id2class_name[value] = key
        #         id2color[value] = colormap[value].tolist()

    if len(class_names) > 0:
        msg_list.append("class_names:{}".format(class_names))
        msg_list.append("class_name_to_id:{}".format(class_name2id))
        for key, value in class_name2id.items():
            msg_list.append("class name:{}, id:{}, color:{}".format(key, value, id2color[value]))

        for msg in msg_list:
            say_sth(msg,print_out=print_out)

        #----draw class name to color index image
        unit = 60
        gap = unit // 2
        qty = len(class_names)
        height = qty* unit + (qty + 4) * gap
        width = unit * 12
        img_index = np.ones([height,width,3],dtype=np.uint8) * 255
        for idx,class_name in enumerate(class_names):
            s_point = (10,unit * 1 + idx * unit + gap * (idx + 1))
            e_point = (s_point[0] + unit * 5,s_point[1] + unit)
            color = []
            for v in colormap[idx]:
                color.append(int(v))#color?????????????????????int or float
            # color = tuple(color)#color??????????????????tuple?????????list?????????
            # print(color)
            cv2.rectangle(img_index,s_point,e_point,color,-1)
            #----add color map
            cv2.putText(img_index, str(color), (s_point[0], s_point[1] + unit//2), cv2.FONT_HERSHEY_PLAIN, 1.5,
                        (255, 255, 255), 2)
            #----add class name text
            cv2.putText(img_index,class_name,(e_point[0]+unit//5,e_point[1]-unit//3),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)


        plt.imshow(img_index)
        plt.axis('off')
        # plt.show()
        if save_dir is not None:
            save_path = os.path.join(save_dir,'label_index.jpg')
            plt.savefig(save_path)
            print("label index image is saved in {}".format(save_path))



    return class_names,class_name2id,id2class_name,id2color

def dict_transform(ori_dict,set_key=False,set_value=False):
    new_dict = dict()
    for key,value in ori_dict.items():
        if set_key:
            key = int(key)
        if set_value:
            value = int(value)
        new_dict[key] = value

    return new_dict

def label_uniqueNum_check(json_dir,cls_name_path,resize,angle=None):

    class_names, class_name2id, id2class_name, id2color = get_classname_id_color(cls_name_path, print_out=True,
                                                                                 save_dir=None)

    count = 0
    tl = tools(print_out=True)
    paths = [file.path for file in os.scandir(json_dir) if file.name.split(".")[-1] == 'json']
    qty = len(paths)

    if len(paths) == 0:
        print("No json files")
    else:
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                content = json.load(f)
                h = content['imageHeight']
                w = content['imageWidth']

            label_shapes = tl.get_label_shapes(path)
            if label_shapes is None:
                continue
            try:
                lbl = tl.shapes_to_label(
                    img_shape=[h, w, 3],
                    shapes=label_shapes,
                    label_name_to_value=class_name2id,
                )
                num_list_1 = list(np.unique(lbl))
                # ----rotate
                if angle is not None:
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    lbl = cv2.warpAffine(lbl, M, (w, h), borderMode=cv2.BORDER_REPLICATE,borderValue=None,flags=cv2.INTER_NEAREST)

                    '''cv2.BORDER_WRAP
                    cv2.BORDER_DEFAULT
                    BORDER_REFLECT
                    BORDER_CONSTANT
                    BORDER_ISOLATED
                    BORDER_TRANSPARENT
                    '''

                lbl_r = cv2.resize(lbl, resize, interpolation=cv2.INTER_NEAREST)

                num_list_2 = list(np.unique(lbl_r))

                if num_list_1 == num_list_2:
                    count += 1
                elif len(num_list_2) < len(num_list_1):  # ???????????????resize????????????????????????pixel????????????
                    wrong_flag = False
                    for num in num_list_2:
                        if num not in num_list_1:
                            wrong_flag = True
                            break
                    if wrong_flag is False:
                        count += 1
                else:
                    print("before unique numbers:", num_list_1)
                    print("after unique numbers:", num_list_2)
            except:
                print("label Error:", path)

    print("unique numbers identical ratio:", count / qty)

class NormDense(tf.keras.layers.Layer):

    def __init__(self, feature_num, classes=1000, output_name=''):
        super(NormDense, self).__init__()
        self.classes = classes
        self.w = self.add_weight(name='norm_dense_w', shape=(feature_num, self.classes),
                                 initializer='random_normal', trainable=True)
        self.output_name = output_name
        print("W shape = ", self.w.shape)

    # def build(self, input_shape):
    #     self.w = self.add_weight(name='norm_dense_w', shape=(input_shape[-1], self.classes),
    #                              initializer='random_normal', trainable=True)

    def call(self, inputs, **kwargs):
        norm_w = tf.nn.l2_normalize(self.w, axis=0)
        x = tf.matmul(inputs, norm_w, name=self.output_name)

        return x

if __name__ == "__main__":
    #----dataloader
    img_dir = r"D:\dataset\optotech\009IRC-FB\20220720_4SEG_Train\img_dir\val\L2_OK_line_pattern"
    tl = tools(print_out=True)
    count = 0
    paths,qty = tl.get_paths(img_dir)
    d_loader = DataLoader(paths,batch_size=7,shuffle=True)
    for i in range(d_loader.iterations):
        batch_paths = d_loader.__next__()
        for path in batch_paths:
            print(path)
            count += 1
    print(i)
    print(count)
    #----contrast_and_brightness_compare
    # img_dir = r"D:\dataset\optotech\silicon_division\PDAP\??????_?????????_particle\ori\L1_particle"
    # paths = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] == 'jpg']

    # contrast_and_brightness_compare(paths,5,br=64,ct=64,standard=64)

    #----file transfer
    # path = r"C:\Users\User\Desktop\train_result\train_result_0.nst"
    # file_transfer(path, cut_num_range=30, random_num_range=10)

    #----file decode
    # path = r"D:\code\model_saver\AE_Seg_132\train_result_0.nst"
    # file_decode_v2(path,random_num_range=10)

    #----tools
    # img_dir = r"D:\dataset\optotech\silicon_division\PDAP\20220517_PDAP_??????0.0.2Data\????????????"
    # t1 = tools(print_out=True)
    # paths,json_paths,qty = t1.get_subdir_paths_withJsonCheck(img_dir)
    # rdms = np.random.randint(0,qty,5)
    # for rdm in rdms:
    #     print(paths[rdm])
    #     print(json_paths[rdm])
    #
    # #----data distribution(with label files)
    # save_dir_list = [r"D:\dataset\optotech\silicon_division\PDAP\20220517_PDAP_??????0.0.2Data\????????????\train",
    #                  r"D:\dataset\optotech\silicon_division\PDAP\20220517_PDAP_??????0.0.2Data\????????????\test"]
    # data_distribution(paths, save_dir_list,json_list=json_paths, ratio=None,select_num=232)

    # ----data distribution(without label files)
    # img_dir = r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\validation\L2_OK_?????????"
    # t1 = tools(print_out=True)
    # paths,qty = t1.get_paths(img_dir)
    # #
    # save_dir_list = [r"D:\dataset\optotech\009IRC-FB\AE\train",
    #                  r"D:\dataset\optotech\009IRC-FB\AE\test"]
    # data_distribution(paths, save_dir_list, json_list=None, ratio=0.8)


    #----get classnames id and color
    # root_dir = r"D:\dataset\optotech\009IRC-FB\classnames.txt"
    # root_dir = r"D:\dataset\optotech\silicon_division\PDAP\?????????(??????+color???+json???)\L2_potion\classnames.txt"
    # class_names,class_name2id,id2class_name,id2color = get_classname_id_color(root_dir,print_out=True,save_dir=None)

    #----create label png
    # tl = tools()
    # tl.class_name2id = class_name_to_id
    # tl.id2color = id_to_color
    # img_dir = r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al?????????\??????\19BR262E01\02\AOI NG???\??????????????????(???NG)"
    # tl.create_label_png(img_dir,save_type='ori+label')

    #----image mask
    # img_source = r"D:\dataset\optotech\silicon_division\PDAP\top_view\PD-55077GR-AP AI Training_2022.01.26\AE_results_544x832_diff10cc60_more_filters\pred_ng_ans_ok\t0_12_-20_MatchLightSet_?????????_Ng_1.jpg"
    # json_path = r"D:\dataset\optotech\silicon_division\PDAP\top_view\PD-55077GR-AP AI Training_2022.01.26\AE_results_544x832_diff10cc60_more_filters\pred_ng_ans_ok\t0_12_-20_MatchLightSet_?????????_Ng_1.json"
    # img_mask(img_source, json_path,zoom_in_value=[75,77,88,88], img_type='path')

    #----check results
    # dir_path = r"D:\code\model_saver\AE_Seg_137"
    # dir_path = r"C:\Users\User\Desktop\train_result"
    # dir_path = r"D:\code\model_saver\Opto_tech\CLS_PDAP_defect_20220112"
    # only2see = ['seg_test_defect_sensitivity_list']
    # check_results(dir_path, encript_flag=False,epoch_range=None,only2see=None)

    #----result comparison
    # result_dict = {
    #     'AE_Seg_137': r'D:\code\model_saver\AE_Seg_137',
    #     'AE_Seg_139': r'D:\code\model_saver\AE_Seg_139',
    # }
    # # extract_name_list = ['train_loss_list', 'test_loss_list']
    # extract_name_list = ['seg_test_defect_recall_list', 'seg_test_defect_sensitivity_list']
    # y_axis_list = ['loss', 'loss']
    # result_comparison(result_dict, extract_name_list, y_axis_list, encript_flag=False)


    #----get_paths_labels
    # img_dir = r"C:\Users\User\Desktop\file_test"
    # label_dict = {"NG":0,"OK":1,"gg":2}

    # paths, labels, stat_dict = get_paths_labels(img_dir ,label_dict ,type='extend')
    # for key,value in stat_dict.items():
    #     if value == 0:
    #         print("??????{}?????????0".format(key))

    #----test code
    # img_dir = r"D:\dataset\optotech\silicon_division\PDAP\top_view\PD-55077GR-AP AI Training_2022.01.26\ori\OK(??????OK)\19BR262E02 ??????OK"
    # save_dir = r"D:\dataset\optotech\silicon_division\PDAP\??????_?????????_particle\ok_parts_test"
    #
    # paths = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] == 'jpg']
    #
    # s_paths = np.random.choice(paths,size=100,replace=False)
    #
    # for path in s_paths:
    #     ext = path.split(".")[-1]
    #     json_path = path.strip(ext) + 'json'
    #     if os.path.exists(json_path):
    #         for ori_path in [path,json_path]:
    #             new_path = ori_path.split("\\")[-1]
    #             new_path = os.path.join(save_dir,new_path)
    #             shutil.copy(ori_path,new_path)
    #     else:
    #         new_path = path.split("\\")[-1]
    #         new_path = os.path.join(save_dir, new_path)
    #         shutil.copy(path, new_path)

    #----check label
    # json_dir = r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\AE_Seg\Seg\train"
    # cls_name_path = r"D:\dataset\optotech\009IRC-FB\classnames.txt"
    # json_dir = r"D:\dataset\optotech\silicon_division\PDAP\??????_?????????_particle\train"
    # cls_name_path = r"D:\dataset\optotech\silicon_division\PDAP\??????_?????????_particle\classnames.txt"
    # resize = (192, 192)
    # angle = 60
    # label_uniqueNum_check(json_dir,cls_name_path,resize,angle=angle)










