import cv2,time,os
import numpy as np
import matplotlib.pyplot as plt

img_format = {'png','PNG','jpg','JPG','JPEG','bmp','BMP'}

def diameter_detection(img ,thrd=[50 ,80] ,print_out=False) :
    msg_list = list()

    label_nums, label_map, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    if label_nums <= 5:
        num_list = [i for i in range(label_nums)]
    else:
        args = np.argsort(stats.T[-1])
        num_list = args[::-1][:5]
    msg_list.append("label_nums:{}".format(label_nums))
    msg_list.append("num_list:{}".format(num_list))

    for label_num in num_list:
        s = stats[label_num]
        msg = "label num {}: x {}, y {}, width {} height {} 連通數量 {}".format(label_num ,s[0] ,s[1] ,s[2] ,s[3] ,s[4])
        msg_list.append(msg)

        w = s[2]
        h = s[3]
        flag = 0
        if w >= thrd[0] and w <= thrd[1]:
            flag += 1
        if h >= thrd[0] and h <= thrd[1]:
            flag += 1
        if flag == 2:
            break
    if flag != 2:
        w = None
        h = None

    if print_out:
        for msg in msg_list:
            print(msg)

    return np.array([w ,h]) ,s[:4]

def cal_mean_std_range(data_dict, std_ratio=2):
    for key, value in data_dict.items():
        value = np.array(value)
        mean = value.mean()
        std = value.std()
        print("{} 計算數量: {}".format(key, len(value)))
        print("   ave:{},std:{}".format(mean, std))
        print("   range: {} ~ {}".format(mean - std_ratio * std, mean + std_ratio * std))

def appearance_detection(img_bgr, inner_d_th, outer_d_th, d_diff_th, w_h_ratio_th, data_dict=None,print_out=False):
    er_code = None
    msg = ''
    img_g_inv = None
    img_copy = None

    img_g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    #----binarization
    img_g_process = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3)

    # ----inner diameter detection
    inner_dia, inner_coors = diameter_detection(img_g_process, thrd=inner_d_th, print_out=print_out)
    if None in inner_dia:
        er_code = 0
        msg = "No inner diameter deteced"
    else:
        if data_dict is not None:
            data_dict['inner_w_h'].extend(inner_dia)
        #----outer diameter detection
        img_g_inv = 255 - img_g_process
        outer_dia, outer_coors = diameter_detection(img_g_inv, thrd=outer_d_th, print_out=print_out)
        if None in outer_dia:
            er_code = 1
            msg = "No outer diameter deteced"
        else:
            if data_dict is not None:
                data_dict['outer_w_h'].extend(outer_dia)

            #----draw rectangles
            img_copy = img_bgr.copy()
            #====inner
            p1 = (inner_coors[0], inner_coors[1])
            p2 = ((p1[0] + inner_coors[2]), (p1[1] + inner_coors[3]))
            cv2.rectangle(img_copy, p1, p2, (0, 0, 255), 1)
            #====outer
            p1 = (outer_coors[0], outer_coors[1])
            p2 = ((p1[0] + outer_coors[2]), (p1[1] + outer_coors[3]))
            cv2.rectangle(img_copy, p1, p2, (255, 255, 0), 1)

            #----diff of inner and outer D
            diff = outer_dia - inner_dia
            if data_dict is not None:
                data_dict['dia_diff'].append(diff)

            # ----deformation
            sort = np.where(diff > d_diff_th)
            outer_ratio = (outer_dia.max() / outer_dia.min()) - 1
            inner_ratio = (inner_dia.max() / inner_dia.min()) - 1
            if data_dict is not None:
                data_dict['w_h_ratio'].append(outer_ratio)
                data_dict['w_h_ratio'].append(inner_ratio)

            if len(sort[0]) > 0:
                er_code = 2
                msg = "d diff > {}".format(d_diff_th)
            elif outer_ratio > w_h_ratio_th:
                er_code = 3
                msg = "Over outer_ratio:{}".format(outer_ratio)
            elif inner_ratio > w_h_ratio_th:
                er_code = 3
                msg = "Over inner_ratio:{}".format(inner_ratio)
    #

    return er_code, msg, img_g_process, img_g_inv, img_copy

def run_main(img_dir,**kwargs):
    defaults = {'inner_d_th':[62, 72],'outer_d_th':[69, 77],'d_diff_th':7.81,'w_h_ratio_th':0.05,'resize':(192,192),
                'to_show':False,'print_out':False,"std_ratio":2}

    for key,value in kwargs.items():
        defaults[key] = value

    inner_d_th = defaults['inner_d_th']
    outer_d_th = defaults['outer_d_th']
    d_diff_th = defaults['d_diff_th']
    w_h_ratio_th = defaults['w_h_ratio_th']
    resize = defaults['resize']
    std_ratio = defaults['std_ratio']
    to_show = defaults['to_show']
    print_out = defaults['print_out']
    error_names = ['no_inner_D', 'no_outer_D', 'd_diff', 'w_h_ratio']
    errors = np.zeros([4], dtype=np.uint16)  # ['no_inner_D', 'no_outer_D','d_diff','w_h_ratio']
    data_dict = {"inner_w_h": [], "outer_w_h": [], 'dia_diff': [], 'w_h_ratio': []}
    path_OK_list = []
    show_num = 4


    d_t = time.time()
    paths = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] in img_format]
    qty_path = len(paths)

    print("讀到圖片張數:",qty_path)
    if qty_path == 0:
        raise ValueError

    for path in paths:
        img = np.fromfile(path, dtype=np.uint8)
        img_ori = cv2.imdecode(img, 1)
        img_gray = cv2.imdecode(img, 0)
        img_gray = cv2.resize(img_gray, resize)
        img_ori = cv2.resize(img_ori, resize)

        er_code, msg, img_g_b, img_g_b_inv, img_copy = appearance_detection(img_ori, inner_d_th, outer_d_th, d_diff_th,
                                                                 w_h_ratio_th,
                                                                 data_dict=data_dict,print_out=print_out)
        if er_code is None:
            path_OK_list.append(path)
        else:
            errors[er_code] += 1
            if print_out:
                print(msg)


            if to_show:
                plt.figure(num=1, figsize=(15, 15))
                plt.subplot(1, show_num, 1)
                plt.imshow(img_gray, cmap='gray')
                plt.title("ori")

                if img_g_b is not None:
                    plt.subplot(1, show_num, 2)
                    plt.imshow(img_g_b, cmap='gray')
                    plt.title("ori_gray_processed")

                if img_g_b_inv is not None:
                    plt.subplot(1, show_num, 3)
                    plt.imshow(img_g_b_inv, cmap='gray')
                    plt.title("ori_gray_processed_inverse")

                if img_copy is not None:
                    plt.subplot(1, show_num, 4)
                    plt.imshow(img_copy[:, :, ::-1])
                    plt.title("rec")
                plt.show()
                # to_show = False
    d_t = time.time() - d_t
    # ----statistics
    for path in path_OK_list:
        print("判定OK:", path.split("\\")[-1])
    print("圖片總數:", len(paths))
    for name, er_qty in zip(error_names, errors):
        print("瑕疵名稱:{}:檢測數量{}".format(name, er_qty))
    print("error ratio = ", errors.sum() / len(paths))

    print("ave time:", d_t / qty_path)
    cal_mean_std_range(data_dict, std_ratio=std_ratio)

if __name__ == "__main__":
    img_dir = r"D:\dataset\optotech\009IRC-FB\0.0.3.1_dataset\training\L2_OK_無分類"
    inner_d_th = [60, 74]
    outer_d_th = [67, 79]
    d_diff_th = 7.95#設定1倍std
    w_h_ratio_th = 0.06
    resize = (192,192)
    std_ratio = 2
    to_show = True
    print_out = False

    run_main(img_dir,inner_d_th=inner_d_th,outer_d_th=outer_d_th,d_diff_th=d_diff_th,w_h_ratio_th=w_h_ratio_th,
             resize=resize,std_ratio=std_ratio,to_show=to_show,print_out=print_out)