import numpy as np
import matplotlib.pyplot as plt
import cv2,os,math

print_out = True
TCPConnected = False
img_format = {'png','PNG','jpg','JPG','JPEG','bmp','BMP'}

class RandomBrightnessContrast():
    def __init__(self,br_ratio=0.05,ct_ratio=0.1,p=0.5,print_out=False):
        self.br_ratio=br_ratio
        self.ct_ratio=ct_ratio
        self.p=p
        self.print_out= print_out

    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        rdm_list = np.random.random(size=3)

        if rdm_list[0] >= (1 - self.p):
            br_ratio = self.br_ratio * rdm_list[1]
            ct_ratio = self.ct_ratio * rdm_list[2]
            ct_ratio = math.tan((45 + 44 * ct_ratio) / 180 * math.pi)

            img = img.astype(np.float32)
            img /= 255
            # ct_ratio *= 47
            # ct_ratio += 1.0
            # print("br ratio:{}, ct ratio:{}".format(br_ratio,ct_ratio))
            ave_pixel = np.mean(img)
            img = (img - ave_pixel * (1 - br_ratio)) * ct_ratio + ave_pixel * (1 + br_ratio)

            img *= 255
            img = cv2.convertScaleAbs(img)
            # img = np.clip(img, 0, 255)
            # img = img.astype(np.uint8)
            msg = "br_ratio:{},ct_ratio:{}".format(br_ratio, ct_ratio)
        else:
            msg = "no process"

        if self.print_out:
            print(msg)

        if label is None:
            return img
        else:
            return img,label

class BrightnessContrast():
    def __init__(self,br_ratio=0.05,ct_ratio=0.1):
        self.br_ratio = br_ratio
        self.ct_ratio = math.tan((45 + 44 * ct_ratio) / 180 * math.pi)

    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        img = img.astype(np.float32)
        img /= 255

        # self.ct_ratio *= 47
        # self.ct_ratio += 1.0
        # print("br ratio:{}, ct ratio:{}".format(br_ratio,ct_ratio))
        ave_pixel = np.mean(img)
        img = (img - ave_pixel * (1 - self.br_ratio)) * self.ct_ratio + ave_pixel * (1 + self.br_ratio)

        img *= 255
        img = cv2.convertScaleAbs(img)
        # img = np.clip(img, 0, 255)
        # img = img.astype(np.uint8)

        if label is None:
            return img
        else:
            return img,label

class RandomBlur():
    def __init__(self,p=0.5):
        self.p=p
        self.kernel_list = [3,3,5,5,7,3,3,5,5,7]

    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        if np.random.random() >= (1 - self.p):
            kernel = tuple(np.random.choice(self.kernel_list, size=2))
            # print("kernel:", kernel)
            if np.random.randint(0, 2) == 0:
                img = cv2.blur(img, kernel)
            else:
                img = cv2.GaussianBlur(img, kernel, 0, 0)


        if label is None:
            return img
        else:
            return img,label

class RandomHorizontalFlip():
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        if np.random.random() >= (1 - self.p):
            img = cv2.flip(img, 1)
            if label is not None:
                label = cv2.flip(label, 1)

        if label is None:
            return img
        else:
            return img,label

class RandomVerticalFlip():
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        if np.random.random() >= (1 - self.p):
            img = cv2.flip(img, 0)
            if label is not None:
                label = cv2.flip(label, 0)

        if label is None:
            return img
        else:
            return img,label

class RandomRotation():
    def __init__(self,degrees=3,p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        if np.random.random() >= (1 - self.p):
            angle = np.random.randint(-self.degrees, self.degrees)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            if label is not None:
                label = cv2.warpAffine(label, M, (w, h),
                                       borderMode=cv2.BORDER_REPLICATE,
                                       flags=cv2.INTER_NEAREST)

        if label is None:
            return img
        else:
            return img, label

class RandomCrop():
    def __init__(self,height=300,width=300):
        self.size=(width,height)
        self.height = height
        self.width = width

    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        h,w = img.shape[:2]
        h_start = 0
        w_start= 0
        corner = np.random.randint(4)
        if h > self.height:
            h_start = np.random.randint((h - self.height))
        if w > self.width:
            w_start = np.random.randint((w - self.width))

        img = self.cut(img,h_start,w_start,corner)
        if label is not None:
            label = self.label_cut(label, h_start, w_start, corner)

        if label is None:
            return img
        else:
            return img,label

    def cut(self,data,h_s,w_s,corner):
        if corner == 0:
            data = data[h_s:h_s+self.height, w_s:w_s+self.width, :]
        elif corner == 1:
            data = data[h_s:h_s+self.height, -(w_s + 1)-self.width:-(w_s + 1), :]
        elif corner == 2:
            data = data[-(h_s + 1)-self.height:-(h_s + 1), w_s:w_s+self.width, :]
        elif corner == 3:
            data = data[-(h_s + 1)-self.height:-(h_s + 1), -(w_s + 1)-self.width:-(w_s + 1), :]
        else:
            data = data[h_s:h_s+self.height, w_s:w_s+self.width, :]

        return data

    def label_cut(self,data,h_s,w_s,corner):
        if corner == 0:
            data = data[h_s:h_s+self.height, w_s:w_s+self.width]
        elif corner == 1:
            data = data[h_s:h_s+self.height, -(w_s + 1)-self.width:-(w_s + 1)]
        elif corner == 2:
            data = data[-(h_s + 1)-self.height:-(h_s + 1), w_s:w_s+self.width]
        elif corner == 3:
            data = data[-(h_s + 1)-self.height:-(h_s + 1), -(w_s + 1)-self.width:-(w_s + 1)]
        else:
            data = data[h_s:h_s+self.height, w_s:w_s+self.width]


        return data

class RandomPNoise():
    def __init__(self,area_range=[30,1000],defect_num=5,pixel_range=[200,250],
                 noise_img_dir=r".\perlin_noise",color_img_dir=r".\car",zoom_in=0,p=0.5,label_class=1):

        self.paths_noise = [file.path for file in os.scandir(noise_img_dir) if file.name.split(".")[-1] == 'png']
        # qty_n = len(paths_noise)
        # msg = "Perlin noise image qty:{}".format(qty_n)
        # say_sth(msg, print_out=print_out)

       #----read nature paths
        self.paths_color = [file.path for file in os.scandir(color_img_dir) if
                       file.name.split(".")[-1] in img_format]
        # qty_c = len(paths_color)
        # msg = "Color image qty:{}".format(qty_c)
        # say_sth(msg, print_out=print_out)


        #----set local var to global
        self.area_range = area_range
        self.defect_num = defect_num
        self.pixel_range = pixel_range
        self.p = p
        self.zoom_in = zoom_in
        self.label_class = label_class

    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        if np.random.random() >= (1 - self.p):
            size = img.shape[:2][::-1]

            img_perlin = read_img(np.random.choice(self.paths_noise),size=size,to_gray=True)
            # ----zoom in
            img_perlin = img_room_in(img_perlin,self.zoom_in)

                # zeros = np.zeros_like(img_perlin)
                # v = self.zoom_in
                # zeros[v:-v, v:-v] = img_perlin[v:-v, v:-v]
                # img_perlin = zeros.copy()
            # img_perlin = np.fromfile(np.random.choice(self.paths_noise), dtype=np.uint8)
            # img_perlin = cv2.imdecode(img_perlin, 0)
            # img_perlin = cv2.resize(img_perlin, size[::-1])

            #----read nature images
            img_nature = read_img(np.random.choice(self.paths_color),size=size)
            # img_nature = np.fromfile(np.random.choice(self.paths_color), dtype=np.uint8)
            # img_nature = cv2.imdecode(img_nature, 1)  # BGR format
            # img_nature = cv2.resize(img_nature, size[::-1])

            # if self.mode == 'nature':
            #     img_nature = np.fromfile(np.random.choice(self.paths_color), dtype=np.uint8)
            #     img_nature = cv2.imdecode(img_nature, 1)  # BGR format
            #     img_nature = cv2.resize(img_nature, size[::-1])
            # else:
            #     a = np.random.randint(self.pixel_range[0], high=self.pixel_range[1], size=size, dtype=np.uint8)
            #     img_nature = np.stack([a, a, a], axis=-1)

            label_num, label_map, stats, centroids = cv2.connectedComponentsWithStats(img_perlin, connectivity=8)
            areas = stats.T[-1]
            b = np.where(areas >= self.area_range[0], areas, self.area_range[1] + 3)
            index_list = np.where(b < self.area_range[1])  # index_list就是符合pixel_range的index
            zeros = np.zeros_like(img_perlin)
            defect_num = np.minimum(len(index_list[0]), self.defect_num)
            # print("實際可執行的defect_num:", defect_num)
            # print("符合pixel_range的數量:", len(index_list[0]))
            for idx in np.random.choice(index_list[0], defect_num, replace=False):
                coors = np.where(label_map == idx)  # coors會是tuple，所以無法使用list extend取得所有座標
                zeros[coors] = 255
                if label is not None:
                    label[coors] = self.label_class

            defect = cv2.bitwise_and(img_nature, img_nature, mask=zeros)
            img_lack = cv2.bitwise_and(img, img, mask=255 - zeros)

            img = img_lack + defect
            img = cv2.convertScaleAbs(img)
            # plt.imshow(img_result[:,:,::-1])
            # plt.show()

        if label is None:
            return img
        else:
            return img,label

class RandomDefect():
    def __init__(self,area_range=[30,1000],defect_num=5,pixel_range=[200,250],
                 noise_img_dir=r".\perlin_noise",zoom_in=0,p=0.5,label_class=1):

        paths_noise = [file.path for file in os.scandir(noise_img_dir) if file.name.split(".")[-1] == 'png']
        qty_n = len(paths_noise)
        msg = "Perlin noise image qty:{}".format(qty_n)
        say_sth(msg, print_out=print_out)

        #----set local var to global
        self.paths_noise = paths_noise
        self.area_range = area_range
        self.defect_num = defect_num
        self.pixel_range = pixel_range
        self.zoom_in = zoom_in
        self.p = p
        self.label_class = label_class

    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        if np.random.random() >= (1 - self.p):
            size = img.shape[:2]

            img_perlin = read_img(np.random.choice(self.paths_noise), size=size[::-1], to_gray=True)

            #----zoom in
            img_perlin = img_room_in(img_perlin,self.zoom_in)

            img_light_defect = np.ones_like(img)
            rdm_number = np.random.randint(self.pixel_range[0], high=self.pixel_range[1], dtype=np.uint8)
            img_light_defect *= rdm_number
            # a = np.random.randint(self.pixel_range[0], high=self.pixel_range[1], size=size, dtype=np.uint8)
            # img_light_defect = np.stack([a, a, a], axis=-1)


            label_num, label_map, stats, centroids = cv2.connectedComponentsWithStats(img_perlin, connectivity=8)
            areas = stats.T[-1]
            b = np.where(areas >= self.area_range[0], areas, self.area_range[1] + 3)
            index_list = np.where(b < self.area_range[1])  # index_list就是符合pixel_range的index
            zeros = np.zeros_like(img_perlin)
            defect_num = np.minimum(len(index_list[0]), self.defect_num)
            # print("實際可執行的defect_num:", defect_num)
            # print("符合pixel_range的數量:", len(index_list[0]))
            for idx in np.random.choice(index_list[0], defect_num, replace=False):
                coors = np.where(label_map == idx)  # coors會是tuple，所以無法使用list extend取得所有座標
                zeros[coors] = 255
                if label is not None:
                    label[coors] = self.label_class

            defect = cv2.bitwise_and(img_light_defect, img_light_defect, mask=zeros)
            img_lack = cv2.bitwise_and(img, img, mask=255 - zeros)

            img = img_lack + defect
            img = cv2.convertScaleAbs(img)


            # img = img.astype(np.int32)
            # if np.random.random() >= 0.5:
            #     img = img - defect
            #     # print("subtract")
            # else:
            #     img = img + defect
            #
            # img = cv2.convertScaleAbs(img)
            # img = np.clip(img,0,255)
            # img = img.astype(np.uint8)
            # plt.imshow(img[:,:,::-1])
            # plt.show()


        if label is None:
            return img
        else:
            return img,label

class RandomVividDefect():
    def __init__(self,defect_num=5,rotation_degrees=20,resize_ratio=20,lower_br_ratio=20,
                 ct_ratio=20,defect_png_dir=r".\defects",zoom_in=[150,150,100,100],#[x_s,x_end,y_s,y_end]
                 margin=5,p=0.5,homogeneity_threshold=5,label_class=1,print_out=False):

        self.paths = [file.path for file in os.scandir(defect_png_dir) if file.name.split(".")[-1] == 'png']
        # qty_n = len(paths_noise)
        # msg = "Perlin noise image qty:{}".format(qty_n)
        # say_sth(msg, print_out=print_out)

       #----read nature paths
        # self.paths_color = [file.path for file in os.scandir(color_img_dir) if
        #                file.name.split(".")[-1] in img_format]
        # qty_c = len(paths_color)
        # msg = "Color image qty:{}".format(qty_c)
        # say_sth(msg, print_out=print_out)


        #----set local var to global
        # self.area_range = area_range
        self.w_mar_zoom = margin + zoom_in[0]
        self.h_mar_zoom = margin + zoom_in[2]
        self.ct_ratio = ct_ratio
        self.defect_num = defect_num
        self.br_ratio = lower_br_ratio
        self.rot_degrees = rotation_degrees
        self.resize_ratio = resize_ratio
        self.margin = margin
        self.p = p
        self.zoom_in = zoom_in
        self.label_class = label_class
        self.kernel_list = [1,3,5,7]
        self.homo_threshold = homogeneity_threshold
        self.print_out = print_out

    def img_png_process(self,img_p):
        img_p = rdm_horizon_flip(img_p, p=self.p)
        img_p = rdm_vertical_flip(img_p, p=self.p)
        img_p = rdm_contrast(img_p, self.ct_ratio, p=self.p, print_out=self.print_out)
        img_p = rdm_blur(img_p, self.kernel_list, p=self.p, print_out=self.print_out)
        img_p = rdm_lower_brightness(img_p, br_ratio=self.br_ratio, p=self.p, print_out=self.print_out)
        img_p = rdm_rotation(img_p, degrees=self.rot_degrees, p=self.p, print_out=self.print_out)
        img_p = rdm_resize(img_p, resize_ratio=self.resize_ratio, p=self.p, print_out=self.print_out)

        return img_p

    def homogeneity_check(self,roi,homo_threshold=1.0,print_out=False):
        if print_out:
            msg_list = []

        if isinstance(homo_threshold, float) or isinstance(homo_threshold, int):
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            std = np.std(roi_gray)

            if std > homo_threshold:
                status = False
            else:
                status = True

            if print_out:
                msg_list.append(f"The homogeneity of roi:{std}")
        else:
            status = True

        if print_out:
            for msg in msg_list:
                print(msg)

        return status

    def coors_overlap_check(self,h_start,h_end,w_start,w_end,coors_list,print_out=False):
        status = False
        if print_out:
            msg_list = []
        #----
        if len(coors_list) == 0:
            status = False
        else:
            for coors in coors_list:
                range_h = range(coors[0], coors[1])
                range_w = range(coors[2], coors[3])
                h_status = False
                w_status = False
                if h_start in range_h or h_end in range_h:
                    h_status = True
                if w_start in range_w or w_end in range_w:
                    w_status = True

                if h_status and w_status:
                    status = True
                    if print_out:
                        msg_list.append("New coors are overlapped")
                    break
                else:
                    status = False

        return status

    def label_overlap_check(self,roi_label):
            status = False

            coors = np.where(roi_label > 0)
            if len(coors[0]) > 0:
                status = True

            return status

    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        #---read defect images
        img_png_list = []
        paths = np.random.choice(self.paths,self.defect_num,replace=False)

        for path in paths:
            img_bgra = np.fromfile(path, dtype=np.uint8)
            img_bgra = cv2.imdecode(img_bgra, -1)

            # ----defect image pre-process
            # img_p = img_bgra.copy()
            # img_p = self.img_png_process(img_p)
            img_png_list.append(self.img_png_process(img_bgra))

        #-----start points
        h_ok, w_ok = img.shape[:-1]

        w_ok_zoom = w_ok - self.zoom_in[1]
        h_ok_zoom = h_ok - self.zoom_in[3]
        coors_list = []

        #----defect paste process
        for img_png in img_png_list:
            img_de = img_png[:, :, :-1]
            mask = img_png[:, :, -1]
            h_de, w_de = img_png.shape[:-1]
            w_ok_zoom_limit = w_ok_zoom - w_de
            h_ok_zoom_limit = h_ok_zoom - h_de
            count = 0
            go_flag = False
            while(go_flag is False):

                #----get start points
                w_start = np.random.randint(self.w_mar_zoom, w_ok_zoom_limit)
                h_start = np.random.randint(self.h_mar_zoom, h_ok_zoom_limit)

                w_end = w_start + w_de
                h_end = h_start + h_de

                #----overlap check
                overlap_check = self.coors_overlap_check(h_start,h_end,w_start,w_end,coors_list,print_out=self.print_out)
                if overlap_check is True:
                    continue

                #----homogeneity check
                roi = img[h_start:h_end, w_start:w_end, :]
                homo_check = self.homogeneity_check(roi, homo_threshold=self.homo_threshold, print_out=self.print_out)
                if homo_check is False:
                    continue

                #----label roi class number check
                if label is None:
                    go_flag = True
                else:
                    roi_label = label[h_start:h_end, w_start:w_end]
                    label_overlap_check = self.label_overlap_check(roi_label)

                    if label_overlap_check:
                        if self.print_out:
                            print("label overlapped")
                        continue
                    else:
                        go_flag = True

                #----img combination
                if go_flag:
                    #----get roi region
                    not_mask = 255 - mask
                    # max_value_roi = np.max(roi)
                    roi = cv2.bitwise_and(roi, roi, mask=not_mask)

                    defect = cv2.bitwise_and(img_de, img_de, mask=mask)
                    #----find max values of 2 areas
                    # max_value_roi = np.max(roi)
                    # max_value_defect = np.max(defect)
                    # coors = np.where(defect > 0)
                    # ave_value_defect = np.mean(defect[coors])
                    # print("max_value_roi:",max_value_roi)
                    # print("max_value_defect:",max_value_defect)
                    # print("ave_value_defect:",ave_value_defect)
                    # if ave_value_defect < 127:
                    #     defect = defect.astype(np.float32)
                    #     defect = defect / max_value_defect * max_value_roi
                    #     defect = cv2.convertScaleAbs(defect)
                    #     print("dark edge process")
                    # else:
                    #     defect = defect.astype(np.float32)
                    #     defect = defect / defect[coors].min() * max_value_roi
                    #     defect = cv2.convertScaleAbs(defect)
                    #     print("light edge process")


                    # if max_value_roi < max_value_defect:
                    #     defect = defect.astype(np.float32)
                    #     defect = defect / max_value_defect * max_value_roi
                    #     defect = cv2.convertScaleAbs(defect)

                    patch = cv2.add(roi,defect)

                    #----blur
                    patch = cv2.GaussianBlur(patch, (3,3), 0, 0)

                    img[h_start:h_end, w_start:w_end, :] = patch
                    coors_list.append([h_start,h_end,w_start,w_end])
                    if label is not None:
                        if self.label_class != 0:
                            # roi_label = np.zeros((h_de,w_de),dtype=np.uint8)
                            roi_label = mask.copy()
                            roi_label = np.where(roi_label == 255,self.label_class,0)
                            # roi_label[coors_label] = self.label_class
                            label[h_start:h_end, w_start:w_end] = roi_label

                count += 1
                #print("count:",count)

                if count >= 100:
                    break

        if label is None:
            return img
        else:
            return img,label

class Resize():
    def __init__(self,height=300, width=300):
        self.size = (width,height)
        self.height = height
        self.width = width

    def resize_label(self,label):
        coor_dict = dict()
        h, w = label.shape
        for label_num in np.unique(label):
            if label_num != 0:
                # ----取出每一種label number的座標
                coor_dict[label_num] = np.where(label == label_num)

        for label_num in coor_dict.keys():
            # ----新建zeros陣列(shape是未resize的label map)
            z_temp = np.zeros_like(label)
            # ----將對應label number的座標處都填上1
            z_temp[coor_dict[label_num]] = 1

            # ----對z_temp進行resize(因為數值都是1，resize不會產生其他的數值)
            z_temp = cv2.resize(z_temp, self.size)
            # ----取出resize後，對應label number的座標值
            coor_dict[label_num] = np.where(z_temp == 1)

        z_temp = np.zeros([self.height, self.width], dtype=np.uint8)
        # print("z_temp shape:",z_temp.shape)
        for label_num in coor_dict.keys():
            z_temp[coor_dict[label_num]] = label_num
        return z_temp



    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        img = cv2.resize(img, self.size, interpolation=cv2.INTER_NEAREST)
        if label is not None:
            label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)
            # label = self.resize_label(label)

        if label is None:
            return img
        else:
            return img,label

class CvtColor():
    def __init__(self,**kwargs):
        self.to_rgb = kwargs.get('to_rgb')
        self.to_gray = kwargs.get('to_gray')

    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        if self.to_rgb:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif self.to_gray:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        if label is None:
            return img
        else:
            return img,label

class Norm():
    def __init__(self,dtype='float32'):
        self.dtype = dtype

    def __call__(self, *args, **kwargs):
        img = kwargs.get('img')
        label = kwargs.get('label')

        img = img.astype(self.dtype)
        img /= 255

        if label is None:
            return img
        else:
            return img,label

def say_sth(msg, print_out=False,header=None):
    if print_out:
        print(msg)
    # if TCPConnected:
    #     TCPClient.send(msg + "\n")

def defect_crop2png(img_dir,defect_type='dark',cc_threshold=20,save_dir=None,to_show=False,to_save=False):
    show_num = 4

    paths = [file.path for file in os.scandir(img_dir) if file.name.split(".")[-1] == 'bmp']

    if len(paths) > 0:
        if to_save:
            if save_dir is None:
                save_dir = os.path.join(os.path.dirname(img_dir), 'crop2png')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        for path in paths:
            img = np.fromfile(path, dtype=np.uint8)
            img_bgr = cv2.imdecode(img, 1)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            #----threshold
            mean = np.mean(img_gray)
            std = np.std(img_gray)
            cut_value = mean - 0 * std
            if defect_type == 'dark':
                method = cv2.THRESH_BINARY_INV
            else:
                method = cv2.THRESH_BINARY

            ret, mask = cv2.threshold(img_gray, cut_value, 255, method)

            #----connected components
            cc_mask = np.zeros_like(mask)
            label_num, label_map, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
            areas = stats.T[-1]
            # print("areas:",areas)
            for i in range(1, label_num):
                if stats[i][-1] > cc_threshold:
                    coors = np.where(label_map == i)
                    cc_mask[coors] = 255

            # ----display
            if to_show:
                plt.figure(figsize=(10, 10))
                plt.subplot(1, show_num, 1)
                plt.imshow(img_bgr[:, :, ::-1])

                plt.subplot(1, show_num, 2)
                plt.imshow(img_gray, cmap='gray')

                plt.subplot(1, show_num, 3)
                plt.imshow(mask, cmap='gray')

                plt.subplot(1, show_num, 4)
                plt.imshow(cc_mask, cmap='gray')

                plt.show()

            #----save
            if to_save:
                img_png = np.concatenate((img_bgr,np.expand_dims(cc_mask,axis=-1)),axis=-1)
                save_path = path.split("\\")[-1].split(".")[0]
                save_path += '.png'
                save_path = os.path.join(save_dir,save_path)
                cv2.imencode('.png', img_png)[1].tofile(save_path)

def rdm_contrast(img_png, ct_ratio, p=0.5,print_out=False):
    if np.random.random() >= (1 - p):
        mask = img_png[:, :, -1]

        ct_level = np.random.randint(0, ct_ratio) / 100
        msg = "Contrast ratio:{}".format(ct_level)
        ct_level = math.tan((45 + 44 * ct_level) / 180 * math.pi)

        img_png = img_png.astype(np.float32)
        ave_pixel = np.mean(img_png[:, :, :-1])

        msg_2 = "ave_pixel:{}".format(ave_pixel)

        img_png = (img_png - ave_pixel) * ct_level + ave_pixel
        img_png = cv2.convertScaleAbs(img_png)
        img_png[:, :, -1] = mask

        if print_out:
            print(msg)
            print(msg_2)
    return img_png

def rdm_blur(img_png, kernel_list, p=0.5, print_out=False):
    if np.random.random() >= (1 - p):
        mask = img_png[:, :, -1]

        kernel = tuple(np.random.choice(kernel_list, size=2))

        if np.random.randint(0, 2) == 0:
            img_png = cv2.blur(img_png, kernel)
        else:
            img_png = cv2.GaussianBlur(img_png, kernel, 0, 0)
        img_png[:, :, -1] = mask

        if print_out:
            print("Blur with kernel {}".format(kernel))

    return img_png

def rdm_lower_brightness(img_png, br_ratio=10, p=0.5, print_out=False):
    if np.random.random() >= (1 - p):
        mask = img_png[:, :, -1]
        br = np.random.randint(-br_ratio, 0)
        br = br / 100 + 1

        img_png = cv2.convertScaleAbs(img_png, alpha=br)
        img_png[:, :, -1] = mask

        if print_out:
            print("lower brightness:", br)

    return img_png

def rdm_rotation(img_png, degrees=10, p=0.5, print_out=False):
    if np.random.random() >= (1 - p):
        mask = img_png[:, :, -1]

        angle = np.random.randint(-degrees, degrees)
        h, w = img_png.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img_png = cv2.warpAffine(img_png, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        mask = cv2.warpAffine(mask, M, (w, h),
                              borderMode=cv2.BORDER_REPLICATE,
                              flags=cv2.INTER_NEAREST)

        img_png[:, :, -1] = mask

        if print_out:
            print("raondom rotation angle:", angle)

    return img_png

def rdm_resize(img_png, resize_ratio=10, p=0.5, print_out=False):
    if np.random.random() >= (1 - p):
        mask = img_png[:, :, -1]

        ratios = np.random.randint(-resize_ratio, resize_ratio)
        #         print(type(ratios))
        ratios /= 100
        ratios += 1

        img_png = cv2.resize(img_png, None, fx=ratios, fy=ratios)

        mask = cv2.resize(mask, None, fx=ratios, fy=ratios, interpolation=cv2.INTER_NEAREST)

        img_png[:, :, -1] = mask

        if print_out:
            print("raondom resize ratios:", ratios)

    return img_png

def rdm_horizon_flip(img_png, p=0.5, print_out=False):
    if np.random.random() >= (1 - p):
        img_png = cv2.flip(img_png, 1)

    return img_png

def rdm_vertical_flip(img_png, p=0.5, print_out=False):
    if np.random.random() >= (1 - p):
        img_png = cv2.flip(img_png, 0)

    return img_png

def read_img(path,size=None,to_rgb=False,to_gray=False):
    #size format (width,height)
    img = None
    if os.path.exists(path):
        img = np.fromfile(path,dtype=np.uint8)
        img = cv2.imdecode(img,1)

        if size is not None:
            img = cv2.resize(img,size)

        if to_rgb:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif to_gray:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def img_room_in(img,zoom_in):

    if zoom_in > 0:
        zeros = np.zeros_like(img)
        v = zoom_in
        if img.ndim == 3:
            zeros[v:-v, v:-v,:] = img[v:-v, v:-v,:]
        else:
            zeros[v:-v, v:-v] = img[v:-v, v:-v]

        img = zeros

    return img



process_dict = dict(RandomBrightnessContrast=RandomBrightnessContrast,
                    RandomBlur=RandomBlur,
                    RandomHorizontalFlip=RandomHorizontalFlip,
                    RandomVerticalFlip=RandomVerticalFlip,
                    RandomCrop=RandomCrop,
                    RandomPNoise=RandomPNoise,
                    RandomDefect=RandomDefect,
                    RandomVividDefect=RandomVividDefect,
                    RandomRotation=RandomRotation,
                    Resize=Resize,
                    CvtColor=CvtColor,
                    Norm=Norm,
                    BrightnessContrast=BrightnessContrast
                    )
def pipeline2transform(pipelines):
    transforms = []
    process_name_list = list(process_dict.keys())
    for pipeline in pipelines:
        if pipeline.get('type') in process_name_list:
            process_name = pipeline['type']
            param_dict = dict()
            for key, value in pipeline.items():
                if key != 'type':
                    param_dict[key] = value

            transforms.append(process_dict[process_name](**param_dict))

    return transforms





if __name__ == "__main__":
    # pipelines = [
    #     dict(type='CvtColor',to_rgb=True),
    #     dict(type='RandomBlur',p=0.5),
    #     dict(type='RandomBrightnessContrast',br_ratio=0.1,ct_ratio=0.1,p=0.5),
    #     dict(type='RandomHorizontalFlip',p=0.5),
    #     dict(type='RandomVerticalFlip',p=0.5),
    #     dict(type='Resize',size=(563, 915)),
    #     dict(type='RandomCrop',size=(512,832))
    # ]
    #
    # transforms = pipeline2transform(pipelines)

    #----
    img_dir = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\defects_but_ok_20220922\crops"
    to_show = False
    to_save = True
    defect_crop2png(img_dir, defect_type='dark', cc_threshold=20, save_dir=None,to_show=to_show,to_save=to_save)