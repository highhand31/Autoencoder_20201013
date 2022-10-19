import os,math,cv2,json,imgviz,shutil,uuid,PIL,time,re
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from image_process import process_dict
from AE_Seg_Util import DataLoader4Seg,get_paths







if __name__ == "__main__":
    # ----augmentation v2
    img_dir = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220901_AOI判定OK\OK(多區OK)\num_100"
    # img_dir = r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train'
    id2class_name = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\classnames.txt"
    # id2class_name = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\classnames_one_defect_class.txt"
    hole_dict = dict(
        defect_num=1,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train\defectCrop2png\hole",
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\hole"
        ],
        label_class=1, p=0.5,
    )
    gold_particle_dict = dict(
        defect_num=1,
        # rotation_degrees=20, resize_ratio=20, lower_br_ratio=10,
        # ct_ratio=20,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train\defectCrop2png\gold_particle",
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\gold_particle"
        ],
        # defect_png_dir=r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\defects_but_ok_20220922\crop2png",
        # defect_png_dir=r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\test\defectCrop2png\hole",
        # defect_png_dir=r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220408新增破洞+金顆粒 資料\1\defectCrop2png\gold_particle",
        # zoom_in=[15, 15, 10, 10],  # [x_s,x_end,y_s,y_end]
        label_class=2, p=0.5,
        # margin=5,homogeneity_threshold=1.0,  print_out=False
    )
    particle_dict = dict(
        defect_num=2,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train\defectCrop2png\particle",
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\particle"
        ],
        label_class=3, p=0.5,
    )
    defect_but_ok_dict = dict(
        defect_num=2,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\gold_residual_but_ok",
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\defect_but_ok"
        ],
        label_class=0, p=0.5,
    )
    defect_but_ok_2_dict = dict(
        defect_num=1,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\defect_parts\小紅點_淺瑕疵\defectCrop2png",
            # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\defect_but_ok"
        ],
        label_class=0, p=0.5,
    )
    area_range = [100, 1000]
    zoom_in = 70
    p = 1.0
    pipelines = [
        dict(type='CvtColor', to_rgb=True),
        dict(type='RandomBlur', p=p),
        dict(type='RandomVividDefect', **hole_dict),
        dict(type='RandomVividDefect', **gold_particle_dict),
        dict(type='RandomVividDefect', **particle_dict),
        dict(type='RandomVividDefect', **defect_but_ok_dict),
        dict(type='RandomVividLightDefect', **defect_but_ok_2_dict),

        dict(type='RandomBrightnessContrast', br_ratio=0.1, ct_ratio=0.3, p=p),
        # dict(type='BrightnessContrast', br_ratio=0.0, ct_ratio=0.3),
        # dict(type='RandomDefect', area_range=area_range, defect_num=3, pixel_range=[7, 12], zoom_in=zoom_in, p=p),
        # dict(type='RandomDefect', area_range=area_range, defect_num=1, pixel_range=[50, 70], zoom_in=zoom_in, p=p),
        # dict(type='RandomDefect', area_range=area_range, defect_num=3, pixel_range=[90, 120], zoom_in=zoom_in, p=p),
        # dict(type='RandomDefect', area_range=area_range, defect_num=1, pixel_range=[140, 160], zoom_in=zoom_in, p=p),
        # dict(type='RandomDefect', area_range=area_range, defect_num=1, pixel_range=[180, 200], zoom_in=zoom_in, p=p),
        # dict(type='RandomDefect', area_range=area_range, defect_num=1, pixel_range=[200,250], zoom_in=zoom_in, p=p,label_class=0),
        # dict(type='RandomPNoise', area_range=area_range, defect_num=3, zoom_in=zoom_in,p=p),

        # dict(type='RandomHorizontalFlip', p=p),
        # dict(type='RandomVerticalFlip', p=p),
        dict(type='RandomRotation', degrees=5, p=p),
        dict(type='Resize', height=530, width=860),
        # dict(type='Resize', height=512,width=832),
        dict(type='RandomCrop', height=512, width=832),
        # dict(type='Norm')
    ]

    show_num = 3
    to_display = True
    # special_process_list = ['rdm_patch', 'rdm_perlin', 'rdm_light_defect']
    paths, qty = get_paths(img_dir)
    print("qty:", qty)
    dataloader = DataLoader4Seg(paths, batch_size=show_num, pipelines=pipelines, to_shuffle=True)
    # tl = tools_v2(pipelines=pipelines,print_out=True)

    dataloader.get_classname_id_color(id2class_name, print_out=True)
    # paths,qty = tl.get_paths(img_dir)
    # tl.set_process(process_dict, setting_dict, print_out=print_out)
    i = 0
    for batch_paths, imgs, labels in dataloader:

        # dataloader.show_data_attributes()

        # ----display
        if to_display:
            i += 1
            combine_data = []

            for i in range(show_num):
                combine_data.append(dataloader.combine_img_label(imgs[i], labels[i]))
            plt.figure(figsize=(10, 10))
            for i in range(show_num):
                plt.subplot(2, show_num, i + 1)
                plt.imshow(imgs[i])
                plt.subplot(2, show_num, i + 1 + show_num)
                plt.imshow(combine_data[i])

            plt.show()