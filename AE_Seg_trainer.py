from AE_Seg import AE_Seg
# from AE_Seg import AE_Seg_v2

def get_AE_arg(**kwargs):
    p = 0.6
    # name_list = ['train_img_dir','test_img_dir','recon_img_dir','model_shape']
    # ----AE
    ae_var = dict()
    # ae_var['train_img_dir'] = [
    #     r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E06\train\OK",
    # ]
    #
    # ae_var['test_img_dir'] = [
    #     r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E06\test\OK",
    # ]
    #
    # ae_var['recon_img_dir'] = r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E06\recon_img"
    # # ====model init
    # ae_var['model_shape'] = [None, 512, 832, 3]#[None,544,832,3]
    ae_var['infer_method'] = "AE_pooling_net_V7"#"AE_pooling_net_V7"  # "AE_JNet"#"AE_transpose_4layer",mit_b0
    # ae_var['infer_method'] = "type_5_7"
    kernel_list = [7, 5, 5, 5, 3, 3, 3]
    filter_list = [16, 24, 32, 40, 48, 56, 64]
    stride_list = [2] * 7
    layer_num = 6
    multi_ratio = 1.5
    ae_var['encode_dict'] = {
        'kernel_list': kernel_list,
        'filter_list': filter_list,
        'stride_list': stride_list,
        'pool_type_list': ['cnn','ave'],#'cnn'
        'pool_kernel_list': kernel_list,
        'layer_list': [6],
        'multi_ratio':multi_ratio,
    }
    ae_var['decode_dict'] = {
        'pool_type_list': ['cnn','ave'],
        'cnn_type':'resnet',
        'kernel_list': kernel_list[::-1],
        'filter_list': filter_list[::-1],
        'stride_list': stride_list,
        'multi_ratio': multi_ratio,
    }
    ae_var['to_reduce'] = False
    ae_var['rot'] = False
    # ae_var['kernel_list'] = [7, 5, 5, 3, 3]
    # ae_var['filter_list'] = [16, 32, 48, 64, 128]
    # ae_var['conv_time'] = 1
    # ae_var['embed_length'] = 144
    # ae_var['scaler'] = 1
    # ae_var['pool_type'] = ['max','ave']  # ['max']
    # ae_var['pool_kernel'] = [7, 2]  # [7, 2]
    ae_var['activation'] = 'relu'
    ae_var['loss_method'] = "ssim"
    ae_var['opti_method'] = "adam"

    # ====train
    # rdm_patch = [0.1, 0.1, 10]  # rdm_patch:[margin_ratio,patch_ratio,size_min]
    # ae_var['ratio'] = 1.0
    ae_var['batch_size'] = 2
    # ae_var['process_dict'] = {"rdm_flip": False, 'rdm_br_ct': True, 'rdm_blur': True,
    #                           'rdm_angle': True, 'rdm_noise': False, 'rdm_shift': True,
    #                           'rdm_patch': False, 'rdm_perlin':False, 'rdm_light_defect':False
    #                           }
    # light_defect_dict = dict(area_range=[500, 1000], defect_num=3, pixel_range=[7, 12], zoom_in=60, p=0.8)
    # ae_var['setting_dict'] = {'rdm_shift': 0.05, 'rdm_angle': 3, 'rdm_patch': rdm_patch, 'rdm_br_ct':[0.05,0.1],
    #                           'rdm_light_defect':light_defect_dict}
    ae_var['train_pipelines'] = [
        dict(type='CvtColor', to_rgb=True),
        dict(type='RandomBrightnessContrast', br_ratio=0.1, ct_ratio=0.3, p=p),
        dict(type='RandomBlur', p=p),
        # dict(type='RandomHorizontalFlip', p=p),
        # dict(type='RandomVerticalFlip', p=p),
        dict(type='RandomRotation', degrees=5, p=p),
        dict(type='Resize', height=int(height * 1.05), width=int(width * 1.05)),
        dict(type='RandomCrop', height=height, width=width),
        dict(type='Norm')
    ]
    ae_var['val_pipelines'] = [
        dict(type='CvtColor', to_rgb=True),
        dict(type='Resize', height=height, width=width),
        dict(type='Norm')
    ]
    # ae_var['aug_times'] = 2
    ae_var['target'] = {'type': 'loss', 'value': 95, 'hit_target_times': 2}

    #----replace args
    for key,value in kwargs.items():
        ae_var[key] = value

    return ae_var

def get_SEG_arg(**kwargs):
    # ----SEG
    seg_var = dict()
    hole_dict = dict(
        defect_num=1,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train\defectCrop2png\hole",
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\hole"
        ],
        label_class=1, p=0.6,
    )
    gold_particle_dict = dict(
        defect_num=1,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train\defectCrop2png\gold_particle",
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\gold_particle"
        ],
        label_class=2, p=0.6,
        # margin=5,homogeneity_threshold=1.0,  print_out=False
    )
    particle_dict = dict(
        defect_num=2,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train\defectCrop2png\particle",
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\particle"
        ],
        label_class=3, p=0.6,
    )
    defect_but_ok_dict = dict(
        defect_num=2,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\gold_residual_but_ok",
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\defect_but_ok"
        ],
        label_class=0, p=0.5,
    )
    light_defect_but_ok_dict = dict(
        defect_num=1,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\defect_parts\小紅點_淺瑕疵\defectCrop2png",
            # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\defect_but_ok"
        ],
        label_class=0, p=0.5,defect_diff_value=4,
    )
    area_range = [100, 1000]
    zoom_in = 70
    p = 0.7
    seg_var['train_pipelines'] = [
        dict(type='CvtColor', to_rgb=True),
        # dict(type='RandomBlur', p=p),
        dict(type='RandomVividDefect', **hole_dict),
        dict(type='RandomVividDefect', **gold_particle_dict),
        dict(type='RandomVividDefect', **particle_dict),
        dict(type='RandomVividDefect', **defect_but_ok_dict),
        dict(type='RandomVividLightDefect', **light_defect_but_ok_dict),
        dict(type='RandomBrightnessContrast', br_ratio=0.1, ct_ratio=0.3, p=p),
        # dict(type='RandomDefect', area_range=area_range, defect_num=1, pixel_range=[200, 250], zoom_in=zoom_in, p=p,
        #      label_class=0),
        # dict(type='RandomDefect', area_range=area_range, defect_num=2, pixel_range=[5, 50], zoom_in=zoom_in, p=p),
        # dict(type='RandomDefect', area_range=area_range, defect_num=1, pixel_range=[50, 90], zoom_in=zoom_in, p=p),
        # dict(type='RandomDefect', area_range=area_range, defect_num=1, pixel_range=[90, 130], zoom_in=zoom_in, p=p),
        # dict(type='RandomDefect', area_range=area_range, defect_num=1, pixel_range=[130, 170], zoom_in=zoom_in, p=p),
        # dict(type='RandomDefect', area_range=area_range, defect_num=1, pixel_range=[170, 210], zoom_in=zoom_in, p=p),
        # dict(type='RandomPNoise', area_range=area_range, defect_num=1, zoom_in=zoom_in, p=p),
        # dict(type='RandomHorizontalFlip', p=p),
        # dict(type='RandomVerticalFlip', p=p),
        dict(type='RandomRotation', degrees=5, p=p),
        dict(type='Resize', height=int(height*1.05), width=int(width*1.05)),
        dict(type='RandomCrop', height=height, width=width),
        dict(type='Norm')
    ]
    seg_var['val_pipelines'] = [
        dict(type='CvtColor', to_rgb=True),
        dict(type='Resize', height=height, width=width),
        dict(type='Norm')
    ]
    # seg_var['id2class_name'] = r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\classnames.txt"
    # seg_var['train_img_seg_dir'] = [
    #     r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E01\02\train"
    #                                 ]
    # seg_var['test_img_seg_dir'] = [
    #     r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E01\02\test"
    #                                ]
    # seg_var['predict_img_dir'] = [r'D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E01\02\predict_img']
    seg_var['to_train_w_AE_paths'] = False
    # seg_var['infer_method'] = "Seg_pooling_net_V10"#'Seg_pooling_net_V4'#'mit_b0'#'Seg_pooling_net_V4'#'Seg_DifNet'
    seg_var['model_name'] = "type_5_10"
    seg_var['encode_dict'] = {
        "first_layer": {
            # "type": "dilated_downSampling",
            # "kernel": 5,
            # "filter": 8,
            # "ratio": [1, 2, 5]
        },
        'kernel_list': [3]*5,
        'filter_list': [32, 48, 64, 80, 96],
        'stride_list': [2]*5,
        'pool_type_list': ['max'],
        'pool_kernel_list': [3]*5,#[5, 5, 5, 5],
        'multi_ratio': 1,
        'activation':'relu'
    }
    seg_var['decode_dict'] = {
        'cnn_type':'',
        'kernel_list': [3]*5,
        'filter_list': [96, 80, 64, 48, 32],
        'stride_list': [2]*5,#[2, 2, 2, 2],
        'multi_ratio': 1,
        'activation': 'relu'
    }
    # seg_var['rot'] = True
    seg_var['kernel_list'] = [3, 3]
    seg_var['filter_list'] = [64, 128]
    seg_var['pool_type'] = ['cnn']
    seg_var['pool_kernel'] = [2]
    seg_var['loss_method'] = "cross_entropy"
    seg_var['opti_method'] = "adam"
    seg_var['learning_rate'] = 1e-4
    # ====train
    seg_var['ratio'] = 1.0
    seg_var['batch_size'] = 2

    # perlin_dict = dict(mode='num_range',area_range=[30,500],
    #                    defect_num=5,pixel_range=[180,250])

    # seg_var['process_dict'] = {"rdm_flip": False,
    #                            'rdm_br_ct': False,
    #                            'rdm_blur': True,
    #                            'rdm_angle': True,
    #                            'rdm_shift': True,
    #                            'rdm_perlin':False
    #                            }
    # seg_var['setting_dict'] = {'rdm_shift': 0.1,
    #                            'rdm_angle': 5,
    #                            'rdm_br_ct': [0.05, 0.05],
    #                            'rdm_perlin':perlin_dict}
    # seg_var['aug_times'] = 2
    seg_var['target_of_best'] = 'recall+sensitivity'
    # seg_var['eval_epochs'] = 2
    # ----replace args
    for key, value in kwargs.items():
        seg_var[key] = value

    return seg_var

def get_commom_arg(**kwargs):
    para_dict = dict()
    para_dict['preprocess_dict'] = {'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}
    # {'ct_ratio': 1.48497, 'bias': 0.25, 'br_ratio': 0.25098}
    # #default {'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}

    # para_dict['show_data_qty'] = True
    para_dict['learning_rate'] = 1e-4
    para_dict['epochs'] = 200
    # ----train
    para_dict['eval_epochs'] = 2
    para_dict['GPU_ratio'] = None

    # para_dict['save_dir'] = r"D:\code\model_saver\AE_Seg_109"
    para_dict['save_pb_name'] = 'infer'
    para_dict['encript_flag'] = True
    para_dict['add_name_tail'] = True
    para_dict['print_out'] = True
    para_dict['to_read_manual_cmd'] = True
    para_dict['to_fix_ae'] = True
    para_dict['to_fix_seg'] = False
    para_dict['use_previous_settings'] = False

    for key, value in kwargs.items():
        para_dict[key] = value

    return para_dict

if __name__ == "__main__":
    height = 128
    width = 128
    #----AE args
    train_img_dir = [
        # r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\training\L1_OK_無分類",
        # r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\training\L2_OK_無分類",
        # r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\training\L4_OK_無分類"
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E06\train\OK",
        # r"D:\dataset\optotech\009IRC-FB\AE\train",
        # r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\training\L2_OK_晶紋"
        r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\ok_parts_train",
        r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220527_比例更改的圖片\OK_train",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220901_AOI判定OK\OK(多區OK)\train"

    ]
    test_img_dir = [
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E06\test\OK",
        # r"D:\dataset\optotech\009IRC-FB\AE\test",
        # r"D:\dataset\optotech\009IRC-FB\AE\test_L1_L2_L4",
        # r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\validation\L1_OK_無分類",
        # r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\validation\L2_OK_無分類",
        # r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\validation\L4_OK_無分類",
        r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\ok_parts_test",
        r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220527_比例更改的圖片\OK_test",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220901_AOI判定OK\OK(多區OK)\test"
    ]
    # recon_img_dir = r"D:\dataset\optotech\009IRC-FB\AE\recon4train"
    # recon_img_dir = r"D:\dataset\optotech\009IRC-FB\AE\recon4train_L4"
    recon_img_dir = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\recon_img"



    #----SEG args
    seg_dict = dict()
    seg_dict['height'] = height
    seg_dict['width'] = width
    seg_dict['train_img_seg_dir'] = [
        # r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\AE_Seg\Seg\train"
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E01\02\train",
        # r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train',
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220408新增破洞+金顆粒 資料\1",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220408新增破洞+金顆粒 資料\2",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_矽電Label_Tidy_data\VRS_Json\train",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\NG(多區NG-輕嚴重)_20220504\selected"
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220901_AOI判定OK\OK(多區OK)\train",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\ok_parts_train",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220527_比例更改的圖片\OK_train",
        r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220901_AOI判定OK\OK(多區OK)\num_100",
        r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220901_AOI判定OK\OK(多區OK)\train",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train_one_defect_class",

    ]
    seg_dict['test_img_seg_dir'] = [
        # r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\AE_Seg\Seg\test"
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E01\02\test",
        r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\test',
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_矽電Label_Tidy_data\VRS_Json\test",

        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\test_one_defect_class",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220408新增破洞+金顆粒 資料\1\one_defect_class",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220408新增破洞+金顆粒 資料\2\one_defect_class"
    ]
    # seg_dict['ok_img_seg_dir'] =

    seg_dict['predict_img_dir'] = [
      # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\defect_parts\小紅點_淺瑕疵\only_fig",
        r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\predict_img"
    ]

    seg_dict['id2class_name'] = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\classnames.txt"
    # seg_dict['id2class_name'] = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\classnames_one_defect_class.txt"
    # id2class_name = r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\AE_Seg\Seg\classnames.txt"
    #r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\classnames.txt"
    # seg_dict['self_create_label'] = False
    # seg_dict['train_with_aug_v2'] = True

    #----common var
    c_dict = dict(
        model_shape=[None, height, width, 3],  # [None, 512, 832, 3]
        preprocess_dict = {'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0},
        epochs=4,
        GPU_ratio=None,
        save_dir=r"D:\code\model_saver\AE_Seg_test",
        to_fix_ae=True,
        to_fix_seg=False,
        encript_flag=False,
                  )
    # {'ct_ratio': 1.48497, 'bias': 0.25, 'br_ratio': 0.25098}
    # #default {'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}


    ae_var = get_AE_arg(train_img_dir=train_img_dir, test_img_dir=test_img_dir, recon_img_dir=recon_img_dir,
                        )

    seg_var = get_SEG_arg(**seg_dict)

    para_dict = get_commom_arg(ae_var=ae_var, seg_var=seg_var, **c_dict)


    AE_Seg_train = AE_Seg(para_dict)
    # AE_Seg_train = AE_Seg_v2(para_dict)
    if AE_Seg_train.status:
        AE_Seg_train.model_init()
        AE_Seg_train.train()