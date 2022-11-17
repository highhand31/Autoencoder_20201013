from Seg import Seg

def get_tempalte_config(**kwargs):
    # ----SEG
    height = 128
    width = 128
    seg_var = dict()
    hole_dict = dict(
        defect_num=1,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\train\defectCrop2png\hole",
            r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\hole"
        ],
        label_class=1, p=0.6,
    )
    gold_particle_dict = dict(
        defect_num=1,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\train\defectCrop2png\gold_particle",
            r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\20220818_AOI_NG\defectCrop2png\gold_particle"
        ],
        label_class=2, p=0.6,
        # margin=5,homogeneity_threshold=1.0,  print_out=False
    )
    particle_dict = dict(
        defect_num=2,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\train\defectCrop2png\particle",
            r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\20220818_AOI_NG\defectCrop2png\particle"
        ],
        label_class=3, p=0.6,
    )
    defect_but_ok_dict = dict(
        defect_num=2,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\20220818_AOI_NG\defectCrop2png\gold_residual_but_ok",
            r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\20220818_AOI_NG\defectCrop2png\defect_but_ok"
        ],
        label_class=0, p=0.5,
    )
    light_defect_but_ok_dict = dict(
        defect_num=1,
        defect_png_source=[
            r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\defect_parts\小紅點_淺瑕疵\defectCrop2png",
            # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220818_AOI_NG\defectCrop2png\defect_but_ok"
        ],
        label_class=0, p=0.5, defect_diff_value=4,
    )
    area_range = [100, 1000]
    zoom_in = 70
    p = 0.7
    seg_var['train_pipelines'] = [
        dict(type='CvtColor', to_rgb=True),
        dict(type='RandomBlur', p=p),
        # dict(type='RandomVividDefect', **hole_dict),
        # dict(type='RandomVividDefect', **gold_particle_dict),
        # dict(type='RandomVividDefect', **particle_dict),
        # dict(type='RandomVividDefect', **defect_but_ok_dict),
        # dict(type='RandomVividLightDefect', **light_defect_but_ok_dict),
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
        dict(type='Resize', height=int(height * 1.05), width=int(width * 1.05)),
        dict(type='RandomCrop', height=height, width=width),
        dict(type='Norm')
    ]
    seg_var['val_pipelines'] = [
        dict(type='CvtColor', to_rgb=True),
        dict(type='Resize', height=height, width=width),
        dict(type='Norm')
    ]

    # seg_var['infer_method'] = "Seg_pooling_net_V10"#'Seg_pooling_net_V4'#'mit_b0'#'Seg_pooling_net_V4'#'Seg_DifNet'
    seg_var['model_name'] = "type_5_8"
    seg_var['encode_dict'] = {
        "first_layer": {
            # "type": "dilated_downSampling",
            # "kernel": 5,
            # "filter": 8,
            # "ratio": [1, 2, 5]
        },
        'kernel_list': [3] * 5,
        'filter_list': [32, 48, 64, 80, 96],
        'stride_list': [2] * 5,
        'pool_type_list': ['max','resize','cnn'],
        'pool_kernel_list': [3] * 5,  # [5, 5, 5, 5],
        'multi_ratio': 1,
        'activation': 'relu'
    }
    seg_var['decode_dict'] = {
        'cnn_type': '',
        'kernel_list': [3] * 5,
        'filter_list': [96, 80, 64, 48, 32],
        'stride_list': [2] * 5,  # [2, 2, 2, 2],
        'multi_ratio': 1,
        'activation': 'relu'
    }
    seg_var['loss_method'] = "cross_entropy"
    seg_var['opti_method'] = "adam"
    seg_var['learning_rate'] = 1e-4
    # ====train
    seg_var['ratio'] = 1.0
    seg_var['batch_size'] = 2
    seg_var['target_of_best'] = 'recall+sensitivity'

    seg_var['learning_rate'] = 1e-4
    seg_var['epochs'] = 200
    # ----train
    seg_var['eval_epochs'] = 2
    seg_var['GPU_ratio'] = None

    seg_var['save_pb_name'] = 'infer'
    seg_var['add_name_tail'] = True
    seg_var['to_read_manual_cmd'] = True
    seg_var['encript_flag'] = True
    seg_var['print_out'] = True
    seg_var['show_parameters'] = False

    if len(kwargs):
        for key, value in kwargs.items():
            seg_var[key] = value

    return seg_var

def set_user_config(**kwargs):
    seg_dict = dict()
    seg_dict['height'] = 512
    seg_dict['width'] = 832
    seg_dict['train_img_seg_dir'] = [
        r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\train",
        r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\20220408新增破洞+金顆粒 資料\1\train",
        r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\20220408新增破洞+金顆粒 資料\2\train"
    ]
    seg_dict['test_img_seg_dir'] = [
        r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\test"
    ]

    # seg_dict['predict_img_dir'] = [
    #     r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\predict_img"
    # ]

    seg_dict['id2class_name'] = r"D:\dataset\optotech\silicon_division\PDAP\PD_55077\classnames.txt"
    seg_dict["epochs"] = 96
    seg_dict["save_dir"] = r"D:\code\model_saver\AE_Seg_test_3"

    seg_dict["encript_flag"] = False
    seg_dict["print_out"] = True

    return seg_dict


if __name__ == "__main__":

    template_config = get_tempalte_config()
    user_config = set_user_config()

    tech = Seg(template_config,user_dict=user_config)
    # AE_Seg_train = AE_Seg_v2(para_dict)
    if tech.status:
        tech.model_init()
        tech.train()