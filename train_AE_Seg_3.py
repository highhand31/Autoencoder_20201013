from AE_Seg import AE_Seg

def get_AE_arg():
    # ----AE
    ae_var = dict()
    ae_var['train_img_dir'] = [
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\ok_parts_train",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220527_比例更改的圖片\OK_train",
        # r"D:\dataset\optotech\silicon_division\PDAP\藥水殘(已完成)\train",
        # r"D:\dataset\optotech\014\20220429_preprocess\train_ok"
        # r"D:\dataset\optotech\014\20220506_preprocess\train_ok"
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\AE\train_OK"
        # r"D:\dataset\optotech\014\20220524_preprocess\014IRA-T_0.0.6_data\OK_train"
        r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E06\train\OK",
    ]

    ae_var['test_img_dir'] = [
        # r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\ok_parts_test',
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220527_比例更改的圖片\OK_test",
        # r'D:\dataset\optotech\silicon_division\PDAP\藥水殘(已完成)\test',
        # r"D:\dataset\optotech\014\20220429_preprocess\test_ok"
        # r"D:\dataset\optotech\014\20220506_preprocess\test_ok"
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\AE\test_OK"
        # r"D:\dataset\optotech\014\20220524_preprocess\014IRA-T_0.0.6_data\OK_test"
        r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E06\test\OK",
    ]
    # ae_var['recon_img_dir'] = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\recon_img"
    # ae_var['recon_img_dir'] = r"D:\dataset\optotech\014\20220429_preprocess\recon_img"
    # ae_var['recon_img_dir'] = r"D:\dataset\optotech\014\20220506_preprocess\recon_img"
    # ae_var['recon_img_dir'] = r"D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\predict_img"
    # ae_var['recon_img_dir'] = r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\AE\recon_img"
    # ae_var['recon_img_dir'] = r"D:\dataset\optotech\014\20220524_preprocess\014IRA-T_0.0.6_data\recon_img"
    ae_var['recon_img_dir'] = r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E06\recon_img"
    # ====model init
    ae_var['model_shape'] = [None, 512, 832, 3]#[None,544,832,3]
    ae_var['infer_method'] = "AE_pooling_net_V7"  # "AE_JNet"#"AE_transpose_4layer"
    ae_var['encode_dict'] = {
        # 'first_layer':{'type':'resize', 'size':[272,416]},
        # 'first_layer':{'type':'patch_embedding', 'patch_size':2,'filter':8},
        # 'first_layer': {'type': 'CNN_downSampling', 'kernel': 5, 'filter': 16},
        # 'first_layer': {'type': 'mixed_downSampling', 'kernel': 5, 'filter': 16, 'pool_type_list':['max','cnn']},
        # 'first_layer': {'type': 'dilated_downSampling', 'kernel': 5, 'filter': 8, 'ratio':[1,2,5,9]},
        'kernel_list': [7, 5, 5, 5, 3, 3, 3],
        'filter_list': [16, 24, 32, 40, 48, 56, 64],
        'stride_list': [2, 2, 2, 2, 2, 2, 2],
        'pool_type_list': ['cnn','max'],
        'pool_kernel_list': [7, 5, 5, 5, 3, 3, 3],
        'layer_list': [6],
        'multi_ratio':1.5,
    }
    ae_var['decode_dict'] = {
        'pool_type_list': ['cnn','max'],
        'cnn_type':'resnet',
        'kernel_list': [3, 3, 3, 5, 5, 5, 7],
        'filter_list': [64, 56, 48, 40, 32, 24, 16],
        'stride_list': [2, 2, 2, 2, 2, 2, 2],
        'multi_ratio': 1.5,
    }
    ae_var['to_reduce'] = False
    ae_var['rot'] = False
    ae_var['kernel_list'] = [7, 5, 5, 3, 3]
    ae_var['filter_list'] = [16, 32, 48, 64, 128]
    ae_var['conv_time'] = 1
    ae_var['embed_length'] = 144
    ae_var['scaler'] = 1
    ae_var['pool_type'] = ['max','ave']  # ['max']
    ae_var['pool_kernel'] = [7, 2]  # [7, 2]
    ae_var['activation'] = 'relu'
    ae_var['loss_method'] = "ssim"
    ae_var['opti_method'] = "adam"

    # ====train
    rdm_patch = [0.25, 0.3, 10]  # rdm_patch:[margin_ratio,patch_ratio,size_min]
    ae_var['ratio'] = 1.0
    ae_var['batch_size'] = 8
    ae_var['process_dict'] = {"rdm_flip": True, 'rdm_br': True, 'rdm_blur': True,
                              'rdm_angle': True, 'rdm_noise': False, 'rdm_shift': True,
                              'rdm_patch': True, 'rdm_perlin':False
                              }
    ae_var['setting_dict'] = {'rdm_shift': 0.05, 'rdm_angle': 3, 'rdm_patch': rdm_patch}
    ae_var['aug_times'] = 2
    ae_var['target'] = {'type': 'loss', 'value': 1.0, 'hit_target_times': 2}

    return ae_var

def get_SEG_arg():
    # ----SEG
    seg_var = dict()
    # seg_var['id2class_name'] = {0:'_background_',1:'hole',2:'gold_particle',3:'particle'}
    # seg_var['id2class_name'] = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\classnames.txt"
    seg_var['id2class_name'] = r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\classnames.txt"
    # seg_var['id2class_name'] = r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\classnames.txt"
    # seg_var['id2class_name'] = {0:'_background_',1:'potion'}
    seg_var['train_img_seg_dir'] = [
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\PD-55092G-AP_0.0.1_dataset\Seg_data\gold_particle\train",
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\PD-55092G-AP_0.0.1_dataset\Seg_data\hole\train",
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\PD-55092G-AP_0.0.1_dataset\Seg_data\Particle\train",
        # r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train',
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220408新增破洞+金顆粒 資料\1",
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220408新增破洞+金顆粒 資料\2"
        # r'D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\train'
        r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E01\02\train"
                                    ]
    seg_var['test_img_seg_dir'] = [
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\PD-55092G-AP_0.0.1_dataset\Seg_data\gold_particle\test",
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\PD-55092G-AP_0.0.1_dataset\Seg_data\hole\test",
        # r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\PD-55092G-AP_0.0.1_dataset\Seg_data\Particle\test",
        # r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\test',
        # r'D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\test'
        r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E01\02\test"
                                   ]
    seg_var['predict_img_dir'] = [r'D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E01\02\predict_img']
    seg_var['to_train_w_AE_paths'] = False
    seg_var['infer_method'] = "Seg_pooling_net_V4"#'Seg_pooling_net_V4'#'mit_b0'#'Seg_pooling_net_V4'#'Seg_DifNet'
    seg_var['encode_dict'] = {
        "first_layer": {
            "type": "dilated_downSampling",
            "kernel": 5,
            "filter": 8,
            "ratio": [1, 2, 5]
        },
        'kernel_list': [3, 3, 3],
        'filter_list': [32, 48, 64],
        'stride_list': [2, 2, 2],
        'pool_type_list': ['max'],
        'pool_kernel_list': [5, 5, 5],
        'multi_ratio': 2,
        # 'kernel_list': [7, 5, 5, 5, 3, 3, 3],
        # 'filter_list': [16, 24, 32, 40, 48, 56, 64],
        # 'stride_list': [2, 2, 2, 2, 2, 2, 2],
        # 'pool_type_list': ['cnn', 'max'],
        # 'pool_kernel_list': [7, 5, 5, 5, 3, 3, 3],
        # 'layer_list': [3],
    }
    seg_var['decode_dict'] = {
        # 'pool_type_list': ['cnn','max'],
        # 'cnn_type':'resnet',
        # 'kernel_list': [3, 3, 3, 5, 5, 5, 7],
        # 'filter_list': [64, 56, 48, 40, 32, 24, 16],
        # 'stride_list': [2, 2, 2, 2, 2, 2, 2],
        # 'pool_type_list': ['max'],
        'cnn_type':'',
        'kernel_list': [3, 3, 3],
        'filter_list': [64, 48, 32],
        'stride_list': [2, 2, 2],
        'multi_ratio': 2,
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
    seg_var['batch_size'] = 1
    seg_var['setting_dict'] = {'rdm_shift': 0.05, 'rdm_angle': 5}

    seg_var['process_dict'] = {"rdm_flip": True,
                               'rdm_br': True,
                               'rdm_blur': True,
                               'rdm_angle': True,
                               'rdm_shift': True,
                               }
    seg_var['aug_times'] = 2
    seg_var['target_of_best'] = 'defect_recall'
    # seg_var['eval_epochs'] = 2

    return seg_var

if __name__ == "__main__":
    para_dict = dict()

    ae_var = get_AE_arg()

    seg_var = get_SEG_arg()

    #----common var
    para_dict['preprocess_dict'] = {'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}
    #{'ct_ratio': 1.48497, 'bias': 0.25, 'br_ratio': 0.25098}
    # #default {'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}

    para_dict['show_data_qty'] = True
    para_dict['learning_rate'] = 1e-4
    para_dict['epochs'] = 200
    #----train
    para_dict['eval_epochs'] = 2
    para_dict['GPU_ratio'] = None

    para_dict['ae_var'] = ae_var
    para_dict['seg_var'] = seg_var
    para_dict['save_dir'] = r"D:\code\model_saver\AE_Seg_109"
    para_dict['save_pb_name'] = 'infer'
    para_dict['encript_flag'] = True
    para_dict['add_name_tail'] = True
    para_dict['print_out'] = True
    para_dict['to_read_manual_cmd'] = True
    para_dict['to_fix_ae'] = True
    para_dict['to_fix_seg'] = False
    para_dict['use_previous_settings'] = False



    AE_Seg_train = AE_Seg(para_dict)
    if AE_Seg_train.status:
        AE_Seg_train.model_init(para_dict)
        AE_Seg_train.train(para_dict)