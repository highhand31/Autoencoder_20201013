from AE_Seg import AE_Seg

def get_AE_arg():
    # ----AE
    ae_var = dict()
    ae_var['train_img_dir'] = [
        # r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\ok_parts_train",
        r"D:\dataset\optotech\silicon_division\PDAP\藥水殘(已完成)\train",
    ]

    ae_var['test_img_dir'] = [
        # r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\ok_parts_test',
        r'D:\dataset\optotech\silicon_division\PDAP\藥水殘(已完成)\test',

    ]
    # ae_var['recon_img_dir'] = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\recon_img"
    ae_var['recon_img_dir'] = r"D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\predict_img"
    # ====model init
    ae_var['model_shape'] = [None, 544, 832, 3]
    ae_var['infer_method'] = "AE_pooling_net_V4"  # "AE_JNet"#"AE_transpose_4layer"
    ae_var['encode_dict'] = {
        'first_layer':{'type':'resize_maxpool', 'size':[272,416]},
        # 'first_layer':{'type':'patch_embedding', 'patch_size':2,'filter':8},
        'kernel_list': [7, 5],
        'filter_list': [32, 48],
        'stride_list': [4, 4],
        'pool_type_list': ['max', 'ave'],
        'pool_kernel_list': [7, 2],
    }
    ae_var['decode_dict'] = {
        'cnn_type':'resnet',
        'kernel_list': [5, 5, 7],
        'filter_list': [80, 64, 48],
        'stride_list': [4, 4, 2],
    }
    ae_var['rot'] = False
    ae_var['kernel_list'] = [7, 5, 5, 3, 3]
    # para_dict['filter_list'] = [64,96,144,192,256]
    ae_var['filter_list'] = [32, 64, 96, 128, 256]
    ae_var['conv_time'] = 1
    ae_var['embed_length'] = 144
    ae_var['scaler'] = 2
    ae_var['pool_type'] = ['max', 'ave']  # ['max']
    ae_var['pool_kernel'] = [7, 2]  # [7, 2]
    ae_var['activation'] = 'relu'
    ae_var['loss_method'] = "ssim"
    ae_var['opti_method'] = "adam"

    # ====train
    rdm_patch = [0.25, 0.3, 10]  # rdm_patch:[margin_ratio,patch_ratio,size_min]
    ae_var['ratio'] = 1.0
    ae_var['batch_size'] = 2
    ae_var['process_dict'] = {"rdm_flip": True, 'rdm_br': True, 'rdm_blur': True,
                              'rdm_angle': True,
                              'rdm_noise': False,
                              'rdm_shift': True,
                              'rdm_patch': True,
                              }
    ae_var['setting_dict'] = {'rdm_shift': 0.1, 'rdm_angle': 10, 'rdm_patch': rdm_patch}
    ae_var['aug_times'] = 2
    ae_var['target'] = {'type': 'loss', 'value': 1.0, 'hit_target_times': 2}

    return ae_var

def get_SEG_arg():
    # ----SEG
    seg_var = dict()
    # seg_var['id2class_name'] = {0:'_background_',1:'hole',2:'gold_particle',3:'particle'}
    # seg_var['id2class_name'] = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\classnames.txt"
    seg_var['id2class_name'] = {0:'_background_',1:'potion'}
    seg_var['train_img_seg_dir'] = [
        # r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train',
        r'D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\train'
                                    ]
    seg_var['test_img_seg_dir'] = [
        # r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\test',
        r'D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\test'
                                   ]
    seg_var['predict_img_dir'] = None  # [r'D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\predict_img']
    seg_var['to_train_w_AE_paths'] = True
    seg_var['infer_method'] = 'Seg_DifNet'
    seg_var['rot'] = True
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
    seg_var['setting_dict'] = {'rdm_shift': 0.1, 'rdm_angle': 10}

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
    para_dict['epochs'] = 300
    #----train
    para_dict['eval_epochs'] = 2
    para_dict['GPU_ratio'] = None

    para_dict['ae_var'] = ae_var
    para_dict['seg_var'] = seg_var
    para_dict['save_dir'] = r"D:\code\model_saver\AE_Seg_33"
    para_dict['save_pb_name'] = 'pb_model'
    para_dict['add_name_tail'] = False
    para_dict['print_out'] = True
    para_dict['to_read_manual_cmd'] = True
    para_dict['to_fix_ae'] = True
    para_dict['to_fix_seg'] = False
    para_dict['use_previous_settings'] = False


    AE_Seg_train = AE_Seg(para_dict)
    if AE_Seg_train.status:
        AE_Seg_train.model_init(para_dict)
        AE_Seg_train.train(para_dict)