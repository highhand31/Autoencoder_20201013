from AE_Seg import AE_Seg



#D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\OK_train

if __name__ == "__main__":
    para_dict = dict()
    dirs = list()

    #----AE
    ae_var = dict()
    ae_var['train_img_dir'] = [
        r"D:\dataset\optotech\silicon_division\PDAP\藥水殘(已完成)\train",
                ]

    ae_var['test_img_dir'] = [
        r'D:\dataset\optotech\silicon_division\PDAP\藥水殘(已完成)\test',
        # r'D:\dataset\optotech\CMIT_009IRC\only_OK\test\L2_OK',
        # r'D:\dataset\optotech\CMIT_009IRC\only_OK\test\L4_OK',
        ]
    ae_var['recon_img_dir'] = r"D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\predict_img"
    #====model init
    ae_var['model_shape'] = [None, 544, 832, 3]
    ae_var['infer_method'] = "AE_pooling_net"  # "AE_JNet"#"AE_transpose_4layer"
    ae_var['rot'] = False
    ae_var['kernel_list'] = [7, 5, 5, 3, 3]
    # para_dict['filter_list'] = [64,96,144,192,256]
    ae_var['filter_list'] = [32, 64, 96, 128, 256]
    ae_var['conv_time'] = 1
    ae_var['embed_length'] = 144
    ae_var['scaler'] = 2
    ae_var['pool_type'] = ['max', 'ave']#['max']
    ae_var['pool_kernel'] = [7,2]#[7, 2]
    ae_var['activation'] = 'relu'
    ae_var['loss_method'] = "ssim"
    ae_var['opti_method'] = "adam"

    #====train
    # ae_var['pause_opt_ae'] = True
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

    #----SEG
    seg_var = dict()
    # seg_var['id2class_name'] = {0:'_background_',1:'hole',2:'gold_particle',3:'particle',4:'probe'}
    seg_var['id2class_name'] = {0:'_background_',1:'potion'}
    seg_var['train_img_seg_dir'] = [r'D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\train',
                                    # r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E01\02\train_OK_parts"
                                    ]
    seg_var['test_img_seg_dir'] = [r'D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\test',
                                   # r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\背面\19BR262E01\02\test_OK_parts"
                                   ]
    seg_var['predict_img_dir'] = [r'D:\dataset\optotech\silicon_division\PDAP\藥水殘(原圖+color圖+json檔)\L2_potion\predict_img']
    seg_var['to_train_w_AE_paths'] = True
    seg_var['infer_method'] = 'Seg_DifNet'
    seg_var['rot'] = True
    seg_var['kernel_list'] = [3,3]
    seg_var['filter_list'] = [64,128]
    seg_var['pool_type'] = ['cnn']
    seg_var['pool_kernel'] = [2]
    seg_var['loss_method'] = "cross_entropy"
    seg_var['opti_method'] = "adam"
    seg_var['learning_rate'] = 1e-4
    #====train
    # seg_var['pause_opt_seg'] = True
    seg_var['ratio'] = 1.0
    seg_var['batch_size'] = 2
    seg_var['setting_dict'] = {'rdm_shift': 0.1, 'rdm_angle': 10}


    seg_var['process_dict'] = {"rdm_flip": True, 'rdm_br': True, 'rdm_blur': True,
                    'rdm_angle': True,
                    'rdm_shift': True,

                    }
    seg_var['aug_times'] = 2
    # seg_var['eval_epochs'] = 2


    #----common var
    para_dict['preprocess_dict'] = {'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}
    para_dict['show_data_qty'] = True
    para_dict['learning_rate'] = 1e-4
    para_dict['epochs'] = 300
    #----train
    para_dict['eval_epochs'] = 2
    para_dict['GPU_ratio'] = None

    para_dict['ae_var'] = ae_var
    para_dict['seg_var'] = seg_var
    para_dict['save_dir'] = r"D:\code\model_saver\AE_Seg_22"
    para_dict['save_pb_name'] = 'pb_model'
    para_dict['add_name_tail'] = False
    para_dict['print_out'] = True
    # if True in process_dict.values():
    #     pass
    # else:
    #     process_dict = None
    # para_dict['process_dict'] = process_dict

    AE_Seg_train = AE_Seg(para_dict)
    if AE_Seg_train.status:
        AE_Seg_train.model_init(para_dict)
        AE_Seg_train.train(para_dict)