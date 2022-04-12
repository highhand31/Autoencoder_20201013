from AE_class_2 import AE



#D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\OK_train

if __name__ == "__main__":
    para_dict = dict()
    dirs = list()

    #----class init
    para_dict['train_img_dir'] = [
        r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\train\OK_pure",
        r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\train\OK_stain",

                ]
    para_dict['test_img_dir'] = [
        r'D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\vali\OK',
        # r'D:\dataset\optotech\CMIT_009IRC\only_OK\test\L2_OK',
        # r'D:\dataset\optotech\CMIT_009IRC\only_OK\test\L4_OK',
        ]
    para_dict['special_img_dir'] = [r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\train\OK_pure"]
    para_dict['recon_img_dir'] = r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\recon"
    para_dict['show_data_qty'] = True

    #----model init
    para_dict['model_shape'] = [None, 192, 192, 3]
    para_dict['preprocess_dict'] = {'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}
    # para_dict['infer_method'] = "AE_pooling_net"#"AE_JNet"#"AE_transpose_4layer"
    para_dict['model_name'] = "type_1_1"
    para_dict['rot'] = False
    para_dict['kernel_list'] = [7,5,3,3,3]
    # para_dict['filter_list'] = [96,128,160,192,320]
    para_dict['filter_list'] = [32,64,96,128,256]
    para_dict['conv_time'] = 1
    para_dict['embed_length'] = 144
    para_dict['scaler'] = 1
    para_dict['pool_type'] = ['max']
    para_dict['pool_kernel'] = [7,2]
    para_dict['activation'] = 'relu'
    para_dict['loss_method'] = "ssim"
    para_dict['loss_method_2'] = None
    para_dict['opti_method'] = "adam"
    para_dict['learning_rate'] = 1e-4
    # para_dict['save_dir'] = r"D:\code\model_saver\AE_st2118_22"
    para_dict['save_dir'] = r"D:\code\model_saver\AE_st2118_test"
    para_dict['save_pb_name'] = 'pb_model'
    para_dict['save_period'] = 1
    para_dict['add_name_tail'] = False
    para_dict['encript_flag'] = False
    para_dict['print_out'] = True
    para_dict['dtype'] = 'float32'

    #----train
    para_dict['epochs'] = 1200
    para_dict['eval_epochs'] = 2
    para_dict['GPU_ratio'] = None
    para_dict['batch_size'] = 8
    para_dict['ratio'] = 1.0
    para_dict['setting_dict'] = {'rdm_shift': 0.1, 'rdm_angle': 10,'rdm_patch':[0.25,0.3,10]}#rdm_patch:[margin_ratio,patch_ratio]

    process_dict = {"rdm_flip": True, 'rdm_br': True,  'rdm_blur': True,
                    'rdm_angle': True,
                    'rdm_noise': False,
                    'rdm_shift': True,
                    'rdm_patch': False,
                    }

    if True in process_dict.values():
        pass
    else:
        process_dict = None
    para_dict['process_dict'] = process_dict

    para_dict['target'] = {'type':'loss','value':0.95,'hit_target_times':2}

    AE_train = AE(para_dict)
    if AE_train.status is True:
        AE_train.model_init(para_dict)
        AE_train.train(para_dict)