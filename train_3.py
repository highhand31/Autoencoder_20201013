from AE_class_2 import AE



#D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\OK_train

if __name__ == "__main__":
    para_dict = dict()
    dirs = list()


    #----class init
    para_dict['train_img_source'] = [
        r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\train\OK_pure",

                ]
    para_dict['vali_img_source'] = [
        r'D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\vali\OK',
        # r'D:\dataset\optotech\CMIT_009IRC\only_OK\test\L2_OK',
        # r'D:\dataset\optotech\CMIT_009IRC\only_OK\test\L4_OK',
        ]
    para_dict['recon_img_dir'] = r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\recon"


    #----model init
    para_dict['model_shape'] = [None, 192, 192, 3]
    para_dict['preprocess_dict'] = {"rot": False, 'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}
    para_dict['infer_method'] = "AE_pooling_net"#"AE_JNet"#"AE_transpose_4layer"
    para_dict['kernel_list'] = [7,5,5,3,3]
    # para_dict['filter_list'] = [64,96,144,192,256]
    para_dict['filter_list'] = [32,64,96,128,256]
    para_dict['conv_time'] = 1
    para_dict['embed_length'] = 144
    para_dict['scaler'] = 1
    para_dict['pool_type'] = ['max','ave']
    para_dict['pool_kernel'] = [7,2]
    para_dict['activation'] = 'relu'
    para_dict['loss_method'] = "ssim"
    para_dict['loss_method_2'] = None
    para_dict['opti_method'] = "adam"
    para_dict['learning_rate'] = 1e-4
    para_dict['save_dir'] = r"D:\code\model_saver\AE_st2118_9"


    para_dict['epochs'] = 1200
    #----train
    para_dict['eval_epochs'] = 2
    para_dict['GPU_ratio'] = None
    para_dict['batch_size'] = 8
    para_dict['ratio'] = 1.0
    para_dict['setting_dict'] = {'rdm_shift': 0.03, 'rdm_angle': 5,'rdm_patch':[0.25,0.3]}#rdm_patch:[margin_ratio,patch_ratio]

    process_dict = {"rdm_flip": True, 'rdm_br': True, 'rdm_crop': False, 'rdm_blur': True,
                    'rdm_angle': False,
                    'rdm_noise': False,
                    'rdm_shift': True,
                    'rdm_patch': True,
                    }

    if True in process_dict.values():
        pass
    else:
        process_dict = None
    para_dict['process_dict'] = process_dict

    AE_train = AE(para_dict)
    AE_train.model_init(para_dict)
    AE_train.train(para_dict)