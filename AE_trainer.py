from AE_class_2 import AE

def get_tempalte_config():
    tech_type = "type_1"
    height = 544
    width = 832
    kernel_list = [7, 5, 5, 5, 3, 3, 3]
    filter_list = [16, 24, 32, 40, 48, 56, 64]
    stride_list = [2] * 7
    multi_ratio = 1.5
    config_dict = dict()
    rdm_patch = [0.25, 0.3, 10]  # rdm_patch:[margin_ratio,patch_ratio,size_min]

    config_dict['must_list'] = ['train_img_dir', 'model_name', 'save_dir', 'epochs']
    # ----about the model
    config_dict['height'] = height
    config_dict['width'] = width
    config_dict['tech_type'] = tech_type
    config_dict['preprocess_dict'] = {'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}
    config_dict['model_name'] = tech_type + "_7"
    config_dict['rot'] = False

    config_dict['encode_dict'] = {
        'kernel_list': kernel_list,
        'filter_list': filter_list,
        'stride_list': stride_list,
        'pool_type_list': ['cnn', 'ave'],  # 'cnn'
        'pool_kernel_list': kernel_list,
        'layer_list': [6],
        'multi_ratio': multi_ratio,
    }
    config_dict['decode_dict'] = {
        'pool_type_list': ['cnn', 'ave'],
        'cnn_type': 'resnet',
        'kernel_list': kernel_list[::-1],
        'filter_list': filter_list[::-1],
        'stride_list': stride_list,
        'multi_ratio': multi_ratio,
    }
    config_dict['activation'] = 'relu'
    config_dict['loss_method'] = "ssim"
    config_dict['opti_method'] = "adam"
    config_dict['learning_rate'] = 1e-4

    # ----about the training
    config_dict['val_pipelines'] = [
        dict(type='CvtColor', to_rgb=True),
        dict(type='Resize', height=height, width=width),
        dict(type='Norm')
    ]
    config_dict['ratio'] = 1.0
    config_dict['batch_size'] = 4
    config_dict['epochs'] = 10
    config_dict['aug_times'] = 2
    config_dict['process_dict'] = {"rdm_flip": True, 'rdm_br': True, 'rdm_blur': True,
                                   'rdm_angle': True,
                                   'rdm_noise': False,
                                   'rdm_shift': True,
                                   'rdm_patch': False,
                                   }
    config_dict['setting_dict'] = {'rdm_shift': 0.1, 'rdm_angle': 10, 'rdm_patch': rdm_patch}
    config_dict['GPU_ratio'] = None

    # ----about weights
    config_dict['target'] = {'type': 'loss', 'value': 99.7, 'hit_target_times': 2}
    config_dict['save_period'] = 1
    config_dict['eval_epochs'] = 1
    config_dict['save_pb_name'] = 'inference'
    config_dict['add_name_tail'] = True
    config_dict['encript_flag'] = True

    # ----about the system
    config_dict['print_out'] = False
    config_dict['show_data_qty'] = False
    config_dict['show_parameters'] = False

    # ----others
    config_dict['special_img_ratio'] = 0.05

    return config_dict

def set_user_config():
    height = 128
    width = 128
    train_img_dir = "D:\dataset\optotech\silicon_division\ST_2118\ST-2118 (Chip) 資料集\OK"
    test_img_dir = "D:\dataset\optotech\silicon_division\ST_2118\ST-2118 (Chip) 資料集\其他(不明NG)"
    # special_img_dir = ""
    save_dir = "D:\code\model_saver\AE_test"
    epochs = 50
    model_name = "type_1_7"
    # encript_flag = True#將訓練結果的相關資料加密
    # print_out = False#列印程式細節(通常給USER使用時設定False)
    # show_parameters = False  # 是否要讓AIE列印出config內容

    # model_shape = [None,height,width,3]
    target = dict(type='loss',value=99.7,hit_target_times=2)

    return dict(
        height=height,
        width=width,
        model_name=model_name,
        train_img_dir=train_img_dir,
        test_img_dir=test_img_dir,
        # special_img_dir=special_img_dir,
        save_dir=save_dir,
        epochs=epochs,
        target=target,
        # encript_flag=encript_flag,
        # print_out=True,
        # show_parameters=show_parameters

    )

if __name__ == "__main__":

    template_config = get_tempalte_config()
    user_config = set_user_config()

    tech = AE(template_config, user_dict=user_config)
    # AE_Seg_train = AE_Seg_v2(para_dict)
    if tech.status:
        tech.model_init()
        tech.train()