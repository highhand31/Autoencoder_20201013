from AE_class_2 import AE



if __name__ == "__main__":
    para_dict = dict()

    #----class init
    para_dict['train_img_dir'] = [
        r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\train\OK_pure",
        r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\train\OK_stain",

                ]
    para_dict['test_img_dir'] = [
        r'D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\vali\OK',

        ]
    para_dict['special_img_dir'] = None# [r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\train\OK_pure"]
    para_dict['recon_img_dir'] = None#r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\recon"

    #----model init
    para_dict['model_shape'] = [None, 128, 128, 3]
    para_dict['preprocess_dict'] = {'ct_ratio': 1, 'bias': 0.5, 'br_ratio': 0}
    para_dict['model_name'] = "type_1_0"
    para_dict['save_dir'] = r"D:\code\model_saver\AE_st2118_test"
    para_dict['save_pb_name'] = 'pb_model'
    para_dict['add_name_tail'] = False
    para_dict['encript_flag'] = False
    para_dict['print_out'] = True

    #----train
    para_dict['epochs'] = 1200
    para_dict['GPU_ratio'] = None
    para_dict['batch_size'] = 8
    para_dict['ratio'] = 1.0
    para_dict['target'] = {'type':'loss','value':1.0,'hit_target_times':2}

    AE_train = AE(para_dict)
    if AE_train.status is True:
        AE_train.model_init(para_dict)
        AE_train.train(para_dict)