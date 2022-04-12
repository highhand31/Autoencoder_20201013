from AE_post_process import AE_crop_defects




if __name__ == "__main__":
    img_source = r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\vali"
    # pb_path = r"D:\code\model_saver\Opto_tech\AE_PDAP_20220103\infer_93.04.nst"
    pb_path = r"D:\code\model_saver\AE_st2118_22\pb_model.pb"
    save_dir = r"D:\dataset\optotech\silicon_division\ST_2118\旺矽_st2118_20211014\分類_20211123\results\AE_2118"

    diff_th = 30
    cc_th = 30
    zoom_in_value = 5
    batch_size = 16

    node_dict = {'input': 'input:0',
                 'loss': "loss_AE:0",
                 'embeddings': 'embeddings:0',
                 'output': "output_AE:0"
                 }

    process_dict = {'ave_filter': False, 'gau_filter': False}
    setting_dict = {'ave_filter': (3, 3), 'gau_filter': (3, 3)}


    AE_crop_defects(img_source, pb_path, diff_th, cc_th, batch_size=batch_size, zoom_in_value=zoom_in_value,
                    node_dict=node_dict, process_dict=process_dict, setting_dict=setting_dict,
                  read_subdir=True,save_dir=save_dir)
