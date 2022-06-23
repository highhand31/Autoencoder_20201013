from AE_post_process import AE_find_defects


if __name__ == "__main__":

    img_source = r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\validation\L2_OK_晶紋"
    pb_path = r"D:\\code\\model_saver\\AE_Seg_113\\infer_20220617151930.nst"

    save_dir = img_source

    diff_th = 50
    cc_th = 50

    img_ori2process = True
    img_ori_p_args = ['gau_blur',(3,3)]  # [process_type,kernel]

    recon2erode = True  # 這個是recon與原圖相減後的erode處理
    erode_args = [(3, 3), 2]  # [kernel,erode times]

    zoom_in_value = None#[33]*4#[45]*4#[75,77,88,88]#5 #[75,77,88,88]
    mask_json_path = None#r"D:\dataset\optotech\silicon_division\PDAP\top_view\PD-55077GR-AP AI Training_2022.01.26\AE_results_544x832_diff10cc60_more_filters\pred_ng_ans_ok\t0_12_-20_MatchLightSet_作用區_Ng_1.json"
    to_mask = False

    batch_size = 8

    node_dict = {'input': 'input:0',
                 'input_ori': 'input_ori:0',
                 'loss': "loss_AE:0",
                 'embeddings': 'embeddings:0',
                 'output': "output_AE:0"
                 }

    process_dict = {'ave_filter': False, 'gau_filter': False}
    setting_dict = {'ave_filter': (3, 3), 'gau_filter': (3, 3)}
    read_subdir = False

    AE_find_defects(img_source, pb_path, diff_th, cc_th, batch_size=batch_size, zoom_in_value=zoom_in_value,
                    to_mask=to_mask,
                    node_dict=node_dict, process_dict=process_dict, setting_dict=setting_dict, cc_type="",
                    save_type='compare', save_recon=True, read_subdir=read_subdir, mask_json_path=mask_json_path)
