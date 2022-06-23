from AE_Seg_prediction import image_similarity


if __name__ == "__main__":
    # img_dir = r"D:\dataset\optotech\009IRC-FB\0.0.3.1_dataset\training\L2_NG_pad外圍殘膠"
    img_dir = r"D:\dataset\optotech\009IRC-FB\20220616-0.0.4.1-2\training\L2_OK_無分類"
    # img_dir = r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\正面\E05\1\MatchLight"
    # img_dir = r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\E02\3\MatchLight"
    # pb_path = r"D:\code\model_saver\AE_Seg_33\infer_best_epoch240.pb"
    # pb_path = r"D:\code\model_saver\AE_Seg_113\infer_91.87.nst"
    # pb_path = r"D:\code\model_saver\AE_Seg_114\infer_94.09.nst"
    # pb_path = r"D:\code\model_saver\AE_Seg_115\infer_91.79.nst"
    # pb_path = r"D:\code\model_saver\AE_Seg_22\pb_model.pb"
    pb_path = r"D:\code\model_saver\AE_Seg_114\infer_94.09.nst"
    node_dict = {
                 'input': 'input:0',
                 'input_ori': 'input_ori:0',
                 'recon': 'output_AE:0',
                 'loss': 'loss_AE:0',
                 }
    image_similarity(img_dir, pb_path, node_dict,to_save_recon=False)
