from AE_Seg_prediction import image_similarity


if __name__ == "__main__":
    img_dir = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\20220527_圖片分析\1CBR059E01(已確認)-11張"
    # img_dir = r"D:\dataset\optotech\silicon_division\PDAP\PD-55077GR-AP Al用照片\正面\E05\1\MatchLight"
    # img_dir = r"D:\dataset\optotech\silicon_division\PDAP\PD-55092\E02\3\MatchLight"
    # pb_path = r"D:\code\model_saver\AE_Seg_33\infer_best_epoch240.pb"
    pb_path = r"D:\code\model_saver\AE_Seg_105\infer_best_epoch39.pb"
    # pb_path = r"D:\code\model_saver\AE_Seg_22\pb_model.pb"
    node_dict = {
                 'input': 'input:0',
                 'input_ori': 'input_ori:0',
                 'recon': 'output_AE:0',
                 'loss': 'loss_AE:0',
                 }
    image_similarity(img_dir, pb_path, node_dict,to_save_recon=True)
